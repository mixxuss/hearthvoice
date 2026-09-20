"""HearthVoice, offline.

Microphone to speaker through three models on the DGX Spark, with whatever
changes in the house published to a local Home Assistant over MQTT. No cloud
service appears anywhere in this pipeline, and the agent refuses to start if
anything is configured to leave the local network.

    python -m hearthvoice.agent               # talk to it
    python -m hearthvoice.agent --devices     # list the house and exit
    python -m hearthvoice.agent --check       # confirm every target is local
    python -m hearthvoice.agent --list-audio  # find a device index
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    OutputAudioRawFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.nvidia.tts import NvidiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from . import engines, tools
from .aec import ReferenceTap, SpeexEchoFilter
from .config import (
    ASR_MODEL,
    ASR_SERVER,
    EVAL_DIR,
    HA_URL,
    LLM_BASE_URL,
    LLM_MODEL,
    MQTT_HOST,
    MQTT_PORT,
    TTS_SERVER,
    TTS_VOICE,
    VIP,
    OfflineViolation,
    check_offline,
)
from .devices import Home
from .metrics import MetricsRecorder
from .mqtt_bridge import MqttBridge
from .prompt import build_system_prompt

# Parakeet only accepts 16 kHz. Magpie's model is natively 22050 Hz, and the
# speaker stream has to be told the same number the synthesiser is producing:
# if they disagree the reply plays at the wrong speed and ends early.
MIC_RATE = 16000
SPEAKER_RATE = 22050

log = logging.getLogger("hearthvoice")


class SpeechEndMarker(FrameProcessor):
    """Starts the clock, and the turn, when the user stops talking.

    Sits at the head of the pipeline, which is the only place
    UserStoppedSpeakingFrame can be seen before anything consumes it. It also
    advances the house's turn counter, which is what stops the model granting
    its own confirmations.
    """

    def __init__(self, recorder: MetricsRecorder, home: Home) -> None:
        super().__init__()
        self.recorder = recorder
        self.home = home
        self._heard_speech = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._heard_speech = True

        # A stop only ends a turn if a start preceded it. The detector also
        # emits a stop when the room falls quiet after the agent's own reply,
        # and starting the clock there measured the gap before the user spoke
        # again: nine seconds of silence reported as recognition latency.
        elif isinstance(frame, UserStoppedSpeakingFrame) and self._heard_speech:
            self._heard_speech = False
            turn = self.recorder.begin_turn()
            turn.speech_ended = time.perf_counter()
            self.home.begin_turn()

        await self.push_frame(frame, direction)


class TranscriptTimer(FrameProcessor):
    """Records the transcript and how long recognition took.

    This has to sit between the recogniser and the context aggregator, and the
    reason is the most expensive thing I got wrong in this project. Pipecat's
    aggregator consumes TranscriptionFrame and does not forward it
    (`llm_response_universal.py`, the `elif isinstance(frame,
    TranscriptionFrame)` branch). A timer placed after the aggregator, as this
    one was for two weeks, can never see a transcript.

    The cost was silent: five recorded sessions, forty-nine turns, every
    transcript an empty string and every end-to-end time None, while the unit
    tests for the arithmetic stayed green because they were handed numbers by
    hand.
    """

    def __init__(self, recorder: MetricsRecorder, home: Home) -> None:
        super().__init__()
        self.recorder = recorder
        self.home = home

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            turn = self.recorder.current
            # No asr_final_s here. In this pipeline the transcript arrives
            # before the frame that marks the end of the user's turn, so the
            # interval between them runs backwards, and measuring it forwards
            # from the previous turn's end reported the silence before the
            # user spoke as recognition latency: about nine seconds, which the
            # summary then averaged and published. A stage that cannot be
            # attributed is left empty and counted as missing.
            turn.transcript = frame.text.strip()
            self.home.note_command(turn.transcript)
            log.info("heard: %s", turn.transcript)

        await self.push_frame(frame, direction)


class SpeechTimer(FrameProcessor):
    """Records when the reply's first audio is ready.

    Sits after synthesis. TTSStartedFrame is the first moment audio exists, so
    this is the end of the interval the user waits through.
    """

    def __init__(self, recorder: MetricsRecorder) -> None:
        super().__init__()
        self.recorder = recorder

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            turn = self.recorder.current
            if turn.speech_ended and turn.reply_s is None:
                # The whole wait, measured directly from one clock rather than
                # assembled out of stages that do not add up.
                turn.reply_s = round(time.perf_counter() - turn.speech_ended, 3)

        await self.push_frame(frame, direction)


class ReplyLogger(FrameProcessor):
    """Keeps the spoken reply on the turn record, for the transcript log."""

    def __init__(self, recorder: MetricsRecorder) -> None:
        super().__init__()
        self.recorder = recorder

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        is_reply = isinstance(frame, TextFrame) and not isinstance(
            frame, TranscriptionFrame
        )
        if is_reply and self.recorder.turns:
            turn = self.recorder.turns[-1]
            # First actual token, not LLMFullResponseStartFrame. That frame is
            # pushed when generation is requested, which is why the old code
            # reported a 35B model answering in one millisecond.
            if turn.speech_ended and turn.llm_ttft_s is None:
                elapsed = time.perf_counter() - turn.speech_ended
                turn.llm_ttft_s = round(max(elapsed - (turn.asr_final_s or 0), 0.0), 3)
            turn.reply += frame.text
        await self.push_frame(frame, direction)


class Speaking:
    """Whether the agent's voice is currently in the room.

    Two moments matter and they are not the same. Synthesis finishing is early:
    audio is still sitting in the speaker buffer and still audible. Playback
    finishing is the real end, and even then a little tail can reach the
    microphone, so a short hold-off follows.
    """

    HOLDOFF_S = 0.7

    def __init__(self) -> None:
        self._active = False
        self._quiet_at = 0.0

    def started(self) -> None:
        if not self._active:
            log.debug("agent speaking")
        self._active = True

    def stopped(self) -> None:
        if self._active:
            log.debug("agent finished, holding off %.1fs", self.HOLDOFF_S)
        self._active = False
        self._quiet_at = time.monotonic() + self.HOLDOFF_S

    @property
    def audible(self) -> bool:
        return self._active or time.monotonic() < self._quiet_at


class ReferenceCapture(FrameProcessor):
    """Copies audio on its way to the speaker, for the echo canceller.

    Sits immediately before the output transport, so it sees every frame that
    is about to be played. The canceller needs to know what we are playing in
    order to subtract it from what the microphone hears.
    """

    def __init__(self, reference: ReferenceTap, mic_rate: int) -> None:
        super().__init__()
        self.reference = reference
        self.mic_rate = mic_rate

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame) and frame.audio:
            # The speaker runs at 22050 Hz and the microphone filter at 16000.
            # Handing Speex a reference at the wrong rate is the same as
            # handing it noise: it cancels nothing and reports no error.
            audio = frame.audio
            if frame.sample_rate != self.mic_rate:
                audio = engines.resample(audio, frame.sample_rate, self.mic_rate)
            self.reference.add(audio)
        await self.push_frame(frame, direction)


class SpeechStart(FrameProcessor):
    """Sits after synthesis. Marks the agent as speaking as early as possible."""

    def __init__(self, state: Speaking) -> None:
        super().__init__()
        self.state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (TTSStartedFrame, BotStartedSpeakingFrame)):
            self.state.started()
        await self.push_frame(frame, direction)


class SpeechEnd(FrameProcessor):
    """Sits after the speaker. Marks the agent quiet once playback is done."""

    def __init__(self, state: Speaking) -> None:
        super().__init__()
        self.state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStoppedSpeakingFrame):
            self.state.stopped()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self.state.started()
        await self.push_frame(frame, direction)


class TranscriptGate(FrameProcessor):
    """Ignores what the recogniser hears while the agent's voice is audible.

    The recogniser turns the agent's own speech, and room noise, into
    transcripts. Each one counts as a new user turn, and a new turn cancels the
    reply in progress, so the agent stops talking part way through.

    Filtering by length was the obvious fix and the wrong one: it threw away
    "Hello" along with the noise. What matters is when a transcript arrives,
    not how long it is. While the agent is audible nothing is listened to;
    while it is quiet every word gets through, however short.

    The cost is real: you have to let the reply finish before speaking again.
    That is the wrong trade for a fluent assistant and the right one for a
    laptop with no echo cancellation, where the microphone cannot tell your
    voice from the agent's.
    """

    def __init__(self, state: Speaking) -> None:
        super().__init__()
        self.state = state
        self.ignored = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and self.state.audible:
            self.ignored += 1
            log.info("ignored own voice: %r", frame.text.strip())
            return

        await self.push_frame(frame, direction)


def build_services():
    """The three Spark models, as pipecat services."""
    stt = NvidiaSTTService(
        api_key=None,
        server=ASR_SERVER,
        use_ssl=False,
        # Self-hosted Riva rejects the stream if you send it a cloud function id.
        model_function_map={"function_id": "", "model_name": ASR_MODEL},
        sample_rate=MIC_RATE,
    )
    llm = OpenAILLMService(
        api_key="not-needed", base_url=LLM_BASE_URL, model=LLM_MODEL
    )
    tts = NvidiaTTSService(
        api_key=None,
        server=TTS_SERVER,
        use_ssl=False,
        model_function_map={"function_id": "",
                            "model_name": "magpie-tts-multilingual"},
        voice_id=TTS_VOICE,
        sample_rate=SPEAKER_RATE,
    )
    return stt, llm, tts


def build_vad() -> SileroVADAnalyzer:
    """Voice detection, deliberately hard of hearing.

    On a laptop the speakers are a few centimetres from the microphone and
    there is no echo cancellation, so the agent hears its own reply. With the
    default settings it treats that as the user interrupting and stops talking
    about a second in. Raising the confidence and volume thresholds means quiet
    leakage from the speakers no longer counts as somebody speaking.
    """
    return SileroVADAnalyzer(params=VADParams(
        confidence=0.8,
        min_volume=0.6,
        start_secs=0.3,
        stop_secs=0.8,
    ))


def build_pipeline(home: Home, recorder: MetricsRecorder,
                   in_device: int | None, out_device: int | None,
                   aec: bool = True) -> Pipeline:
    stt, llm, tts = build_services()
    tools.register(home, llm)
    speaking = Speaking()
    reference = ReferenceTap()
    echo_filter = SpeexEchoFilter(reference) if aec else None

    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt(home)}],
        tools=tools.pipecat_schema(),
    )
    aggregators = LLMContextAggregatorPair(context)

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=MIC_RATE,
            audio_out_sample_rate=SPEAKER_RATE,
            vad_analyzer=build_vad(),
            audio_in_filter=echo_filter,
            # Overridable so the live path can be driven from a loopback device
            # rather than a person at the microphone: same pipeline, same VAD,
            # repeatable input.
            input_device_index=in_device,
            output_device_index=out_device,
        )
    )

    return Pipeline([
        transport.input(),
        SpeechEndMarker(recorder, home),
        stt,
        TranscriptTimer(recorder, home),   # before the aggregator eats it
        TranscriptGate(speaking),
        aggregators.user(),
        llm,
        ReplyLogger(recorder),
        tts,
        SpeechStart(speaking),
        SpeechTimer(recorder),
        ReferenceCapture(reference, MIC_RATE),
        transport.output(),
        SpeechEnd(speaking),
        aggregators.assistant(),
    ])


async def run(in_device: int | None = None, out_device: int | None = None,
              drift: bool = True, interrupt: bool = False,
              aec: bool = True) -> None:
    log.info("offline check passed: %s", ", ".join(check_offline()))
    log.info("audio: mic %d Hz, speaker %d Hz, voice %s",
             MIC_RATE, SPEAKER_RATE, TTS_VOICE)

    home = Home()
    bridge = MqttBridge(home, host=MQTT_HOST, port=MQTT_PORT)
    bridge.start()
    log.info("published %d entities to Home Assistant", len(list(home.all())))

    recorder = MetricsRecorder(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        EVAL_DIR,
        {"stt": ASR_MODEL, "llm": LLM_MODEL,
         "tts": f"magpie/{TTS_VOICE}", "host": VIP},
    )

    task = PipelineTask(
        build_pipeline(home, recorder, in_device, out_device, aec=aec),
        # Interruptions are off by default. Without echo cancellation the
        # agent hears itself through the speakers and cuts its own reply short.
        # Turn them on with --interrupt when wearing headphones.
        params=PipelineParams(allow_interruptions=interrupt, enable_metrics=True),
    )
    # The drift task publishes to MQTT from the event loop. It is a suspect in
    # the stall described in KNOWN_ISSUES.md, so it can be turned off to
    # isolate that.
    drift_task = asyncio.create_task(_drift_sensors(home)) if drift else None

    print("\n  HearthVoice is listening. Talk to it. Ctrl-C to stop.")
    print(f"  Dashboard: {HA_URL}\n")

    try:
        await PipelineRunner(handle_sigint=True).run(task)
    finally:
        if drift_task:
            drift_task.cancel()
        path = recorder.write()
        print(recorder.table())
        print(f"\n  written to {path}\n")
        bridge.stop()


async def _drift_sensors(home: Home) -> None:
    """Nudge the read-only sensors so the dashboard is not frozen on video."""
    try:
        while True:
            await asyncio.sleep(20)
            home.tick_sensors()
    except asyncio.CancelledError:
        pass


def _print_audio_devices() -> None:
    import pyaudio

    audio = pyaudio.PyAudio()
    for index in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(index)
        if info["maxInputChannels"] or info["maxOutputChannels"]:
            print(f"  [{index:2}] {info['name']:<34} "
                  f"in={info['maxInputChannels']} out={info['maxOutputChannels']}")
    audio.terminate()


def _enable_stack_dump() -> None:
    """Let a stuck agent be asked where it is stuck.

    The live-path stall produces no crash and no error, so there is nothing to
    read afterwards. Sending SIGUSR1 makes every thread print its stack, which
    is the only way to tell a blocked event loop from one that is simply not
    being given any work.
    """
    import faulthandler
    import signal

    # Keep the handle open for the life of the process: faulthandler writes
    # from a signal handler and cannot open a file at that point. The dump
    # goes to its own file because the harness tears the agent down as soon
    # as it has its results, which truncated it when it went to stderr.
    path = Path(EVAL_DIR) / "stackdump.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w")
    globals()["_STACK_DUMP_FILE"] = handle
    faulthandler.register(signal.SIGUSR1, file=handle,
                          all_threads=True, chain=False)


def main() -> None:
    _enable_stack_dump()
    parser = argparse.ArgumentParser(description="HearthVoice offline voice agent")
    parser.add_argument("--devices", action="store_true",
                        help="list the house and exit")
    parser.add_argument("--check", action="store_true",
                        help="confirm every target is local, then exit")
    parser.add_argument("--list-audio", action="store_true",
                        help="list audio devices and exit")
    parser.add_argument("--in-device", type=int, help="input device index")
    parser.add_argument("--out-device", type=int, help="output device index")
    parser.add_argument("--no-aec", action="store_true",
                        help="turn off echo cancellation")
    parser.add_argument("--interrupt", action="store_true",
                        help="allow barge-in (use headphones, or it interrupts itself)")
    parser.add_argument("--no-drift", action="store_true",
                        help="stop the sensors drifting (isolates a suspected bug)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_audio:
        return _print_audio_devices()

    if args.check:
        for line in check_offline():
            print(f"  {line}")
        print("\n  every model runs on hardware we control")
        return

    if args.devices:
        for entity in Home().all():
            print(f"  {entity.entity_id:34} {entity.describe()}")
        return

    try:
        asyncio.run(run(args.in_device, args.out_device,
                        drift=not args.no_drift, interrupt=args.interrupt,
                        aec=not args.no_aec))
    except OfflineViolation as e:
        print(f"\n{e}\n", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
