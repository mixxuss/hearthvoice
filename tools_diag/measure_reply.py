"""Measure how long the agent actually speaks for.

Reproduces the laptop setup without anyone talking. The agent listens on the
real microphone, and a command is played out of the real speakers so the
microphone picks it up through the air, the same way it would from a person.
The agent's reply goes to the BlackHole loopback rather than the speakers, and
is recorded there, so its length can be measured instead of judged by ear.

A recording much shorter than the reply needs means the reply is being cut
short. A recording that matches means the problem is somewhere else.

This is a scratch diagnostic, not part of the system or the evaluation.

    python tools_diag/measure_reply.py
"""

from __future__ import annotations

import array
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import pyaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearthvoice import engines  # noqa: E402

COMMAND = "Turn on the living room lights."
RECORD_SECONDS = 14
RATE = 22050


def find(pa: pyaudio.PyAudio, name: str, output: bool) -> int:
    key = "maxOutputChannels" if output else "maxInputChannels"
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        if name.lower() in info["name"].lower() and info[key] > 0:
            return index
    raise RuntimeError(f"no device matching {name}")


def record(pa: pyaudio.PyAudio, device: int, seconds: int, out: list) -> None:
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                     input_device_index=device, frames_per_buffer=1024)
    frames = [stream.read(1024, exception_on_overflow=False)
              for _ in range(int(RATE / 1024 * seconds))]
    stream.stop_stream()
    stream.close()
    out.append(b"".join(frames))


def speech_duration(pcm: bytes, rate: int) -> float:
    """Seconds of audio above the noise floor, so silence is not counted."""
    samples = array.array("h")
    samples.frombytes(pcm)
    window = rate // 50
    loud = sum(
        1
        for i in range(0, len(samples) - window, window)
        if max(abs(s) for s in samples[i : i + window]) > 500
    )
    return round(loud * window / rate, 2)


def main() -> None:
    pa = pyaudio.PyAudio()
    mic = pa.get_default_input_device_info()["index"]
    speakers = find(pa, "MacBook Pro Speakers", output=True)
    loop_in = find(pa, "BlackHole", output=False)
    loop_out = find(pa, "BlackHole", output=True)

    print(f"  agent listens on  [{mic}] real microphone")
    print(f"  agent speaks to   [{loop_out}] BlackHole, recorded")
    print(f"  command played on [{speakers}] real speakers\n")

    agent = subprocess.Popen(
        [sys.executable, "-m", "hearthvoice.agent",
         "--in-device", str(mic), "--out-device", str(loop_out), "-v"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    time.sleep(22)

    recorded: list[bytes] = []
    recorder = threading.Thread(
        target=record, args=(pa, loop_in, RECORD_SECONDS, recorded)
    )
    recorder.start()
    time.sleep(0.5)

    audio, _ = engines.speak(COMMAND)
    pcm = engines.resample(audio, engines.TTS_RATE, RATE)
    out = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True,
                  output_device_index=speakers, frames_per_buffer=1024)
    out.write(pcm)
    out.stop_stream()
    out.close()
    print(f"  played: {COMMAND}")

    recorder.join()
    agent.terminate()
    try:
        agent.wait(timeout=15)
    except subprocess.TimeoutExpired:
        agent.kill()
    log = agent.stdout.read() if agent.stdout else ""
    pa.terminate()

    reply_audio = recorded[0] if recorded else b""
    path = Path("eval/diag_reply.wav")
    path.parent.mkdir(exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(reply_audio)

    heard = [line.split("heard: ", 1)[1].strip()
             for line in log.splitlines() if "heard: " in line]
    tool_calls = [line for line in log.splitlines() if "hearthvoice.tools" in line]

    reference, _ = engines.speak("Living room light on.")
    reference_seconds = len(reference) / 2 / engines.TTS_RATE

    print(f"\n  heard       : {heard or 'nothing'}")
    print(f"  tool calls  : {len(tool_calls)}")
    print(f"  agent spoke : {speech_duration(reply_audio, RATE)} s")
    print(f"  for comparison, a one-sentence reply takes about "
          f"{reference_seconds:.2f} s")
    print(f"  recording   : {path}")

    log_path = Path("eval/diag_agent.log")
    log_path.write_text(log)
    print(f"  agent log   : {log_path}")


if __name__ == "__main__":
    main()
