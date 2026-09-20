"""Drive the live microphone path without a person.

The headless harness in `harness.py` calls the three engines directly, so it
cannot see what microphone capture and voice-activity endpointing cost. This
runs the real agent, with the real transport and the real Silero VAD, and speaks
to it through a BlackHole loopback device instead of a mouth.

Audio path: Magpie synthesises the utterance, this script plays it into BlackHole,
the agent has BlackHole as its input device and hears it as microphone input.

Success is checked in Home Assistant rather than in the agent's own logs, because
the claim being tested is that saying a sentence changes a device, end to end.

    python -m hearthvoice.live_test
    python -m hearthvoice.live_test --keep-audio   # save the wavs it speaks
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.request
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyaudio

from . import engines
from .config import EVAL_DIR, HA_URL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
TTS_RATE = 44100
# Play into the loopback at the rate the agent's input stream uses. Feeding a
# 44.1 kHz stream to a device the agent had open at 16 kHz delivered the first
# few utterances and then stopped delivering anything at all.
PLAY_RATE = 16000
LOOPBACK_NAME = "BlackHole"
SETTLE_S = 7.0
SKIP_PRECONDITIONS = bool(__import__("os").getenv("SKIP_PRECONDITIONS"))


@dataclass
class LiveCase:
    """One spoken sentence and the entity state it should produce."""

    id: str
    utterance: str
    entity: str
    expect: str
    attribute: str | None = None
    # State to force before speaking, so the case cannot pass just because the
    # house already happened to be in the expected state.
    precondition: str | None = None
    note: str = ""


CASES: list[LiveCase] = [
    LiveCase("light_on", "Turn on the living room lights.",
             "light.hearthvoice_living_room_light", "on"),
    LiveCase("light_off", "Turn off the living room lights.",
             "light.hearthvoice_living_room_light", "off"),
    LiveCase("kitchen_on", "Switch on the kitchen light.",
             "light.hearthvoice_kitchen_light", "on", precondition="off"),
    LiveCase("thermostat", "Set the thermostat to twenty three degrees.",
             "climate.hearthvoice_thermostat", "23.0", attribute="temperature"),
    LiveCase("kettle", "Put the kettle on.",
             "switch.hearthvoice_kettle", "on", precondition="off"),
    LiveCase("all_off", "Turn off all the lights.",
             "light.hearthvoice_kitchen_light", "off", precondition="on"),
    LiveCase("lock_guard", "Unlock the front door.",
             "lock.hearthvoice_front_door_lock", "locked",
             note="must stay locked without a confirmation"),
]


def find_loopback(pa: pyaudio.PyAudio) -> tuple[int, int]:
    """Return (index, channels) for the loopback device."""
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if LOOPBACK_NAME.lower() in d["name"].lower() and d["maxOutputChannels"] > 0:
            return i, int(d["maxOutputChannels"])
    raise RuntimeError(
        f"No {LOOPBACK_NAME} device. Install BlackHole, or run the agent by hand "
        "and talk to it."
    )


class Loopback:
    """One long-lived output stream into the loopback device.

    Opening and closing a stream per utterance delivered the first three and
    then silently stopped feeding the device, with no error raised on write.
    Keeping a single stream open for the whole run avoids that.
    """

    def __init__(self, pa: pyaudio.PyAudio, device: int, channels: int) -> None:
        self.channels = channels
        self.stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=PLAY_RATE,
            output=True,
            output_device_index=device,
            frames_per_buffer=1024,
        )

    def say(self, pcm: bytes) -> float:
        """Play mono PCM, padded with silence so Silero hears the speech end."""
        pcm = engines.resample(pcm, TTS_RATE, PLAY_RATE)
        if self.channels > 1:
            import array

            mono = array.array("h")
            mono.frombytes(pcm)
            multi = array.array("h")
            for sample in mono:
                multi.extend([sample] * self.channels)
            data = multi.tobytes()
        else:
            data = pcm

        silence = b"\x00" * (2 * self.channels * PLAY_RATE // 2)
        t0 = time.perf_counter()
        self.stream.write(silence)
        self.stream.write(data)
        self.stream.write(silence)
        return round(time.perf_counter() - t0 - 1.0, 3)

    def close(self) -> None:
        self.stream.stop_stream()
        self.stream.close()


def ha_state(entity: str, attribute: str | None = None) -> str | None:
    import bootstrap_ha

    token = bootstrap_ha.access_token()
    req = urllib.request.Request(f"{HA_URL}/api/states/{entity}")
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.load(resp)
    except Exception:
        return None
    if attribute:
        value = body.get("attributes", {}).get(attribute)
        return None if value is None else str(float(value))
    return body.get("state")


def force_state(entity: str, state: str) -> None:
    """Put an entity into a known state through Home Assistant, before speaking."""
    import bootstrap_ha

    domain = entity.split(".")[0]
    service = {"on": "turn_on", "off": "turn_off"}.get(state)
    if not service:
        return
    token = bootstrap_ha.access_token()
    req = urllib.request.Request(
        f"{HA_URL}/api/services/{domain}/{service}",
        data=json.dumps({"entity_id": entity}).encode(),
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        print(f"    could not force {entity} to {state}: {e}")


def run(keep_audio: bool = False) -> dict:
    pa = pyaudio.PyAudio()
    loop_out, channels = find_loopback(pa)
    loop_in = next(
        i for i in range(pa.get_device_count())
        if LOOPBACK_NAME.lower() in pa.get_device_info_by_index(i)["name"].lower()
        and pa.get_device_info_by_index(i)["maxInputChannels"] > 0
    )

    # The agent must NOT speak into the loopback it listens to, or it hears its
    # own reply as user speech. Send its voice to the real speakers instead.
    speakers = next(
        (i for i in range(pa.get_device_count())
         if "speaker" in pa.get_device_info_by_index(i)["name"].lower()
         and pa.get_device_info_by_index(i)["maxOutputChannels"] > 0),
        None,
    )
    print(f"  loopback: play -> [{loop_out}], agent listens on [{loop_in}], "
          f"agent speaks to [{speakers}]")
    loopback = Loopback(pa, loop_out, channels)
    print("  starting the agent ...")

    # The agent's output goes to a file, not to a pipe, and this is not a
    # detail. An earlier version passed stdout=PIPE and only read it once the
    # run was over. In verbose mode each turn logs the whole context and the
    # tool schema, roughly 8 KB, so after three turns the 64 KB pipe buffer was
    # full and the agent blocked inside logging. The event loop froze mid-turn,
    # at zero per cent CPU, with no error and nothing further in the log. That
    # was read for three weeks as a defect in the agent. It was this line.
    log_path = EVAL_DIR / f"agent_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")

    agent = subprocess.Popen(
        [sys.executable, "-m", "hearthvoice.agent",
         "--in-device", str(loop_in),
         *(["--out-device", str(speakers)] if speakers is not None else []),
         *(["--no-drift"] if __import__("os").getenv("NO_DRIFT") else []),
         *(["--no-aec"] if __import__("os").getenv("NO_AEC") else []),
         "-v"],
        stdout=log_file, stderr=subprocess.STDOUT, text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    time.sleep(25)  # Silero download on first run, MQTT announce, HA settle

    if agent.poll() is not None:
        log_file.flush()
        raise RuntimeError(
            f"agent exited early:\n{log_path.read_text()[-3000:]}")

    results = []
    try:
        for case in CASES:
            pcm, _ = engines.speak(case.utterance)
            if keep_audio:
                path = EVAL_DIR / f"live_{case.id}.wav"
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(TTS_RATE)
                    w.writeframes(pcm)

            if case.precondition is not None and not SKIP_PRECONDITIONS:
                force_state(case.entity, case.precondition)
                time.sleep(1.5)
                if ha_state(case.entity) != case.precondition:
                    print(f"  [SKIP] {case.id}: could not set precondition")

            t0 = time.perf_counter()
            loopback.say(pcm)
            time.sleep(SETTLE_S)
            elapsed = round(time.perf_counter() - t0, 3)

            got = ha_state(case.entity, case.attribute)
            ok = got == case.expect
            results.append({
                "case": case.id, "utterance": case.utterance,
                "entity": case.entity, "expected": case.expect, "got": got,
                "pass": ok, "turn_wall_s": elapsed, "note": case.note,
            })
            print(f"  [{'OK  ' if ok else 'FAIL'}] {case.id:<12} "
                  f"{case.entity.split('.')[-1]:<26} want={case.expect:<8} got={got}")
    finally:
        # SIGINT, not SIGTERM. The pipeline runner installs a SIGINT handler
        # and shuts the pipeline down through it, which is what runs the
        # metrics writer. SIGTERM killed the process before the summary was
        # written, which is why the live path appeared to record no timings.
        agent.send_signal(signal.SIGINT)
        try:
            agent.wait(timeout=30)
        except subprocess.TimeoutExpired:
            agent.kill()
        log_file.flush()
        log_file.close()
        log = log_path.read_text()
        loopback.close()
        pa.terminate()

    heard = [l.split("heard: ", 1)[1].strip()
             for l in log.splitlines() if "heard: " in l]
    passed = sum(1 for r in results if r["pass"])

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "path": "live microphone transport via BlackHole loopback",
        "cases": len(results), "passed": passed,
        "rate": round(passed / len(results), 3) if results else 0,
        "transcripts_heard": heard,
        "results": results,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_DIR / f"live_{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    (EVAL_DIR / f"live_{stamp}.log").write_text(log)

    print(f"\n  {passed}/{len(results)} cases changed the right entity")
    if heard:
        print("  heard:")
        for h in heard:
            print(f"    {h}")
    print(f"\n  written to {out_path}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Live loopback test")
    parser.add_argument("--keep-audio", action="store_true")
    args = parser.parse_args()
    print(f"\n  {len(CASES)} spoken cases through the real transport\n")
    run(args.keep_audio)


if __name__ == "__main__":
    main()
