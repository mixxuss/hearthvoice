"""Compare the deployed models against local alternatives.

The project template asks for evidence that models were tested and chosen on
criteria rather than simply picked. This runs that comparison on the same
twelve commands the evaluation harness uses.

Recognition: Parakeet on the Spark against faster-whisper in three sizes on the
laptop CPU. Both are local, which is the comparison that matters here, since a
cloud engine would win on nothing this project cares about.

Synthesis: Magpie on the Spark against Piper on the laptop. Piper is the
default choice in the open self-hosting world and the one the literature review
names, so it is the alternative a reader will ask about.

Word error rate is reported twice. Recognisers differ in how they write down
what they heard: "twenty one" comes back as "21" from one and as words from
another, which counts as an error without anyone having misheard anything. The
normalised figure spells digits back out, so the comparison is about
recognition rather than formatting.

    python tools_diag/model_comparison.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hearthvoice import engines  # noqa: E402
from hearthvoice.config import EVAL_DIR  # noqa: E402
from hearthvoice.harness import SUITE  # noqa: E402

WHISPER_SIZES = ["tiny.en", "base.en", "small.en"]
PIPER_VOICE = "en_US-lessac-medium"

DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "20": "twenty", "21": "twenty one", "30": "thirty", "50": "fifty",
}


def normalise(text: str, spell_digits: bool = False) -> list[str]:
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    words = text.split()
    if spell_digits:
        words = [w for word in words for w in DIGITS.get(word, word).split()]
    return words


def word_error_rate(reference: str, hypothesis: str, spell: bool = False) -> float:
    """Levenshtein distance over words, divided by the reference length."""
    ref = normalise(reference, spell)
    hyp = normalise(hypothesis, spell)
    if not ref:
        return 0.0

    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return round(previous[-1] / len(ref), 3)


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def _score(engine: str, where: str, clips, transcripts, elapsed) -> dict:
    references = [reference for reference, _ in clips]
    raw = [word_error_rate(r, t) for r, t in zip(references, transcripts)]
    spelled = [word_error_rate(r, t, spell=True)
               for r, t in zip(references, transcripts)]
    return {
        "engine": engine,
        "where": where,
        "wer": round(sum(raw) / len(raw), 3),
        "wer_normalised": round(sum(spelled) / len(spelled), 3),
        "mean_latency_s": round(sum(elapsed) / len(elapsed), 3),
        "max_latency_s": round(max(elapsed), 3),
        "perfect_clips": sum(1 for w in spelled if w == 0.0),
        "of": len(spelled),
        "transcripts": transcripts,
    }


def compare_recognition(clips: list[tuple[str, bytes]]) -> list[dict]:
    rows = []

    print("  Parakeet on the Spark ...")
    transcripts, elapsed = [], []
    for _, audio in clips:
        text, seconds = engines.transcribe(audio)
        transcripts.append(text)
        elapsed.append(seconds)
    rows.append(_score("parakeet-1.1b", "DGX Spark, streaming gRPC",
                       clips, transcripts, elapsed))

    from faster_whisper import WhisperModel

    for size in WHISPER_SIZES:
        print(f"  faster-whisper {size} on the laptop CPU ...")
        model = WhisperModel(size, device="cpu", compute_type="int8")
        transcripts, elapsed = [], []
        for _, audio in clips:
            pcm = engines.resample(audio, engines.TTS_RATE, 16000)
            path = EVAL_DIR / "_cmp.wav"
            _write_wav(path, pcm, 16000)
            started = time.perf_counter()
            segments, _ = model.transcribe(str(path), beam_size=1)
            text = " ".join(s.text for s in segments).strip()
            elapsed.append(round(time.perf_counter() - started, 3))
            transcripts.append(text)
        rows.append(_score(f"faster-whisper {size}", "laptop CPU, int8",
                           clips, transcripts, elapsed))

    return rows


def _piper(sentences: list[str]) -> dict:
    from piper import PiperVoice
    from piper.download_voices import download_voice

    voices = Path(__file__).resolve().parent.parent / ".local_voices"
    voices.mkdir(parents=True, exist_ok=True)
    if not (voices / f"{PIPER_VOICE}.onnx").is_file():
        print(f"  downloading {PIPER_VOICE} ...")
        download_voice(PIPER_VOICE, voices)

    print("  Piper on the laptop CPU ...")
    voice = PiperVoice.load(str(voices / f"{PIPER_VOICE}.onnx"))
    ttfb, total, seconds = [], [], []
    for sentence in sentences:
        started = time.perf_counter()
        first = None
        samples = 0
        for chunk in voice.synthesize(sentence):
            if first is None:
                first = round(time.perf_counter() - started, 3)
            samples += len(chunk.audio_int16_bytes) // 2
        total.append(round(time.perf_counter() - started, 3))
        ttfb.append(first or 0.0)
        seconds.append(round(samples / chunk.sample_rate, 2))

    return {
        "engine": f"Piper ({PIPER_VOICE})", "where": "laptop CPU",
        "ttfb_s": round(sum(ttfb) / len(ttfb), 3),
        "total_s": round(sum(total) / len(total), 3),
        "audio_s": round(sum(seconds) / len(seconds), 2),
    }


def compare_synthesis(sentences: list[str]) -> list[dict]:
    rows = []

    print("  Magpie on the Spark ...")
    ttfb, total, seconds = [], [], []
    for sentence in sentences:
        started = time.perf_counter()
        audio, first = engines.speak(sentence)
        total.append(round(time.perf_counter() - started, 3))
        ttfb.append(first)
        seconds.append(round(len(audio) / 2 / engines.TTS_RATE, 2))
    rows.append({
        "engine": "Magpie (Brian)", "where": "DGX Spark, streaming gRPC",
        "ttfb_s": round(sum(ttfb) / len(ttfb), 3),
        "total_s": round(sum(total) / len(total), 3),
        "audio_s": round(sum(seconds) / len(seconds), 2),
    })

    try:
        rows.append(_piper(sentences))
    except Exception as e:
        rows.append({"engine": f"Piper ({PIPER_VOICE})", "where": "laptop CPU",
                     "error": f"{type(e).__name__}: {str(e)[:140]}"})
    return rows


def main() -> None:
    utterances = [s.utterance for s in SUITE]
    print(f"\n  {len(utterances)} utterances\n")

    print("  synthesising the reference clips with Magpie ...")
    clips = [(u, engines.speak(u)[0]) for u in utterances]

    asr = compare_recognition(clips)
    tts = compare_synthesis(utterances[:6])

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "utterances": len(utterances),
        "note": ("The reference audio is synthetic, produced by Magpie, so "
                 "these are clean-speech figures. They say nothing about "
                 "accented or atypical speech, which this project measures "
                 "nowhere."),
        "asr": asr,
        "tts": tts,
    }
    path = EVAL_DIR / "model_comparison.json"
    path.write_text(json.dumps(report, indent=2))

    print("\n  recognition")
    print(f"    {'engine':<26}{'WER':>7}{'norm':>8}{'mean s':>9}{'clean':>9}")
    for row in asr:
        print(f"    {row['engine']:<26}{row['wer']:>7.3f}"
              f"{row['wer_normalised']:>8.3f}{row['mean_latency_s']:>9.3f}"
              f"{row['perfect_clips']:>6}/{row['of']}")

    print("\n  synthesis")
    for row in tts:
        if "error" in row:
            print(f"    {row['engine']:<26} {row['error']}")
        else:
            print(f"    {row['engine']:<26}ttfb {row['ttfb_s']:.3f} s   "
                  f"total {row['total_s']:.3f} s")

    print(f"\n  written to {path}\n")


if __name__ == "__main__":
    main()
