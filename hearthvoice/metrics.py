"""Per-stage timing for the pipeline.

Written to replace the LiveKit-era summary writer, which averaged cancelled TTS
segments (recorded as -1.0) in with real measurements and so reported a
time-to-first-byte lower than any figure actually observed. `mean()` here drops
negatives and says how many it dropped, because a metric that quietly discards
data is how the first version went wrong.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SENTINEL = -1.0


@dataclass
class Turn:
    """One user utterance and the reply to it."""

    index: int
    started_at: float = field(default_factory=time.time)
    # perf_counter reading for when the user stopped talking. Every stage time
    # is measured from here.
    speech_ended: float | None = None
    transcript: str = ""
    reply: str = ""
    asr_final_s: float | None = None
    llm_ttft_s: float | None = None
    tts_ttfb_s: float | None = None
    # Speech end to first audio out, measured directly rather than added up.
    # On the live path the stages cannot all be attributed (the recogniser's
    # transcript arrives before the turn-end frame, so there is no interval to
    # measure it over), but the thing the user actually waits through can still
    # be timed end to end.
    reply_s: float | None = None
    tool_calls: list[str] = field(default_factory=list)

    @property
    def end_to_end_s(self) -> float | None:
        """Speech ends to first audio out.

        None unless all three stages produced a real measurement. A turn where
        the reply was interrupted has no end-to-end time, and inventing one by
        treating the missing stage as zero is what produced the bad numbers in
        the first version of this project.
        """
        if self.reply_s is not None and self.reply_s >= 0:
            return round(self.reply_s, 3)
        parts = [self.asr_final_s, self.llm_ttft_s, self.tts_ttfb_s]
        if any(p is None or p < 0 for p in parts):
            return None
        return round(sum(parts), 3)


def mean(values: list[float | None]) -> tuple[float | None, int]:
    """Average the real measurements. Returns (mean, number discarded)."""
    usable = [v for v in values if v is not None and v >= 0]
    discarded = len(values) - len(usable)
    if not usable:
        return None, discarded
    return round(statistics.fmean(usable), 3), discarded


def percentile(values: list[float | None], p: float) -> float | None:
    usable = sorted(v for v in values if v is not None and v >= 0)
    if not usable:
        return None
    if len(usable) == 1:
        return round(usable[0], 3)
    pos = (len(usable) - 1) * p
    lo, hi = int(pos), min(int(pos) + 1, len(usable) - 1)
    return round(usable[lo] + (usable[hi] - usable[lo]) * (pos - lo), 3)


class MetricsRecorder:
    """Collects turns and writes the session summary."""

    def __init__(self, session_id: str, out_dir: Path, models: dict[str, str]) -> None:
        self.session_id = session_id
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.models = models
        self.turns: list[Turn] = []
        self._current: Turn | None = None

    def begin_turn(self) -> Turn:
        self._current = Turn(index=len(self.turns) + 1)
        self.turns.append(self._current)
        return self._current

    @property
    def current(self) -> Turn:
        return self._current or self.begin_turn()

    def real_turns(self) -> list[Turn]:
        """Turns where the user actually said something.

        The detector occasionally opens a turn that never receives a
        transcript, usually at the end of a session. One of those, carrying a
        reply time and an empty transcript, was being averaged in with the real
        ones while the first genuine command was being dropped for having no
        reply time, which moved the reported live mean by 0.16 s in the
        flattering direction. Counting only turns with a transcript is the
        honest population.
        """
        return [t for t in self.turns if t.transcript.strip()]

    def summary(self) -> dict[str, Any]:
        turns = self.real_turns()
        asr = [t.asr_final_s for t in turns]
        llm = [t.llm_ttft_s for t in turns]
        tts = [t.tts_ttfb_s for t in turns]
        e2e = [t.end_to_end_s for t in turns]

        asr_mean, asr_drop = mean(asr)
        llm_mean, llm_drop = mean(llm)
        tts_mean, tts_drop = mean(tts)
        e2e_mean, e2e_drop = mean(e2e)

        complete = [t for t in turns if t.end_to_end_s is not None]
        target_met = [t for t in complete if t.end_to_end_s <= 2.0]

        return {
            "session": self.session_id,
            "models": self.models,
            "turns_recorded": len(self.turns),
            "turns_with_speech": len(turns),
            "turns_complete": len(complete),
            "asr_final_s": {"mean": asr_mean, "p95": percentile(asr, 0.95),
                            "missing": asr_drop},
            "llm_ttft_s": {"mean": llm_mean, "p95": percentile(llm, 0.95),
                           "missing": llm_drop},
            "tts_ttfb_s": {"mean": tts_mean, "p95": percentile(tts, 0.95),
                           "missing": tts_drop},
            "end_to_end_s": {"mean": e2e_mean, "p95": percentile(e2e, 0.95),
                             "missing": e2e_drop},
            "target_2s": {
                "met": len(target_met),
                "of": len(complete),
                "rate": round(len(target_met) / len(complete), 3) if complete else None,
            },
            "per_turn": [asdict(t) for t in self.turns],
        }

    def write(self) -> Path:
        path = self.out_dir / f"summary_{self.session_id}.json"
        path.write_text(json.dumps(self.summary(), indent=2))
        return path

    def table(self) -> str:
        s = self.summary()

        def row(label: str, key: str) -> str:
            d = s[key]
            m = "n/a" if d["mean"] is None else f"{d['mean']:.3f}"
            p = "n/a" if d["p95"] is None else f"{d['p95']:.3f}"
            miss = f"  ({d['missing']} missing)" if d["missing"] else ""
            return f"  {label:<34}{m:>8} s{p:>10} s{miss}"

        lines = [
            "",
            f"  session {s['session']}   {s['turns_recorded']} turns "
            f"({s['turns_complete']} complete)",
            f"  {'stage':<34}{'mean':>10}{'p95':>12}",
            "  " + "-" * 58,
            row("ASR, speech end to transcript", "asr_final_s"),
            row("LLM, first token", "llm_ttft_s"),
            row("TTS, first audio", "tts_ttfb_s"),
            row("End to end", "end_to_end_s"),
        ]
        t = s["target_2s"]
        if t["of"]:
            lines.append(
                f"\n  under the 2 s target: {t['met']}/{t['of']} turns "
                f"({t['rate'] * 100:.0f}%)"
            )
        return "\n".join(lines)
