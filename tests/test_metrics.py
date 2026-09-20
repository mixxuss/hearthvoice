"""Tests for the averaging bug that produced a published figure lower than
any measurement taken.

Chapter 4.6 describes it: cancelled speech segments are recorded as -1.0, and
averaging them in with real measurements gave 0.653 seconds when no turn was
faster than 1.310. These tests pin the behaviour so it cannot come back.
"""

from hearthvoice.metrics import Turn, mean


class TestMeanIgnoresSentinels:
    def test_the_original_bug(self):
        """The exact values from the session in chapter 4.6."""
        observed = [-1.0, 2.159, -1.0, 1.795, 1.310]
        average, dropped = mean(observed)
        assert dropped == 2
        assert average == 1.755
        assert average > min(v for v in observed if v >= 0) * 0.9

    def test_nothing_usable_returns_nothing(self):
        assert mean([-1.0, -1.0]) == (None, 2)
        assert mean([None, None]) == (None, 2)

    def test_missing_and_cancelled_both_counted_as_dropped(self):
        average, dropped = mean([1.0, None, -1.0, 3.0])
        assert (average, dropped) == (2.0, 2)


class TestEndToEnd:
    def test_a_turn_missing_a_stage_has_no_end_to_end(self):
        assert Turn(index=1, asr_final_s=0.2, llm_ttft_s=0.3).end_to_end_s is None

    def test_a_cancelled_stage_does_not_become_zero(self):
        turn = Turn(index=1, asr_final_s=0.2, llm_ttft_s=0.3, tts_ttfb_s=-1.0)
        assert turn.end_to_end_s is None

    def test_a_complete_turn_adds_up(self):
        turn = Turn(index=1, asr_final_s=0.2, llm_ttft_s=0.3, tts_ttfb_s=0.1)
        assert turn.end_to_end_s == 0.6
