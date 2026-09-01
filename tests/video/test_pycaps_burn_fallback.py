"""Fallback handling for the burn failures that cannot degrade.

A burn-step failure used to keep the caption-less video and report success
under `fallback_ffmpeg`, so subtitle-less videos shipped silently. Failures
now route through `_handle_pycaps_burn_failure`, where only `warn_and_skip`
keeps the caption-less video and `raise` and `fallback_ffmpeg` both abort.

Two of the four burn failures always reach that helper: a missing
transcript leaves nothing to build captions from and a missing assembled
video leaves nothing to burn them onto, so neither can degrade. A third,
a pycaps render failure, reaches it under every policy unless its FFmpeg
fallback succeeds. The fourth -- pycaps having vanished since the run
that recorded the engine -- degrades the same way and reaches the same
outcome under every policy. It differs only in mechanism: it resolves
the policy inline rather than through the helper. Both degrading cases are covered in
`test_ffmpeg_fallback_on_burn_failure.py`, which is why the message
below is a missing-transcript one rather than the render-failure message
this file used to carry.
"""

import pytest

from src.video.producer.context import PipelineError
from src.video.producer.steps import _handle_pycaps_burn_failure

MSG = "Pycaps mode requested but whisper transcript is missing at /tmp/x.json"


def test_warn_and_skip_keeps_caption_less_video(caplog):
    """warn_and_skip returns without raising (caller keeps the bare video)."""
    with caplog.at_level("WARNING"):
        _handle_pycaps_burn_failure("warn_and_skip", MSG)  # returns, no raise
    assert any("caption-less" in r.message for r in caplog.records)


@pytest.mark.parametrize("policy", ["raise", "fallback_ffmpeg"])
def test_an_undegradable_failure_aborts(policy):
    """Both policies abort the failures that cannot fall back.

    A render failure reaches this helper under `fallback_ffmpeg` only when the
    FFmpeg burn itself fails; normally it degrades instead. That fall-through
    is deliberate, so do not read it as dead code. What still reaches it is a missing
    transcript or a missing assembled video, which is what `MSG` now is.
    """
    with pytest.raises(PipelineError, match="transcript is missing"):
        _handle_pycaps_burn_failure(policy, MSG)
