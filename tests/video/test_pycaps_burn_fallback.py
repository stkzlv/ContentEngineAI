"""Fallback handling for the burn failures that cannot degrade.

A burn-step failure used to keep the caption-less video and report success
under `fallback_ffmpeg`, so subtitle-less videos shipped silently. Failures
now route through `_handle_pycaps_burn_failure`, where only `warn_and_skip`
keeps the caption-less video and `raise` and `fallback_ffmpeg` both abort.

Two of the four burn failures reach that helper. A missing transcript leaves
nothing to build captions from and a missing assembled video leaves nothing
to burn them onto, so neither can degrade. The other two -- a pycaps render
failure, and pycaps having vanished since the run that recorded the engine --
have both on disk and burn captions with FFmpeg instead; those are covered in
`test_ffmpeg_fallback_on_burn_failure.py`, which is why the message below is
a missing-transcript one rather than the render-failure message this file
used to carry.
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

    A render failure no longer reaches this helper under `fallback_ffmpeg`;
    it burns with FFmpeg instead. What still reaches it is a missing
    transcript or a missing assembled video, which is what `MSG` now is.
    """
    with pytest.raises(PipelineError, match="transcript is missing"):
        _handle_pycaps_burn_failure(policy, MSG)
