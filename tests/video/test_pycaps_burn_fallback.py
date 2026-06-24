"""Fallback handling for a runtime pycaps burn failure.

A runtime pycaps render failure (e.g. the CSS renderer timing out with no
display) used to keep the caption-less assembled video and report success under
`fallback_ffmpeg`, so subtitle-less videos shipped silently. The burn step now
routes a runtime failure through `_handle_pycaps_burn_failure`: only
`warn_and_skip` keeps the caption-less video; `raise` and `fallback_ffmpeg` both
abort. `fallback_ffmpeg` still degrades to ffmpeg for the pycaps-*unavailable*
case, which is handled earlier in `step_generate_subtitles`, not here.
"""

import pytest

from src.video.producer.context import PipelineError
from src.video.producer.steps import _handle_pycaps_burn_failure

MSG = "pycaps render failed: boom. template=word-focus, renderer=css"


def test_warn_and_skip_keeps_caption_less_video(caplog):
    """warn_and_skip returns without raising (caller keeps the bare video)."""
    with caplog.at_level("WARNING"):
        result = _handle_pycaps_burn_failure("warn_and_skip", MSG)
    assert result is None
    assert any("caption-less" in r.message for r in caplog.records)


@pytest.mark.parametrize("policy", ["raise", "fallback_ffmpeg"])
def test_runtime_failure_aborts(policy):
    """Both raise and fallback_ffmpeg abort a runtime burn failure."""
    with pytest.raises(PipelineError, match="pycaps render failed"):
        _handle_pycaps_burn_failure(policy, MSG)
