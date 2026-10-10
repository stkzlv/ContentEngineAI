"""No caption effect scales a word past 1.15x (REQ-VID-162).

The research's safety limit for caption motion. Checked against what ships:
the pycaps templates in the bundled pool, read from the installed library,
and the FFmpeg pulse setting in the bundled config.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
MAX_SCALE = 1.15


def _bundled() -> dict:
    raw: dict = yaml.safe_load((REPO / "config" / "subtitles.yaml").read_text())
    return raw


@pytest.mark.unit
@pytest.mark.req("REQ-VID-162")
def test_the_ffmpeg_pulse_stays_within_the_limit() -> None:
    assert _bundled()["subtitle_effects"]["pulse_scale_max"] <= MAX_SCALE * 100


@pytest.mark.unit
@pytest.mark.req("REQ-VID-162")
def test_no_pool_template_scales_a_word_past_the_limit() -> None:
    pytest.importorskip(
        "pycaps", reason="optional group not installed (poetry install --with pycaps)"
    )
    from pycaps.template import TemplateFactory, TemplateLoader

    pool = _bundled()["subtitle_settings"]["pycaps"]["template_pool"]
    assert pool
    scaled = 0
    for name in pool:
        builder = TemplateLoader(TemplateFactory().create(name)).load(False)
        for animator in builder._caps_pipeline._animators:
            animation = animator._animation
            overshoot = getattr(animation, "_overshoot", None)
            peak = 1.0 + (overshoot.amount if overshoot else 0.0)
            scaled += overshoot is not None
            assert peak <= MAX_SCALE, (name, type(animation).__name__, peak)
            init = getattr(animation, "_init_scale", 1.0)
            assert init <= MAX_SCALE, (name, type(animation).__name__, init)
    # The getattr defaults would pass anything if pycaps renamed the fields;
    # `explosive` overshoots, so at least one must be seen.
    assert scaled, "no overshoot read; the attribute names may have changed"
