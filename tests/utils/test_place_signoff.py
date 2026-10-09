"""The sign-off is moved to directly before the CTA (REQ-CNT-045)."""

from __future__ import annotations

import pytest

from src.utils.script_signoff import place_signoff

SIGNOFF = "That's the real picture."
CTA = "Share with someone who needs this."


@pytest.mark.req("REQ-CNT-045")
def test_an_early_signoff_moves_before_the_cta() -> None:
    script = f"It lasts a week. {SIGNOFF} Steel beats silicone. {CTA}"

    assert place_signoff(script, SIGNOFF, CTA) == (
        f"It lasts a week. Steel beats silicone. {SIGNOFF} {CTA}"
    )


@pytest.mark.req("REQ-CNT-045")
@pytest.mark.parametrize(
    "script",
    [
        f"It lasts a week. Steel beats silicone. {SIGNOFF} {CTA}",  # in place
        f"It lasts a week. Steel beats silicone. {CTA}",  # not spoken
        f"It lasts a week. {SIGNOFF} Steel beats silicone.",  # no CTA at the end
    ],
)
def test_otherwise_the_script_is_unchanged(script: str) -> None:
    assert place_signoff(script, SIGNOFF, CTA) == script


@pytest.mark.req("REQ-CNT-045")
def test_a_signoff_with_other_punctuation_still_moves() -> None:
    script = f"It lasts a week. That's the real picture! Steel beats silicone. {CTA}"

    assert place_signoff(script, SIGNOFF, CTA) == (
        f"It lasts a week. Steel beats silicone. That's the real picture! {CTA}"
    )


@pytest.mark.req("REQ-CNT-045")
def test_no_draw_changes_nothing() -> None:
    script = f"It lasts a week. {SIGNOFF} Steel beats silicone. {CTA}"

    assert place_signoff(script, None, CTA) == script
    assert place_signoff(script, SIGNOFF, None) == script


@pytest.mark.req("REQ-CNT-045")
def test_the_script_step_places_the_drawn_signoff() -> None:
    import inspect

    from src.video.producer import steps

    source = inspect.getsource(steps.step_generate_script)
    assert (
        "place_signoff(\n                sanitize_script(script_text), signature.signoff"
        in (source)
    )
