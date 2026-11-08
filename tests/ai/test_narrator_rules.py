"""Narrator rules kept apart from the narrator profile (design 0032)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ai.llm_settings import ScriptTemplateConfig, reusable_examples
from src.video.config import config
from src.video.producer.orchestration import config_for_record

SNAPSHOTS = Path(__file__).parent.parent / "fixtures" / "narrator"


def _commentary(templates):
    record = SimpleNamespace(topic="A take", topic_kind="commentary")
    cfg = config.model_copy(deep=True)
    cfg.llm_settings.script_templates = templates
    return config_for_record(cfg, record).llm_settings.script_templates


@pytest.mark.req("REQ-CNT-171")
def test_the_bundled_prompts_are_byte_identical_to_the_old_profiles() -> None:
    t = config.llm_settings.script_templates

    assert t.narrator_for(False) == (SNAPSHOTS / "product.txt").read_text()
    assert t.narrator_for(True) == (SNAPSHOTS / "topic.txt").read_text()
    commentary = _commentary(t.model_copy(deep=True)).narrator_for(True)
    assert commentary == (SNAPSHOTS / "commentary.txt").read_text()


@pytest.mark.req("REQ-CNT-171")
def test_an_overlay_that_sets_only_a_profile_keeps_the_rules() -> None:
    t = config.llm_settings.script_templates.model_copy(deep=True)
    t.narrator_profile_commentary = "Narrator profile: a different person."

    text = _commentary(t).narrator_for(True)

    assert text.startswith("Narrator profile: a different person.\n\nCTA:")
    assert "{TARGET_SECONDS}" not in text  # filled from the commentary target
    assert "Do not invent a figure" in text


@pytest.mark.req("REQ-CNT-171")
def test_an_empty_topic_rules_key_falls_back_to_the_product_rules() -> None:
    t = ScriptTemplateConfig(
        narrator_profile="P.", narrator_rules="R.", narrator_profile_topic="T."
    )

    assert t.narrator_for(True) == "T.\n\nR."
    assert ScriptTemplateConfig(narrator_profile="P.").narrator_for(True) == "P."


@pytest.mark.req("REQ-CNT-172")
def test_the_bundled_config_raises_no_narrator_warning(caplog) -> None:
    raw = config.llm_settings.script_templates.model_dump()

    with caplog.at_level(logging.WARNING, logger="src.ai.llm_settings"):
        ScriptTemplateConfig(**raw)

    assert "narrator" not in caplog.text


@pytest.mark.req("REQ-CNT-172")
def test_a_narrator_without_a_length_target_is_named(caplog) -> None:
    raw = config.llm_settings.script_templates.model_dump()
    raw["narrator_profile_commentary"] = "Narrator profile: a different person."
    raw["narrator_rules_commentary"] = "CTA: end on the call to action."

    with caplog.at_level(logging.WARNING, logger="src.ai.llm_settings"):
        ScriptTemplateConfig(**raw)

    assert "The commentary narrator carries no length target" in caplog.text
    assert "The product narrator" not in caplog.text


@pytest.mark.req("REQ-CNT-172")
def test_a_quoted_line_a_script_could_reuse_is_named(caplog) -> None:
    raw = config.llm_settings.script_templates.model_dump()
    raw["narrator_profile_commentary"] = (
        'Close like this: "So who answers the next question?"'
    )

    with caplog.at_level(logging.WARNING, logger="src.ai.llm_settings"):
        ScriptTemplateConfig(**raw)

    assert "So who answers the next question?" in caplog.text


@pytest.mark.req("REQ-CNT-172", "REQ-CNT-159")
def test_a_shape_with_a_slot_or_a_short_label_is_no_reusable_line() -> None:
    assert reusable_examples('"it weighs [figure] in total."') == []
    assert reusable_examples('"heavier than I expected"') == []
    assert reusable_examples('"Steel beats plastic for clamps."') == [
        "Steel beats plastic for clamps."
    ]
