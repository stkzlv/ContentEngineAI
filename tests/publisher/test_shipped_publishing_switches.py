"""The bundled config turns the TikTok AI label off and short titles on."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.publisher.config import parse_tiktok_settings

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.req("REQ-CMP-016", "REQ-OPS-121")
def test_the_bundled_config_keeps_the_tiktok_ai_label_on() -> None:
    """The public default discloses; an installation that meets TikTok's
    exemption turns it off in its overlay (decision 0013).
    """
    raw = yaml.safe_load((ROOT / "config" / "publisher.yaml").read_text())

    assert parse_tiktok_settings(raw["tiktok_settings"]).video_made_with_ai is True


@pytest.mark.req("REQ-PUB-008", "REQ-CNT-149")
def test_the_bundled_config_turns_short_product_titles_on() -> None:
    from src.video.config import load_video_config_modular

    assert load_video_config_modular().description_settings.short_product_titles


@pytest.mark.req("REQ-OPS-121")
def test_the_bundled_config_names_no_installation() -> None:
    """Schedule, funnel and author voice are an installation's own (design 0026)."""
    publisher = yaml.safe_load((ROOT / "config" / "publisher.yaml").read_text())
    schedule = publisher["recurring_schedule"]
    assert schedule["timezone"] == "UTC" and len(schedule["slots"]) == 1
    assert publisher["link_in_bio"]["enabled"] is False
    for template in publisher["first_comment"]["platforms"].values():
        assert "bio" not in template.lower()

    from src.video.config import load_video_config_modular

    templates = load_video_config_modular().llm_settings.script_templates
    for line in templates.cta_options + templates.cta_options_topic:
        assert "bio" not in line.lower() and "link" not in line.lower(), line
    assert templates.signature.enabled is False
