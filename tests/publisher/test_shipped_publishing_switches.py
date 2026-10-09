"""The bundled config turns the TikTok AI label off and short titles on."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.publisher.config import parse_tiktok_settings

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.req("REQ-CMP-017")
def test_the_bundled_config_turns_the_tiktok_ai_label_off() -> None:
    raw = yaml.safe_load((ROOT / "config" / "publisher.yaml").read_text())

    assert parse_tiktok_settings(raw["tiktok_settings"]).video_made_with_ai is False


@pytest.mark.req("REQ-PUB-008", "REQ-CNT-149")
def test_the_bundled_config_turns_short_product_titles_on() -> None:
    from src.video.config import load_video_config_modular

    assert load_video_config_modular().description_settings.short_product_titles
