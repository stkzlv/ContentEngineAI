"""A publisher switch takes only a YAML boolean (REQ-PUB-160).

The config dataclasses take values as written, so a quoted "false" was a
non-empty string and read as on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.publisher.config import load_publisher_config

KEY = {"api_key": "sk_test_" + "x" * 40}


def _load(tmp_path: Path, text: str):
    path = tmp_path / "publisher.yaml"
    path.write_text(text, encoding="utf-8")
    return load_publisher_config(config_path=path, cli_overrides=KEY)


@pytest.mark.unit
@pytest.mark.req("REQ-PUB-160")
class TestStrictBooleans:
    def test_real_booleans_load(self, tmp_path):
        config = _load(
            tmp_path,
            "youtube_profile_pointer: true\ncleanup:\n  enabled: false\n"
            "covers:\n  enabled: true\n",
        )
        assert config.youtube_profile_pointer is True
        assert config.cleanup_config.enabled is False
        assert config.covers_config.enabled is True

    @pytest.mark.parametrize(
        ("text", "name"),
        [
            ('youtube_profile_pointer: "false"\n', "youtube_profile_pointer"),
            ('immediate_publish: "no"\n', "immediate_publish"),
            ('cleanup:\n  enabled: "false"\n', "cleanup_config.enabled"),
            ('covers:\n  enabled: "true"\n', "covers_config.enabled"),
            ("link_in_bio:\n  enabled: 1\n", "link_in_bio_config.enabled"),
            (
                'tiktok_settings:\n  video_made_with_ai: "false"\n',
                "tiktok_settings.video_made_with_ai",
            ),
        ],
    )
    def test_a_quoted_or_numeric_switch_is_refused(self, tmp_path, text, name):
        with pytest.raises(ValueError, match=name.replace(".", r"\.")):
            _load(tmp_path, text)

    def test_the_shipped_config_passes(self, monkeypatch):
        monkeypatch.delenv("LATE_API_KEY", raising=False)
        load_publisher_config(cli_overrides=KEY)
