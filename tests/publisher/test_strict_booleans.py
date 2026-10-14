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
            ('cleanup:\n  enabled: "false"\n', "cleanup.enabled"),
            ('covers:\n  enabled: "true"\n', "covers.enabled"),
            ("link_in_bio:\n  enabled: 1\n", "link_in_bio.enabled"),
            (
                'tiktok_settings:\n  video_made_with_ai: "false"\n',
                "tiktok_settings.video_made_with_ai",
            ),
        ],
    )
    def test_a_quoted_or_numeric_switch_is_refused(self, tmp_path, text, name):
        with pytest.raises(ValueError, match=name.replace(".", r"\.")):
            _load(tmp_path, text)

    @pytest.mark.parametrize(
        ("text", "name"),
        [
            # A refused sibling sends the section to its defaults, which used
            # to drop the quoted switch unseen and turn cleanup on.
            (
                'cleanup:\n  enabled: "false"\n  keep_published_days: -1\n',
                "cleanup.enabled",
            ),
            ('covers:\n  enabled: "true"\n  unknown_key: 1\n', "covers.enabled"),
            (
                'tiktok_settings:\n  video_made_with_ai: "false"\n  unknown_key: 1\n',
                "tiktok_settings.video_made_with_ai",
            ),
            (
                'recurring_schedule:\n  enabled: "false"\n'
                "schedule_validation:\n  max_posts_per_day: -5\n",
                "recurring_schedule.enabled",
            ),
        ],
    )
    def test_a_section_that_falls_back_still_refuses_the_switch(
        self, tmp_path, text, name
    ):
        with pytest.raises(ValueError, match=name.replace(".", r"\.")):
            _load(tmp_path, text)

    def test_the_shipped_config_passes(self, monkeypatch):
        monkeypatch.delenv("LATE_API_KEY", raising=False)
        load_publisher_config(cli_overrides=KEY)


@pytest.mark.unit
@pytest.mark.req("REQ-PUB-160")
def test_every_parsed_section_is_checked_raw():
    # A section added to the parser but not to `_raw_sections` would let a
    # quoted switch through whenever a sibling key falls back.
    import re

    from src.publisher import config as config_mod

    source = Path(config_mod.__file__).read_text(encoding="utf-8")
    # A section is read as a mapping (`.get("name", {})`), tiktok_settings
    # through its own parser.
    parsed = set(re.findall(r'\.get\("([a-z_]+)", \{\}\)', source))
    parsed |= set(
        re.findall(r'parse_tiktok_settings\(result\.get\("([a-z_]+)"', source)
    )
    # Accounts hold credentials and names, no switches.
    parsed -= {"accounts"}
    assert parsed <= set(config_mod._raw_sections())
