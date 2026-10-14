"""The prompts name a bio link only where the installation has one.

REQ-CNT-150: `youtube_profile_pointer` says whether the profiles carry a link.
With it on, the metadata and caption prompts are the text they always were;
off, none of them asks for or shows "Link in bio".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ai import description_generator
from src.scraper.amazon.models import ProductData
from src.utils import youtube_description
from src.utils.youtube_description import pointer_placeholders, profile_pointer_on

PROMPTS = Path(__file__).resolve().parents[2] / "src" / "ai" / "prompts"
CAPTION_PROMPTS = ["youtube_metadata.md", "tiktok_caption.md", "instagram_caption.md"]
EXTRA = {"CAPTION_STYLE": "style", "EMOJI_ENABLED": "yes"}

PRODUCT = ProductData(
    title="Test Gadget Wireless Charger",
    description="Charges fast.",
    asin="B0TESTPPT1",
    url="https://example.com/x",
    price="1",
    keyword="charger",
    platform="amazon",
)


def _config(tmp_path: Path, monkeypatch, text: str | None) -> None:
    path = tmp_path / "publisher.yaml"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(youtube_description, "PUBLISHER_CONFIG", path)


def _render(name: str) -> str:
    template = (PROMPTS / name).read_text(encoding="utf-8")
    return description_generator.format_prompt(
        template, PRODUCT, video_script="Script.", extra_placeholders=EXTRA
    )


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-150")
class TestTheSetting:
    def test_on(self, tmp_path, monkeypatch):
        _config(tmp_path, monkeypatch, "youtube_profile_pointer: true\n")

        assert profile_pointer_on() is True

    @pytest.mark.parametrize(
        "text",
        [
            "youtube_profile_pointer: false\n",
            "{}\n",
            'youtube_profile_pointer: "true"\n',
            "",
        ],
    )
    def test_anything_but_a_yaml_true_is_off(self, tmp_path, monkeypatch, text):
        _config(tmp_path, monkeypatch, text)

        assert profile_pointer_on() is False

    def test_a_missing_file_is_off(self, tmp_path, monkeypatch):
        _config(tmp_path, monkeypatch, None)

        assert profile_pointer_on() is False

    def test_the_private_overlay_turns_it_on(self, tmp_path, monkeypatch):
        _config(tmp_path, monkeypatch, "youtube_profile_pointer: false\n")
        (tmp_path / "publisher.private.yaml").write_text(
            "youtube_profile_pointer: true\n", encoding="utf-8"
        )
        monkeypatch.setenv("CONFIG_OVERLAYS", "on")

        assert profile_pointer_on() is True


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-150")
class TestThePrompts:
    @pytest.mark.parametrize("name", CAPTION_PROMPTS)
    def test_off_no_prompt_names_a_bio_link(self, tmp_path, monkeypatch, name):
        _config(tmp_path, monkeypatch, "youtube_profile_pointer: false\n")

        prompt = _render(name)

        assert "bio" not in prompt.lower()
        assert "profile pointer" not in prompt
        assert "{" not in prompt.split("**Output Format")[0][-200:]

    @pytest.mark.parametrize("name", CAPTION_PROMPTS)
    def test_on_the_prompt_is_its_original_text(self, tmp_path, monkeypatch, name):
        # Each placeholder's "on" value is the text the prompt carried before
        # it became a setting; putting them back must reproduce that text.
        _config(tmp_path, monkeypatch, "youtube_profile_pointer: true\n")
        template = (PROMPTS / name).read_text(encoding="utf-8")
        literal = template
        for key, value in pointer_placeholders(on=True).items():
            literal = literal.replace("{" + key + "}", value)

        assert _render(name) == description_generator.format_prompt(
            literal, PRODUCT, video_script="Script.", extra_placeholders=EXTRA
        )

    def test_on_the_youtube_prompt_asks_for_the_pointer(self, tmp_path, monkeypatch):
        _config(tmp_path, monkeypatch, "youtube_profile_pointer: true\n")

        prompt = _render("youtube_metadata.md")

        assert prompt.count("\n\nLink in bio.\n\nHASHTAGS") == 3
        assert '"Link in bio." on the final line' in prompt
        assert 'end with "Link in bio."]' in prompt
        assert "https://" not in prompt

    def test_off_the_youtube_prompt_still_forbids_a_url(self, tmp_path, monkeypatch):
        _config(tmp_path, monkeypatch, "youtube_profile_pointer: false\n")

        prompt = _render("youtube_metadata.md")

        assert "never write a URL." in prompt
        assert "skip the closing line.]" in prompt

    def test_a_prompt_without_the_placeholders_reads_no_config(
        self, tmp_path, monkeypatch
    ):
        # video_description.md names no bio link; a broken publisher config
        # must not reach it.
        _config(tmp_path, monkeypatch, "youtube_profile_pointer: [\n")

        assert "Test Gadget" in _render("video_description.md")


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-150")
@pytest.mark.req("REQ-CNT-033")
def test_the_default_script_prompt_closes_on_the_configured_cta():
    from src.ai.script_generator import format_prompt

    template = (PROMPTS / "video_script.md").read_text(encoding="utf-8")
    prompt = format_prompt(template, PRODUCT, "aud", cta_rule="- End on: CTA LINE")

    assert "- End on: CTA LINE" in prompt
    assert "bio" not in prompt.lower()
