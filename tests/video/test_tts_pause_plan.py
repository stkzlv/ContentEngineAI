"""Context-varied pauses instead of one pause after every sentence (#439).

The bundled profiles put the same `[short pause]` after every sentence, and
uniform pause length is a robotic tell. A pause plan pauses longer where the
script turns, not at all after the hook, and jitters the rest by product.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from src.video.config import SILENT_TTS_TAGS, PausePlan
from src.video.config.audio_models import (
    GoogleCloudTTSSettings,
    GoogleCloudVoiceCriteria,
    TextMarkupRule,
    TTSConfig,
    VoiceProfileConfig,
)
from src.video.tts import TTSManager, apply_pause_plan

SCRIPT = (
    "Your phone gets hot when you charge it. The case traps the heat. "
    "Phones throttle when warm. It feels slow after that.\n"
    "Take the case off first. Plug it back in. Let it breathe. "
    "Put the case back when it is cool.\n"
    "Save this for the next time it happens."
)
TAG = re.compile(r"\[[a-z ]+\]")


def _tags_between_sentences(text: str) -> list[str]:
    """The tag (or "") at each sentence boundary, in order."""
    gaps = re.split(r"(?<=[.!?])", text)[1:-1]
    return [m.group(0) if (m := TAG.search(g)) else "" for g in gaps]


class TestWhereThePausesGo:
    def test_no_jitter_is_the_plain_plan(self) -> None:
        tags = _tags_between_sentences(
            apply_pause_plan(SCRIPT, PausePlan(jitter=0), "B0X")
        )
        assert tags == [
            "",  # after the hook
            "[short pause]",
            "[short pause]",
            "[medium pause]",  # paragraph break
            "[short pause]",
            "[short pause]",
            "[short pause]",
            "[medium pause]",  # before the closing line
        ]

    def test_the_hook_and_closing_tags_win_over_a_line_break(self) -> None:
        """One sentence per line, the shape the topic scripts take: the hook
        and the next-to-last sentence both end a paragraph.
        """
        script = "Hook line.\nMiddle one.\nMiddle two.\nFollow for more."
        plan = PausePlan(
            after_hook="",
            paragraph="[long pause]",
            before_last="[medium pause]",
            jitter=0,
        )
        tags = _tags_between_sentences(apply_pause_plan(script, plan, "B0X"))
        assert tags == ["", "[long pause]", "[medium pause]"]

    def test_an_unpunctuated_hook_line_keeps_its_break(self) -> None:
        out = apply_pause_plan(
            "POV: your router at 2am\nIt drops every call. Here is why.",
            PausePlan(jitter=0),
            "B0X",
        )
        assert out.startswith("POV: your router at 2am\nIt drops")

    def test_nothing_follows_the_last_sentence(self) -> None:
        out = apply_pause_plan(SCRIPT, PausePlan(), "B0X")
        assert out.endswith("Save this for the next time it happens.")

    def test_a_single_sentence_is_untouched(self) -> None:
        assert apply_pause_plan("Just one.", PausePlan(), "B0X") == "Just one."


class TestTheJitter:
    def test_it_is_reproducible_per_product(self) -> None:
        plan = PausePlan(jitter=1.0)
        assert apply_pause_plan(SCRIPT, plan, "B0A") == apply_pause_plan(
            SCRIPT, plan, "B0A"
        )

    def test_it_varies_across_products(self) -> None:
        plan = PausePlan(jitter=1.0)
        outs = {apply_pause_plan(SCRIPT, plan, f"B0{i:04d}") for i in range(20)}
        assert len(outs) > 1

    def test_it_only_touches_ordinary_boundaries(self) -> None:
        for i in range(20):
            tags = _tags_between_sentences(
                apply_pause_plan(SCRIPT, PausePlan(jitter=1.0), f"B0{i:04d}")
            )
            assert tags[0] == ""
            assert tags[3] == "[medium pause]"
            assert tags[-1] == "[medium pause]"
            assert set(tags[1:3] + tags[4:7]) <= {"", "[medium pause]"}

    def test_no_product_means_no_jitter(self) -> None:
        plan = PausePlan(jitter=1.0)
        assert apply_pause_plan(SCRIPT, plan, None) == apply_pause_plan(
            SCRIPT, PausePlan(jitter=0), None
        )


class TestOnlySilentTagsAreAccepted:
    @pytest.mark.parametrize("tag", ["[uhm]", "[scared]", "[breath]", "pause"])
    def test_a_spoken_or_unmeasured_tag_is_refused(self, tag: str) -> None:
        with pytest.raises(ValidationError):
            PausePlan(sentence=tag)

    @pytest.mark.parametrize("tag", sorted(SILENT_TTS_TAGS) + [""])
    def test_a_measured_tag_or_nothing_is_accepted(self, tag: str) -> None:
        assert PausePlan(paragraph=tag).paragraph == tag

    def test_an_unknown_key_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            PausePlan.model_validate({"sentense": "[short pause]"})


class TestFallbackProvidersNeverReadATagAloud:
    @pytest.mark.parametrize("tag", sorted(SILENT_TTS_TAGS))
    def test_every_silent_tag_is_stripped(self, tag: str) -> None:
        assert TTSManager._strip_markup(f"One. {tag} Two.") == "One. Two."

    def test_stripping_keeps_the_paragraph_breaks(self) -> None:
        planned = apply_pause_plan(
            "Hook.\nStep two\nNext sentence.\nClose.", PausePlan(jitter=0), "B0X"
        )
        stripped = TTSManager._strip_markup(planned)
        assert re.search(r"Step two[^\S\n]*\nNext sentence\.", stripped)
        assert "[" not in stripped


def _config(profile: VoiceProfileConfig) -> TTSConfig:
    return TTSConfig(
        provider_order=["google_cloud"],
        google_cloud=GoogleCloudTTSSettings(
            language_code="en-US",
            voice_selection_criteria=[GoogleCloudVoiceCriteria(language_code="en-US")],
        ),
        voice_profiles={"p": profile},
        default_voice_profile="p",
    )


class TestTheProfileWiring:
    @pytest.mark.asyncio
    async def test_a_plan_replaces_the_markup_rules(self) -> None:
        profile = VoiceProfileConfig(
            provider="gemini",
            markup_rules=[
                TextMarkupRule(pattern=r"\.\s+", insert_after="[long pause] ")
            ],
            pause_plan=PausePlan(jitter=0),
        )
        manager = TTSManager(_config(profile), {}, product_id="B0X")
        out = Path("/tmp/x.wav")  # noqa: S108
        with patch(
            "src.video.tts._generate_gemini_speech", new_callable=AsyncMock
        ) as gemini:
            gemini.return_value = (out, "Charon")
            await manager.generate_speech(SCRIPT, out)
        sent = gemini.call_args.args[0]
        assert "[long pause]" not in sent
        assert sent == apply_pause_plan(SCRIPT, PausePlan(jitter=0), "B0X")

    @pytest.mark.asyncio
    async def test_the_fallback_gets_the_text_without_tags(self) -> None:
        profile = VoiceProfileConfig(provider="gemini", pause_plan=PausePlan(jitter=0))
        manager = TTSManager(_config(profile), {}, product_id="B0X")
        out = Path("/tmp/x.wav")  # noqa: S108
        with (
            patch(
                "src.video.tts._generate_gemini_speech", new_callable=AsyncMock
            ) as gemini,
            patch(
                "src.video.tts._generate_google_cloud_speech", new_callable=AsyncMock
            ) as cloud,
        ):
            gemini.return_value = (None, None)
            cloud.return_value = (out, "en-US-Chirp3")
            await manager.generate_speech(SCRIPT, out)
        assert "[" not in cloud.call_args.args[0]
