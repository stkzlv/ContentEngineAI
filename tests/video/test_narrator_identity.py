"""A narrator identity pins what every render of its kind keeps the same.

REQ-CNT-168 (the block and its kinds), REQ-CNT-169 (the sign-off on every
render), REQ-CNT-170 (the variety report reads pins as settings).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ai.llm_settings import SignatureConfig, SignaturePools
from src.ai.script_generator import select_signature
from src.scraper.amazon.models import ProductData
from src.utils.script_signoff import place_signoff
from src.video.config.audio_models import SignatureSting
from src.video.config.identity_models import NarratorIdentity, content_kind
from src.video.producer.orchestration import config_for_record
from src.video.producer.topic_input import TopicSpec, build_topic_product
from src.video.render_choices import dominance_alerts, report

CTA = "Follow for more fixes like this."


@pytest.fixture
def config():
    from src.video.config_adapter import load_video_config_modular

    return load_video_config_modular()


def _product() -> ProductData:
    return ProductData(
        title="A thing", price="", url="", platform="test", asin="B0TEST0001"
    )


def _topic(kind: str = "tutorial") -> ProductData:
    return build_topic_product(TopicSpec(title="A subject", kind=kind))


@pytest.mark.req("REQ-CNT-168")
class TestTheBlock:
    def test_the_kinds(self):
        assert content_kind(_product()) == "product"
        assert content_kind(_topic()) == "topic"
        assert content_kind(_topic("commentary")) == "commentary"

    def test_an_empty_identity_keeps_the_config(self, config):
        assert config_for_record(config, _product()) is config
        assert config_for_record(config, _topic()) is config

    def test_every_pin_lands_on_a_copy(self, config, tmp_path: Path):
        bed = tmp_path / "bed.mp3"
        config.narrator_identity.product = NarratorIdentity(
            voice_profile="pinned_voice",
            signoff="That's the take.",
            sting=SignatureSting(path=tmp_path / "sting.wav", position="end"),
            music_files=[bed],
            caption_template="pinned_template",
        )
        record = config_for_record(config, _product())
        assert record is not config
        assert record.tts_config.default_voice_profile == "pinned_voice"
        assert record.tts_config.voice_profile_pool == []
        signature = record.llm_settings.script_templates.signature
        assert signature.product.pinned_signoff == "That's the take."
        assert signature.topic.pinned_signoff == ""
        assert record.audio_settings.signature_sting.position == "end"
        assert record.audio_settings.background_music_paths == [bed]
        assert record.audio_settings.audio_providers == []
        assert record.subtitle_settings["pycaps"]["template_name"] == "pinned_template"
        assert record.subtitle_settings["pycaps"]["template_pool"] == []
        # The shared config is untouched for the next record.
        assert config.tts_config.default_voice_profile != "pinned_voice"
        assert config.audio_settings.audio_providers

    def test_music_queries_replace_the_providers_queries(self, config):
        config.narrator_identity.topic = NarratorIdentity(music_queries=["dark synth"])
        record = config_for_record(config, _topic())
        queries = [
            p.settings["search_queries"]
            for p in record.audio_settings.audio_providers
            if "search_queries" in p.settings
        ]
        assert queries and all(q == ["dark synth"] for q in queries)
        assert record.audio_settings.freesound_search_query == "dark synth"

    def test_each_kind_reads_its_own_identity(self, config):
        config.narrator_identity.commentary = NarratorIdentity(
            voice_profile="take_voice"
        )
        assert config_for_record(config, _topic()) is config
        commentary = config_for_record(config, _topic("commentary"))
        assert commentary.tts_config.default_voice_profile == "take_voice"
        # The commentary family still applies with the identity.
        templates = commentary.llm_settings.script_templates
        assert templates.topic_templates == templates.commentary_templates


@pytest.mark.req("REQ-CNT-169")
class TestThePinnedSignoff:
    def test_it_is_drawn_even_with_the_signature_off(self):
        signature = SignatureConfig(
            enabled=False, product=SignaturePools(pinned_signoff="That's the take.")
        )
        assert select_signature(signature, "B0TEST0001").signoff == "That's the take."

    def test_it_ignores_the_use_rate(self):
        signature = SignatureConfig(
            enabled=True,
            product=SignaturePools(
                use_rate=0.0, signoffs=["Other one."], pinned_signoff="Pinned."
            ),
        )
        assert select_signature(signature, "B0TEST0001").signoff == "Pinned."

    def test_a_missing_pinned_signoff_is_inserted_before_the_cta(self):
        script = f"The fix is the channel. Set it to six. {CTA}"
        placed = place_signoff(script, "And you're done", CTA, insert=True)
        assert (
            placed == f"The fix is the channel. Set it to six. And you're done. {CTA}"
        )

    def test_without_insert_a_missing_signoff_stays_missing(self):
        script = f"The fix is the channel. Set it to six. {CTA}"
        assert place_signoff(script, "And you're done.", CTA) == script

    def test_a_spoken_signoff_is_not_doubled(self):
        script = f"The fix is the channel. And you're done. {CTA}"
        assert place_signoff(script, "And you're done.", CTA, insert=True) == script


@pytest.mark.req("REQ-CNT-170")
class TestTheVarietyReport:
    def _rows(self, n: int, pinned: list[str]) -> list[dict]:
        return [
            {"product_id": f"B0TEST{i:04d}", "voice_profile": "same", "pinned": pinned}
            for i in range(n)
        ] + [{"product_id": "B0TESTOTHER", "voice_profile": "other", "pinned": []}]

    def test_a_pinned_dimension_does_not_alert(self):
        assert dominance_alerts(self._rows(6, ["voice_profile"]), 0.6) == []

    def test_the_same_rows_unpinned_alert(self):
        alerts = dominance_alerts(self._rows(6, []), 0.6)
        assert [a[0] for a in alerts] == ["voice_profile"]

    def test_the_report_names_the_pin(self):
        lines = report(self._rows(6, ["voice_profile"]), 0.6, 0.9)
        assert any(
            "voice_profile: same 6, other 1 (pinned in 6)" in line for line in lines
        )

    def test_pinned_dimensions(self):
        identity = NarratorIdentity(
            voice_profile="v", caption_template="c", music_files=[Path("a.mp3")]
        )
        assert identity.pinned_dimensions() == [
            "voice_profile",
            "caption_template",
            "music",
        ]
        two_beds = NarratorIdentity(music_files=[Path("a.mp3"), Path("b.mp3")])
        assert two_beds.pinned_dimensions() == []
