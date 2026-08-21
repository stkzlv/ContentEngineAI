"""Tests for rendering from a topic instead of a scraped product.

Split the same way as the stock-keyword tests in `tests/test_video_config.py`:
one group proves the record is built correctly, a separate group proves the
values actually reach the search. A field that is set but never read looks
identical to a working one.
"""

import pytest
import yaml

from src.scraper.amazon.models import ProductData
from src.scraper.base.models import Platform
from src.utils import sanitize_filename
from src.video.producer.steps import resolve_topic_keywords
from src.video.producer.topic_input import (
    TOPIC_ID_PREFIX,
    TopicInputError,
    TopicSpec,
    build_topic_product,
    load_topics_file,
    topic_product_id,
    topic_slug,
)


@pytest.mark.unit
class TestTopicField:
    """`topic` is what distinguishes a topic record from a scraped one."""

    def test_defaults_to_none_on_a_scraped_record(self):
        product = ProductData(
            title="A product",
            price="$9.99",
            url="https://x/dp/B0",
            platform=Platform.AMAZON,
        )
        assert product.topic is None

    def test_survives_to_dict(self):
        """`data.json` for a topic is written from `to_dict`.

        A field missing from that hand-written dict is absent from the file and
        therefore gone on the next resume, with nothing to signal it.
        """
        product = build_topic_product(TopicSpec(title="Why wifi drops"))
        assert product.to_dict()["topic"] == "Why wifi drops"

    def test_round_trips_through_a_record_rebuild(self):
        """`ProductData(**data)` is how a run is resumed from disk."""
        original = build_topic_product(TopicSpec(title="Why wifi drops"))
        payload = original.to_dict()
        # `to_dict` serialises enums to their values; the rebuild takes them raw.
        payload["platform"] = Platform.AMAZON
        payload.pop("status", None)
        rebuilt = ProductData(**payload)
        assert rebuilt.topic == original.topic


@pytest.mark.unit
class TestTopicSlug:
    def test_is_deterministic(self):
        assert topic_slug("Why your wifi drops") == topic_slug("Why your wifi drops")

    def test_readable_part_ignores_case_and_punctuation(self):
        assert topic_slug("Why your WiFi drops!") == topic_slug("why your wifi drops")

    def test_identifiers_never_collide_across_distinct_titles(self):
        """The readable slug alone is not unique, and a collision is silent.

        It truncates, and it drops everything outside the Latin alphabet, so two
        long titles sharing a prefix land on one identifier and any two non-Latin
        titles both reduce to "untitled". The second run then inherits the
        first's completed pipeline state, skips every step, and reports success
        while returning the first topic's video.
        """
        titles = [
            "Why your home wifi keeps dropping out during video calls and how to fix it",
            "Why your home wifi keeps dropping out during video calls and what to try first",
            "Почему падает интернет",
            "为什么你的无线网络会掉线",
        ]
        ids = [topic_product_id(t) for t in titles]
        assert len(set(ids)) == len(titles), ids

    def test_identifier_is_stable_for_the_same_title(self):
        """Resume depends on it: a re-run must find its own directory."""
        assert topic_product_id("Why wifi drops") == topic_product_id("Why wifi drops")

    @pytest.mark.parametrize(
        "title",
        [
            "Fix a leaky faucet: the cheap way",
            "Why your WiFi keeps dropping!",
            'Slashes / colons : quotes " and *stars*',
            "Em — dashes and  double  spaces",
            "ПОЧЕМУ WI-FI",
            "!!!",
            "a" * 300,
        ],
    )
    def test_identifier_is_a_fixed_point_of_sanitize_filename(self, title):
        """The directory is sanitized but the video filename is not.

        `get_product_paths` sanitizes the id for the directory and interpolates
        the raw id into the filename. If they differ, the video lands beside the
        directory the publisher looks in, and nothing finds it.
        """
        product_id = topic_product_id(title)
        assert sanitize_filename(product_id) == product_id

    def test_empty_after_stripping_falls_back(self):
        assert topic_slug("!!!") == "untitled"

    def test_is_prefixed(self):
        assert topic_product_id("Why wifi drops").startswith(TOPIC_ID_PREFIX)


@pytest.mark.unit
class TestBuildTopicProduct:
    def test_identifier_names_the_run(self):
        product = build_topic_product(TopicSpec(title="Why wifi drops"))
        assert product.asin is not None
        assert product.asin.startswith("topic-why-wifi-drops-")

    def test_listing_fields_are_left_empty(self):
        """A topic has no listing, so these carry nothing rather than a guess."""
        product = build_topic_product(TopicSpec(title="Why wifi drops"))
        assert product.price == ""
        assert product.url == ""
        assert product.images == []
        assert product.videos == []

    def test_description_reaches_the_record(self):
        """The script generator reads only the title and the description."""
        product = build_topic_product(
            TopicSpec(title="Why wifi drops", description="Router placement.")
        )
        assert product.description == "Router placement."


@pytest.mark.unit
class TestTopicSpecValidation:
    """`--topic` builds a spec inline, so validation has to live on the spec.

    Otherwise the flag accepts what the file loader rejects, and the two entry
    points disagree about what a valid topic is.
    """

    @pytest.mark.parametrize("title", ["", "   ", "\t\n"])
    def test_blank_titles_are_rejected(self, title):
        with pytest.raises(TopicInputError):
            TopicSpec(title=title)

    def test_title_is_trimmed(self):
        assert TopicSpec(title="  Why wifi drops  ").title == "Why wifi drops"


@pytest.mark.unit
class TestResolveTopicKeywords:
    """What the stock search actually receives.

    The provider joins the terms into a single query string, so a phrase split
    into separate words searches for something else entirely.
    """

    def test_scraped_product_yields_nothing(self):
        product = ProductData(
            title="A product",
            price="$9.99",
            url="https://x/dp/B0",
            platform=Platform.AMAZON,
        )
        assert resolve_topic_keywords(product) == []

    def test_phrases_survive_the_round_trip(self):
        product = build_topic_product(
            TopicSpec(title="Why wifi drops", keywords=["wifi router", "home network"])
        )
        assert resolve_topic_keywords(product) == ["wifi router", "home network"]

    def test_topic_without_keywords_yields_nothing(self):
        product = build_topic_product(TopicSpec(title="Why wifi drops"))
        assert resolve_topic_keywords(product) == []


@pytest.mark.unit
class TestLoadTopicsFile:
    def _write(self, tmp_path, payload):
        path = tmp_path / "topics.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def test_parses_a_well_formed_file(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                {
                    "title": "Why wifi drops",
                    "description": "Router placement.",
                    "keywords": ["wifi router"],
                },
                {"title": "Loud laptop fan"},
            ],
        )
        specs = load_topics_file(path)
        assert [s.title for s in specs] == ["Why wifi drops", "Loud laptop fan"]
        assert specs[0].keywords == ["wifi router"]
        assert specs[1].description == ""

    @pytest.mark.parametrize(
        "payload",
        [
            [{"description": "no title"}],
            [{"title": ""}],
            [{"title": "ok", "keywords": "not-a-list"}],
            [{"title": "ok", "descrption": "typo"}],
            ["a bare string"],
            {"title": "not a list"},
        ],
    )
    def test_malformed_entries_raise(self, tmp_path, payload):
        """Raised, not skipped.

        A topics file is written by hand. Dropping a bad line silently renders
        fewer videos than asked for, and nothing says which one went missing.
        """
        path = self._write(tmp_path, payload)
        with pytest.raises(TopicInputError):
            load_topics_file(path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "topics.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(TopicInputError):
            load_topics_file(path)

    def test_duplicate_titles_raise(self, tmp_path):
        """The same title twice would render into one directory, twice."""
        path = self._write(
            tmp_path, [{"title": "Why WiFi drops"}, {"title": "Why WiFi drops"}]
        )
        with pytest.raises(TopicInputError, match="identifier"):
            load_topics_file(path)

    def test_titles_differing_only_in_case_do_not_collide(self, tmp_path):
        """They share a readable slug but not an identifier, by design."""
        path = self._write(
            tmp_path, [{"title": "Why WiFi drops"}, {"title": "why wifi drops!"}]
        )
        assert len(load_topics_file(path)) == 2


@pytest.mark.unit
class TestStockProfile:
    """The bundled stock-only profile the topic path renders through."""

    def test_is_configured_for_stock_only(self):
        from src.video.config import config

        profile = config.video_profiles["slideshow_stock"]
        assert profile.use_scraped_images is False
        assert profile.use_scraped_videos is False
        assert profile.use_stock_images is True

    def test_gathers_enough_images_for_a_video_free_profile(self):
        """Below the floor the run is reported SKIPPED, not failed.

        That reads as a missing product rather than a misconfigured profile, so
        the shortfall is easy to misdiagnose.
        """
        from src.video.config import config

        profile = config.video_profiles["slideshow_stock"]
        assert profile.stock_image_count >= config.video_settings.min_images_if_no_video

    def test_declares_empty_keywords_rather_than_inheriting(self):
        """Omitting the key inherits product-oriented global terms.

        Those get concatenated into the same query string as the topic's own
        terms, which is what makes a topic search return product stock footage.
        """
        from src.video.config import config

        assert config.video_profiles["slideshow_stock"].stock_media_keywords == []

    def test_is_excluded_from_random_selection(self):
        """A random product batch drawing this profile would find no imagery."""
        from src.video.producer.utils import EXCLUDED_RANDOM_PROFILES

        assert "slideshow_stock" in EXCLUDED_RANDOM_PROFILES
