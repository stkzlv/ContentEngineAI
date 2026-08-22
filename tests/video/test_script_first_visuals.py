"""Tests for choosing stock footage from the script rather than the title.

A stock-only render has no product photography, so the search phrases are the
entire visual layer. These tests cover the two halves that decide what the
viewer sees: which order the steps run in, and what the phrases end up being.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ai.script_generator import sanitize_visual_search_phrases
from src.video.producer.state import (
    STEP_GATHER_VISUALS,
    STEP_GENERATE_SCRIPT,
    VALID_STEPS,
    resolved_step_order,
)
from src.video.producer.steps import (
    _fetch_stock_across_queries,
    _resolve_script_visual_phrases,
    _uses_script_visual_phrases,
    next_share,
)
from src.video.producer.utils import draws_visuals_from_script

STOCK_ONLY = SimpleNamespace(use_scraped_images=False, use_scraped_videos=False)
SCRAPED = SimpleNamespace(use_scraped_images=True, use_scraped_videos=False)
SCRAPED_VIDEO = SimpleNamespace(use_scraped_images=False, use_scraped_videos=True)


@pytest.mark.unit
class TestWhichProfilesReorder:
    def test_a_stock_only_profile_draws_from_the_script(self):
        assert draws_visuals_from_script(STOCK_ONLY)

    def test_a_profile_with_product_photos_does_not(self):
        assert not draws_visuals_from_script(SCRAPED)

    def test_scraped_video_alone_is_still_product_footage(self):
        """Both flags have to be off. One product source is still a source."""
        assert not draws_visuals_from_script(SCRAPED_VIDEO)


@pytest.mark.unit
class TestStepOrder:
    def test_a_product_profile_keeps_the_stored_order(self):
        assert resolved_step_order(SCRAPED) == VALID_STEPS

    def test_a_stock_profile_gathers_visuals_after_the_script(self):
        order = resolved_step_order(STOCK_ONLY)
        assert order.index(STEP_GENERATE_SCRIPT) < order.index(STEP_GATHER_VISUALS)

    def test_reordering_moves_a_step_rather_than_adding_one(self):
        """Every step still runs exactly once, and none is lost.

        Resume truncation slices this list, so a duplicated or dropped name
        would silently decide that a step never has to run again.
        """
        order = resolved_step_order(STOCK_ONLY)
        assert sorted(order) == sorted(VALID_STEPS)
        assert len(order) == len(set(order))

    def test_the_steps_after_the_script_keep_their_places(self):
        """Only the visuals move. Voiceover still follows the script."""
        order = resolved_step_order(STOCK_ONLY)
        moved = [s for s in order if s != STEP_GATHER_VISUALS]
        assert moved == [s for s in VALID_STEPS if s != STEP_GATHER_VISUALS]

    def test_the_shared_list_is_not_mutated(self):
        """`VALID_STEPS` is module state; reordering in place would leak."""
        before = list(VALID_STEPS)
        resolved_step_order(STOCK_ONLY)
        resolved_step_order(SCRAPED)
        assert before == VALID_STEPS


@pytest.mark.unit
class TestSanitizePhrases:
    def test_reads_one_phrase_per_line(self):
        raw = "wifi router on a shelf\nfrustrated person with laptop"
        assert sanitize_visual_search_phrases(raw) == [
            "wifi router on a shelf",
            "frustrated person with laptop",
        ]

    def test_strips_bullets_and_numbering(self):
        raw = "1. hand holding smartphone\n- phone on bedside table"
        assert sanitize_visual_search_phrases(raw) == [
            "hand holding smartphone",
            "phone on bedside table",
        ]

    def test_drops_a_preamble_line(self):
        raw = "Here are three phrases:\nwifi router on a shelf"
        assert sanitize_visual_search_phrases(raw) == ["wifi router on a shelf"]

    def test_drops_prose_rather_than_searching_it(self):
        """A sentence is not a query.

        Punctuation means the model answered in prose, and the library would
        return whatever matched the longest run of words in it.
        """
        raw = "The script talks about routers, so show a router.\nrouter on a desk"
        assert sanitize_visual_search_phrases(raw) == ["router on a desk"]

    def test_a_single_word_is_a_category_not_a_shot(self):
        """Every result for "phone" looks like a catalogue page."""
        assert sanitize_visual_search_phrases("phone\nphone on a desk") == [
            "phone on a desk"
        ]

    def test_an_over_long_phrase_is_dropped_not_truncated(self):
        """Cutting a phrase can invert what it depicts.

        "person resetting router cables" trimmed to two words is "person
        resetting", which returns something else entirely. One fewer shot beats
        one wrong shot.
        """
        raw = "a person carefully resetting the router cables\nrouter on a desk"
        assert sanitize_visual_search_phrases(raw, max_words=4) == ["router on a desk"]

    def test_caps_the_number_of_phrases(self):
        raw = "\n".join(f"thing number {i}" for i in range(10))
        assert len(sanitize_visual_search_phrases(raw, max_phrases=3)) == 3

    def test_deduplicates_case_insensitively(self):
        raw = "Router On A Desk\nrouter on a desk\nlaptop on a desk"
        assert sanitize_visual_search_phrases(raw) == [
            "Router On A Desk",
            "laptop on a desk",
        ]

    def test_empty_or_unusable_output_yields_nothing(self):
        """The caller reads an empty list as "keep the terms you had"."""
        assert sanitize_visual_search_phrases("") == []
        assert sanitize_visual_search_phrases(None) == []
        assert sanitize_visual_search_phrases("I cannot help with that.") == []


@pytest.mark.unit
class TestNextShare:
    def test_asks_each_search_for_a_fair_share(self):
        assert next_share(6, 3) == 2

    def test_rounds_up_so_the_last_search_covers_the_rest(self):
        assert next_share(8, 3) == 3

    def test_the_final_search_is_asked_for_everything_missing(self):
        assert next_share(5, 1) == 5

    def test_nothing_missing_asks_for_nothing(self):
        assert next_share(0, 3) == 0
        assert next_share(-2, 3) == 0


class _FakeItem:
    def __init__(self, path: str, type_: str = "image"):
        self.path = Path(path)
        self.type = type_


class _FakeFetcher:
    """Records each search and returns items named after the query."""

    def __init__(
        self,
        per_query: dict[str, list[str]] | None = None,
        video_paths: set[str] | None = None,
    ):
        self.calls: list[tuple[list[str], int, int]] = []
        self.per_query = per_query
        self.video_paths = video_paths or set()

    async def fetch_and_download_stock(
        self, keywords, image_count, video_count, assets_dir, session
    ):
        self.calls.append((list(keywords), image_count, video_count))
        query = " ".join(keywords)
        if self.per_query is not None:
            return [
                _FakeItem(p, "video" if p in self.video_paths else "image")
                for p in self.per_query.get(query, [])
            ]
        return [_FakeItem(f"{query}-{i}.jpg") for i in range(image_count)]


@pytest.mark.unit
class TestFetchAcrossQueries:
    async def test_searches_each_phrase_separately(self):
        """Joined into one query string, the phrases collapse onto one shot."""
        fetcher = _FakeFetcher()
        await _fetch_stock_across_queries(
            fetcher,
            [["router", "on", "desk"], ["person", "with", "laptop"]],
            4,
            0,
            Path("assets"),
            None,
        )
        assert [q for q, _, _ in fetcher.calls] == [
            ["router", "on", "desk"],
            ["person", "with", "laptop"],
        ]

    async def test_splits_the_requested_count_across_searches(self):
        fetcher = _FakeFetcher()
        result = await _fetch_stock_across_queries(
            fetcher, [["a", "b"], ["c", "d"], ["e", "f"]], 8, 0, Path("assets"), None
        )
        assert [images for _, images, _ in fetcher.calls] == [3, 3, 2]
        assert len(result) == 8

    async def test_an_empty_search_is_made_up_by_the_next_one(self):
        """A fixed slice would leave the render short by that slice.

        Falling from 8 images to 5 can cross the media-requirements floor,
        which skips the render rather than shortening it.
        """
        fetcher = _FakeFetcher(
            {"a b": [], "c d": [f"c{i}.jpg" for i in range(4)], "e f": ["e0.jpg"]}
        )
        result = await _fetch_stock_across_queries(
            fetcher, [["a", "b"], ["c", "d"], ["e", "f"]], 8, 0, Path("assets"), None
        )
        assert [images for _, images, _ in fetcher.calls] == [3, 4, 4]
        assert len(result) == 5

    async def test_a_duplicate_is_counted_as_missing_not_as_found(self):
        """Deduplication happens before the shortfall is measured.

        Counting the discarded copy would leave the render one image short
        while reporting the full count.
        """
        fetcher = _FakeFetcher(
            {"a b": ["shared.jpg"], "c d": ["shared.jpg"], "e f": ["e0.jpg", "e1.jpg"]}
        )
        result = await _fetch_stock_across_queries(
            fetcher, [["a", "b"], ["c", "d"], ["e", "f"]], 3, 0, Path("assets"), None
        )
        assert [images for _, images, _ in fetcher.calls] == [1, 1, 2]
        assert [str(item.path) for item in result] == ["shared.jpg", "e0.jpg", "e1.jpg"]

    async def test_the_same_item_from_two_searches_appears_once(self):
        """Both searches download to the same path.

        Keeping both would show the identical photograph twice in one render.
        """
        fetcher = _FakeFetcher(
            {"a b": ["shared.jpg", "one.jpg"], "c d": ["shared.jpg", "two.jpg"]}
        )
        result = await _fetch_stock_across_queries(
            fetcher, [["a", "b"], ["c", "d"]], 4, 0, Path("assets"), None
        )
        assert [str(item.path) for item in result] == [
            "shared.jpg",
            "one.jpg",
            "two.jpg",
        ]

    async def test_an_empty_search_does_not_fail_the_others(self):
        fetcher = _FakeFetcher({"a b": [], "c d": ["two.jpg"]})
        result = await _fetch_stock_across_queries(
            fetcher, [["a", "b"], ["c", "d"]], 4, 0, Path("assets"), None
        )
        assert [str(item.path) for item in result] == ["two.jpg"]

    async def test_images_and_videos_are_counted_separately(self):
        """A search returning only photographs must not satisfy the video
        budget, or a video profile silently renders without any.
        """
        fetcher = _FakeFetcher(
            {"a b": ["a0.jpg"], "c d": ["c0.jpg", "c0.mp4"]}, video_paths={"c0.mp4"}
        )
        await _fetch_stock_across_queries(
            fetcher, [["a", "b"], ["c", "d"]], 2, 2, Path("assets"), None
        )
        assert fetcher.calls == [(["a", "b"], 1, 1), (["c", "d"], 1, 2)]

    async def test_a_single_query_behaves_as_one_search(self):
        """The product path passes one keyword list and must be unchanged."""
        fetcher = _FakeFetcher()
        await _fetch_stock_across_queries(
            fetcher, [["wireless", "earbuds"]], 5, 2, Path("assets"), None
        )
        assert fetcher.calls == [(["wireless", "earbuds"], 5, 2)]


def _phrase_ctx(*, script="A script.", stock_only=True, enabled=True, topic="A topic"):
    """A context carrying only what `_resolve_script_visual_phrases` reads."""
    profile = SimpleNamespace(
        use_scraped_images=not stock_only, use_scraped_videos=False
    )
    return SimpleNamespace(
        profile=profile,
        script=script,
        product=SimpleNamespace(topic=topic, title="t", description="d"),
        secrets={},
        session=None,
        debug_mode=False,
        config=SimpleNamespace(
            llm_settings=SimpleNamespace(
                visual_search_terms=SimpleNamespace(
                    enabled=enabled, max_phrases=3, max_words_per_phrase=5
                ),
                script_templates=SimpleNamespace(narrator_for=lambda _is_topic: ""),
            ),
            api_settings=None,
        ),
    )


@pytest.mark.unit
class TestPhraseResolutionGates:
    """When the render must NOT ask, and must NOT pay for the asking.

    Each of these leaves the previously shipped search terms in place, which is
    the whole non-regression guarantee: the feature can only add matching, never
    remove a render.
    """

    async def test_a_product_profile_never_asks(self, monkeypatch):
        """The script does not exist yet at this point in the product order."""
        called = []
        monkeypatch.setattr(
            "src.video.producer.steps.generate_visual_search_phrases",
            lambda *a, **k: called.append(1),
        )
        result = await _resolve_script_visual_phrases(_phrase_ctx(stock_only=False))
        assert result == []
        assert not called

    async def test_disabled_config_never_asks(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            "src.video.producer.steps.generate_visual_search_phrases",
            lambda *a, **k: called.append(1),
        )
        result = await _resolve_script_visual_phrases(_phrase_ctx(enabled=False))
        assert result == []
        assert not called

    async def test_no_script_never_asks(self, monkeypatch):
        """Nothing to derive terms from, so asking would spend a call on air."""
        called = []
        monkeypatch.setattr(
            "src.video.producer.steps.generate_visual_search_phrases",
            lambda *a, **k: called.append(1),
        )
        result = await _resolve_script_visual_phrases(_phrase_ctx(script=""))
        assert result == []
        assert not called

    async def test_a_failed_generation_yields_no_phrases(self, monkeypatch):
        """`generate_visual_search_phrases` swallows its own errors.

        An empty list is how the caller is told to keep the terms it had, so a
        no-key or provider failure must not surface as an exception here.
        """

        async def _empty(*a, **k):
            return []

        monkeypatch.setattr(
            "src.video.producer.steps.generate_visual_search_phrases", _empty
        )
        assert await _resolve_script_visual_phrases(_phrase_ctx()) == []

    async def test_a_stock_profile_with_a_script_asks(self, monkeypatch):
        async def _phrases(*a, **k):
            return ["bread on counter"]

        monkeypatch.setattr(
            "src.video.producer.steps.generate_visual_search_phrases", _phrases
        )
        assert await _resolve_script_visual_phrases(_phrase_ctx()) == [
            "bread on counter"
        ]


@pytest.mark.unit
class TestPreloaderGate:
    """The preloader prefetches from title words before the script exists.

    On the script-first path the render searches for something else, so
    consulting it would spend a download on a query no longer being made.
    """

    def test_skipped_when_the_script_supplies_the_phrases(self):
        assert _uses_script_visual_phrases(_phrase_ctx()) is True

    def test_kept_for_a_product_profile(self):
        assert _uses_script_visual_phrases(_phrase_ctx(stock_only=False)) is False

    def test_kept_when_the_feature_is_off(self):
        """Disabled means the old search terms are used, so the old prefetch
        is the right one to consult.
        """
        assert _uses_script_visual_phrases(_phrase_ctx(enabled=False)) is False


@pytest.mark.unit
class TestStaleArtifactRemoval:
    """Dropping a step from the state is not enough to make it re-run.

    `step_gather_visuals` short-circuits on `gathered_visuals.json` existing,
    not on the state, so a truncation that leaves the file behind pairs fresh
    narration with the footage searched from the script that was lost. That is
    the exact mismatch the script-first order exists to prevent.
    """

    def _state(self, tmp_path):
        visuals = tmp_path / "gathered_visuals.json"
        script = tmp_path / "script.txt"
        visuals.write_text("{}", encoding="utf-8")
        script.write_text("old narration", encoding="utf-8")
        return (
            visuals,
            script,
            {
                "generate_script": {
                    "status": "done",
                    "artifacts": {"script_file": str(script)},
                },
                "gather_visuals": {
                    "status": "done",
                    "artifacts": {"gathered_visuals_file": str(visuals)},
                },
            },
        )

    def test_a_dropped_steps_output_is_deleted(self, tmp_path):
        from src.video.producer.state import _discard_stale_artifacts

        visuals, script, state = self._state(tmp_path)
        _discard_stale_artifacts(state, valid_steps=[])
        assert not visuals.exists()
        assert not script.exists()

    def test_a_surviving_steps_output_is_kept(self, tmp_path):
        """Truncation drops the tail, and the head must still be usable."""
        from src.video.producer.state import _discard_stale_artifacts

        visuals, script, state = self._state(tmp_path)
        _discard_stale_artifacts(state, valid_steps=["generate_script"])
        assert script.exists()
        assert not visuals.exists()

    def test_an_already_missing_artifact_is_not_an_error(self, tmp_path):
        """The artifact whose absence triggered the truncation is one of these."""
        from src.video.producer.state import _discard_stale_artifacts

        state = {
            "generate_script": {
                "status": "done",
                "artifacts": {"script_file": str(tmp_path / "gone.txt")},
            }
        }
        _discard_stale_artifacts(state, valid_steps=[])

    def test_non_dict_state_entries_are_skipped(self, tmp_path):
        """Top-level scalars such as `pillar` sit beside the step dicts."""
        from src.video.producer.state import _discard_stale_artifacts

        _discard_stale_artifacts({"pillar": "utility"}, valid_steps=[])
