"""Correction topics (REQ-CNT-175) and their first comment (REQ-PUB-166).

A topic marked `kind: correction` names the published record it corrects,
renders with the correction template in the commentary identity, is refused
before rendering when that record has no queued or published post, and its
first comment links the original post on each platform.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.pipeline.config import GlobalBatchConfig
from src.publisher.first_comment import build_first_comment, resolve_correction_links
from src.publisher.models import FirstCommentConfig
from src.video.config import config
from src.video.config.identity_models import content_kind
from src.video.producer.topic_input import (
    TopicInputError,
    TopicSpec,
    build_topic_product,
    materialise_topics,
    specs_from_args,
    specs_from_mappings,
    topic_product_id,
)

ORIGINAL = "B0TEST0001"
TITLE = "The router myth"


def _history(outputs: Path, product_id: str = ORIGINAL, post_id: str = "p1") -> None:
    state = outputs / "state"
    state.mkdir(parents=True, exist_ok=True)
    posts = {
        f"{product_id}:{platform}": {
            "product_id": product_id,
            "platform": platform,
            "post_id": post_id,
            "published_at": "2026-01-01T00:00:00+00:00",
        }
        for platform in ("youtube", "instagram")
    }
    (state / "publish_history.json").write_text(json.dumps({"posts": posts}))


def _correction(corrects: str = ORIGINAL) -> TopicSpec:
    return TopicSpec(
        title="Correction: the router figure",
        description="The vendor data counts 2,040 answers, not 3,000.",
        kind="correction",
        corrects=corrects,
    )


@pytest.mark.req("REQ-CNT-175")
class TestTheRecord:
    def test_a_correction_names_what_it_corrects(self) -> None:
        with pytest.raises(TopicInputError):
            TopicSpec(title="x", kind="correction")
        with pytest.raises(TopicInputError):
            TopicSpec(title="x", kind="commentary", corrects=ORIGINAL)

    def test_a_topics_file_entry_carries_corrects(self) -> None:
        (spec,) = specs_from_mappings(
            [{"title": "x", "kind": "correction", "corrects": ORIGINAL}], "file"
        )

        assert (spec.kind, spec.corrects) == ("correction", ORIGINAL)

    def test_the_flag_needs_the_correction_kind(self) -> None:
        common = {
            "topic": "x",
            "topic_description": "",
            "topic_keywords": "",
            "topics_file": None,
        }
        with pytest.raises(TopicInputError):
            specs_from_args(**common, topic_kind="commentary", corrects=ORIGINAL)

        (spec,) = specs_from_args(**common, topic_kind="correction", corrects=ORIGINAL)
        assert spec.corrects == ORIGINAL

    def test_the_record_carries_its_kind_and_target(self) -> None:
        record = build_topic_product(_correction())

        assert record.to_dict()["topic_kind"] == "correction"
        assert record.to_dict()["corrects"] == ORIGINAL
        assert content_kind(record) == "commentary"


@pytest.mark.req("REQ-CNT-175")
class TestTheRefusal:
    def test_a_record_never_published_is_refused(self, tmp_path) -> None:
        with pytest.raises(TopicInputError, match=ORIGINAL):
            materialise_topics([_correction()], config, "p", outputs_dir=tmp_path)

        assert not (tmp_path / topic_product_id(_correction().title)).exists()

    def test_a_queued_record_passes(self, tmp_path) -> None:
        _history(tmp_path)

        ((topic_dir, product),) = materialise_topics(
            [_correction()], config, "p", outputs_dir=tmp_path
        )

        assert json.loads((topic_dir / "data.json").read_text())["corrects"] == ORIGINAL

    def test_a_backfilled_row_without_its_id_field_counts(self, tmp_path) -> None:
        state = tmp_path / "state"
        state.mkdir()
        row = {"platform": "youtube", "post_id": "p1", "backfilled_from_provider": True}
        (state / "publish_history.json").write_text(
            json.dumps({"posts": {f"{ORIGINAL}:youtube": row}})
        )

        materialise_topics([_correction()], config, "p", outputs_dir=tmp_path)

    def test_the_batch_fails_only_that_record(self, tmp_path) -> None:
        from src.pipeline.global_batch import GlobalPipelineOrchestrator

        batch = GlobalBatchConfig(
            topics=[_correction(), TopicSpec(title="Why your wifi keeps dropping")],
            profile="slideshow_stock",
            outputs_dir=tmp_path,
            skip_publish=True,
        )

        summary = GlobalPipelineOrchestrator(batch)._materialise_topics_phase()

        assert summary.failed == 1 and summary.successful == 1


@pytest.mark.req("REQ-CNT-175")
class TestTheRender:
    def test_it_draws_the_correction_template_in_the_commentary_voice(self) -> None:
        from src.video.producer.orchestration import config_for_record

        templates = config_for_record(
            config, build_topic_product(_correction())
        ).llm_settings.script_templates

        assert templates.topic_templates == ["correction"]
        commentary = config.llm_settings.script_templates
        assert (
            templates.narrator_profile_topic == commentary.narrator_profile_commentary
        )
        assert templates.target_length.topic == commentary.target_length.commentary

    def test_no_correction_templates_refuses_the_record(self) -> None:
        from src.video.producer.context import PipelineError
        from src.video.producer.orchestration import config_for_record

        cfg = config.model_copy(deep=True)
        cfg.llm_settings.script_templates.correction_templates = []

        with pytest.raises(PipelineError):
            config_for_record(cfg, build_topic_product(_correction()))

    def test_a_product_never_draws_the_correction_template(self) -> None:
        from src.ai.script_generator import select_script_template

        assert "correction" in config.llm_settings.script_templates.correction_templates
        product_pool = {
            select_script_template(config.llm_settings, f"B0TEST{n:04d}").stem
            for n in range(300)
        }
        assert len(product_pool) > 3
        assert "correction" not in product_pool


def _write_record(outputs: Path) -> str:
    product_id = topic_product_id(_correction().title)
    (outputs / product_id).mkdir(parents=True)
    (outputs / product_id / "data.json").write_text(
        json.dumps(build_topic_product(_correction()).to_dict())
    )
    return product_id


@pytest.mark.req("REQ-PUB-166")
class TestTheFirstComment:
    @pytest.mark.asyncio
    async def test_it_links_the_original_post_per_platform(self, tmp_path) -> None:
        from src.publisher.product_registry import RegistryEntry, save_registry

        _history(tmp_path)
        save_registry(
            [RegistryEntry(product_id=ORIGINAL, title=TITLE, url="", affiliate_url="")],
            tmp_path,
        )
        product_id = _write_record(tmp_path)
        publisher = AsyncMock()
        publisher.get_post_platforms.return_value = [
            {"platform": "youtube", "platform_post_url": "https://youtu.be/x"},
            {"platform": "instagram", "platform_post_url": None},
        ]

        links = await resolve_correction_links(
            publisher, product_id, tmp_path, ["youtube", "instagram"]
        )
        fc = FirstCommentConfig(enabled=True, platforms={"youtube": "{closing_line}"})

        youtube = build_first_comment(
            fc, "youtube", product_id, tmp_path, correction_links=links
        )
        instagram = build_first_comment(
            fc, "instagram", product_id, tmp_path, correction_links=links
        )

        assert youtube == f"Corrects: {TITLE} https://youtu.be/x"
        assert instagram == f"Corrects: {TITLE}"
        publisher.get_post_platforms.assert_awaited_once_with("p1")

    @pytest.mark.asyncio
    async def test_any_other_record_gets_no_links(self, tmp_path) -> None:
        (tmp_path / "B0TEST0002").mkdir()
        (tmp_path / "B0TEST0002" / "data.json").write_text(json.dumps({"title": "x"}))

        assert (
            await resolve_correction_links(
                AsyncMock(), "B0TEST0002", tmp_path, ["youtube"]
            )
            == {}
        )

    def test_the_posts_own_first_comment_still_wins(self, tmp_path) -> None:
        product_id = _write_record(tmp_path)
        (tmp_path / product_id / "metadata.json").write_text(
            json.dumps({"first_comment": {"youtube": "My own words."}})
        )
        fc = FirstCommentConfig(enabled=True, platforms={"youtube": "{closing_line}"})

        comment = build_first_comment(
            fc,
            "youtube",
            product_id,
            tmp_path,
            correction_links={"youtube": (TITLE, "https://youtu.be/x")},
        )

        assert comment == "My own words."


@pytest.mark.req("REQ-PUB-166")
@pytest.mark.asyncio
async def test_the_provider_leg_carries_its_post_url() -> None:
    from unittest.mock import patch

    from late.models import PostGetResponse

    from src.publisher.late.client import LatePublisher

    resp = PostGetResponse.model_validate(
        {
            "post": {
                "_id": "p1",
                "platforms": [
                    {
                        "platform": "youtube",
                        "platformPostId": "y1",
                        "platformPostUrl": "https://youtu.be/y1",
                        "status": "published",
                    },
                    {
                        "platform": "instagram",
                        "platformPostId": None,
                        "status": "pending",
                    },
                ],
            }
        }
    )
    pub = LatePublisher(api_key="test")  # noqa: S106 - test stub
    with patch.object(pub, "_posts_get_safe", return_value=resp):
        legs = await pub.get_post_platforms("p1")

    assert [leg["platform_post_url"] for leg in legs] == ["https://youtu.be/y1", None]


@pytest.mark.req("REQ-CNT-175")
@pytest.mark.asyncio
async def test_a_correction_keeps_to_its_figures_and_asks_nothing() -> None:
    from tests.ai.test_commentary_checks import _generate, _take

    statement = "That is what changes."
    out, calls = await _generate([_take("1,016", statement)], kind="correction")
    assert out is not None and calls == 1, "no closing question is asked for"

    exact = _take("1,016", statement)
    out, calls = await _generate(
        [_take("about a thousand", statement), exact], kind="correction"
    )
    assert out == exact and calls == 2, "an unbacked figure is still retried"


LINKS = {"youtube": (TITLE, "https://youtu.be/x")}


@pytest.mark.req("REQ-PUB-166")
@pytest.mark.asyncio
@pytest.mark.parametrize("platform_specific", [False, True])
async def test_each_publish_mode_passes_the_links(
    tmp_path: Path, platform_specific: bool
) -> None:
    from unittest.mock import patch

    from src.publisher import publish_modes
    from src.publisher.models import Platform, PublishMetadata

    meta = PublishMetadata(
        platform=Platform.YOUTUBE, title="t", description="d", hashtags=["x"]
    )
    publisher = AsyncMock()
    publisher.publish.return_value = {"post_id": "p", "status": "scheduled"}
    seen: list[object] = []

    def build(*_args, correction_links=None, **_kwargs):
        seen.append(correction_links)

    with (
        patch.object(publish_modes, "load_platform_metadata", return_value=meta),
        patch.object(publish_modes, "resolve_correction_links", return_value=LINKS),
        patch.object(
            publish_modes, "_build_platform_contents_with_comments", side_effect=build
        ),
    ):
        await publish_modes.publish_product(
            publisher=publisher,
            media_id="m",
            product_id="B0TEST0002",
            platforms=[{"platform": "youtube", "account_id": "a"}],
            outputs_dir=tmp_path,
            platform_specific=platform_specific,
        )

    assert seen == [LINKS]


@pytest.mark.req("REQ-PUB-166")
@pytest.mark.asyncio
async def test_the_schedule_passes_the_links(tmp_path: Path) -> None:
    from datetime import UTC, datetime
    from unittest.mock import patch

    from src.publisher import schedule
    from src.publisher.models import Platform, ScheduleConfig

    manager = schedule.ScheduleManager(
        schedule_path=tmp_path / "schedule.json", config=ScheduleConfig()
    )
    publisher = AsyncMock()
    seen: list[object] = []

    def first_comments(*_args, correction_links=None, **_kwargs):
        seen.append(correction_links)
        raise OSError("stop here")

    with (
        patch.object(manager, "_publish_targets", AsyncMock(return_value=[])),
        patch.object(manager, "_captions_for", return_value=({}, {}, False)),
        patch.object(manager, "_first_comments_for", side_effect=first_comments),
        patch.object(schedule, "resolve_correction_links", return_value=LINKS),
    ):
        await manager._schedule_one(
            video=tmp_path / "B0TEST0002" / "video.mp4",
            product_id="B0TEST0002",
            platforms=[Platform.YOUTUBE],
            publisher=publisher,
            next_time=datetime.now(UTC),
            next_idx=0,
            occupied_slot_times=set(),
            outputs_dir=tmp_path,
            cleanup_manager=None,
            link_in_bio_config=None,
        )

    assert seen == [LINKS]


@pytest.mark.req("REQ-CNT-175")
class TestReviewCases:
    def test_a_correction_asks_nothing_in_its_cta(self) -> None:
        from src.video.producer.orchestration import config_for_record

        t = config_for_record(
            config, build_topic_product(_correction())
        ).llm_settings.script_templates

        assert (
            t.cta_options_topic
            == config.llm_settings.script_templates.cta_options_correction
        )
        assert not any("answer" in cta.lower() for cta in t.cta_options_topic)

    def test_commentary_off_leaves_corrections_on(self) -> None:
        from src.video.producer.orchestration import config_for_record

        cfg = config.model_copy(deep=True)
        cfg.llm_settings.script_templates.commentary_templates = []

        templates = config_for_record(
            cfg, build_topic_product(_correction())
        ).llm_settings.script_templates

        assert templates.topic_templates == ["correction"]

    def test_the_producer_refuses_corrects_without_a_topic(self, monkeypatch) -> None:
        import asyncio

        from src.video.producer import cli

        monkeypatch.setattr(
            "sys.argv",
            ["producer", "products.json", "slideshow_stock", "--corrects", ORIGINAL],
        )
        with pytest.raises(SystemExit) as exc:
            asyncio.run(cli.main())

        assert exc.value.code == 2
