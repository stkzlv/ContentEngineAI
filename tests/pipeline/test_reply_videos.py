"""Reply videos (REQ-CNT-176).

The producer builds a record from a viewer comment and the published record
it was left on, renders it in the commentary identity with the reply template,
opens on a card showing the comment, and leaves it in `drafts/`, where nothing
publishes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.utils.outputs_paths import DRAFTS_DIR_NAME, is_product_directory
from src.video.config import config
from src.video.config.identity_models import content_kind
from src.video.config.visual_models import ReplyCardSettings
from src.video.producer.topic_input import (
    REPLY_ID_PREFIX,
    TopicInputError,
    build_reply_product,
    move_reply_to_drafts,
    reply_product_id,
)

PARENT = "B0TEST0001"
COMMENT = "Does this still work if the router is ten years old?"


def _published(outputs: Path, record: str = PARENT, title: str = "") -> None:
    state = outputs / "state"
    state.mkdir(parents=True, exist_ok=True)
    row = {"platform": "youtube", "post_id": "p1", "product_id": record}
    (state / "publish_history.json").write_text(
        json.dumps({"posts": {f"{record}:youtube": row}})
    )
    if title:
        (outputs / record).mkdir(parents=True, exist_ok=True)
        (outputs / record / "data.json").write_text(json.dumps({"title": title}))


@pytest.mark.req("REQ-CNT-176")
class TestTheRecord:
    def test_it_carries_the_comment_and_what_the_video_said(self, tmp_path) -> None:
        _published(tmp_path, title="Why your wifi drops at night")
        (tmp_path / PARENT / "script.json").write_text(
            json.dumps({"script": "Old routers throttle when they overheat."})
        )

        record = build_reply_product(COMMENT, PARENT, tmp_path, author="viewer1")

        assert record.topic_kind == "reply"
        assert (record.reply_comment, record.reply_author, record.reply_to) == (
            COMMENT,
            "viewer1",
            PARENT,
        )
        assert COMMENT in record.description
        assert "Why your wifi drops at night" in record.description
        assert "Old routers throttle" in record.description
        assert (record.asin or "").startswith(REPLY_ID_PREFIX)
        assert content_kind(record) == "commentary"

    def test_a_cleaned_parent_falls_back_to_the_registry(self, tmp_path) -> None:
        from src.publisher.product_registry import RegistryEntry, save_registry

        _published(tmp_path)
        save_registry(
            [
                RegistryEntry(
                    product_id=PARENT, title="The router", url="", affiliate_url=""
                )
            ],
            tmp_path,
        )

        record = build_reply_product(COMMENT, PARENT, tmp_path)

        assert "The router" in record.description
        assert "What that video said" not in record.description

    def test_an_unpublished_record_is_refused(self, tmp_path) -> None:
        with pytest.raises(TopicInputError, match=PARENT):
            build_reply_product(COMMENT, PARENT, tmp_path)

    def test_an_empty_comment_is_refused(self, tmp_path) -> None:
        _published(tmp_path)
        with pytest.raises(TopicInputError):
            build_reply_product("   ", PARENT, tmp_path)

    def test_the_id_is_stable_per_comment(self) -> None:
        assert reply_product_id(PARENT, COMMENT) == reply_product_id(PARENT, COMMENT)
        assert reply_product_id(PARENT, COMMENT) != reply_product_id(PARENT, "Other")


def _reply(tmp_path: Path):
    _published(tmp_path)
    return build_reply_product(COMMENT, PARENT, tmp_path)


@pytest.mark.req("REQ-CNT-176")
class TestTheRender:
    def test_it_draws_the_reply_template_in_the_commentary_voice(
        self, tmp_path
    ) -> None:
        from src.video.producer.orchestration import config_for_record

        t = config_for_record(config, _reply(tmp_path)).llm_settings.script_templates
        base = config.llm_settings.script_templates

        assert t.topic_templates == ["reply"]
        assert t.cta_options_topic == base.cta_options_reply
        assert t.target_length.topic == base.target_length.commentary

    def test_no_reply_templates_refuses_the_record(self, tmp_path) -> None:
        from src.video.producer.context import PipelineError
        from src.video.producer.orchestration import config_for_record

        cfg = config.model_copy(deep=True)
        cfg.llm_settings.script_templates.reply_templates = []

        with pytest.raises(PipelineError):
            config_for_record(cfg, _reply(tmp_path))

    def test_a_product_never_draws_the_reply_template(self) -> None:
        from src.ai.script_generator import select_script_template

        pool = {
            select_script_template(config.llm_settings, f"B0TEST{n:04d}").stem
            for n in range(300)
        }
        assert "reply" not in pool


@pytest.mark.req("REQ-CNT-176")
class TestTheCard:
    def _apply(self, tmp_path: Path, settings: ReplyCardSettings, author="viewer1"):
        from src.video.assembler.overlay_builder import apply_reply_card_overlay

        return apply_reply_card_overlay(
            ["[0:v]scale=1080:1920[v0]", "[v0]copy[v_out]"],
            settings,
            COMMENT,
            author,
            72,
            1080,
            tmp_path,
        )

    def test_it_draws_the_comment_over_the_opening(self, tmp_path) -> None:
        out = self._apply(tmp_path, ReplyCardSettings())

        assert out[-1] == "[v_reply]copy[v_out]"
        assert "between(t\\,0\\,3.000)" in out[-2]
        drawn = " ".join(p.read_text() for p in sorted(tmp_path.glob("reply_line_*")))
        assert drawn.startswith("Does this still work")
        assert "viewer1" not in drawn

    def test_the_author_is_drawn_only_when_asked(self, tmp_path) -> None:
        self._apply(tmp_path, ReplyCardSettings(show_author=True))

        assert (tmp_path / "reply_line_0.txt").read_text() == "@viewer1"

    def test_off_draws_nothing(self, tmp_path) -> None:
        out = self._apply(tmp_path, ReplyCardSettings(enabled=False))

        assert out == ["[0:v]scale=1080:1920[v0]", "[v0]copy[v_out]"]

    def test_a_long_comment_is_cut_to_two_lines(self) -> None:
        from src.video.assembler.overlay_builder import reply_card_lines

        lines = reply_card_lines("word " * 80, 64, 900, 2)

        assert len(lines) == 2
        assert lines[-1].endswith("...")

    def test_the_bundled_card_ships_on_without_the_handle(self) -> None:
        card = config.video_settings.reply_card
        assert card.enabled and not card.show_author


@pytest.mark.req("REQ-CNT-176")
class TestTheDrafts:
    def _config(self, root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            global_output_root_path=root,
            get_product_paths=lambda pid, profile: {"product_root": root / pid},
        )

    def test_a_rendered_reply_moves_into_drafts(self, tmp_path) -> None:
        record = _reply(tmp_path)
        rendered = tmp_path / record.asin
        rendered.mkdir()
        (rendered / "video.mp4").write_bytes(b"x")

        moved = move_reply_to_drafts(
            self._config(tmp_path), record, "p", rendered / "video.mp4"
        )

        assert not rendered.exists()
        assert Path(moved) == tmp_path / DRAFTS_DIR_NAME / record.asin / "video.mp4"
        assert Path(moved).read_bytes() == b"x"

    def test_a_re_render_replaces_the_draft(self, tmp_path) -> None:
        record = _reply(tmp_path)
        old = tmp_path / DRAFTS_DIR_NAME / record.asin
        old.mkdir(parents=True)
        (old / "stale.txt").write_text("old")
        (tmp_path / record.asin).mkdir()
        (tmp_path / record.asin / "video.mp4").write_bytes(b"new")

        move_reply_to_drafts(
            self._config(tmp_path), record, "p", tmp_path / record.asin / "video.mp4"
        )

        assert not (old / "stale.txt").exists()

    def test_nothing_publishes_or_cleans_a_draft(self, tmp_path) -> None:
        from src.publisher.batch import BatchPublisher

        draft = tmp_path / DRAFTS_DIR_NAME / "reply-x"
        draft.mkdir(parents=True)
        (draft / "video_reply-x_slideshow_stock.mp4").write_bytes(b"x")

        assert not is_product_directory(tmp_path / DRAFTS_DIR_NAME)
        assert (
            BatchPublisher(MagicMock(), outputs_dir=tmp_path)._discover_videos() == []
        )


@pytest.mark.req("REQ-CNT-176")
class TestTheFlags:
    @pytest.mark.parametrize(
        "argv",
        [
            ["slideshow_stock", "--reply-to-comment", COMMENT],
            ["slideshow_stock", "--reply-record", PARENT],
            ["slideshow_stock", "--reply-author", "v"],
            [
                "slideshow_stock",
                "--reply-to-comment",
                COMMENT,
                "--reply-record",
                PARENT,
                "--topic",
                "x",
            ],
        ],
    )
    def test_bad_combinations_are_refused(self, monkeypatch, argv) -> None:
        import asyncio

        from src.video.producer import cli

        monkeypatch.setattr("sys.argv", ["producer", *argv])
        with pytest.raises(SystemExit) as exc:
            asyncio.run(cli.main())

        assert exc.value.code == 2


@pytest.mark.req("REQ-CNT-176")
@pytest.mark.asyncio
async def test_a_reply_keeps_to_its_figures_and_asks_nothing() -> None:
    from tests.ai.test_commentary_checks import _generate, _take

    statement = "That is the answer."
    out, calls = await _generate([_take("1,016", statement)], kind="reply")
    assert out is not None and calls == 1, "no closing question is asked for"

    exact = _take("1,016", statement)
    out, calls = await _generate(
        [_take("about a thousand", statement), exact], kind="reply"
    )
    assert out == exact and calls == 2, "an unbacked figure is still retried"
