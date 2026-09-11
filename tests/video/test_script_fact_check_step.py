"""The fact check as the pipeline reaches it (#380).

The module's own tests prove the guards. These prove the wiring: that the
check runs where the script has just been written and not on a resume, that
the record lands somewhere the run can be audited from afterwards, and that a
checker which cannot run at all still ships a video.

The last one is the important one. A grounded call is the pipeline's only
dependency on a search backend, and it sits after the paid script call and
before every other paid step.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.config import config
from src.video.producer.context import PipelineContext
from src.video.producer.state import get_video_run_paths

TOPIC = "Why one usb port charges faster"
CTA = config.llm_settings.script_templates.cta_options_topic[0]
SCRIPT = (
    "Your laptop charges slower on one port because the ports are not the same. "
    "One of them carries the higher wattage and the others do not. "
    "Look along the edge for a small lightning bolt printed beside a port. "
    "If no port has one, the laptop does not do fast charging at all. "
    "The cable matters as much as the port, so try a different one first. "
    f"{CTA}"
)
FIXED = SCRIPT.replace("a small lightning bolt", "a small battery or bolt icon")
FLAGGED = (
    "VERDICT: FLAGGED\n"
    "CLAIM: Look along the edge for a small lightning bolt printed beside a port.\n"
    "REASON: Vendors mark the port with a battery icon as often as a bolt.\n"
    "FIX: The marking is a battery or a bolt depending on the vendor."
)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """A context whose product is a topic, on a real run directory."""
    monkeypatch.setattr(config, "global_output_root_path", tmp_path)
    paths = get_video_run_paths(config, "topic-usb", "slideshow_stock")
    product = MagicMock(
        asin="topic-usb", product_id="topic-usb", topic=TOPIC, title=TOPIC
    )
    c = PipelineContext(
        product=product,
        profile=config.video_profiles["slideshow_stock"],
        profile_name="slideshow_stock",
        config=config,
        secrets={"GEMINI_API_KEY": "k"},
        session=MagicMock(),
        run_paths=paths,
        debug_mode=False,
    )
    c.script = SCRIPT
    paths["script_file"].parent.mkdir(parents=True, exist_ok=True)
    paths["script_file"].write_text(SCRIPT, encoding="utf-8")
    return c


def record_of(ctx) -> dict:
    text = ctx.run_paths["script_fact_check"].read_text(encoding="utf-8")
    record: dict = json.loads(text)
    return record


def genai_client(text: str) -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=MagicMock(text=text))
    client.aio.aclose = AsyncMock()
    return client


@pytest.mark.unit
class TestTheRunDirectory:
    def test_the_record_path_is_registered_for_the_runtime(self, ctx) -> None:
        """`ctx.run_paths` is built by `get_video_run_paths`, which keeps its
        own dict; a key added only to the config-layer paths raises KeyError
        at render time and nowhere earlier.
        """
        assert ctx.run_paths["script_fact_check"].name == "script_fact_check.json"

    def test_it_lands_in_the_run_temp_directory(self, ctx) -> None:
        assert (
            ctx.run_paths["script_fact_check"].parent
            == ctx.run_paths["script_file"].parent
        )

    @pytest.mark.asyncio
    async def test_it_is_not_recorded_as_a_step_artifact(self, ctx) -> None:
        """A recorded artifact that does not exist drops its step and every
        step after it on resume. This file is absent whenever the check is off
        -- which is the code default and every product render -- so recording
        it would re-run the description, the voiceover, the subtitles, the
        music, the assembly and the burn on a finished render.
        """
        from src.video.producer.state import (
            STEP_GENERATE_SCRIPT,
            _update_state_after_step,
        )

        await _update_state_after_step(ctx, STEP_GENERATE_SCRIPT)

        artifacts = ctx.state[STEP_GENERATE_SCRIPT]["artifacts"]
        assert "script_file" in artifacts
        assert not any("fact_check" in key for key in artifacts)


@pytest.mark.unit
class TestTheStep:
    @pytest.mark.asyncio
    async def test_a_dead_checker_still_ships_the_script(self, ctx) -> None:
        """The whole design in one test: no search backend, no key that works,
        no library -- and the render carries on with the script it generated.
        """
        from src.video.producer.steps import _ensure_fact_checked

        with patch("google.genai.Client", side_effect=ImportError("no genai")):
            await _ensure_fact_checked(ctx, None)

        assert ctx.script == SCRIPT
        assert ctx.run_paths["script_file"].read_text(encoding="utf-8") == SCRIPT
        assert record_of(ctx)["ran"] is False

    @pytest.mark.asyncio
    async def test_an_accepted_revision_reaches_the_file_on_disk(self, ctx) -> None:
        """The script is written before the check, so a revision that only
        updated `ctx.script` would ship the original: the assembler reads the
        context, but a resume and every debug read take the file.
        """
        from src.video.producer.steps import _ensure_fact_checked

        with (
            patch("google.genai.Client", return_value=genai_client(FLAGGED)),
            patch(
                "src.ai.platform_metadata.utilities.generate_with_llm",
                AsyncMock(return_value=FIXED),
            ),
        ):
            await _ensure_fact_checked(ctx, None)

        assert ctx.script == FIXED
        assert ctx.run_paths["script_file"].read_text(encoding="utf-8") == FIXED

    @pytest.mark.asyncio
    async def test_the_state_entry_is_a_string(self, ctx) -> None:
        """The state loader tells step entries from scalars with
        `isinstance(info, dict)`, so a dict here is read as a step record.
        """
        from src.video.producer.steps import _ensure_fact_checked

        with patch("google.genai.Client", return_value=genai_client("VERDICT: OK")):
            await _ensure_fact_checked(ctx, None)

        assert ctx.state["script_fact_check"] == "flagged=0 revised=0"

    @pytest.mark.asyncio
    async def test_a_clean_verdict_still_writes_the_record(self, ctx) -> None:
        from src.video.producer.steps import _ensure_fact_checked

        with patch("google.genai.Client", return_value=genai_client("VERDICT: OK")):
            await _ensure_fact_checked(ctx, None)

        assert record_of(ctx)["ran"] is True
        assert record_of(ctx)["flagged"] == []

    @pytest.mark.asyncio
    async def test_the_raw_answer_is_kept(self, ctx) -> None:
        """What lets a flagged render be audited afterwards, rather than only
        through the throwaway rig that produced the numbers in the config.

        A debug render keeps it; a successful normal run deletes the whole
        intermediate directory, this file with it.
        """
        from src.video.producer.steps import _ensure_fact_checked

        with (
            patch("google.genai.Client", return_value=genai_client(FLAGGED)),
            patch(
                "src.ai.platform_metadata.utilities.generate_with_llm",
                AsyncMock(return_value=FIXED),
            ),
        ):
            await _ensure_fact_checked(ctx, None)

        assert "battery icon as often as a bolt" in record_of(ctx)["raw_answer"]

    @pytest.mark.asyncio
    async def test_a_product_render_uses_the_listing_arm(self, ctx) -> None:
        """A web search resolves a product claim against a different SKU, a
        review or a successor model, so the product arm rules against the
        scraped listing instead, ungrounded (#383). The routing is by the
        record's kind, not a knob.
        """
        from src.ai.script_fact_check import FactCheckResult
        from src.video.producer.steps import _ensure_fact_checked

        ctx.product.topic = None
        ctx.product.title = "USB-C Charger 65W"
        ctx.product.description = "65W output, two ports."
        with (
            patch(
                "src.ai.script_fact_check.check_product_script",
                AsyncMock(return_value=FactCheckResult(ran=True, flagged=[])),
            ) as listing_check,
            patch(
                "src.ai.script_fact_check.check_script",
                AsyncMock(return_value=FactCheckResult(ran=True, flagged=[])),
            ) as grounded_check,
        ):
            await _ensure_fact_checked(ctx, None)

        assert listing_check.await_count == 1
        assert grounded_check.await_count == 0
        assert record_of(ctx)["arm"] == "product"

    @pytest.mark.asyncio
    async def test_the_product_arm_can_be_switched_off(self, ctx) -> None:
        from src.video.producer.steps import _ensure_fact_checked

        ctx.product.topic = None
        off = config.llm_settings.model_copy(
            update={
                "script_fact_check": (
                    config.llm_settings.script_fact_check.model_copy(
                        update={"products": False}
                    )
                )
            }
        )
        with (
            patch.object(ctx.config, "llm_settings", off),
            patch("google.genai.Client") as client,
        ):
            await _ensure_fact_checked(ctx, None)

        assert client.call_count == 0
        assert not ctx.run_paths["script_fact_check"].exists()

    @pytest.mark.asyncio
    async def test_a_disabled_check_writes_nothing_at_all(self, ctx) -> None:
        from src.video.producer.steps import _ensure_fact_checked

        off = config.llm_settings.model_copy(
            update={
                "script_fact_check": (
                    config.llm_settings.script_fact_check.model_copy(
                        update={"enabled": False}
                    )
                )
            }
        )
        with (
            patch.object(ctx.config, "llm_settings", off),
            patch("google.genai.Client") as client,
        ):
            await _ensure_fact_checked(ctx, None)

        assert client.call_count == 0
        assert not ctx.run_paths["script_fact_check"].exists()

    @pytest.mark.asyncio
    async def test_a_resumed_script_is_not_re_checked(self) -> None:
        """The call sits inside the generation branch. A resume over an
        existing script must pay nothing -- the check is per generated script,
        not per run, or a retried render is billed twice for one script.
        """
        source = __import__("inspect").getsource(
            __import__(
                "src.video.producer.steps", fromlist=["step_generate_script"]
            ).step_generate_script
        )
        before, _, after = source.partition("_ensure_fact_checked(ctx, pillar)")
        assert "Loading existing script from previous run" in before
        assert before.rindex("else:") > before.rindex("Loading existing script")
        assert "_ensure_hook_headline" in after
