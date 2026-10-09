"""The sample stage: text-only scripts under each variant (REQ-OPS-109).

Each script is written by the producer's own script step, in process, into
a scratch outputs root, so a sample sees the step list, the fact check and
the hook headline exactly as a render would. Nothing is scraped or rendered.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp

from src.ai.script_generator import ScriptGenerationError
from src.scraper.amazon.models import ProductData
from src.utils.outputs_paths import with_outputs_root
from src.video.config import VideoConfig, load_video_config_modular
from src.video.producer.context import (
    PipelineContext,
    PipelineError,
    TopicNotSourcedError,
)
from src.video.producer.state import get_video_run_paths
from src.video.producer.topic_input import TopicSpec, build_topic_product
from src.video.producer.utils import collect_producer_secrets

logger = logging.getLogger(__name__)

TASK_TEMPLATE = "topic_answer_first"


def is_task(title: str) -> bool:
    """A topic phrased as a task ("How to ...") rather than a problem."""
    return title.strip().lower().startswith("how to ")


def variant_config(base: VideoConfig, name: str, title: str | None) -> VideoConfig:
    """The base config with a variant's overrides, in memory only.

    `title` is the topic's, None for a product: the variants change topic
    scripts only, so a product is sampled under `shipped` alone.
    """
    config = base.model_copy(deep=True)
    if name == "free_form":
        config.llm_settings.topic_scripts.step_list.enabled = False
    elif name == "task_answer_first" and title and is_task(title):
        # A step list replaces the template, so the template is compared on
        # the free-form path.
        config.llm_settings.topic_scripts.step_list.enabled = False
        config.llm_settings.script_templates.fixed_template = TASK_TEMPLATE
    elif name not in ("shipped", "task_answer_first"):
        raise ValueError(f"Unknown variant: {name}")
    return config


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


async def sample_one(
    config: VideoConfig,
    product: ProductData,
    profile: str,
    secrets: dict[str, str],
    session: aiohttp.ClientSession,
) -> dict[str, Any]:
    """Run the script step for one product and collect what it wrote."""
    from src.video.producer import steps
    from src.video.producer.state import STEP_GENERATE_SCRIPT

    product_id = product.asin or ""
    paths = get_video_run_paths(config, product_id, profile)
    ctx = PipelineContext(
        product=product,
        profile=config.video_profiles[profile],
        profile_name=profile,
        config=config,
        secrets=secrets,
        session=session,
        run_paths=paths,
        debug_mode=True,
    )
    record: dict[str, Any] = {
        "id": product_id,
        "title": product.title,
        "keyword": getattr(product, "keyword", "") or "",
    }
    try:
        await steps.step_generate_script(ctx)
    except (
        PipelineError,
        TopicNotSourcedError,
        ScriptGenerationError,
        aiohttp.ClientError,
        OSError,
        ValueError,
    ) as e:
        # One failed sample is a finding, not the end of the stage. A topic
        # the step-list filter drops is recorded the same way.
        logger.warning("Sample %s failed: %s", product_id, e)
        record["error"] = str(e)
        return record
    temp = Path(paths["intermediate_base"])
    step_list = _read_json(temp / "step_list.json")
    record.update(
        {
            "script": ctx.script or "",
            "template": ctx.state.get("script_template"),
            "cta": ctx.state.get("cta"),
            "hook_headline": ctx.state.get("hook_headline"),
            "signoff": ctx.state.get("signoff"),
            "steps": len(step_list.get("steps", []))
            if isinstance(step_list, dict)
            else None,
            "explainer": bool(step_list.get("explainer"))
            if isinstance(step_list, dict)
            else False,
            "fact_check": _read_json(temp / "script_fact_check.json"),
            "step": STEP_GENERATE_SCRIPT,
        }
    )
    return record


def recent_products(outputs_root: Path, count: int) -> list[ProductData]:
    """The most recently scraped products, newest first."""
    from src.video.producer.cli import discover_products_for_batch

    found = discover_products_for_batch(outputs_root)
    found.sort(key=lambda item: (item[0] / "data.json").stat().st_mtime, reverse=True)
    return [product for _, product in found[:count]]


async def run_sample(
    variants: list[str],
    topics: list[TopicSpec],
    products: list[ProductData],
    profile: str,
    out_dir: Path,
    load: Callable[..., VideoConfig] = load_video_config_modular,
) -> list[dict[str, Any]]:
    """Sample every topic under every variant and every product once."""
    records: list[dict[str, Any]] = []
    # Before any model call and before any earlier sample is cleared.
    if profile not in load().video_profiles:
        raise ValueError(f"sample.profile {profile!r} names no video profile")
    async with aiohttp.ClientSession() as session:
        for variant in variants:
            root = out_dir / "samples" / variant
            # The script step reuses a script already on disk, so a second
            # run into the same directory would measure the first run's. A
            # clear that fails raises: a stale sample must not pass silently.
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(root)
            base = load(cli_overrides=with_outputs_root({}, root))
            secrets = collect_producer_secrets(base)
            for spec in topics:
                config = variant_config(base, variant, spec.title)
                record = await sample_one(
                    config, build_topic_product(spec), profile, secrets, session
                )
                records.append(
                    {
                        "variant": variant,
                        "kind": "topic",
                        "search": spec.search,
                        **record,
                    }
                )
            if variant != "shipped":
                continue
            for product in products:
                record = await sample_one(base, product, profile, secrets, session)
                records.append({"variant": variant, "kind": "product", **record})
    return records


def sample(
    variants: list[str],
    topics: list[TopicSpec],
    products: list[ProductData],
    profile: str,
    out_dir: Path,
) -> list[dict[str, Any]]:
    return asyncio.run(run_sample(variants, topics, products, profile, out_dir))
