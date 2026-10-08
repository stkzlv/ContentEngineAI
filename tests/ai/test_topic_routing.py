"""A task topic is routed to the task-first template (REQ-CNT-161, #701)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from src.ai.llm_settings import TopicRoutingConfig, is_task_title
from src.ai.script_generator import select_script_template
from src.video.config import config

IDS = [f"topic-{n}" for n in range(40)]
TASK = "How to back up your iPhone to iCloud"
SYMPTOM = "Why your laptop fan runs when idle"


def _settings(routing: bool):
    settings = config.llm_settings.model_copy(deep=True)
    settings.script_templates.topic_routing.enabled = routing
    settings.script_templates.fixed_template = None
    return settings


def _drawn(settings, title: str) -> set[str]:
    return {
        select_script_template(settings, pid, None, is_topic=True, title=title).stem
        for pid in IDS
    }


@pytest.mark.req("REQ-CNT-161")
@pytest.mark.parametrize(
    ("title", "task"),
    [
        (TASK, True),
        ("how to clear app cache on Android", True),
        ("  How To Sign Out Of Gmail", True),
        (SYMPTOM, False),
        ("Howto guide", False),
        ("", False),
        (None, False),
    ],
)
def test_a_task_title_starts_with_how_to(title: str | None, task: bool) -> None:
    assert is_task_title(title) is task


@pytest.mark.req("REQ-CNT-161")
def test_on_a_task_topic_draws_only_the_task_template() -> None:
    assert _drawn(_settings(routing=True), TASK) == {"topic_answer_first"}


@pytest.mark.req("REQ-CNT-161")
def test_on_another_topic_keeps_the_whole_pool() -> None:
    assert _drawn(_settings(routing=True), SYMPTOM) == {
        "topic_answer_first",
        "topic_symptom_cause",
        "topic_mistake_fix",
    }


@pytest.mark.req("REQ-CNT-161")
def test_off_a_task_topic_draws_as_before() -> None:
    assert len(_drawn(_settings(routing=False), TASK)) == 3


@pytest.mark.req("REQ-CNT-161")
def test_a_product_is_never_routed() -> None:
    settings = _settings(routing=True)

    drawn = {
        select_script_template(settings, pid, None, is_topic=False, title=TASK).stem
        for pid in IDS
    }

    assert "topic_answer_first" not in drawn and len(drawn) > 3


@pytest.mark.req("REQ-CNT-161")
def test_routing_needs_a_task_template() -> None:
    with pytest.raises(ValidationError):
        TopicRoutingConfig(task_templates=[])


@pytest.mark.req("REQ-CNT-161")
@pytest.mark.asyncio
async def test_the_script_step_routes_by_the_topics_title() -> None:
    from src.ai import script_generator
    from src.scraper.amazon.models import ProductData
    from src.scraper.base.models import Platform

    settings = _settings(routing=True)
    settings.script_validation.lint.enabled = False
    settings.script_validation.reject_copied_examples = False
    cta = settings.script_templates.cta_options_for(True)[0]
    script = "Open Settings and tap your name, then iCloud. " * 10 + cta
    templates = set()
    for pid in IDS[:12]:
        topic = ProductData(
            title=TASK, price="", url="https://e.com", platform=Platform.AMAZON
        )
        topic.topic = TASK
        call = AsyncMock(return_value=script)
        with patch.object(script_generator, "_call_llm_api_with_retry", call):
            _, template, _ = await script_generator.generate_script(
                topic,
                settings,
                {settings.api_key_env_var: "k"},
                AsyncMock(),
                {},
                False,
                product_id=pid,
            )
        templates.add(template)

    assert templates == {"topic_answer_first"}


@pytest.mark.req("REQ-CNT-161")
def test_task_templates_missing_from_the_pool_fall_back_with_a_warning(
    caplog,
) -> None:
    settings = _settings(routing=True)
    settings.script_templates.topic_routing.task_templates = ["no_such_template"]

    with caplog.at_level("WARNING"):
        drawn = _drawn(settings, TASK)

    assert len(drawn) == 3
    assert "No task template" in caplog.text
