"""The grounded step-list call has room to finish, and a timeout is named."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.ai.llm_settings import StepListConfig
from src.video.config import config


@pytest.mark.req("REQ-VID-121")
def test_the_timeout_clears_a_measured_call() -> None:
    # A grounded call measured 45-55 s; at 60 most pool topics timed out.
    assert StepListConfig().timeout_seconds >= 120
    assert config.llm_settings.topic_scripts.step_list.timeout_seconds >= 120


@pytest.mark.req("REQ-VID-121")
@pytest.mark.asyncio
async def test_a_timed_out_call_logs_its_type(caplog) -> None:
    from src.ai import step_list as module

    class _Models:
        async def generate_content(self, **_: object) -> None:
            await asyncio.sleep(5)

    client = SimpleNamespace(aio=SimpleNamespace(models=_Models()))
    settings = SimpleNamespace(model="m", max_steps=6, timeout_seconds=0.01)
    with (
        patch("google.genai.Client", return_value=client),
        caplog.at_level(logging.WARNING),
    ):
        result = await module.build_step_list(
            "How to x", "", api_key="k", settings=settings
        )

    assert result is None
    assert "TimeoutError" in caplog.text
