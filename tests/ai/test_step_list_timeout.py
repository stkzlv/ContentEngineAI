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


def _client(outcomes: list[object]) -> SimpleNamespace:
    """A client whose calls take each outcome in turn: raise it or return it."""
    calls: list[int] = []

    class _Models:
        async def generate_content(self, **_: object) -> object:
            calls.append(1)
            outcome = outcomes[len(calls) - 1]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    return SimpleNamespace(aio=SimpleNamespace(models=_Models()), calls=calls)


_ANSWER = SimpleNamespace(
    text='{"start_screen": "Settings", "platform": "iOS 26", "forks": false, '
    '"steps": [{"action": "Open Settings", "ui_path": "Settings", '
    '"expected": "Settings opens", "source": "https://support.apple.com/x"}], '
    '"common_mistake": null, "topic_check": {"specific": true, '
    '"searchable": true, "demonstrable": true, "non_default": true, '
    '"advice": "none"}}'
)


async def _build(client: SimpleNamespace, attempts: int) -> object:
    from src.ai import step_list as module

    settings = SimpleNamespace(
        model="m", max_steps=6, timeout_seconds=5, attempts=attempts
    )
    with patch("google.genai.Client", return_value=client):
        return await module.build_step_list(
            "How to x", "", api_key="k", settings=settings
        )


@pytest.mark.req("REQ-VID-121")
@pytest.mark.asyncio
async def test_a_timed_out_call_is_tried_again() -> None:
    client = _client([TimeoutError(), _ANSWER])

    result = await _build(client, attempts=2)

    assert result is not None and len(client.calls) == 2


@pytest.mark.req("REQ-VID-121")
@pytest.mark.asyncio
async def test_an_answered_call_is_not_repeated() -> None:
    client = _client([_ANSWER, _ANSWER])

    assert await _build(client, attempts=2) is not None
    assert len(client.calls) == 1


@pytest.mark.req("REQ-VID-121")
@pytest.mark.asyncio
async def test_retries_stop_at_the_configured_attempts() -> None:
    client = _client([TimeoutError(), TimeoutError(), _ANSWER])

    assert await _build(client, attempts=2) is None
    assert len(client.calls) == 2


@pytest.mark.req("REQ-VID-121")
@pytest.mark.asyncio
async def test_a_client_error_is_not_retried() -> None:
    from google.genai import errors

    # No credit or a bad request fails the same way on a second call.
    error = errors.ClientError(402, {"error": {"message": "credits depleted"}})
    client = _client([error, _ANSWER])

    assert await _build(client, attempts=2) is None
    assert len(client.calls) == 1


@pytest.mark.req("REQ-VID-121")
def test_step_lists_ship_on_with_a_retry() -> None:
    shipped = config.llm_settings.topic_scripts.step_list
    assert shipped.enabled is True
    assert shipped.attempts >= 2
