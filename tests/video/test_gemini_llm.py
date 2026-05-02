"""Unit tests for the pycaps Gemini adapter.

The adapter implements pycaps' ``Llm`` ABC and is wired into pycaps'
``LlmProvider`` singleton from ``step_burn_pycaps_subtitles`` when AI
tagging is enabled. Tests use a fake ``google.genai.Client`` so no real
network calls happen.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "pycaps",
    reason="pycaps optional group is not installed (poetry install --with pycaps)",
)

from src.video.pycaps_engine.gemini_llm import GeminiLlm  # noqa: E402


def _install_fake_genai(monkeypatch: pytest.MonkeyPatch, response_text: str = "OK"):
    """Install a fake ``google.genai`` module that records calls.

    Returns the ``Client`` mock so tests can assert against
    ``client.models.generate_content`` invocations.
    """
    fake_genai = types.ModuleType("google.genai")
    client_instance = MagicMock(name="GeminiClient")
    response = MagicMock(text=response_text)
    client_instance.models.generate_content.return_value = response
    fake_genai.Client = MagicMock(return_value=client_instance)

    fake_google_pkg = types.ModuleType("google")
    fake_google_pkg.genai = fake_genai

    # Patch into sys.modules so ``from google import genai`` inside the adapter
    # resolves to our fake.
    monkeypatch.setitem(sys.modules, "google", fake_google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    return fake_genai, client_instance


def test_is_enabled_false_without_key():
    adapter = GeminiLlm(api_key=None)
    assert adapter.is_enabled() is False


def test_is_enabled_true_with_key():
    adapter = GeminiLlm(api_key="test-key")
    assert adapter.is_enabled() is True


def test_send_message_raises_when_no_key():
    adapter = GeminiLlm(api_key=None)
    with pytest.raises(RuntimeError, match="without an API key"):
        adapter.send_message("hello")


def test_send_message_round_trip(monkeypatch: pytest.MonkeyPatch):
    fake_genai, client = _install_fake_genai(monkeypatch, response_text="tagged")
    adapter = GeminiLlm(api_key="k", model="gemini-2.5-flash")

    result = adapter.send_message("the most impactful word", model=None)

    assert result == "tagged"
    fake_genai.Client.assert_called_once_with(api_key="k")
    client.models.generate_content.assert_called_once_with(
        model="gemini-2.5-flash", contents="the most impactful word"
    )
    assert adapter.call_count == 1


def test_send_message_uses_explicit_model_arg(monkeypatch: pytest.MonkeyPatch):
    _, client = _install_fake_genai(monkeypatch)
    adapter = GeminiLlm(api_key="k", model="gemini-2.5-flash")

    adapter.send_message("hi", model="gemini-2.5-pro")

    client.models.generate_content.assert_called_once_with(
        model="gemini-2.5-pro", contents="hi"
    )


def test_send_message_returns_empty_string_when_response_text_none(
    monkeypatch: pytest.MonkeyPatch,
):
    """Gemini can return responses with no `.text` (safety blocks, empty)."""
    _, client = _install_fake_genai(monkeypatch)
    client.models.generate_content.return_value = MagicMock(text=None)
    adapter = GeminiLlm(api_key="k")

    assert adapter.send_message("x") == ""


def test_client_is_lazy(monkeypatch: pytest.MonkeyPatch):
    """`google.genai` must not be imported until first send_message call."""
    fake_genai, _ = _install_fake_genai(monkeypatch)
    adapter = GeminiLlm(api_key="k")

    fake_genai.Client.assert_not_called()
    adapter.send_message("hello")
    fake_genai.Client.assert_called_once()


def test_client_reused_across_calls(monkeypatch: pytest.MonkeyPatch):
    fake_genai, _ = _install_fake_genai(monkeypatch)
    adapter = GeminiLlm(api_key="k")

    adapter.send_message("a")
    adapter.send_message("b")

    assert fake_genai.Client.call_count == 1
    assert adapter.call_count == 2


def test_on_error_skip_returns_empty_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    _, client = _install_fake_genai(monkeypatch)
    client.models.generate_content.side_effect = RuntimeError("safety block")
    adapter = GeminiLlm(api_key="k", on_error="skip")

    with caplog.at_level("WARNING"):
        result = adapter.send_message("oops")

    assert result == ""
    assert "GeminiLlm.send_message failed" in caplog.text
    assert adapter.call_count == 1


def test_on_error_raise_propagates(monkeypatch: pytest.MonkeyPatch):
    _, client = _install_fake_genai(monkeypatch)
    client.models.generate_content.side_effect = RuntimeError("safety block")
    adapter = GeminiLlm(api_key="k", on_error="raise")

    with pytest.raises(RuntimeError, match="safety block"):
        adapter.send_message("oops")
