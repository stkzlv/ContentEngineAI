"""Gemini adapter for pycaps AI word tagging.

Plugs ContentEngineAI's existing Gemini key into the pycaps ``LlmProvider``
singleton so templates with ``tagger_rules`` of ``type: ai`` (e.g. the built-in
``neo-minimal`` and ``explosive`` presets) call Gemini instead of OpenAI.

Pycaps' base ``Llm`` ABC declares ``send_message(message: str, model: str)``
but its real call sites pass only ``prompt`` (see
``pycaps/tag/tagger/external_llm_tagger.py``). We mirror the existing ``Gpt``
adapter and default ``model`` so single-arg calls work.

Constructed lazily — ``google.genai`` is not imported until the first
``send_message`` call. Keeps non-pycaps runs free of the import cost.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from google.genai import Client

logger = logging.getLogger(__name__)

# Inherit from pycaps' Llm ABC when the optional group is installed; fall back
# to a no-op stub so this module imports cleanly in the default install. The
# Llm base only declares two abstract methods, both of which we implement, so
# the runtime parent doesn't change behavior. Pycaps consumers duck-type via
# LlmProvider.get().send_message(...) — no isinstance checks involved.
try:
    from pycaps.ai.llm import Llm as _LlmBase
except ImportError:

    class _LlmBase:  # type: ignore[no-redef]
        """Stub parent used when pycaps is not installed."""


class GeminiLlm(_LlmBase):
    """pycaps ``Llm`` implementation backed by ``google-genai``.

    Args:
    ----
        api_key: Gemini API key. ``None`` disables the adapter; ``is_enabled``
            returns ``False`` and pycaps' AI tagger silently skips the rule.
        model: Default Gemini model. Pycaps usually calls ``send_message``
            with one positional arg, so this default is what the API ends
            up using in practice.
        on_error: ``"skip"`` swallows API errors and returns an empty string
            (pycaps' tagger then drops the tag for that segment).
            ``"raise"`` re-raises so the caller's fallback policy decides.

    """

    def __init__(
        self,
        api_key: str | None,
        model: str = "gemini-2.5-flash",
        on_error: Literal["raise", "skip"] = "skip",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._on_error = on_error
        self._client: Client | None = None
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Number of ``send_message`` invocations since construction."""
        return self._call_count

    def is_enabled(self) -> bool:
        return bool(self._api_key)

    def send_message(self, prompt: str, model: str | None = None) -> str:
        if not self._api_key:
            raise RuntimeError(
                "GeminiLlm.send_message called without an API key. "
                "Check the Gemini key is in ctx.secrets before registering."
            )

        client = self._get_client()
        target_model = model or self._model
        self._call_count += 1
        try:
            resp: Any = client.models.generate_content(
                model=target_model, contents=prompt
            )
        except Exception as e:  # noqa: BLE001 - downstream library raises a wide tree
            if self._on_error == "raise":
                raise
            logger.warning(
                "GeminiLlm.send_message failed (model=%s): %s. "
                "Returning empty response so the AI tagger drops this segment.",
                target_model,
                e,
            )
            return ""

        text = getattr(resp, "text", None)
        return text or ""

    def _get_client(self) -> Client:
        if self._client is not None:
            return self._client
        from google import genai

        self._client = genai.Client(api_key=self._api_key)
        return self._client
