"""Filters for the OpenRouter free-model pool.

The pool is whatever OpenRouter currently lists as free. It is unaudited, it
changes under the account, and two kinds of entry in it cannot do this
pipeline's job:

- A model that does not emit text. The pool carries music and image models
  beside the chat ones, and a request to one either errors or returns
  something no validator downstream is written against.
- A model that reasons in its visible output. Its monologue about writing a
  description is longer and wordier than a description, so every length floor
  a validator applies is cleared by a wide margin (#404).

Both are declared by the models API, so they are decided here from the
response rather than from a blocklist that has to chase a moving pool. The
blocklist stays for the case this cannot see: a model that answers in the
right shape and answers badly.

Unknown is treated as usable throughout. A response shape this does not
recognise must not empty the pool, because the pool is the fallback that keeps
a batch running when the primary provider is rate-limited.
"""

from __future__ import annotations

from typing import Any

__all__ = ["model_reject_reason"]


def _output_modality_reason(model: dict[str, Any]) -> str | None:
    """Reject a model that advertises any output modality other than text."""
    architecture = model.get("architecture")
    if not isinstance(architecture, dict):
        return None
    modalities = architecture.get("output_modalities")
    if not isinstance(modalities, list) or not modalities:
        return None
    if [m for m in modalities if m != "text"]:
        return f"outputs {'/'.join(str(m) for m in modalities)}, not text alone"
    return None


def _reasoning_reason(model: dict[str, Any]) -> str | None:
    """Reject a model whose reasoning is on unless a caller turns it off.

    Three spellings of the same thing appear in the live response, and a model
    matching any of them emits its reasoning by default: ``mandatory``,
    ``default_enabled``, and a ``default_effort`` above ``none``. A model that
    merely *supports* reasoning is kept, because it does not use it unless
    asked to and nothing here asks.
    """
    reasoning = model.get("reasoning")
    if not isinstance(reasoning, dict):
        return None
    if reasoning.get("mandatory") is True:
        return "reasons in its output and cannot be turned off"
    if reasoning.get("default_enabled") is True:
        return "reasons in its output by default"
    effort = reasoning.get("default_effort")
    if isinstance(effort, str) and effort.lower() not in ("", "none"):
        return f"reasons in its output by default (effort '{effort}')"
    return None


def model_reject_reason(model: dict[str, Any]) -> str | None:
    """Return why this discovered model is unusable here, or None if it is.

    ``model`` is one entry of the OpenRouter ``/models`` response.
    """
    return _output_modality_reason(model) or _reasoning_reason(model)
