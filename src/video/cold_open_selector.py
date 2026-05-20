"""Cold-open variant selection (Phase 1.2e).

Picks one named cold-open variant per product using a deterministic salted
MD5 hash on the product ID — same pattern as script template, font, colour,
voice profile, and pycaps template selection elsewhere in the project.

v1 records the selected variant in ``pipeline_state.json`` so downstream
analytics can segment retention by variant. Visual differentiation between
variants (different pre-motion peaks, hook durations, title-card styles) is
expected to follow once a baseline render exists to A/B against. Until then,
all variants render identically — the framework lets analytics call out
which variant the LLM and assembler produced.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

_DEFAULT_VARIANT = "mid_zoom_title_card"
_SALT = "cold_open_variant"


def select_cold_open_variant(product_id: str, pool: list[str] | None) -> str:
    """Return one cold-open variant name from ``pool`` keyed on ``product_id``.

    Empty / None pool returns the default variant. Single-entry pool returns
    that entry. The hex slice [0:8] matches the pycaps template selector;
    salt differentiates the selections so they don't collide.
    """
    if not pool:
        return _DEFAULT_VARIANT
    if len(pool) == 1:
        return pool[0]
    digest = hashlib.md5(  # noqa: S324
        f"{product_id}:{_SALT}".encode()
    ).hexdigest()
    index = int(digest[0:8], 16) % len(pool)
    return pool[index]
