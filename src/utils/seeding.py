"""Per-product random draws that are the same in every process.

Python's `hash()` of a string is salted per process, so a choice seeded from
it changes from one run to the next. A choice seeds from the md5 of
`<product_id>:<purpose>` instead, and draws from its own generator rather
than reseeding the global one, which would change every later draw.
"""

from __future__ import annotations

import hashlib
import random


def product_rng(product_id: str, purpose: str) -> random.Random:
    """A generator seeded by the product and what the draw is for."""
    digest = hashlib.md5(
        f"{product_id}:{purpose}".encode(), usedforsecurity=False
    ).hexdigest()
    return random.Random(int(digest[:16], 16))  # noqa: S311
