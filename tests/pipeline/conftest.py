"""Shared fixtures for the batch pipeline tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _clear_publisher_settings_cache() -> Iterator[None]:
    """Drop the batch's cached publisher config around every test.

    `_publisher_settings` is cached so all four call sites in one run see the
    same object, which is the property the module is meant to have. Across
    tests that same cache is a leak: whichever test loads first decides what
    every later one reads, including a deliberately broken config whose
    fallback would then be served to a test expecting the real file.
    """
    from src.pipeline.global_batch import _publisher_settings

    _publisher_settings.cache_clear()
    yield
    _publisher_settings.cache_clear()
