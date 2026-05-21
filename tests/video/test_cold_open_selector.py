"""Tests for the Phase 1.2e cold-open variant selector."""

from __future__ import annotations

from src.video.cold_open_selector import select_cold_open_variant


def test_empty_pool_returns_default() -> None:
    assert select_cold_open_variant("B0X", []) == "mid_zoom_title_card"
    assert select_cold_open_variant("B0X", None) == "mid_zoom_title_card"


def test_single_entry_pool_returns_only_entry() -> None:
    assert select_cold_open_variant("B0X", ["only_one"]) == "only_one"


def test_deterministic_per_product() -> None:
    """Same product_id always returns the same variant."""
    pool = ["a", "b", "c"]
    first = select_cold_open_variant("B0DETERMINISTIC", pool)
    second = select_cold_open_variant("B0DETERMINISTIC", pool)
    third = select_cold_open_variant("B0DETERMINISTIC", pool)
    assert first == second == third
    assert first in pool


def test_distinct_products_spread_across_pool() -> None:
    """Across many product_ids, the selector covers each variant."""
    pool = ["a", "b", "c"]
    chosen = {select_cold_open_variant(f"B0PRODUCT{i:04d}", pool) for i in range(200)}
    assert chosen == set(pool), f"Selector missed variants: {set(pool) - chosen}"


def test_returns_member_of_pool() -> None:
    pool = ["x", "y", "z"]
    for i in range(50):
        assert select_cold_open_variant(f"ASIN_{i}", pool) in pool
