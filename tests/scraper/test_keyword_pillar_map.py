"""Tests for keyword-to-pillar mapping in batch config (#82)."""

from __future__ import annotations

from unittest.mock import patch

from src.scraper.amazon.models import BatchConfig, SearchParameters


def _make_config(
    keywords: list[str] | None = None,
    keyword_pillar_map: dict[str, str] | None = None,
) -> BatchConfig:
    return BatchConfig(
        product_ids=[],
        keywords=keywords or [],
        fail_fast=False,
        search_params=SearchParameters(),
        max_products=10,
        products_per_keyword=1,
        keyword_pillar_map=keyword_pillar_map or {},
    )


class TestBatchConfigPillarFor:
    def test_returns_pillar_for_mapped_keyword(self) -> None:
        config = _make_config(
            keywords=["USB C hub"],
            keyword_pillar_map={"USB C hub": "value"},
        )
        assert config.pillar_for("USB C hub") == "value"

    def test_returns_none_for_unmapped_keyword(self) -> None:
        config = _make_config(
            keywords=["random thing"],
            keyword_pillar_map={"USB C hub": "value"},
        )
        assert config.pillar_for("random thing") is None

    def test_empty_map_returns_none(self) -> None:
        config = _make_config(keywords=["USB C hub"])
        assert config.pillar_for("USB C hub") is None


class TestLoadBatchConfigDictKeywords:
    """Config loader parses dict-shaped keywords and builds the pillar map."""

    def test_dict_keywords_flatten_and_map(self) -> None:
        yaml_config = {
            "batch": {
                "keywords": {
                    "value": ["USB C hub", "smart plug"],
                    "novelty": ["smart ring"],
                },
                "product_ids": [],
            },
            "scrapers": {"amazon": {"max_products": 10}},
        }
        with patch("src.scraper.amazon.config.CONFIG", yaml_config):
            from src.scraper.amazon.config import load_batch_config

            bc = load_batch_config()
        # The keyword list keeps its spelling, because it is what gets
        # searched; the map is keyed by the matching form, because it is what
        # gets looked up.
        assert bc.keywords == ["USB C hub", "smart plug", "smart ring"]
        assert bc.keyword_pillar_map == {
            "usb c hub": "value",
            "smart plug": "value",
            "smart ring": "novelty",
        }
        assert bc.pillar_for("USB C hub") == "value"
        assert bc.pillar_for("smart ring") == "novelty"

    def test_flat_list_keywords_backward_compat(self) -> None:
        yaml_config = {
            "batch": {
                "keywords": ["USB C hub", "smart ring"],
                "product_ids": [],
            },
            "scrapers": {"amazon": {"max_products": 10}},
        }
        with patch("src.scraper.amazon.config.CONFIG", yaml_config):
            from src.scraper.amazon.config import load_batch_config

            bc = load_batch_config()
        assert bc.keywords == ["USB C hub", "smart ring"]
        assert bc.keyword_pillar_map == {}
        assert bc.pillar_for("USB C hub") is None

    def test_cli_keywords_override_yaml(self) -> None:
        yaml_config = {
            "batch": {
                "keywords": {
                    "value": ["USB C hub"],
                },
                "product_ids": [],
            },
            "scrapers": {"amazon": {"max_products": 10}},
        }
        with patch("src.scraper.amazon.config.CONFIG", yaml_config):
            from src.scraper.amazon.config import load_batch_config

            bc = load_batch_config(cli_keywords=["custom keyword"])
        assert bc.keywords == ["custom keyword"]
        # CLI keywords have no pillar map entry (pillar only from YAML dict)
        assert bc.pillar_for("custom keyword") is None


class TestBaseProductDataPillar:
    def test_pillar_field_defaults_none(self) -> None:
        from src.scraper.amazon.models import ProductData
        from src.scraper.base.models import Platform

        p = ProductData(
            title="Test",
            price="$10",
            url="https://example.com",
            platform=Platform.AMAZON,
        )
        assert p.pillar is None

    def test_pillar_field_serializes(self) -> None:
        from src.scraper.amazon.models import ProductData
        from src.scraper.base.models import Platform

        p = ProductData(
            title="Test",
            price="$10",
            url="https://example.com",
            platform=Platform.AMAZON,
            pillar="value",
        )
        d = p.to_dict()
        assert d["pillar"] == "value"
