"""Tests for global batch config CLI override logic."""

import argparse
import tempfile
from pathlib import Path

import pytest
import yaml

from src.pipeline.config import load_global_batch_config


def _write_yaml(tmp_path: Path, data: dict) -> str:
    """Write a pipeline YAML and return its path."""
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False))
    return str(path)


def _make_cli(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse.Namespace with given overrides."""
    defaults = {
        "product_ids": None,
        "keywords": None,
        "max_products": None,
        "products_per_keyword": None,
        "min_price": None,
        "max_price": None,
        "min_rating": None,
        "prime_only": False,
        "profile": None,
        "random_profile": False,
        "profile_pool": None,
        "fail_fast": False,
        "process_all_products": False,
        "outputs_dir": None,
        "debug": False,
        "skip_publish": False,
        "platforms": None,
        "schedule_time": None,
        "fail_fast_publish": False,
        "platform_specific": False,
        "voice_profile": None,
        "script_template": None,
        "resume": False,
        "dry_run": False,
        "output_format": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture()
def yaml_with_keywords(tmp_path):
    """YAML with 3 keywords and no product_ids."""
    return _write_yaml(
        tmp_path,
        {
            "global_batch": {
                "product_ids": [],
                "keywords": ["smart ring", "mini projector", "portable charger"],
                "max_products": 10,
                "products_per_keyword": 1,
            }
        },
    )


@pytest.fixture()
def yaml_with_both(tmp_path):
    """YAML with both product_ids and keywords."""
    return _write_yaml(
        tmp_path,
        {
            "global_batch": {
                "product_ids": ["B0YAMLASIN1"],
                "keywords": ["yaml keyword"],
            }
        },
    )


class TestCLIOverride:
    """CLI inputs should suppress YAML defaults for the other input type."""

    def test_cli_product_ids_suppresses_yaml_keywords(self, yaml_with_keywords):
        cli = _make_cli(product_ids=["B0CLIASIN1"])
        config = load_global_batch_config(cli, config_path=yaml_with_keywords)

        assert config.product_ids == ["B0CLIASIN1"]
        assert config.keywords == []

    def test_cli_keywords_suppresses_yaml_product_ids(self, yaml_with_both):
        cli = _make_cli(keywords=["cli keyword"])
        config = load_global_batch_config(cli, config_path=yaml_with_both)

        assert config.keywords == ["cli keyword"]
        assert config.product_ids == []

    def test_cli_both_uses_both(self, yaml_with_keywords):
        cli = _make_cli(product_ids=["B0CLIASIN1"], keywords=["cli keyword"])
        config = load_global_batch_config(cli, config_path=yaml_with_keywords)

        assert config.product_ids == ["B0CLIASIN1"]
        assert config.keywords == ["cli keyword"]

    def test_no_cli_uses_yaml_defaults(self, yaml_with_keywords):
        cli = _make_cli()
        config = load_global_batch_config(cli, config_path=yaml_with_keywords)

        assert config.product_ids == []
        assert config.keywords == ["smart ring", "mini projector", "portable charger"]

    def test_no_cli_uses_yaml_both(self, yaml_with_both):
        cli = _make_cli()
        config = load_global_batch_config(cli, config_path=yaml_with_both)

        assert config.product_ids == ["B0YAMLASIN1"]
        assert config.keywords == ["yaml keyword"]

    def test_cli_product_ids_empty_list_treated_as_no_input(self, yaml_with_keywords):
        """Empty list from CLI is falsy, so falls through to YAML."""
        cli = _make_cli(product_ids=[])
        config = load_global_batch_config(cli, config_path=yaml_with_keywords)

        # Empty list is falsy, so no CLI inputs -> use YAML
        assert config.keywords == ["smart ring", "mini projector", "portable charger"]
