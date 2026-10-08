"""The quality report groups posts by duration band (REQ-PUB-084, design 0025)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.publisher.analytics import PostMetrics, duration_band
from src.publisher.models import AnalyticsConfig

EDGES = [20.0, 30.0, 45.0, 60.0]


@pytest.mark.req("REQ-PUB-084")
@pytest.mark.parametrize(
    ("seconds", "band"),
    [
        (12, "<20s"),
        (20, "20-30s"),
        (29.99, "20-30s"),
        (34.5, "30-45s"),
        ("50.0", "45-60s"),
        (60, ">60s"),
        (77, ">60s"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_a_length_falls_in_one_band(seconds: object, band: str) -> None:
    assert duration_band(seconds, EDGES) == band


@pytest.mark.req("REQ-PUB-084")
def test_the_bands_ship_as_the_design_sets_them() -> None:
    import yaml

    from src.utils.outputs_paths import get_project_root

    shipped = yaml.safe_load(
        (get_project_root() / "config" / "publisher.yaml").read_text()
    )["analytics"]
    assert AnalyticsConfig(**shipped).duration_bands_sec == [20, 30, 45, 60]
    assert AnalyticsConfig().duration_bands_sec == EDGES


@pytest.mark.req("REQ-PUB-084")
@pytest.mark.parametrize(
    "bands", [[], [30, 20], [20, 20], [0, 20], [-5], [True, 30], "20,30", ["20"]]
)
def test_bands_that_cannot_partition_are_rejected(bands: object) -> None:
    assert AnalyticsConfig(duration_bands_sec=[20, 30]).duration_bands_sec == [20, 30]
    with pytest.raises(ValueError, match="duration_bands_sec"):
        AnalyticsConfig(duration_bands_sec=bands)


@pytest.mark.req("REQ-PUB-084")
def test_the_report_groups_posts_by_band(monkeypatch, caplog) -> None:
    import src.publisher.product_registry as registry
    import src.utils.render_choices_store as store
    from src.publisher.late import cli as late_cli

    rows = [
        {"product_id": "A", "video_duration_sec": 25.0},
        {"product_id": "B", "video_duration_sec": 28.0},
        {"product_id": "C"},  # rendered before durations were recorded
    ]
    metrics = [
        PostMetrics(post_id=p, platform_metrics={"tiktok": {"views": v}})
        for p, v in (("a", 100), ("b", 300), ("c", 50))
    ]
    monkeypatch.setattr(store, "load_recent", lambda outputs, n: rows)
    monkeypatch.setattr(store, "latest_per_product", lambda rows: rows)
    monkeypatch.setattr(
        registry,
        "load_registry",
        lambda outputs: [SimpleNamespace(product_id="D", content_format="review")],
    )
    monkeypatch.setattr(late_cli, "load_metrics", lambda outputs: metrics)
    monkeypatch.setattr(
        late_cli, "_load_product_map", lambda outputs: {"a": "A", "b": "B", "c": "C"}
    )

    with caplog.at_level(logging.INFO):
        late_cli._log_quality_segments(Path("."), EDGES)

    assert "duration_band=20-30s [2 post(s)]: tiktok.views=200.0(n=2)" in caplog.text
    assert "duration_band=unknown [1 post(s)]: tiktok.views=50.0(n=1)" in caplog.text
