"""Tests for the performance report generator tool."""

import json
import tempfile
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.utils.performance import PerformanceHistoryManager, PipelineRunMetrics
from tools.performance_report import (
    PerformanceReportGenerator,
    _percentile,
    _report_to_csv,
)


def _recent_ts(days_ago: int = 0) -> str:
    """Return an ISO timestamp for N days ago (UTC)."""
    dt = datetime.now(tz=UTC) - timedelta(days=days_ago)
    return dt.isoformat()


def _make_run(
    run_id: str = "r1",
    product_id: str = "P1",
    profile_name: str = "slideshow_images1",
    duration: float = 10.0,
    success: bool = True,
    timestamp: str | None = None,
    step_durations: list[tuple[str, float]] | None = None,
) -> PipelineRunMetrics:
    """Helper to build a PipelineRunMetrics for tests."""
    if timestamp is None:
        timestamp = _recent_ts(0)
    if step_durations is None:
        step_durations = [
            ("gather_visuals", duration / 2),
            ("generate_script", duration / 2),
        ]

    steps = []
    for name, dur in step_durations:
        steps.append(
            {
                "step_name": name,
                "start_time": 1000,
                "end_time": 1000 + dur,
                "duration": dur,
                "memory_start": 100,
                "memory_peak": 150,
                "memory_end": 120,
                "cpu_percent": 30,
                "io_read_bytes": 0,
                "io_write_bytes": 0,
                "errors": [],
                "metadata": {},
            }
        )

    return PipelineRunMetrics(
        run_id=run_id,
        product_id=product_id,
        profile_name=profile_name,
        start_timestamp=timestamp,
        end_timestamp=timestamp,
        total_duration=duration,
        total_memory_delta=5.0,
        peak_memory=200.0,
        total_cpu_percent=30.0,
        step_metrics=steps,
        success=success,
    )


def _seed_history(
    hm: PerformanceHistoryManager,
    count: int = 5,
    profile: str = "slideshow_images1",
    base_duration: float = 10.0,
) -> None:
    """Populate a history manager with test runs."""
    for i in range(count):
        hm.save_run_metrics(
            _make_run(
                run_id=f"run-{i}",
                product_id=f"P{i % 3}",
                profile_name=profile,
                duration=base_duration + i,
                timestamp=_recent_ts(days_ago=count - i),
            )
        )


class TestPercentile:
    """Test the _percentile helper."""

    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([5.0], 50) == 5.0
        assert _percentile([5.0], 99) == 5.0

    def test_median(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 50) == 3.0

    def test_p95(self):
        vals = list(range(1, 101))
        p95 = _percentile([float(v) for v in vals], 95)
        assert 95 <= p95 <= 96


class TestSummaryReport:
    """Test summary report generation."""

    def test_summary_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            _seed_history(hm, count=5)

            gen = PerformanceReportGenerator(hm)
            report = gen.generate_summary_report(limit=50)

            assert report["report_type"] == "summary"
            assert report["data_range"]["total_runs"] == 5
            assert report["success_metrics"]["success_rate_percent"] == 100.0

            duration = report["performance_metrics"]["duration"]
            assert "p50_seconds" in duration
            assert "p95_seconds" in duration
            assert "p99_seconds" in duration
            assert duration["minimum_seconds"] <= duration["p50_seconds"]
            assert duration["p50_seconds"] <= duration["maximum_seconds"]

    def test_summary_empty_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            gen = PerformanceReportGenerator(hm)
            report = gen.generate_summary_report()
            assert "error" in report

    def test_step_analysis_has_percentiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            _seed_history(hm, count=10)

            gen = PerformanceReportGenerator(hm)
            report = gen.generate_summary_report()
            steps = report["step_analysis"]

            for _step_name, stats in steps.items():
                assert "p50_duration" in stats
                assert "p95_duration" in stats
                assert "p99_duration" in stats


class TestTrendsReport:
    """Test trends report generation."""

    def test_trends_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            _seed_history(hm, count=5)

            gen = PerformanceReportGenerator(hm)
            report = gen.generate_trends_report(days=30)

            assert report["report_type"] == "trends"
            assert len(report["trend_data"]) > 0
            assert "step_trends" in report

            # Each day should have step trends too
            step_trends = report["step_trends"]
            assert "gather_visuals" in step_trends

    def test_trends_product_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            _seed_history(hm, count=6)

            gen = PerformanceReportGenerator(hm)
            report = gen.generate_trends_report(product_id="P0", days=30)

            assert report["filters"]["product_id"] == "P0"
            assert report["filters"]["total_runs"] == 2  # P0 appears at i=0,3

    def test_trends_empty_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            gen = PerformanceReportGenerator(hm)
            report = gen.generate_trends_report()
            assert "error" in report


class TestDetailedReport:
    """Test detailed report generation."""

    def test_detailed_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            _seed_history(hm, count=3)

            gen = PerformanceReportGenerator(hm)
            report = gen.generate_detailed_report(limit=10)

            assert report["report_type"] == "detailed"
            assert len(report["runs"]) == 3

            run = report["runs"][0]
            assert "run_id" in run
            assert "step_details" in run
            assert len(run["step_details"]) == 2

    def test_detailed_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            gen = PerformanceReportGenerator(hm)
            report = gen.generate_detailed_report()
            assert "error" in report


class TestComparisonReport:
    """Test profile comparison report."""

    def test_comparison_multiple_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))

            for i in range(5):
                hm.save_run_metrics(
                    _make_run(
                        run_id=f"a-{i}",
                        profile_name="slideshow_images1",
                        duration=10.0 + i,
                        timestamp=_recent_ts(days_ago=10 - i),
                    )
                )
            for i in range(5):
                hm.save_run_metrics(
                    _make_run(
                        run_id=f"b-{i}",
                        profile_name="video_sequential",
                        duration=20.0 + i,
                        timestamp=_recent_ts(days_ago=10 - i),
                    )
                )

            gen = PerformanceReportGenerator(hm)
            report = gen.generate_comparison_report()

            assert report["report_type"] == "comparison"
            assert "slideshow_images1" in report["profiles"]
            assert "video_sequential" in report["profiles"]

            slide_stats = report["profiles"]["slideshow_images1"]
            assert slide_stats["run_count"] == 5
            assert "duration" in slide_stats
            assert "p50" in slide_stats["duration"]
            assert "p95" in slide_stats["duration"]

    def test_comparison_single_profile(self):
        """Should return error when only one profile exists."""
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            _seed_history(hm, count=3)

            gen = PerformanceReportGenerator(hm)
            report = gen.generate_comparison_report()
            assert "error" in report

    def test_comparison_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            gen = PerformanceReportGenerator(hm)
            report = gen.generate_comparison_report()
            assert "error" in report


class TestRegressionDetection:
    """Test regression detection."""

    def test_detect_regressions_found(self):
        """Detect a clear regression in step duration."""
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))

            # Previous runs: fast (older)
            for i in range(5):
                hm.save_run_metrics(
                    _make_run(
                        run_id=f"old-{i}",
                        duration=10.0,
                        step_durations=[
                            ("gather_visuals", 5.0),
                            ("generate_script", 5.0),
                        ],
                        timestamp=_recent_ts(days_ago=15 - i),
                    )
                )

            # Recent runs: much slower (newer)
            for i in range(5):
                hm.save_run_metrics(
                    _make_run(
                        run_id=f"new-{i}",
                        duration=50.0,
                        step_durations=[
                            ("gather_visuals", 25.0),
                            ("generate_script", 25.0),
                        ],
                        timestamp=_recent_ts(days_ago=5 - i),
                    )
                )

            gen = PerformanceReportGenerator(hm)
            report = gen.detect_regressions(window=5, threshold_factor=2.0)

            assert report["status"] == "regressions_found"
            assert len(report["step_regressions"]) > 0
            assert report["pipeline_regression"] is not None

    def test_detect_regressions_ok(self):
        """No regressions when performance is stable."""
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))

            for i in range(10):
                hm.save_run_metrics(
                    _make_run(
                        run_id=f"r-{i}",
                        duration=10.0,
                        timestamp=_recent_ts(days_ago=10 - i),
                    )
                )

            gen = PerformanceReportGenerator(hm)
            report = gen.detect_regressions(window=5)

            assert report["status"] == "ok"
            assert len(report["step_regressions"]) == 0
            assert report["pipeline_regression"] is None

    def test_detect_regressions_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            _seed_history(hm, count=3)

            gen = PerformanceReportGenerator(hm)
            report = gen.detect_regressions(window=5)
            assert "error" in report


class TestCSVExport:
    """Test CSV export format."""

    def test_csv_detailed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            _seed_history(hm, count=3)

            gen = PerformanceReportGenerator(hm)
            report = gen.generate_detailed_report(limit=10)
            csv_output = _report_to_csv(report)

            lines = csv_output.strip().split("\n")
            assert len(lines) == 4  # header + 3 data rows
            assert "run_id" in lines[0]
            assert "product_id" in lines[0]

    def test_csv_trends_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            _seed_history(hm, count=5)

            gen = PerformanceReportGenerator(hm)
            report = gen.generate_trends_report(days=30)
            csv_output = _report_to_csv(report)

            lines = csv_output.strip().split("\n")
            assert len(lines) > 1
            assert "date" in lines[0]

    def test_csv_empty_detailed(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            gen = PerformanceReportGenerator(hm)
            report = gen.generate_detailed_report()
            csv_output = _report_to_csv(report)
            # Error report falls back to JSON
            assert "error" in csv_output

    def test_csv_unsupported_type_falls_back_to_json(self):
        report = {"report_type": "summary", "data": "test"}
        csv_output = _report_to_csv(report)
        parsed = json.loads(csv_output)
        assert parsed["report_type"] == "summary"
