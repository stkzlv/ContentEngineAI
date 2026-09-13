"""Tests for performance monitoring utilities."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.utils.performance import (
    PerformanceHistoryManager,
    PerformanceMetrics,
    PerformanceMonitor,
    PipelineRunMetrics,
    async_timer,
    performance_monitor,
    timer,
)


class TestPerformanceMetrics:
    """Test performance metrics data container."""

    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            step_name="test_step",
            start_time=1000.0,
            end_time=1002.5,
            duration=2.5,
            memory_start=100.0,
            memory_peak=150.0,
            memory_end=120.0,
            cpu_percent=45.5,
        )

        assert metrics.step_name == "test_step"
        assert metrics.duration == 2.5
        assert metrics.duration_ms == 2500.0
        assert metrics.memory_delta == 20.0

    def test_metrics_with_defaults(self):
        """Test metrics with default values."""
        metrics = PerformanceMetrics(
            step_name="test",
            start_time=1000.0,
            end_time=1001.0,
            duration=1.0,
            memory_start=100.0,
            memory_peak=100.0,
            memory_end=100.0,
            cpu_percent=10.0,
        )

        assert metrics.io_read_bytes == 0
        assert metrics.io_write_bytes == 0
        assert metrics.errors == []
        assert metrics.metadata == {}


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor.metrics == []
        assert monitor.current_step is None
        assert monitor.pipeline_start is None

    def test_monitor_custom_interval(self):
        """Test monitor with custom memory_monitor_interval."""
        monitor = PerformanceMonitor(memory_monitor_interval=0.5)
        assert monitor.memory_monitor_interval == 0.5

    def test_start_pipeline(self):
        """Test pipeline start tracking."""
        monitor = PerformanceMonitor()
        monitor.start_pipeline()

        assert monitor.pipeline_start is not None
        assert monitor.metrics == []

    @patch("src.utils.performance.psutil.Process")
    def test_reset_clears_state(self, mock_process_class):
        """Test reset() clears all pipeline state."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        monitor.start_pipeline(
            run_id="test-run", product_id="P1", profile_name="profile1"
        )
        monitor.metrics.append(
            PerformanceMetrics(
                step_name="s",
                start_time=0,
                end_time=1,
                duration=1,
                memory_start=0,
                memory_peak=0,
                memory_end=0,
                cpu_percent=0,
            )
        )

        monitor.reset()

        assert monitor.metrics == []
        assert monitor.pipeline_start is None
        assert monitor.current_step is None
        assert monitor.current_run_id is None
        assert monitor.current_product_id is None
        assert monitor.current_profile_name is None

    @patch("src.utils.performance.psutil.Process")
    def test_reset_sets_history_manager(self, mock_process_class):
        """Test reset() can set a new history manager."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        assert monitor.history_manager is None

        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            monitor.reset(history_manager=hm)
            assert monitor.history_manager is hm

    @patch("src.utils.performance.psutil.Process")
    def test_get_memory_usage(self, mock_process_class):
        """Test memory usage measurement."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100 MB
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        memory_usage = monitor.get_memory_usage()
        assert memory_usage == 100.0

    @patch("src.utils.performance.psutil.Process")
    def test_get_cpu_percent(self, mock_process_class):
        """Test CPU usage measurement."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 25.5
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        cpu_usage = monitor.get_cpu_percent()
        assert cpu_usage == 25.5

    @patch("src.utils.performance.psutil.Process")
    def test_get_io_stats(self, mock_process_class):
        """Test I/O statistics measurement."""
        mock_process = Mock()
        mock_process.io_counters.return_value.read_bytes = 1000
        mock_process.io_counters.return_value.write_bytes = 2000
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        read_bytes, write_bytes = monitor.get_io_stats()
        assert read_bytes == 1000
        assert write_bytes == 2000

    @pytest.mark.asyncio
    @patch("src.utils.performance.psutil.Process")
    async def test_measure_step_context_manager(self, mock_process_class):
        """Test step measurement context manager."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100 MB
        mock_process.cpu_percent.return_value = 30.0
        mock_process.io_counters.return_value.read_bytes = 1000
        mock_process.io_counters.return_value.write_bytes = 2000
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        async with monitor.measure_step("test_step", test_metadata="value"):
            await asyncio.sleep(0.01)  # Simulate work

        assert len(monitor.metrics) == 1
        metric = monitor.metrics[0]
        assert metric.step_name == "test_step"
        assert metric.duration > 0
        assert metric.metadata == {"test_metadata": "value"}

    @pytest.mark.asyncio
    @patch("src.utils.performance.psutil.Process")
    async def test_measure_step_with_exception(self, mock_process_class):
        """Test step measurement with exception handling."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 100
        mock_process.cpu_percent.return_value = 30.0
        mock_process.io_counters.return_value.read_bytes = 1000
        mock_process.io_counters.return_value.write_bytes = 2000
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        with pytest.raises(ValueError):
            async with monitor.measure_step("test_step"):
                raise ValueError("Test error")

        assert len(monitor.metrics) == 1
        metric = monitor.metrics[0]
        assert len(metric.errors) == 1
        assert "Test error" in metric.errors[0]

    def test_get_pipeline_summary_empty(self):
        """Test pipeline summary with no metrics."""
        monitor = PerformanceMonitor()
        summary = monitor.get_pipeline_summary()
        assert summary == {}

    @patch("src.utils.performance.psutil.Process")
    def test_get_pipeline_summary_with_metrics(self, mock_process_class):
        """Test pipeline summary with metrics."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        monitor.start_pipeline()

        # Add mock metrics
        metric1 = PerformanceMetrics(
            step_name="step1",
            start_time=1000,
            end_time=1002,
            duration=2.0,
            memory_start=100,
            memory_peak=120,
            memory_end=110,
            cpu_percent=25.0,
        )
        metric2 = PerformanceMetrics(
            step_name="step2",
            start_time=1002,
            end_time=1005,
            duration=3.0,
            memory_start=110,
            memory_peak=130,
            memory_end=115,
            cpu_percent=35.0,
        )
        monitor.metrics = [metric1, metric2]

        summary = monitor.get_pipeline_summary()

        assert "total_duration" in summary
        assert (
            summary["total_memory_delta_mb"] == 15.0
        )  # (110-100) + (115-110) = 10 + 5
        assert summary["average_cpu_percent"] == 30.0  # (25 + 35) / 2
        assert summary["steps_completed"] == 2
        assert summary["longest_step"]["name"] == "step2"

    def test_save_metrics(self):
        """Test saving metrics to file."""
        monitor = PerformanceMonitor()
        monitor.start_pipeline()

        # Add mock metric
        metric = PerformanceMetrics(
            step_name="test_step",
            start_time=1000,
            end_time=1002,
            duration=2.0,
            memory_start=100,
            memory_peak=120,
            memory_end=110,
            cpu_percent=25.0,
        )
        monitor.metrics = [metric]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            monitor.save_metrics(output_path)

            assert output_path.exists()

            with output_path.open() as f:
                data = json.load(f)

            assert "pipeline_summary" in data
            assert "step_metrics" in data
            assert len(data["step_metrics"]) == 1

    @patch("src.utils.performance.psutil.Process")
    def test_check_thresholds_no_warnings(self, mock_process_class):
        """Test check_thresholds when everything is within limits."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        monitor.metrics = [
            PerformanceMetrics(
                step_name="fast_step",
                start_time=0,
                end_time=1,
                duration=1.0,
                memory_start=100,
                memory_peak=200,
                memory_end=150,
                cpu_percent=50,
            )
        ]

        warnings = monitor.check_thresholds(
            timing_threshold_sec=5.0, memory_warning_mb=1000
        )
        assert warnings == []

    @patch("src.utils.performance.psutil.Process")
    def test_check_thresholds_timing_exceeded(self, mock_process_class):
        """Test check_thresholds warns on slow steps."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        monitor.metrics = [
            PerformanceMetrics(
                step_name="slow_step",
                start_time=0,
                end_time=10,
                duration=10.0,
                memory_start=100,
                memory_peak=200,
                memory_end=150,
                cpu_percent=50,
            )
        ]

        warnings = monitor.check_thresholds(
            timing_threshold_sec=5.0, memory_warning_mb=1000
        )
        assert len(warnings) == 1
        assert "slow_step" in warnings[0]
        assert "10.0s" in warnings[0]

    @patch("src.utils.performance.psutil.Process")
    def test_check_thresholds_memory_exceeded(self, mock_process_class):
        """Test check_thresholds warns on high memory."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        monitor.metrics = [
            PerformanceMetrics(
                step_name="hungry_step",
                start_time=0,
                end_time=1,
                duration=1.0,
                memory_start=100,
                memory_peak=1500,
                memory_end=150,
                cpu_percent=50,
            )
        ]

        warnings = monitor.check_thresholds(
            timing_threshold_sec=5.0, memory_warning_mb=1000
        )
        assert len(warnings) == 1
        assert "hungry_step" in warnings[0]
        assert "1500" in warnings[0]

    @patch("src.utils.performance.psutil.Process")
    def test_finish_pipeline_saves_to_history(self, mock_process_class):
        """Test finish_pipeline writes to history manager."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            monitor = PerformanceMonitor(history_manager=hm)
            monitor.start_pipeline(
                run_id="run-1", product_id="P1", profile_name="profile1"
            )
            monitor.metrics = [
                PerformanceMetrics(
                    step_name="s1",
                    start_time=1000,
                    end_time=1002,
                    duration=2.0,
                    memory_start=100,
                    memory_peak=120,
                    memory_end=110,
                    cpu_percent=25,
                )
            ]

            monitor.finish_pipeline(success=True)

            runs = hm.get_run_history()
            assert len(runs) == 1
            assert runs[0].run_id == "run-1"
            assert runs[0].success is True

    @patch("src.utils.performance.psutil.Process")
    def test_finish_pipeline_no_history_manager(self, mock_process_class):
        """Test finish_pipeline is a no-op without history manager."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()  # no history_manager
        monitor.start_pipeline(run_id="run-1", product_id="P1", profile_name="profile1")
        # Should not raise
        monitor.finish_pipeline(success=True)


class TestPerformanceHistoryManager:
    """Test performance history storage and retrieval."""

    def _make_run(
        self,
        run_id: str = "r1",
        product_id: str = "P1",
        profile_name: str = "prof1",
        duration: float = 10.0,
        success: bool = True,
        timestamp: str = "2025-01-15T10:00:00+00:00",
    ) -> PipelineRunMetrics:
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
            step_metrics=[
                {
                    "step_name": "gather_visuals",
                    "start_time": 1000,
                    "end_time": 1005,
                    "duration": duration / 2,
                    "memory_start": 100,
                    "memory_peak": 150,
                    "memory_end": 120,
                    "cpu_percent": 30,
                    "io_read_bytes": 0,
                    "io_write_bytes": 0,
                    "errors": [],
                    "metadata": {},
                }
            ],
            success=success,
        )

    def test_save_and_load_round_trip(self):
        """Test saving and loading metrics preserves data."""
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            run = self._make_run(run_id="round-trip")
            hm.save_run_metrics(run)

            loaded = hm.get_run_history()
            assert len(loaded) == 1
            assert loaded[0].run_id == "round-trip"
            assert loaded[0].total_duration == 10.0
            assert loaded[0].success is True

    def test_cleanup_enforces_max_runs(self):
        """Test that cleanup keeps only max_runs entries."""
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp), max_runs=3)

            # Save 5 runs (cleanup runs every 10 saves, so force it)
            for i in range(5):
                run = self._make_run(
                    run_id=f"r{i}",
                    timestamp=f"2025-01-{15 + i:02d}T10:00:00+00:00",
                )
                hm.save_run_metrics(run)

            hm.force_cleanup()

            loaded = hm.get_run_history()
            assert len(loaded) == 3
            # Should keep the 3 newest
            run_ids = {r.run_id for r in loaded}
            assert "r4" in run_ids
            assert "r3" in run_ids
            assert "r2" in run_ids

    def test_product_filtering(self):
        """Test get_metrics_for_product returns only matching runs."""
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            hm.save_run_metrics(self._make_run(run_id="r1", product_id="A"))
            hm.save_run_metrics(self._make_run(run_id="r2", product_id="B"))
            hm.save_run_metrics(self._make_run(run_id="r3", product_id="A"))

            a_runs = hm.get_metrics_for_product("A")
            assert len(a_runs) == 2
            assert all(r.product_id == "A" for r in a_runs)

            b_runs = hm.get_metrics_for_product("B")
            assert len(b_runs) == 1

    def test_empty_history(self):
        """Test loading from nonexistent history file returns empty."""
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            assert hm.get_run_history() == []
            assert hm.get_metrics_for_product("X") == []

    def test_corrupt_jsonl_handling(self):
        """Test that corrupt lines are skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))

            # Write one good and one bad line
            good_run = self._make_run(run_id="good")
            hm.save_run_metrics(good_run)

            # Append a corrupt line
            with open(hm.history_file, "a") as f:
                f.write("{this is not valid json}\n")
                f.write('{"run_id": "bad", "missing_fields": true}\n')

            loaded = hm.get_run_history()
            assert len(loaded) == 1
            assert loaded[0].run_id == "good"

    def test_limit_on_get_run_history(self):
        """Test limit parameter on get_run_history."""
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            for i in range(5):
                hm.save_run_metrics(
                    self._make_run(
                        run_id=f"r{i}",
                        timestamp=f"2025-01-{15 + i:02d}T10:00:00+00:00",
                    )
                )

            loaded = hm.get_run_history(limit=2)
            assert len(loaded) == 2


class TestPipelineRunMetrics:
    """Test PipelineRunMetrics creation and factory methods."""

    def test_from_pipeline_summary_with_metrics(self):
        """Test creating run metrics from pipeline summary."""
        metrics = [
            PerformanceMetrics(
                step_name="step1",
                start_time=1000,
                end_time=1005,
                duration=5.0,
                memory_start=100,
                memory_peak=200,
                memory_end=150,
                cpu_percent=40,
            ),
            PerformanceMetrics(
                step_name="step2",
                start_time=1005,
                end_time=1008,
                duration=3.0,
                memory_start=150,
                memory_peak=300,
                memory_end=180,
                cpu_percent=60,
            ),
        ]

        run = PipelineRunMetrics.from_pipeline_summary(
            run_id="test-run",
            product_id="PROD1",
            profile_name="slideshow",
            start_time=1000.0,
            end_time=1008.0,
            metrics=metrics,
        )

        assert run.run_id == "test-run"
        assert run.total_duration == 8.0
        assert run.peak_memory == 300  # max of 200 and 300
        assert run.total_memory_delta == 80  # 180 - 100
        assert run.total_cpu_percent == 50.0  # (40 + 60) / 2
        assert run.success is True
        assert len(run.step_metrics) == 2

    def test_from_pipeline_summary_empty_metrics(self):
        """Test creating run metrics with no step metrics."""
        run = PipelineRunMetrics.from_pipeline_summary(
            run_id="empty",
            product_id="P1",
            profile_name="prof",
            start_time=1000.0,
            end_time=1005.0,
            metrics=[],
        )

        assert run.total_duration == 5.0
        assert run.peak_memory == 0
        assert run.total_memory_delta == 0
        assert run.total_cpu_percent == 0
        assert run.step_metrics == []

    def test_from_pipeline_summary_with_error(self):
        """Test creating run metrics with error message."""
        run = PipelineRunMetrics.from_pipeline_summary(
            run_id="fail",
            product_id="P1",
            profile_name="prof",
            start_time=1000.0,
            end_time=1002.0,
            metrics=[],
            success=False,
            error_message="Something broke",
        )

        assert run.success is False
        assert run.error_message == "Something broke"


class TestTimingDecorators:
    """Test timing decorator functionality."""

    @pytest.mark.asyncio
    async def test_async_timer_decorator(self):
        """Test async timing decorator."""

        @async_timer
        async def test_async_function():
            await asyncio.sleep(0.01)
            return "result"

        result = await test_async_function()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_async_timer_with_exception(self):
        """Test async timer with exception."""

        @async_timer
        async def test_async_function():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await test_async_function()

    def test_timer_decorator(self):
        """Test synchronous timing decorator."""

        @timer
        def test_function():
            import time

            time.sleep(0.01)
            return "result"

        result = test_function()
        assert result == "result"

    def test_timer_with_exception(self):
        """Test timer with exception."""

        @timer
        def test_function():
            import time

            time.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_function()


class TestGlobalMonitor:
    """Test global performance monitor instance."""

    def test_global_monitor_exists(self):
        """Test that global monitor instance exists."""
        assert performance_monitor is not None
        assert isinstance(performance_monitor, PerformanceMonitor)
