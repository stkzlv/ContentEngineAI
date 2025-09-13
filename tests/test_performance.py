"""Tests for performance monitoring utilities."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.utils.performance import (
    PerformanceMetrics,
    PerformanceMonitor,
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

    def test_start_pipeline(self):
        """Test pipeline start tracking."""
        monitor = PerformanceMonitor()
        monitor.start_pipeline()

        assert monitor.pipeline_start is not None
        assert monitor.metrics == []

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

            import json

            with output_path.open() as f:
                data = json.load(f)

            assert "pipeline_summary" in data
            assert "step_metrics" in data
            assert len(data["step_metrics"]) == 1


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
