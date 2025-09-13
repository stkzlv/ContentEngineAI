"""Performance measurement and monitoring utilities.

This module provides tools for measuring and tracking performance metrics
during video production pipeline execution. It includes timing utilities,
memory monitoring, and pipeline profiling capabilities.
"""

import asyncio
import functools
import json
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""

    step_name: str
    start_time: float
    end_time: float
    duration: float
    memory_start: float
    memory_peak: float
    memory_end: float
    cpu_percent: float
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    network_sent: int = 0
    network_recv: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def memory_delta(self) -> float:
        """Memory usage delta in MB."""
        return self.memory_end - self.memory_start

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000


@dataclass
class PipelineRunMetrics:
    """Complete metrics for a single pipeline run."""

    run_id: str
    product_id: str
    profile_name: str
    start_timestamp: str
    end_timestamp: str
    total_duration: float
    total_memory_delta: float
    peak_memory: float
    total_cpu_percent: float
    step_metrics: list[dict[str, Any]]  # Serialized PerformanceMetrics
    success: bool
    error_message: str | None = None

    @classmethod
    def from_pipeline_summary(
        cls,
        run_id: str,
        product_id: str,
        profile_name: str,
        start_time: float,
        end_time: float,
        metrics: list[PerformanceMetrics],
        success: bool = True,
        error_message: str | None = None,
    ) -> "PipelineRunMetrics":
        """Create from pipeline summary data."""
        start_timestamp = datetime.fromtimestamp(start_time, tz=UTC).isoformat()
        end_timestamp = datetime.fromtimestamp(end_time, tz=UTC).isoformat()

        # Calculate aggregate metrics
        total_duration = end_time - start_time
        memory_start = metrics[0].memory_start if metrics else 0
        memory_end = metrics[-1].memory_end if metrics else 0
        peak_memory = max(m.memory_peak for m in metrics) if metrics else 0
        avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics) if metrics else 0

        return cls(
            run_id=run_id,
            product_id=product_id,
            profile_name=profile_name,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            total_duration=total_duration,
            total_memory_delta=memory_end - memory_start,
            peak_memory=peak_memory,
            total_cpu_percent=avg_cpu,
            step_metrics=[asdict(m) for m in metrics],
            success=success,
            error_message=error_message,
        )


class PerformanceHistoryManager:
    """Manages historical performance metrics storage and retrieval."""

    def __init__(self, history_dir: Path, max_runs: int = 100):
        self.history_dir = Path(history_dir)
        self.max_runs = max_runs
        self.history_file = self.history_dir / "performance_history.jsonl"

        # Ensure directory exists
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def save_run_metrics(self, run_metrics: PipelineRunMetrics) -> None:
        """Save a pipeline run's metrics to history."""
        try:
            # Append to JSONL file
            with open(self.history_file, "a") as f:
                f.write(json.dumps(asdict(run_metrics)) + "\n")

            # Maintain max_runs limit
            self._cleanup_old_runs()

            logger.debug(f"Saved run metrics for {run_metrics.run_id}")

        except Exception as e:
            logger.error(f"Failed to save run metrics: {e}")

    def _cleanup_old_runs(self) -> None:
        """Remove old runs to maintain max_runs limit."""
        if not self.history_file.exists():
            return

        try:
            # Read all runs
            runs = []
            with open(self.history_file) as f:
                for line in f:
                    if line.strip():
                        runs.append(json.loads(line))

            # Keep only the most recent max_runs
            if len(runs) > self.max_runs:
                # Sort by start_timestamp and keep the newest
                runs.sort(key=lambda x: x["start_timestamp"], reverse=True)
                runs = runs[: self.max_runs]

                # Rewrite the file
                with open(self.history_file, "w") as f:
                    for run in runs:
                        f.write(json.dumps(run) + "\n")

                logger.debug(f"Cleaned up old runs, keeping {len(runs)} most recent")

        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {e}")

    def get_run_history(self, limit: int | None = None) -> list[PipelineRunMetrics]:
        """Get historical run metrics."""
        if not self.history_file.exists():
            return []

        runs = []
        try:
            with open(self.history_file) as f:
                for line in f:
                    if line.strip():
                        run_data = json.loads(line)
                        runs.append(PipelineRunMetrics(**run_data))

            # Sort by timestamp (newest first)
            runs.sort(key=lambda x: x.start_timestamp, reverse=True)

            if limit:
                runs = runs[:limit]

            return runs

        except Exception as e:
            logger.error(f"Failed to load run history: {e}")
            return []

    def get_metrics_for_product(
        self, product_id: str, limit: int = 10
    ) -> list[PipelineRunMetrics]:
        """Get metrics for a specific product."""
        all_runs = self.get_run_history()
        product_runs = [run for run in all_runs if run.product_id == product_id]
        return product_runs[:limit]


class PerformanceMonitor:
    """Monitors and tracks performance metrics during pipeline execution."""

    def __init__(self, history_manager: PerformanceHistoryManager | None = None):
        self.metrics: list[PerformanceMetrics] = []
        self.current_step: str | None = None
        self.pipeline_start: float | None = None
        self.process = psutil.Process()
        self.history_manager = history_manager

        # Pipeline context for history tracking
        self.current_run_id: str | None = None
        self.current_product_id: str | None = None
        self.current_profile_name: str | None = None

    def start_pipeline(
        self,
        run_id: str | None = None,
        product_id: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        """Mark the start of pipeline execution."""
        self.pipeline_start = time.time()
        self.metrics.clear()

        # Set context for history tracking
        self.current_run_id = run_id
        self.current_product_id = product_id
        self.current_profile_name = profile_name

        logger.debug(f"Pipeline performance monitoring started for run {run_id}")

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        memory_info = self.process.memory_info()
        memory_mb: float = memory_info.rss / 1024 / 1024
        return memory_mb

    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        cpu_percent: float = self.process.cpu_percent()
        return cpu_percent

    def get_io_stats(self) -> tuple[int, int]:
        """Get I/O read and write bytes."""
        io_counters = self.process.io_counters()
        return io_counters.read_bytes, io_counters.write_bytes

    @asynccontextmanager
    async def measure_step(self, step_name: str, **metadata):
        """Context manager for measuring pipeline step performance."""
        start_time = time.time()
        memory_start = self.get_memory_usage()
        io_read_start, io_write_start = self.get_io_stats()

        # Start CPU monitoring
        self.process.cpu_percent()  # Initialize CPU measurement

        memory_peak = memory_start
        errors = []

        # Memory monitoring task
        async def monitor_memory():
            nonlocal memory_peak
            while True:
                try:
                    current_memory = self.get_memory_usage()
                    memory_peak = max(memory_peak, current_memory)
                    await asyncio.sleep(0.1)  # Check every 100ms
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Memory monitoring error: {e}")
                    break

        monitor_task = asyncio.create_task(monitor_memory())

        try:
            self.current_step = step_name
            logger.debug(f"Starting performance measurement for step: {step_name}")
            yield self
        except Exception as e:
            errors.append(str(e))
            raise
        finally:
            monitor_task.cancel()
            from contextlib import suppress

            with suppress(asyncio.CancelledError):
                await monitor_task

            end_time = time.time()
            memory_end = self.get_memory_usage()
            cpu_percent = self.get_cpu_percent()
            io_read_end, io_write_end = self.get_io_stats()

            metrics = PerformanceMetrics(
                step_name=step_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_start=memory_start,
                memory_peak=memory_peak,
                memory_end=memory_end,
                cpu_percent=cpu_percent,
                io_read_bytes=io_read_end - io_read_start,
                io_write_bytes=io_write_end - io_write_start,
                errors=errors,
                metadata=metadata,
            )

            self.metrics.append(metrics)
            self.current_step = None

            logger.info(
                f"Step '{step_name}' completed in {metrics.duration:.2f}s "
                f"(Memory: {memory_start:.1f}→{memory_end:.1f}MB, "
                f"Peak: {memory_peak:.1f}MB, CPU: {cpu_percent:.1f}%)"
            )

    def get_pipeline_summary(self) -> dict[str, Any]:
        """Get a summary of pipeline performance metrics."""
        if not self.metrics or self.pipeline_start is None:
            return {}

        total_duration = time.time() - self.pipeline_start
        total_memory_delta = sum(m.memory_delta for m in self.metrics)
        total_io_read = sum(m.io_read_bytes for m in self.metrics)
        total_io_write = sum(m.io_write_bytes for m in self.metrics)
        avg_cpu = sum(m.cpu_percent for m in self.metrics) / len(self.metrics)

        step_durations = {m.step_name: m.duration for m in self.metrics}
        longest_step = max(self.metrics, key=lambda m: m.duration)

        total_errors = sum(len(m.errors) for m in self.metrics)

        return {
            "total_duration": total_duration,
            "total_memory_delta_mb": total_memory_delta,
            "total_io_read_mb": total_io_read / 1024 / 1024,
            "total_io_write_mb": total_io_write / 1024 / 1024,
            "average_cpu_percent": avg_cpu,
            "step_durations": step_durations,
            "longest_step": {
                "name": longest_step.step_name,
                "duration": longest_step.duration,
            },
            "total_errors": total_errors,
            "steps_completed": len(self.metrics),
        }

    def save_metrics(self, output_path: Path) -> None:
        """Save performance metrics to a JSON file."""
        import json

        data = {
            "pipeline_summary": self.get_pipeline_summary(),
            "step_metrics": [
                {
                    "step_name": m.step_name,
                    "duration": m.duration,
                    "duration_ms": m.duration_ms,
                    "memory_start": m.memory_start,
                    "memory_peak": m.memory_peak,
                    "memory_end": m.memory_end,
                    "memory_delta": m.memory_delta,
                    "cpu_percent": m.cpu_percent,
                    "io_read_mb": m.io_read_bytes / 1024 / 1024,
                    "io_write_mb": m.io_write_bytes / 1024 / 1024,
                    "errors": m.errors,
                    "metadata": m.metadata,
                }
                for m in self.metrics
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Performance metrics saved to {output_path}")

    def finish_pipeline(
        self, success: bool = True, error_message: str | None = None
    ) -> None:
        """Mark the end of pipeline execution and save to history if configured."""
        if not self.history_manager or not self.pipeline_start:
            return

        if (
            not self.current_run_id
            or not self.current_product_id
            or not self.current_profile_name
        ):
            logger.warning("Missing pipeline context for history tracking")
            return

        try:
            # Create pipeline run metrics
            run_metrics = PipelineRunMetrics.from_pipeline_summary(
                run_id=self.current_run_id,
                product_id=self.current_product_id,
                profile_name=self.current_profile_name,
                start_time=self.pipeline_start,
                end_time=time.time(),
                metrics=self.metrics,
                success=success,
                error_message=error_message,
            )

            # Save to history
            self.history_manager.save_run_metrics(run_metrics)
            logger.debug(f"Pipeline run {self.current_run_id} saved to history")

        except Exception as e:
            logger.error(f"Failed to save pipeline run to history: {e}")

    def get_history_manager(self) -> PerformanceHistoryManager | None:
        """Get the associated history manager."""
        return self.history_manager


def async_timer(func: Callable) -> Callable:
    """Decorator to time async function execution."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise

    return wrapper


def timer(func: Callable) -> Callable:
    """Decorator to time synchronous function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise

    return wrapper


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
