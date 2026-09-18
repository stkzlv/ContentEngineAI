"""Performance measurement and monitoring utilities.

Measures each render step (wall time, memory, CPU, I/O) and keeps one JSONL
row per pipeline run under `outputs/performance_history/`, which
`tools/performance_report.py` reads.

What the numbers mean, because two of them used to mean less than they said:

- Memory is the RSS of the whole process tree: this process plus every
  child (ffmpeg, the subtitle renderer's Chromium, the STT subprocess). The
  children are where a render's memory goes, and the notes' OOM entries are
  about them; the Python process alone is a fraction of it. The peak is
  sampled from a thread, so it is read even while a step blocks the event
  loop.
- `cpu_percent` on a step is the process tree's CPU time during the step
  over the step's wall time, times 100. It is well-defined under parallel
  steps, where a sample of "CPU since the last sample" was not.
- A run's `kind` says what kind of invocation wrote it: `render` for a full
  pipeline, `step` for a `--step` debug run. The report tool reads renders
  only, since a two-second single-step run in the same file used to drag
  every average down.
"""

import dataclasses
import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)

RUN_KIND_RENDER = "render"
RUN_KIND_STEP = "step"

# The step every full render ends on. A legacy row (no `kind`) that recorded
# it was a render; one that did not was a single-step debug run.
_RENDER_MARKER_STEP = "assemble_video"


@dataclass
class PerformanceMetrics:
    """Measurements for one pipeline step."""

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
    """Complete metrics for a single pipeline run: one history row."""

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
    kind: str = RUN_KIND_RENDER
    skipped: bool = False
    failed_step: str | None = None

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
        kind: str = RUN_KIND_RENDER,
        skipped: bool = False,
    ) -> "PipelineRunMetrics":
        """Build a row from a run's step metrics.

        `total_memory_delta` is the net change from the first step's start
        to the last step's end, the one definition `get_pipeline_summary`
        also uses. `failed_step` is the first step that recorded an error.
        """
        start_timestamp = datetime.fromtimestamp(start_time, tz=UTC).isoformat()
        end_timestamp = datetime.fromtimestamp(end_time, tz=UTC).isoformat()

        peak_memory = max((m.memory_peak for m in metrics), default=0.0)
        avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics) if metrics else 0
        failed_step = next((m.step_name for m in metrics if m.errors), None)

        return cls(
            run_id=run_id,
            product_id=product_id,
            profile_name=profile_name,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            total_duration=end_time - start_time,
            total_memory_delta=_net_memory_delta(metrics),
            peak_memory=peak_memory,
            total_cpu_percent=avg_cpu,
            step_metrics=[asdict(m) for m in metrics],
            success=success,
            error_message=error_message,
            kind=kind,
            skipped=skipped,
            failed_step=failed_step,
        )

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "PipelineRunMetrics":
        """Load a history row, tolerating fields added or dropped since.

        Unknown keys are ignored rather than failing the row, and a row
        written before `kind` existed is classified by what it recorded: a
        run that reached the assembly step was a render, anything else a
        single-step debug run.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in row.items() if k in known}
        if "kind" not in data:
            steps = {s.get("step_name") for s in data.get("step_metrics") or []}
            data["kind"] = (
                RUN_KIND_RENDER if _RENDER_MARKER_STEP in steps else RUN_KIND_STEP
            )
        return cls(**data)


def _net_memory_delta(metrics: list[PerformanceMetrics]) -> float:
    """Memory change across a run: last step's end minus first step's start."""
    if not metrics:
        return 0.0
    return metrics[-1].memory_end - metrics[0].memory_start


class PerformanceHistoryManager:
    """One JSONL file of run rows, capped at `max_runs`, newest kept."""

    def __init__(self, history_dir: Path, max_runs: int = 100):
        self.history_dir = Path(history_dir)
        self.max_runs = max_runs
        self.history_file = self.history_dir / "performance_history.jsonl"

    def save_run_metrics(self, run_metrics: PipelineRunMetrics) -> None:
        """Append a run and trim the file to `max_runs`.

        Trimmed on every save, not every Nth: the orchestrator builds a new
        manager per render, so a per-instance counter never reached N and
        the file grew without bound.
        """
        try:
            self.history_dir.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "a") as f:
                f.write(json.dumps(asdict(run_metrics)) + "\n")
            self._trim()
            logger.debug("Saved run metrics for %s", run_metrics.run_id)
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save run metrics: %s", e)

    def _rows(self) -> Iterator[dict[str, Any]]:
        with open(self.history_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping corrupt history line: %s", e)

    def _trim(self) -> None:
        """Keep only the newest `max_runs` rows, by start timestamp."""
        if not self.history_file.exists():
            return
        try:
            rows = list(self._rows())
            if len(rows) <= self.max_runs:
                return
            rows.sort(key=lambda x: x.get("start_timestamp", ""), reverse=True)
            rows = rows[: self.max_runs]
            with open(self.history_file, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            logger.debug("Trimmed history to the %d most recent runs", len(rows))
        except OSError as e:
            logger.error("Failed to trim run history: %s", e)

    def get_run_history(
        self, limit: int | None = None, kind: str | None = RUN_KIND_RENDER
    ) -> list[PipelineRunMetrics]:
        """Runs newest first; `kind=None` returns every kind."""
        if not self.history_file.exists():
            return []

        runs = []
        try:
            for row in self._rows():
                try:
                    runs.append(PipelineRunMetrics.from_row(row))
                except TypeError as e:
                    logger.warning("Skipping corrupt history line: %s", e)
        except OSError as e:
            logger.error("Failed to load run history: %s", e)
            return []

        if kind is not None:
            runs = [r for r in runs if r.kind == kind]
        runs.sort(key=lambda x: x.start_timestamp, reverse=True)
        if limit:
            runs = runs[:limit]
        return runs

    def get_metrics_for_product(
        self, product_id: str, limit: int = 10
    ) -> list[PipelineRunMetrics]:
        """Renders of one product, newest first."""
        all_runs = self.get_run_history()
        product_runs = [run for run in all_runs if run.product_id == product_id]
        return product_runs[:limit]


class _PeakSampler:
    """Samples a callable on a thread and keeps the maximum it saw.

    A thread rather than an asyncio task so the peak is still read while a
    step blocks the event loop, which the heavy steps do.
    """

    def __init__(self, read, interval: float, initial: float):
        self._read = read
        self._interval = max(interval, 0.01)
        self.peak = initial
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.peak = max(self.peak, self._read())
            except psutil.Error as e:
                logger.warning("Memory sampling error: %s", e)
                return

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class PerformanceMonitor:
    """Monitors and tracks performance metrics during pipeline execution."""

    def __init__(
        self,
        history_manager: PerformanceHistoryManager | None = None,
        memory_monitor_interval: float = 0.1,
    ):
        self.metrics: list[PerformanceMetrics] = []
        self.current_step: str | None = None
        self.pipeline_start: float | None = None
        self.process = psutil.Process()
        self.history_manager = history_manager
        self.memory_monitor_interval = memory_monitor_interval

        # Pipeline context for history tracking
        self.current_run_id: str | None = None
        self.current_product_id: str | None = None
        self.current_profile_name: str | None = None
        self.current_kind: str = RUN_KIND_RENDER

    def reset(
        self,
        history_manager: PerformanceHistoryManager | None = None,
        memory_monitor_interval: float | None = None,
    ) -> None:
        """Reset monitor state for a new pipeline run.

        Optionally sets a new history manager and/or memory monitor interval.
        Use this instead of directly mutating attributes between batch runs.
        """
        self.metrics.clear()
        self.current_step = None
        self.pipeline_start = None
        self.current_run_id = None
        self.current_product_id = None
        self.current_profile_name = None
        self.current_kind = RUN_KIND_RENDER
        if history_manager is not None:
            self.history_manager = history_manager
        if memory_monitor_interval is not None:
            self.memory_monitor_interval = memory_monitor_interval

    def start_pipeline(
        self,
        run_id: str | None = None,
        product_id: str | None = None,
        profile_name: str | None = None,
        kind: str = RUN_KIND_RENDER,
    ) -> None:
        """Mark the start of pipeline execution."""
        self.pipeline_start = time.time()
        self.metrics.clear()

        # Set context for history tracking
        self.current_run_id = run_id
        self.current_product_id = product_id
        self.current_profile_name = profile_name
        self.current_kind = kind

        logger.debug("Pipeline performance monitoring started for run %s", run_id)

    def get_memory_usage(self) -> float:
        """RSS of this process and every child, in MB."""
        rss = self.process.memory_info().rss
        try:
            for child in self.process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    continue  # exited between listing and reading
        except psutil.Error as e:
            logger.debug("Could not list child processes: %s", e)
        return rss / 1024 / 1024

    def _cpu_seconds(self) -> float:
        """CPU time consumed so far by this process and its children."""
        t = self.process.cpu_times()
        return (
            t.user
            + t.system
            + getattr(t, "children_user", 0.0)
            + getattr(t, "children_system", 0.0)
        )

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
        cpu_start = self._cpu_seconds()

        sampler = _PeakSampler(
            self.get_memory_usage, self.memory_monitor_interval, memory_start
        )
        sampler.start()
        errors: list[str] = []

        try:
            self.current_step = step_name
            logger.debug("Starting performance measurement for step: %s", step_name)
            yield self
        except Exception as e:
            errors.append(str(e))
            raise
        finally:
            sampler.stop()

            end_time = time.time()
            duration = end_time - start_time
            memory_end = self.get_memory_usage()
            memory_peak = max(sampler.peak, memory_start, memory_end)
            cpu_seconds = self._cpu_seconds() - cpu_start
            cpu_percent = (cpu_seconds / duration * 100) if duration > 0 else 0.0
            io_read_end, io_write_end = self.get_io_stats()

            metrics = PerformanceMetrics(
                step_name=step_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
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

            logger.debug(
                "Step '%s' completed in %.2fs "
                "(Memory: %.1f->%.1fMB, Peak: %.1fMB, CPU: %.1f%%)",
                step_name,
                metrics.duration,
                memory_start,
                memory_end,
                memory_peak,
                cpu_percent,
            )

    def get_pipeline_summary(self) -> dict[str, Any]:
        """Get a summary of pipeline performance metrics."""
        if not self.metrics or self.pipeline_start is None:
            return {}

        total_duration = time.time() - self.pipeline_start
        total_io_read = sum(m.io_read_bytes for m in self.metrics)
        total_io_write = sum(m.io_write_bytes for m in self.metrics)
        avg_cpu = sum(m.cpu_percent for m in self.metrics) / len(self.metrics)

        step_durations = {m.step_name: m.duration for m in self.metrics}
        longest_step = max(self.metrics, key=lambda m: m.duration)

        total_errors = sum(len(m.errors) for m in self.metrics)

        return {
            "total_duration": total_duration,
            "total_memory_delta_mb": _net_memory_delta(self.metrics),
            "peak_memory_mb": max(m.memory_peak for m in self.metrics),
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

        logger.debug("Performance metrics saved to %s", output_path)

    def check_thresholds(
        self, timing_threshold_sec: float, memory_warning_mb: int
    ) -> list[str]:
        """Warnings for steps over the timing or memory threshold.

        No defaults here: the thresholds live on `DebugSettings`, and a
        second set of defaults in this method drifted from them.
        """
        warnings: list[str] = []

        for m in self.metrics:
            if m.duration > timing_threshold_sec:
                warnings.append(
                    f"Step '{m.step_name}' took {m.duration:.1f}s"
                    f" (threshold: {timing_threshold_sec:.1f}s)"
                )
            if m.memory_peak > memory_warning_mb:
                warnings.append(
                    f"Step '{m.step_name}' peak memory"
                    f" {m.memory_peak:.0f}MB (threshold: {memory_warning_mb}MB)"
                )

        return warnings

    def finish_pipeline(
        self,
        success: bool = True,
        error_message: str | None = None,
        skipped: bool = False,
    ) -> None:
        """Mark the end of pipeline execution and save to history if configured.

        A skip (insufficient media) is recorded as `skipped`, not as a
        failure: it is an expected outcome for some products, and counting
        it as failed made the success rate say something else.
        """
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
            run_metrics = PipelineRunMetrics.from_pipeline_summary(
                run_id=self.current_run_id,
                product_id=self.current_product_id,
                profile_name=self.current_profile_name,
                start_time=self.pipeline_start,
                end_time=time.time(),
                metrics=self.metrics,
                success=success,
                error_message=error_message,
                kind=self.current_kind,
                skipped=skipped,
            )
            self.history_manager.save_run_metrics(run_metrics)
            logger.debug("Pipeline run %s saved to history", self.current_run_id)
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save pipeline run to history: %s", e)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
