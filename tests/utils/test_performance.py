"""Tests for performance monitoring utilities.

The history used to say less than it claimed: peak memory was the Python
process alone, the trim never ran, a `--step` debug run looked like a
render, and a skip counted as a failure. Each of those is pinned here
against the real shapes the file holds.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.utils.performance import (
    RUN_KIND_RENDER,
    RUN_KIND_STEP,
    PerformanceHistoryManager,
    PerformanceMetrics,
    PerformanceMonitor,
    PipelineRunMetrics,
    performance_monitor,
)


def _metric(
    name: str = "step",
    duration: float = 1.0,
    memory_start: float = 100.0,
    memory_peak: float = 120.0,
    memory_end: float = 110.0,
    cpu_percent: float = 25.0,
    errors: list[str] | None = None,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        step_name=name,
        start_time=1000.0,
        end_time=1000.0 + duration,
        duration=duration,
        memory_start=memory_start,
        memory_peak=memory_peak,
        memory_end=memory_end,
        cpu_percent=cpu_percent,
        errors=errors or [],
    )


def _process(rss_mb: float = 100.0, children_mb: tuple[float, ...] = ()) -> Mock:
    """A psutil.Process stand-in with a process tree and CPU times."""
    proc = Mock()
    proc.memory_info.return_value.rss = int(rss_mb * 1024 * 1024)
    kids = []
    for mb in children_mb:
        kid = Mock()
        kid.memory_info.return_value.rss = int(mb * 1024 * 1024)
        kids.append(kid)
    proc.children.return_value = kids
    proc.cpu_times.return_value = SimpleNamespace(
        user=1.0, system=0.5, children_user=2.0, children_system=0.5
    )
    proc.io_counters.return_value.read_bytes = 1000
    proc.io_counters.return_value.write_bytes = 2000
    return proc


class TestPerformanceMetrics:
    def test_metrics_creation(self):
        metrics = _metric(duration=2.5, memory_start=100.0, memory_end=120.0)
        assert metrics.duration_ms == 2500.0
        assert metrics.memory_delta == 20.0

    def test_metrics_with_defaults(self):
        metrics = _metric()
        assert metrics.io_read_bytes == 0
        assert metrics.io_write_bytes == 0
        assert metrics.errors == []
        assert metrics.metadata == {}


class TestMemoryIsTheWholeProcessTree:
    """ffmpeg, Chromium and the STT subprocess are where a render's memory
    goes; the Python process alone was a fraction of what the OOM entries
    in the notes describe.
    """

    @patch("src.utils.performance.psutil.Process")
    def test_children_are_counted(self, mock_process_class):
        mock_process_class.return_value = _process(100.0, (1500.0, 400.0))
        monitor = PerformanceMonitor()
        assert monitor.get_memory_usage() == 2000.0

    @patch("src.utils.performance.psutil.Process")
    def test_a_child_that_exits_mid_read_is_skipped(self, mock_process_class):
        import psutil

        proc = _process(100.0, (300.0,))
        gone = Mock()
        gone.memory_info.side_effect = psutil.NoSuchProcess(pid=1)
        proc.children.return_value.append(gone)
        mock_process_class.return_value = proc
        assert PerformanceMonitor().get_memory_usage() == 400.0

    @patch("src.utils.performance.psutil.Process")
    def test_get_io_stats(self, mock_process_class):
        mock_process_class.return_value = _process()
        assert PerformanceMonitor().get_io_stats() == (1000, 2000)


class TestMeasureStep:
    @pytest.mark.asyncio
    @patch("src.utils.performance.psutil.Process")
    async def test_records_the_step_with_metadata(self, mock_process_class):
        mock_process_class.return_value = _process()
        monitor = PerformanceMonitor()

        async with monitor.measure_step("test_step", test_metadata="value"):
            await asyncio.sleep(0.01)

        (metric,) = monitor.metrics
        assert metric.step_name == "test_step"
        assert metric.duration > 0
        assert metric.metadata == {"test_metadata": "value"}

    @pytest.mark.asyncio
    @patch("src.utils.performance.psutil.Process")
    async def test_cpu_is_tree_cpu_time_over_wall_time(self, mock_process_class):
        """4 CPU-seconds of process tree over a 2s step is 200%, whatever a
        sample of "CPU since the last sample" would have said.
        """
        proc = _process()
        proc.cpu_times.side_effect = [
            SimpleNamespace(
                user=1.0, system=0.0, children_user=0.0, children_system=0.0
            ),
            SimpleNamespace(
                user=2.0, system=1.0, children_user=2.0, children_system=0.0
            ),
        ]
        mock_process_class.return_value = proc
        monitor = PerformanceMonitor()
        with patch("src.utils.performance.time.time", side_effect=[1000.0, 1002.0]):
            async with monitor.measure_step("busy"):
                pass
        assert monitor.metrics[0].cpu_percent == pytest.approx(200.0)

    @pytest.mark.asyncio
    @patch("src.utils.performance.psutil.Process")
    async def test_peak_is_sampled_while_the_loop_is_blocked(self, mock_process_class):
        """The sampler is a thread: a step that never yields still gets its
        peak read, which an asyncio sampler could not do.
        """
        proc = _process(100.0)
        readings = iter([100.0, 900.0, 900.0, 900.0, 900.0, 100.0])

        def rss():
            mb = next(readings, 100.0)
            return int(mb * 1024 * 1024)

        proc.memory_info.side_effect = lambda: SimpleNamespace(rss=rss())
        mock_process_class.return_value = proc
        monitor = PerformanceMonitor(memory_monitor_interval=0.01)

        async with monitor.measure_step("blocking"):
            import time

            time.sleep(0.15)  # blocks the event loop on purpose

        assert monitor.metrics[0].memory_peak == 900.0

    @pytest.mark.asyncio
    @patch("src.utils.performance.psutil.Process")
    async def test_an_exception_is_recorded_and_re_raised(self, mock_process_class):
        mock_process_class.return_value = _process()
        monitor = PerformanceMonitor()

        with pytest.raises(ValueError):
            async with monitor.measure_step("test_step"):
                raise ValueError("Test error")

        (metric,) = monitor.metrics
        assert metric.errors == ["Test error"]


class TestPerformanceMonitor:
    def test_monitor_initialization(self):
        monitor = PerformanceMonitor()
        assert monitor.metrics == []
        assert monitor.current_step is None
        assert monitor.pipeline_start is None
        assert monitor.current_kind == RUN_KIND_RENDER

    def test_start_pipeline_records_the_kind(self):
        monitor = PerformanceMonitor()
        monitor.start_pipeline(kind=RUN_KIND_STEP)
        assert monitor.pipeline_start is not None
        assert monitor.current_kind == RUN_KIND_STEP

    @patch("src.utils.performance.psutil.Process")
    def test_reset_clears_state(self, mock_process_class):
        mock_process_class.return_value = _process()
        monitor = PerformanceMonitor()
        monitor.start_pipeline(
            run_id="r", product_id="P1", profile_name="p", kind=RUN_KIND_STEP
        )
        monitor.metrics.append(_metric())

        monitor.reset()

        assert monitor.metrics == []
        assert monitor.pipeline_start is None
        assert monitor.current_run_id is None
        assert monitor.current_kind == RUN_KIND_RENDER

    def test_reset_sets_history_manager(self):
        monitor = PerformanceMonitor()
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            monitor.reset(history_manager=hm)
            assert monitor.history_manager is hm

    def test_get_pipeline_summary_empty(self):
        assert PerformanceMonitor().get_pipeline_summary() == {}

    def test_summary_memory_delta_is_net_not_a_sum(self):
        """The file's row and the log line used to disagree: the row was
        net first-to-last, the summary a sum of per-step deltas.
        """
        monitor = PerformanceMonitor()
        monitor.start_pipeline()
        monitor.metrics = [
            _metric("step1", 2.0, memory_start=100, memory_peak=120, memory_end=110),
            _metric("step2", 3.0, memory_start=110, memory_peak=130, memory_end=115),
        ]
        summary = monitor.get_pipeline_summary()
        assert summary["total_memory_delta_mb"] == 15.0
        assert summary["peak_memory_mb"] == 130
        assert summary["average_cpu_percent"] == 25.0
        assert summary["longest_step"]["name"] == "step2"
        assert summary["steps_completed"] == 2

    def test_save_metrics(self):
        monitor = PerformanceMonitor()
        monitor.start_pipeline()
        monitor.metrics = [_metric()]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            monitor.save_metrics(output_path)
            data = json.loads(output_path.read_text())
        assert "pipeline_summary" in data
        assert len(data["step_metrics"]) == 1

    @pytest.mark.parametrize(
        ("metric", "fragment"),
        [
            (_metric("slow_step", duration=10.0), "10.0s"),
            (_metric("hungry_step", memory_peak=1500), "1500"),
        ],
        ids=["timing", "memory"],
    )
    def test_check_thresholds_warns(self, metric, fragment):
        monitor = PerformanceMonitor()
        monitor.metrics = [metric]
        warnings = monitor.check_thresholds(
            timing_threshold_sec=5.0, memory_warning_mb=1000
        )
        assert len(warnings) == 1
        assert metric.step_name in warnings[0]
        assert fragment in warnings[0]

    def test_check_thresholds_quiet_within_limits(self):
        monitor = PerformanceMonitor()
        monitor.metrics = [_metric("fast", duration=1.0, memory_peak=200)]
        assert (
            monitor.check_thresholds(timing_threshold_sec=5.0, memory_warning_mb=1000)
            == []
        )


class TestFinishPipeline:
    def _monitor(
        self, tmp: str
    ) -> tuple[PerformanceMonitor, PerformanceHistoryManager]:
        hm = PerformanceHistoryManager(history_dir=Path(tmp))
        monitor = PerformanceMonitor(history_manager=hm)
        monitor.start_pipeline(run_id="run-1", product_id="P1", profile_name="prof")
        monitor.metrics = [_metric("s1")]
        return monitor, hm

    def test_saves_a_render_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor, hm = self._monitor(tmp)
            monitor.finish_pipeline(success=True)
            (run,) = hm.get_run_history()
        assert run.run_id == "run-1"
        assert run.success is True
        assert run.kind == RUN_KIND_RENDER
        assert run.skipped is False
        assert run.failed_step is None

    def test_a_skip_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor, hm = self._monitor(tmp)
            monitor.finish_pipeline(
                success=False, error_message="not enough media", skipped=True
            )
            (run,) = hm.get_run_history()
        assert run.success is False
        assert run.skipped is True

    def test_a_failure_names_its_step(self):
        """25 rows in the real file say only "Parallel pipeline execution
        failed"; the step that raised was in the step metrics all along.
        """
        with tempfile.TemporaryDirectory() as tmp:
            monitor, hm = self._monitor(tmp)
            monitor.metrics = [
                _metric("generate_script"),
                _metric("create_voiceover", errors=["TTS quota"]),
            ]
            monitor.finish_pipeline(
                success=False, error_message="Parallel pipeline execution failed"
            )
            (run,) = hm.get_run_history()
        assert run.failed_step == "create_voiceover"

    def test_a_step_run_is_marked_as_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            monitor = PerformanceMonitor(history_manager=hm)
            monitor.start_pipeline(
                run_id="dbg", product_id="P1", profile_name="prof", kind=RUN_KIND_STEP
            )
            monitor.metrics = [_metric("gather_visuals")]
            monitor.finish_pipeline(success=True)
            assert hm.get_run_history() == []
            (run,) = hm.get_run_history(kind=None)
        assert run.kind == RUN_KIND_STEP

    def test_no_history_manager_is_a_no_op(self):
        monitor = PerformanceMonitor()
        monitor.start_pipeline(run_id="r", product_id="P1", profile_name="p")
        monitor.finish_pipeline(success=True)


def _run(
    run_id: str = "r1",
    product_id: str = "P1",
    profile_name: str = "prof1",
    duration: float = 10.0,
    success: bool = True,
    timestamp: str = "2025-01-15T10:00:00+00:00",
    kind: str = RUN_KIND_RENDER,
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
        kind=kind,
    )


class TestPerformanceHistoryManager:
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            hm.save_run_metrics(_run(run_id="round-trip"))
            (loaded,) = hm.get_run_history()
        assert loaded.run_id == "round-trip"
        assert loaded.total_duration == 10.0

    def test_the_file_never_exceeds_max_runs(self):
        """Trimmed on every save. The orchestrator builds a new manager per
        render, so a per-instance every-Nth counter never fired and the
        real file sat at 310 rows against a cap of 100.
        """
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                hm = PerformanceHistoryManager(history_dir=Path(tmp), max_runs=3)
                hm.save_run_metrics(
                    _run(
                        run_id=f"r{i}", timestamp=f"2025-01-{15 + i:02d}T10:00:00+00:00"
                    )
                )
            lines = hm.history_file.read_text().splitlines()
            loaded = hm.get_run_history()
        assert len(lines) == 3
        assert {r.run_id for r in loaded} == {"r2", "r3", "r4"}

    def test_reading_does_not_create_the_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "never"
            hm = PerformanceHistoryManager(history_dir=target)
            assert hm.get_run_history() == []
            assert not target.exists()

    def test_renders_only_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            hm.save_run_metrics(_run(run_id="full"))
            hm.save_run_metrics(_run(run_id="dbg", kind=RUN_KIND_STEP))
            assert [r.run_id for r in hm.get_run_history()] == ["full"]
            assert len(hm.get_run_history(kind=None)) == 2

    def test_a_legacy_row_is_classified_by_its_steps(self):
        """Rows written before `kind` existed: reaching assembly means a
        render; a lone gather_visuals means a --step run.
        """
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            render = _run(run_id="old-render")
            render.step_metrics.append(
                dict(render.step_metrics[0], step_name="assemble_video")
            )
            probe = _run(run_id="old-probe")
            hm.history_dir.mkdir(exist_ok=True)
            with open(hm.history_file, "w") as f:
                for row in (render, probe):
                    data = {k: v for k, v in row.__dict__.items() if k != "kind"}
                    data.pop("skipped")
                    data.pop("failed_step")
                    f.write(json.dumps(data) + "\n")
            by_id = {r.run_id: r for r in hm.get_run_history(kind=None)}
        assert by_id["old-render"].kind == RUN_KIND_RENDER
        assert by_id["old-probe"].kind == RUN_KIND_STEP
        assert by_id["old-render"].skipped is False

    def test_unknown_keys_do_not_lose_the_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            hm.history_dir.mkdir(exist_ok=True)
            data = _run(run_id="future").__dict__ | {"some_new_field": 1}
            hm.history_file.write_text(json.dumps(data) + "\n")
            (loaded,) = hm.get_run_history()
        assert loaded.run_id == "future"

    def test_product_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            hm.save_run_metrics(_run(run_id="r1", product_id="A"))
            hm.save_run_metrics(_run(run_id="r2", product_id="B"))
            hm.save_run_metrics(_run(run_id="r3", product_id="A"))
            assert len(hm.get_metrics_for_product("A")) == 2
            assert len(hm.get_metrics_for_product("B")) == 1

    def test_empty_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            assert hm.get_run_history() == []
            assert hm.get_metrics_for_product("X") == []

    def test_corrupt_jsonl_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            hm.save_run_metrics(_run(run_id="good"))
            with open(hm.history_file, "a") as f:
                f.write("{this is not valid json}\n")
                f.write('{"run_id": "bad", "missing_fields": true}\n')
            (loaded,) = hm.get_run_history()
        assert loaded.run_id == "good"

    def test_limit_on_get_run_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            hm = PerformanceHistoryManager(history_dir=Path(tmp))
            for i in range(5):
                hm.save_run_metrics(
                    _run(
                        run_id=f"r{i}", timestamp=f"2025-01-{15 + i:02d}T10:00:00+00:00"
                    )
                )
            assert len(hm.get_run_history(limit=2)) == 2


class TestPipelineRunMetrics:
    def test_from_pipeline_summary_with_metrics(self):
        metrics = [
            _metric(
                "step1",
                5.0,
                memory_start=100,
                memory_peak=200,
                memory_end=150,
                cpu_percent=40,
            ),
            _metric(
                "step2",
                3.0,
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
        assert run.total_duration == 8.0
        assert run.peak_memory == 300
        assert run.total_memory_delta == 80  # 180 - 100
        assert run.total_cpu_percent == 50.0
        assert run.success is True
        assert run.kind == RUN_KIND_RENDER
        assert len(run.step_metrics) == 2

    def test_from_pipeline_summary_empty_metrics(self):
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
        run = PipelineRunMetrics.from_pipeline_summary(
            run_id="fail",
            product_id="P1",
            profile_name="prof",
            start_time=1000.0,
            end_time=1002.0,
            metrics=[_metric("assemble_video", errors=["ffmpeg exit 1"])],
            success=False,
            error_message="Something broke",
        )
        assert run.success is False
        assert run.error_message == "Something broke"
        assert run.failed_step == "assemble_video"


class TestGlobalMonitor:
    def test_global_monitor_exists(self):
        assert isinstance(performance_monitor, PerformanceMonitor)
