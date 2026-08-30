"""The TTS warmers looked broken because the metric reporting them was.

The end-of-run summary showed both warmer tasks with a duration equal to the
whole pipeline's wall clock, finishing "at the same instant the pipeline
ended" -- which reads as a warmer that never won its race. Measured on a real
render, the google_cloud warmer completed in 0.54s and `create_voiceover`
started 8ms later. It won comfortably.

`BackgroundTask.duration` was `time.time() - start_time`, recomputed on every
read. The completion log read it immediately and got the truth; the summary
read it minutes later and got elapsed-since-start for a task that had long
finished.

Separately, the coqui warmer ran on every render even though `provider_order`
omits coqui and the dependency was dropped, so nothing could ever use what it
warmed.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.utils.background_processing import BackgroundProcessor, TTSWarmer


class TestDurationIsFrozenAtCompletion:
    @pytest.mark.asyncio
    async def test_a_finished_task_stops_accruing_time(self):
        """The defect: reading later gave a larger number every time."""
        processor = BackgroundProcessor()

        async def quick():
            return "done"

        await processor.start_task(task_id="t", name="Quick", coro_func=quick)
        await asyncio.sleep(0.05)

        first = processor.completed_tasks[-1].duration
        await asyncio.sleep(0.15)
        second = processor.completed_tasks[-1].duration

        assert first == second, (
            "duration grows after the task finished, so any later reader -- "
            "the end-of-run summary included -- reports elapsed pipeline time "
            "instead of how long the task took"
        )

    @pytest.mark.asyncio
    async def test_the_reported_duration_is_the_task_duration(self):
        processor = BackgroundProcessor()

        async def quick():
            await asyncio.sleep(0.02)

        await processor.start_task(task_id="t", name="Quick", coro_func=quick)
        await asyncio.sleep(0.3)

        summary = processor.get_summary()
        reported = summary["recent_completed"][-1]["duration"]

        assert reported < 0.2, (
            f"the summary reports {reported:.2f}s for a 0.02s task; that is "
            "elapsed time since it started, which is what made the warmers "
            "look like they ran the whole pipeline"
        )

    @pytest.mark.asyncio
    async def test_a_running_task_still_reports_elapsed_time(self):
        """Freezing must not make an in-flight task report zero."""
        processor = BackgroundProcessor()
        release = asyncio.Event()

        async def slow():
            await release.wait()

        await processor.start_task(task_id="t", name="Slow", coro_func=slow)
        await asyncio.sleep(0.05)

        running = processor.active_tasks["t"]
        assert running.end_time is None
        assert running.duration > 0

        release.set()
        await asyncio.sleep(0.05)

    def test_end_time_defaults_to_unset(self):
        """A task that never completes must not read as instantaneous."""
        from src.utils.background_processing import BackgroundTask

        task = BackgroundTask(
            task_id="t",
            name="n",
            task=SimpleNamespace(done=lambda: False),
            start_time=time.time() - 5,
        )

        assert task.end_time is None
        assert task.duration >= 5


class TestOnlyReachableProvidersAreWarmed:
    """`enabled` is not the same as "this run can reach it"."""

    def _tts_config(self, provider_order):
        return SimpleNamespace(
            provider_order=provider_order,
            coqui=SimpleNamespace(enabled=True),
            google_cloud=SimpleNamespace(enabled=True),
            voice_profiles={},
        )

    @pytest.mark.asyncio
    async def test_a_provider_outside_the_order_is_not_warmed(self):
        """The shipped case: coqui has a config block, `provider_order` omits
        it, and the dependency was dropped -- so nothing could ever use what
        the warmer loaded.
        """
        processor = AsyncMock()
        processor.start_task = AsyncMock(return_value=object())
        warmer = TTSWarmer(processor)

        await warmer.warm_tts_models(
            SimpleNamespace(tts_config=self._tts_config(["google_cloud"]))
        )

        warmed = {
            call.kwargs["metadata"]["provider"]
            for call in processor.start_task.await_args_list
        }
        assert warmed == {"google_cloud"}, f"warmed {warmed}"

    @pytest.mark.asyncio
    async def test_a_provider_inside_the_order_is_warmed(self):
        """The counterpart, so the test above cannot pass by warming nothing."""
        processor = AsyncMock()
        processor.start_task = AsyncMock(return_value=object())
        warmer = TTSWarmer(processor)

        await warmer.warm_tts_models(
            SimpleNamespace(tts_config=self._tts_config(["coqui", "google_cloud"]))
        )

        warmed = {
            call.kwargs["metadata"]["provider"]
            for call in processor.start_task.await_args_list
        }
        assert warmed == {"coqui", "google_cloud"}

    @pytest.mark.asyncio
    async def test_an_empty_order_warms_nothing(self):
        processor = AsyncMock()
        processor.start_task = AsyncMock(return_value=object())
        warmer = TTSWarmer(processor)

        await warmer.warm_tts_models(SimpleNamespace(tts_config=self._tts_config([])))

        assert processor.start_task.await_count == 0

    @pytest.mark.asyncio
    async def test_the_bundled_config_warms_only_google_cloud(self):
        """Against the real config, which is what a stock render uses."""
        from src.video.config import config as real_config

        processor = AsyncMock()
        processor.start_task = AsyncMock(return_value=object())
        warmer = TTSWarmer(processor)

        await warmer.warm_tts_models(real_config)

        warmed = {
            call.kwargs["metadata"]["provider"]
            for call in processor.start_task.await_args_list
        }
        assert "coqui" not in warmed, (
            "coqui is warmed on a stock render although `provider_order` "
            "omits it and the dependency is not installed"
        )
