"""A product with too little media is reported skipped, not failed.

`create_video_for_product` catches `InsufficientMediaError` and returns
"SKIPPED"; the producer CLI and the global batch both count that separately
from a failure. The path could not be reached: `PipelineGraph.execute_step`
wrapped every step in `except Exception` and turned it into a FAILED result,
so an under-mediaed product was reported as a failed render naming a step
that had worked, and a real step failure was indistinguishable from it.

These drive the real graph rather than asserting where the code sits: a
handler moved above the generic one still passes a source check while a
second swallow site downstream undoes it.
"""

import contextlib

import pytest

from src.video.pipeline_graph import PipelineGraph, StepStatus
from src.video.producer.context import InsufficientMediaError


class _BoomError(RuntimeError):
    """An ordinary step failure."""


@pytest.mark.asyncio
class TestPropagatedExceptions:
    async def test_a_declared_exception_reaches_the_caller(self):
        graph = PipelineGraph(propagate=(InsufficientMediaError,))

        async def gather(ctx):
            raise InsufficientMediaError("only 2 images")

        graph.add_step("gather_visuals", gather)
        with pytest.raises(InsufficientMediaError):
            await graph.execute_pipeline(context=object())

    async def test_it_survives_a_parallel_level(self):
        """`gather(return_exceptions=True)` turns a raise into a value."""
        graph = PipelineGraph(propagate=(InsufficientMediaError,))

        async def gather(ctx):
            raise InsufficientMediaError("only 2 images")

        async def fine(ctx):
            return "ok"

        graph.add_step("gather_visuals", gather)
        graph.add_step("generate_script", fine)
        with pytest.raises(InsufficientMediaError):
            await graph.execute_pipeline(context=object())

    async def test_an_ordinary_failure_is_still_a_failed_step(self):
        graph = PipelineGraph(propagate=(InsufficientMediaError,))

        async def boom(ctx):
            raise _BoomError("whisper timed out")

        graph.add_step("generate_subtitles", boom)
        results = await graph.execute_pipeline(context=object())
        assert [r.status for r in results] == [StepStatus.FAILED]
        assert isinstance(results[0].error, _BoomError)

    async def test_nothing_propagates_by_default(self):
        """A graph that declares nothing keeps the old behaviour."""
        graph = PipelineGraph()

        async def gather(ctx):
            raise InsufficientMediaError("only 2 images")

        graph.add_step("gather_visuals", gather)
        results = await graph.execute_pipeline(context=object())
        assert results[0].status == StepStatus.FAILED


class TestTheProducerGraphDeclaresIt:
    """The wiring, not just the mechanism."""

    def test_the_producer_propagates_media_rejections(self, monkeypatch):
        import src.video.producer.orchestration as orch

        seen = {}

        class _Spy(PipelineGraph):
            def __init__(self, *args, **kwargs):
                seen["propagate"] = kwargs.get("propagate", ())
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(orch, "PipelineGraph", _Spy)

        import asyncio
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        ctx = MagicMock()
        ctx.profile = SimpleNamespace(
            use_scraped_images=True,
            use_scraped_videos=True,
            use_stock_images=False,
            use_stock_videos=False,
        )
        ctx.run_paths = {"state_file": MagicMock(exists=lambda: False)}
        # The steps run against a stub; only the graph's construction is
        # under test here.
        with contextlib.suppress(Exception):
            asyncio.run(orch.execute_pipeline_parallel(ctx))
        assert InsufficientMediaError in seen["propagate"]
