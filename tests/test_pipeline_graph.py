"""Tests for pipeline graph and parallel execution framework."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.video.pipeline_graph import (
    PipelineGraph,
    PipelineStep,
    StepResult,
    StepStatus,
    create_video_pipeline_graph,
)


@dataclass
class MockContext:
    """Mock pipeline context for testing."""

    state: dict[str, Any]
    data: dict[str, Any]

    def __init__(self):
        self.state = {}
        self.data = {}


class TestPipelineGraph:
    """Test cases for PipelineGraph class."""

    def test_add_step_basic(self):
        """Test adding a basic step with no dependencies."""
        graph = PipelineGraph()
        mock_func = AsyncMock()

        graph.add_step("test_step", mock_func)

        assert "test_step" in graph.steps
        assert graph.steps["test_step"].name == "test_step"
        assert graph.steps["test_step"].function == mock_func
        assert graph.steps["test_step"].dependencies == set()
        assert graph.steps["test_step"].status == StepStatus.PENDING

    def test_add_step_with_dependencies(self):
        """Test adding a step with dependencies."""
        graph = PipelineGraph()
        mock_func1 = AsyncMock()
        mock_func2 = AsyncMock()

        # Add dependency first
        graph.add_step("step1", mock_func1)
        graph.add_step("step2", mock_func2, {"step1"})

        assert graph.steps["step2"].dependencies == {"step1"}

    def test_add_step_invalid_dependency(self):
        """Test adding a step with non-existent dependency raises error."""
        graph = PipelineGraph()
        mock_func = AsyncMock()

        with pytest.raises(ValueError, match="Dependency 'nonexistent' not found"):
            graph.add_step("test_step", mock_func, {"nonexistent"})

    def test_compute_execution_order_linear(self):
        """Test execution order computation for linear dependencies."""
        graph = PipelineGraph()

        # Create linear dependency chain: A -> B -> C
        graph.add_step("step_a", AsyncMock())
        graph.add_step("step_b", AsyncMock(), {"step_a"})
        graph.add_step("step_c", AsyncMock(), {"step_b"})

        order = graph.compute_execution_order()

        assert order == [["step_a"], ["step_b"], ["step_c"]]

    def test_compute_execution_order_parallel(self):
        """Test execution order computation for parallel opportunities."""
        graph = PipelineGraph()

        # Create diamond dependency: A -> (B,C) -> D
        graph.add_step("step_a", AsyncMock())
        graph.add_step("step_b", AsyncMock(), {"step_a"})
        graph.add_step("step_c", AsyncMock(), {"step_a"})
        graph.add_step("step_d", AsyncMock(), {"step_b", "step_c"})

        order = graph.compute_execution_order()

        assert len(order) == 3
        assert order[0] == ["step_a"]
        assert set(order[1]) == {"step_b", "step_c"}  # Order within level may vary
        assert order[2] == ["step_d"]

    def test_compute_execution_order_circular_dependency(self):
        """Test circular dependency detection."""
        graph = PipelineGraph()

        # Create circular dependency: A -> B -> A
        graph.add_step("step_a", AsyncMock())
        graph.add_step("step_b", AsyncMock(), {"step_a"})

        # Manually create circular dependency (bypass validation)
        graph.steps["step_a"].dependencies = {"step_b"}

        with pytest.raises(ValueError, match="Circular dependency detected"):
            graph.compute_execution_order()

    def test_compute_execution_order_empty(self):
        """Test execution order computation for empty graph."""
        graph = PipelineGraph()
        order = graph.compute_execution_order()
        assert order == []

    @pytest.mark.asyncio
    async def test_execute_step_success(self):
        """Test successful step execution."""
        graph = PipelineGraph()
        mock_func = AsyncMock()
        context = MockContext()

        graph.add_step("test_step", mock_func)

        result = await graph.execute_step("test_step", context)

        assert result.step_name == "test_step"
        assert result.status == StepStatus.COMPLETED
        assert result.duration > 0
        assert result.error is None
        assert graph.steps["test_step"].status == StepStatus.COMPLETED
        mock_func.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_execute_step_failure(self):
        """Test step execution with failure."""
        graph = PipelineGraph()
        test_error = Exception("Test error")
        mock_func = AsyncMock(side_effect=test_error)
        context = MockContext()

        graph.add_step("test_step", mock_func)

        result = await graph.execute_step("test_step", context)

        assert result.step_name == "test_step"
        assert result.status == StepStatus.FAILED
        assert result.duration > 0
        assert result.error == test_error
        assert graph.steps["test_step"].status == StepStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_step_skip_completed(self):
        """Test step execution skipping already completed steps."""
        graph = PipelineGraph()
        mock_func = AsyncMock()
        context = MockContext()
        context.state = {"test_step": {"status": "done"}}

        graph.add_step("test_step", mock_func)

        result = await graph.execute_step("test_step", context)

        assert result.step_name == "test_step"
        assert result.status == StepStatus.SKIPPED
        assert result.duration == 0
        mock_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_level_single_step(self):
        """Test executing a level with single step."""
        graph = PipelineGraph()
        mock_func = AsyncMock()
        context = MockContext()

        graph.add_step("test_step", mock_func)

        results = await graph.execute_level(["test_step"], context)

        assert len(results) == 1
        assert results[0].step_name == "test_step"
        assert results[0].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_level_parallel_steps(self):
        """Test executing multiple steps in parallel."""
        graph = PipelineGraph()
        mock_func1 = AsyncMock()
        mock_func2 = AsyncMock()
        context = MockContext()

        graph.add_step("step1", mock_func1)
        graph.add_step("step2", mock_func2)

        results = await graph.execute_level(["step1", "step2"], context)

        assert len(results) == 2
        step_names = {r.step_name for r in results}
        assert step_names == {"step1", "step2"}
        assert all(r.status == StepStatus.COMPLETED for r in results)
        mock_func1.assert_called_once_with(context)
        mock_func2.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_execute_level_with_failure(self):
        """Test executing level with one step failing."""
        graph = PipelineGraph()
        mock_func1 = AsyncMock()
        mock_func2 = AsyncMock(side_effect=Exception("Test error"))
        context = MockContext()

        graph.add_step("step1", mock_func1)
        graph.add_step("step2", mock_func2)

        results = await graph.execute_level(["step1", "step2"], context)

        assert len(results) == 2
        successful = [r for r in results if r.status == StepStatus.COMPLETED]
        failed = [r for r in results if r.status == StepStatus.FAILED]
        assert len(successful) == 1
        assert len(failed) == 1

    @pytest.mark.asyncio
    async def test_execute_pipeline_success(self):
        """Test full pipeline execution success."""
        graph = PipelineGraph()
        mock_func1 = AsyncMock()
        mock_func2 = AsyncMock()
        mock_func3 = AsyncMock()
        context = MockContext()

        # A -> B -> C
        graph.add_step("step_a", mock_func1)
        graph.add_step("step_b", mock_func2, {"step_a"})
        graph.add_step("step_c", mock_func3, {"step_b"})

        results = await graph.execute_pipeline(context)

        assert len(results) == 3
        assert all(r.status == StepStatus.COMPLETED for r in results)

        # Verify execution order
        step_names = [r.step_name for r in results]
        assert step_names == ["step_a", "step_b", "step_c"]

    @pytest.mark.asyncio
    async def test_execute_pipeline_fail_fast(self):
        """Test pipeline execution with fail_fast=True."""
        graph = PipelineGraph()
        mock_func1 = AsyncMock()
        mock_func2 = AsyncMock(side_effect=Exception("Test error"))
        mock_func3 = AsyncMock()
        context = MockContext()

        # A -> B -> C (B fails)
        graph.add_step("step_a", mock_func1)
        graph.add_step("step_b", mock_func2, {"step_a"})
        graph.add_step("step_c", mock_func3, {"step_b"})

        results = await graph.execute_pipeline(context, fail_fast=True)

        # Should have results for A and B only (C not executed)
        assert len(results) == 2
        step_names = {r.step_name for r in results}
        assert step_names == {"step_a", "step_b"}

        # A should succeed, B should fail
        assert results[0].status == StepStatus.COMPLETED
        assert results[1].status == StepStatus.FAILED

        # C should not be called
        mock_func3.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_pipeline_continue_on_failure(self):
        """Test pipeline execution with fail_fast=False."""
        graph = PipelineGraph()
        mock_func1 = AsyncMock()
        mock_func2 = AsyncMock(side_effect=Exception("Test error"))
        mock_func3 = AsyncMock()
        context = MockContext()

        # A, B (parallel, B fails), C (depends on A only)
        graph.add_step("step_a", mock_func1)
        graph.add_step("step_b", mock_func2)
        graph.add_step("step_c", mock_func3, {"step_a"})

        results = await graph.execute_pipeline(context, fail_fast=False)

        assert len(results) == 3
        completed = [r for r in results if r.status == StepStatus.COMPLETED]
        failed = [r for r in results if r.status == StepStatus.FAILED]
        assert len(completed) == 2  # A and C
        assert len(failed) == 1  # B

    def test_get_parallelization_summary(self):
        """Test parallelization summary generation."""
        graph = PipelineGraph()

        # Diamond pattern: A -> (B,C) -> D
        graph.add_step("step_a", AsyncMock())
        graph.add_step("step_b", AsyncMock(), {"step_a"})
        graph.add_step("step_c", AsyncMock(), {"step_a"})
        graph.add_step("step_d", AsyncMock(), {"step_b", "step_c"})

        summary = graph.get_parallelization_summary()

        assert summary["total_steps"] == 4
        assert summary["sequential_levels"] == 3
        assert summary["max_parallel_steps"] == 2
        assert summary["theoretical_speedup"] == 4 / 3  # ~1.33x
        assert len(summary["execution_order"]) == 3

    def test_get_parallelization_summary_empty(self):
        """Test parallelization summary for empty graph."""
        graph = PipelineGraph()
        summary = graph.get_parallelization_summary()

        assert summary["total_steps"] == 0
        assert summary["sequential_levels"] == 0
        assert summary["max_parallel_steps"] == 0
        assert summary["theoretical_speedup"] == 1.0


class TestVideoProductionGraph:
    """Test cases for video production pipeline graph."""

    def test_create_video_pipeline_graph(self):
        """Test creation of video production pipeline graph."""
        graph = create_video_pipeline_graph()

        # Verify all steps are present
        expected_steps = {
            "gather_visuals",
            "generate_script",
            "create_voiceover",
            "generate_subtitles",
            "download_music",
            "assemble_video",
        }
        assert set(graph.steps.keys()) == expected_steps

        # Verify dependencies are correct
        assert graph.steps["gather_visuals"].dependencies == set()
        assert graph.steps["generate_script"].dependencies == {"gather_visuals"}
        assert graph.steps["create_voiceover"].dependencies == {"generate_script"}
        assert graph.steps["generate_subtitles"].dependencies == {"create_voiceover"}
        assert graph.steps["download_music"].dependencies == {"create_voiceover"}
        assert graph.steps["assemble_video"].dependencies == {
            "gather_visuals",
            "generate_script",
            "create_voiceover",
            "generate_subtitles",
            "download_music",
        }

    def test_video_pipeline_execution_order(self):
        """Test execution order for video production pipeline."""
        graph = create_video_pipeline_graph()
        order = graph.compute_execution_order()

        # Should have 4 levels:
        # 1. gather_visuals
        # 2. generate_script
        # 3. create_voiceover
        # 4. (generate_subtitles, download_music) - parallel
        # 5. assemble_video
        assert len(order) == 5
        assert order[0] == ["gather_visuals"]
        assert order[1] == ["generate_script"]
        assert order[2] == ["create_voiceover"]
        assert set(order[3]) == {"generate_subtitles", "download_music"}
        assert order[4] == ["assemble_video"]

    def test_video_pipeline_parallelization_potential(self):
        """Test parallelization potential for video production pipeline."""
        graph = create_video_pipeline_graph()
        summary = graph.get_parallelization_summary()

        # Should show potential for 2 steps to run in parallel
        assert summary["total_steps"] == 6
        assert summary["sequential_levels"] == 5
        assert summary["max_parallel_steps"] == 2
        # Theoretical speedup: 6 total steps / 5 sequential levels = 1.2x
        assert abs(summary["theoretical_speedup"] - 1.2) < 0.01


@pytest.mark.asyncio
async def test_step_result_dataclass():
    """Test StepResult dataclass functionality."""
    result = StepResult(
        step_name="test_step", status=StepStatus.COMPLETED, duration=1.5
    )

    assert result.step_name == "test_step"
    assert result.status == StepStatus.COMPLETED
    assert result.duration == 1.5
    assert result.error is None
    assert result.outputs is None


@pytest.mark.asyncio
async def test_pipeline_step_dataclass():
    """Test PipelineStep dataclass functionality."""
    mock_func = AsyncMock()
    step = PipelineStep(
        name="test_step", dependencies={"dep1", "dep2"}, function=mock_func
    )

    assert step.name == "test_step"
    assert step.dependencies == {"dep1", "dep2"}
    assert step.function == mock_func
    assert step.status == StepStatus.PENDING
    assert step.result is None
