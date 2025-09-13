"""Pipeline dependency graph and parallel execution framework.

This module implements a dependency-aware pipeline execution system that can
run independent steps in parallel while respecting data dependencies between steps.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a pipeline step."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of a pipeline step execution."""

    step_name: str
    status: StepStatus
    duration: float
    error: Exception | None = None
    outputs: dict[str, Any] | None = None


@dataclass
class PipelineStep:
    """Definition of a pipeline step."""

    name: str
    dependencies: set[str]
    function: Callable
    status: StepStatus = StepStatus.PENDING
    result: StepResult | None = None


class PipelineGraph:
    """Manages pipeline step dependencies and parallel execution."""

    def __init__(self):
        self.steps: dict[str, PipelineStep] = {}
        self.execution_order: list[list[str]] = []

    def add_step(
        self, name: str, function: Callable, dependencies: set[str] | None = None
    ) -> None:
        """Add a step to the pipeline graph.

        Args:
        ----
            name: Unique step identifier
            function: Async function to execute for this step
            dependencies: Set of step names this step depends on

        """
        dependencies = dependencies or set()

        # Validate dependencies exist
        for dep in dependencies:
            if dep not in self.steps:
                raise ValueError(f"Dependency '{dep}' not found for step '{name}'")

        self.steps[name] = PipelineStep(
            name=name, dependencies=dependencies, function=function
        )

        logger.debug(f"Added step '{name}' with dependencies: {dependencies}")

    def has_step(self, name: str) -> bool:
        """Check if a step exists in the pipeline.

        Args:
        ----
            name: Name of the step to check

        Returns:
        -------
            True if step exists, False otherwise

        """
        return name in self.steps

    def skip_step(self, name: str) -> None:
        """Mark a step as skipped.

        Args:
        ----
            name: Name of the step to skip

        """
        if name in self.steps:
            self.steps[name].status = StepStatus.SKIPPED
            logger.debug(f"Mapped step '{name}' as skipped")

    def compute_execution_order(self) -> list[list[str]]:
        """Compute the optimal execution order for parallel execution.

        Returns
        -------
            List of lists, where each inner list contains steps that can run in parallel

        """
        if not self.steps:
            return []

        # Topological sort with parallelization
        in_degree = {name: len(step.dependencies) for name, step in self.steps.items()}
        execution_levels = []

        while in_degree:
            # Find all steps with no remaining dependencies
            ready_steps = [name for name, degree in in_degree.items() if degree == 0]

            if not ready_steps:
                # Circular dependency detected
                remaining_steps = list(in_degree.keys())
                raise ValueError(
                    f"Circular dependency detected among steps: {remaining_steps}"
                )

            execution_levels.append(ready_steps)

            # Remove ready steps and update dependencies
            for step_name in ready_steps:
                del in_degree[step_name]

                # Reduce in-degree for dependent steps
                for remaining_name, remaining_step in self.steps.items():
                    if (
                        remaining_name in in_degree
                        and step_name in remaining_step.dependencies
                    ):
                        in_degree[remaining_name] -= 1

        self.execution_order = execution_levels
        logger.info(f"Computed execution order: {execution_levels}")
        return execution_levels

    async def execute_step(self, step_name: str, context: Any) -> StepResult:
        """Execute a single pipeline step.

        Args:
        ----
            step_name: Name of the step to execute
            context: Pipeline context object to pass to the step function

        Returns:
        -------
            StepResult with execution details

        """
        step = self.steps[step_name]
        step.status = StepStatus.RUNNING

        start_time = asyncio.get_event_loop().time()

        try:
            logger.info(f"Starting execution of step: {step_name}")

            # Check if step should be skipped (already completed in state)
            if (
                hasattr(context, "state")
                and context.state.get(step_name, {}).get("status") == "done"
            ):
                step.status = StepStatus.SKIPPED
                result = StepResult(
                    step_name=step_name, status=StepStatus.SKIPPED, duration=0.0
                )
                logger.info(f"Step '{step_name}' skipped - already completed")
                return result

            # Execute the step function
            await step.function(context)

            duration = asyncio.get_event_loop().time() - start_time
            step.status = StepStatus.COMPLETED

            result = StepResult(
                step_name=step_name, status=StepStatus.COMPLETED, duration=duration
            )

            step.result = result
            logger.info(f"Step '{step_name}' completed successfully in {duration:.2f}s")
            return result

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            step.status = StepStatus.FAILED

            result = StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                duration=duration,
                error=e,
            )

            step.result = result
            logger.error(f"Step '{step_name}' failed after {duration:.2f}s: {e}")
            return result

    async def execute_level(
        self, step_names: list[str], context: Any
    ) -> list[StepResult]:
        """Execute a level of steps in parallel.

        Args:
        ----
            step_names: List of step names to execute in parallel
            context: Pipeline context object

        Returns:
        -------
            List of StepResult objects

        """
        if len(step_names) == 1:
            # Single step - no need for parallel execution
            result = await self.execute_step(step_names[0], context)
            return [result]

        logger.info(f"Executing {len(step_names)} steps in parallel: {step_names}")

        # Execute all steps in parallel
        tasks = [self.execute_step(step_name, context) for step_name in step_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        step_results: list[StepResult] = []
        for i, result_item in enumerate(results):
            if isinstance(result_item, BaseException):
                # Convert exception to failed result
                step_result = StepResult(
                    step_name=step_names[i],
                    status=StepStatus.FAILED,
                    duration=0.0,
                    error=cast(Exception, result_item),
                )
                step_results.append(step_result)
            else:
                # result_item is guaranteed to be StepResult here
                step_results.append(cast(StepResult, result_item))

        return step_results

    async def execute_pipeline(
        self, context: Any, fail_fast: bool = True
    ) -> list[StepResult]:
        """Execute the entire pipeline with optimal parallelization.

        Args:
        ----
            context: Pipeline context object to pass to step functions
            fail_fast: If True, stop execution on first failure

        Returns:
        -------
            List of all step results

        """
        if not self.execution_order:
            self.compute_execution_order()

        all_results = []

        for level_index, level_steps in enumerate(self.execution_order):
            logger.info(
                f"Executing level {level_index + 1}/{len(self.execution_order)}: "
                f"{level_steps}"
            )

            level_results = await self.execute_level(level_steps, context)
            all_results.extend(level_results)

            # Check for failures
            failed_steps = [r for r in level_results if r.status == StepStatus.FAILED]
            if failed_steps and fail_fast:
                logger.error(f"Pipeline failed at level {level_index + 1}")
                for failed_result in failed_steps:
                    logger.error(
                        f"  - {failed_result.step_name}: {failed_result.error}"
                    )
                break

        completed_steps = [r for r in all_results if r.status == StepStatus.COMPLETED]
        failed_steps = [r for r in all_results if r.status == StepStatus.FAILED]
        skipped_steps = [r for r in all_results if r.status == StepStatus.SKIPPED]

        logger.info(
            f"Pipeline execution completed: "
            f"{len(completed_steps)} completed, "
            f"{len(skipped_steps)} skipped, "
            f"{len(failed_steps)} failed"
        )

        return all_results

    def get_parallelization_summary(self) -> dict[str, Any]:
        """Get a summary of the parallelization opportunities.

        Returns
        -------
            Dictionary with parallelization statistics

        """
        if not self.execution_order:
            self.compute_execution_order()

        total_steps = len(self.steps)
        sequential_levels = len(self.execution_order)
        max_parallel = (
            max(len(level) for level in self.execution_order)
            if self.execution_order
            else 0
        )

        # Calculate potential speedup (theoretical)
        theoretical_speedup = (
            total_steps / sequential_levels if sequential_levels > 0 else 1.0
        )

        return {
            "total_steps": total_steps,
            "sequential_levels": sequential_levels,
            "max_parallel_steps": max_parallel,
            "theoretical_speedup": theoretical_speedup,
            "execution_order": self.execution_order,
        }


def create_video_pipeline_graph() -> PipelineGraph:
    """Create the standard video production pipeline graph.

    Returns
    -------
        Configured PipelineGraph with video production steps

    """
    from src.video.producer import (
        step_assemble_video,
        step_create_voiceover,
        step_download_music,
        step_gather_visuals,
        step_generate_script,
        step_generate_subtitles,
    )

    graph = PipelineGraph()

    # Add steps with their dependencies
    graph.add_step("gather_visuals", step_gather_visuals)
    graph.add_step("generate_script", step_generate_script, {"gather_visuals"})
    graph.add_step("create_voiceover", step_create_voiceover, {"generate_script"})

    # These two steps can run in parallel after voiceover is ready
    graph.add_step("generate_subtitles", step_generate_subtitles, {"create_voiceover"})
    graph.add_step("download_music", step_download_music, {"create_voiceover"})

    # Final assembly depends on all previous steps
    graph.add_step(
        "assemble_video",
        step_assemble_video,
        {
            "gather_visuals",
            "generate_script",
            "create_voiceover",
            "generate_subtitles",
            "download_music",
        },
    )

    return graph
