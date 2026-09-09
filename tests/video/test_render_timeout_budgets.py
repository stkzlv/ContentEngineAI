"""The render's budget bounds the limits derived inside it (#398, #402).

Two timeouts govern a render and they were derived independently, with the
inner one able to exceed the outer: with the shipped settings a 59-second
voiceover earned Whisper 1008s inside a 900s pipeline. Whisper finished
inside its own budget having spent most of the run's, the pipeline timeout
fired during assembly, and the log blamed the pipeline rather than the step
that spent the time.
"""

import ast
from pathlib import Path

import pytest

from src.utils.pipeline_deadline import (
    clear_pipeline_deadline,
    remaining_pipeline_seconds,
    set_pipeline_deadline,
)
from src.video.config.llm_settings import LLMSettings  # noqa: F401  (config import)
from src.video.stt_functions import _calculate_timeout, _timeout_schedule


@pytest.fixture(autouse=True)
def _no_leaked_deadline():
    clear_pipeline_deadline()
    yield
    clear_pipeline_deadline()


class _Whisper:
    """The three fields the timeout arithmetic reads."""

    base_timeout_sec = 120
    duration_multiplier = 15.0
    max_timeout_sec = 1800
    timeout_retry_attempts = 1
    timeout_retry_multiplier = 2.0


class TestTheDeadline:
    def test_unset_reads_as_no_bound_not_as_no_time(self):
        assert remaining_pipeline_seconds() is None

    def test_a_budget_starts_the_clock(self):
        set_pipeline_deadline(900)
        remaining = remaining_pipeline_seconds()
        assert remaining is not None
        assert 890 < remaining <= 900

    def test_a_non_positive_budget_clears_rather_than_expires(self):
        """A caller applying no pipeline timeout must not cap every step."""
        set_pipeline_deadline(0)
        assert remaining_pipeline_seconds() is None
        set_pipeline_deadline(None)
        assert remaining_pipeline_seconds() is None


class TestTheSttLimitIsBounded:
    def test_the_documented_case(self):
        """59.2s of audio earns more than the whole 900s pipeline allowed.

        #398 quotes 1007.5s; the arithmetic it describes gives 120 + 59.2*15
        = 1008.0, so the figure asserted here is the computed one.
        """
        unbounded = _calculate_timeout(59.2, _Whisper())
        assert unbounded == pytest.approx(1008.0)
        assert unbounded > 900  # the whole pipeline's shipped budget, then

        set_pipeline_deadline(900)
        bounded = _calculate_timeout(59.2, _Whisper())
        assert bounded < 900
        assert bounded < unbounded

    def test_a_short_clip_keeps_its_own_smaller_limit(self):
        """The bound is a ceiling, not a replacement."""
        set_pipeline_deadline(2700)
        assert _calculate_timeout(20.0, _Whisper()) == pytest.approx(420.0)

    def test_no_deadline_leaves_the_configured_ceiling(self):
        assert _calculate_timeout(1000.0, _Whisper()) == pytest.approx(1800)

    def test_the_retry_schedule_cannot_widen_past_the_budget(self):
        """A retry promised time the render does not have cannot use it."""
        set_pipeline_deadline(700)
        limits = _timeout_schedule(_calculate_timeout(59.2, _Whisper()), _Whisper())
        assert limits
        assert all(limit <= 700 for limit in limits)

    def test_widening_still_happens_when_there_is_room(self):
        set_pipeline_deadline(2700)
        limits = _timeout_schedule(_calculate_timeout(30.0, _Whisper()), _Whisper())
        assert len(limits) == 2
        assert limits[1] > limits[0]


class TestTheDeadlineIsSetWhereTheTimeoutIsApplied:
    """The bound is inert unless whoever applies the timeout starts the clock.

    Both entry points are the same shape, and the project's own alignment
    rule is that they drift. Neither is drivable without a rendered product,
    so the call sites are read.
    """

    @pytest.mark.parametrize(
        "module",
        ["src/video/producer/cli.py", "src/pipeline/global_batch.py"],
    )
    def test_the_call_site_starts_the_clock(self, module):
        tree = ast.parse(Path(module).read_text())

        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set_pipeline_deadline"
        ]
        assert len(calls) == 1, (
            f"{module} calls set_pipeline_deadline {len(calls)} time(s); "
            "the deadline must be started exactly where the pipeline timeout "
            "is applied"
        )

        (arg,) = calls[0].args
        assert isinstance(arg, ast.Attribute)
        assert (
            arg.attr == "pipeline_timeout_sec"
        ), "the deadline must carry the same budget the wait_for enforces"


class TestTheShippedNumbers:
    """#402: the defaults and their documented reason."""

    def test_the_two_yaml_values_agree_with_the_models(self):
        from src.video.config import config

        assert config.pipeline_timeout_sec == 2700
        assert config.ffmpeg_settings.final_assembly_timeout_sec == 1800

    def test_assembly_fits_inside_the_pipeline_budget(self):
        """`final_assembly_timeout_sec` sits inside `pipeline_timeout_sec`.

        Raising either alone leaves the other binding, which is how the first
        fix attempt still lost the render.
        """
        from src.video.config import config

        assert (
            config.ffmpeg_settings.final_assembly_timeout_sec
            < config.pipeline_timeout_sec
        )

    def test_both_numbers_carry_their_reason(self):
        """Documented in terms of what makes them binding, not as bare numbers."""
        core = Path("config/core.yaml").read_text()
        perf = Path("config/performance.yaml").read_text()
        assert "contention" in core
        assert (
            "final_assembly_timeout_sec" in core
        ), "core.yaml must say the assembly timeout sits inside this one"
        assert (
            "pipeline_timeout_sec" in perf
        ), "performance.yaml must say this sits inside the pipeline budget"
