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
from src.video.stt_functions import (
    _attempt_limit,
    _calculate_timeout,
    _timeout_schedule,
)


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

    def test_the_retry_is_clamped_against_the_budget_it_actually_has(self):
        """The schedule is built once; the budget shrinks as attempts run.

        With a 2700s budget 300s in, the schedule is [1008, 1800]. Attempt 1
        spending its whole 1008s leaves 1392s, so handing the retry its
        scheduled 1800s promises time the render no longer has, and the outer
        timeout then cancels the run with this step's limit unreached.
        """
        set_pipeline_deadline(2400)
        schedule = _timeout_schedule(_calculate_timeout(59.2, _Whisper()), _Whisper())
        assert schedule == [pytest.approx(1008.0), pytest.approx(1800.0)]

        # Attempt 1 spends its whole limit.
        set_pipeline_deadline(2400 - schedule[0])
        remaining = remaining_pipeline_seconds()
        assert remaining is not None
        limit, _ = _attempt_limit(schedule[1], _Whisper())
        assert limit <= remaining
        assert limit < schedule[1]

    def test_the_reason_describes_the_limit_not_a_later_clock(self):
        """Recomputing the flag when the attempt fails can only mislead.

        The budget only shrinks, so a later read turns False into True and
        the error names `pipeline_timeout_sec` for a limit the formula set --
        a knob that changes nothing.
        """
        set_pipeline_deadline(2400)
        schedule = _timeout_schedule(_calculate_timeout(59.2, _Whisper()), _Whisper())

        _, capped = _attempt_limit(schedule[0], _Whisper())
        assert capped is False, "1008s came from base + duration * multiplier"

        set_pipeline_deadline(2400 - schedule[0])
        _, capped_later = _attempt_limit(schedule[1], _Whisper())
        assert capped_later is True, "the retry really is capped by the budget"

    def test_a_formula_limit_is_not_blamed_on_the_budget(self):
        """The distinguishing case: budget below max_timeout_sec, not binding.

        `_stt_ceiling` alone answers "is the budget binding now", which is a
        different question from "did the budget set this limit". With 1700s
        left of the render and a 33.4s voiceover, the limit is the formula's
        621s and the budget is merely below `max_timeout_sec`. Reporting that
        as budget-capped sends the operator to `pipeline_timeout_sec`, which
        changes nothing.
        """
        set_pipeline_deadline(1700)
        derived = _calculate_timeout(33.4, _Whisper())
        assert derived == pytest.approx(621.0)

        limit, capped = _attempt_limit(derived, _Whisper())
        assert limit == pytest.approx(621.0)
        assert capped is False

        # ...and the budget really binding is still reported as such.
        set_pipeline_deadline(400)
        limit, capped = _attempt_limit(derived, _Whisper())
        assert limit <= 400
        assert capped is True

    def test_an_exhausted_budget_yields_no_time_rather_than_a_negative(self):
        set_pipeline_deadline(1)
        import time as _time

        _time.sleep(1.05)
        limit, capped = _attempt_limit(600.0, _Whisper())
        assert limit == 0.0
        assert capped is True

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


class TestTheReportedReasonComesFromOneHelper:
    """Both log lines must agree about what set the limit.

    The error path took its flag from `_attempt_limit` while the info line
    recomputed it from `_stt_ceiling`, so one run's log gave two
    contradictory causes for one number.
    """

    def test_the_info_line_uses_the_attempt_helper(self):
        source = Path("src/video/stt_functions.py").read_text()
        assert "transcription_timeout, capped_by_run = _attempt_limit(" in source, (
            "the info line must take the limit and its reason from the same "
            "helper the loop uses, not recompute the flag"
        )

    def test_the_flag_is_never_recomputed_from_the_ceiling(self):
        tree = ast.parse(Path("src/video/stt_functions.py").read_text())

        bare_ceiling_reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_stt_ceiling"
        ]
        # Only `_calculate_timeout`, `_timeout_schedule` and `_attempt_limit`
        # may read it; a fourth caller is a second opinion about the reason.
        assert len(bare_ceiling_reads) == 3, (
            f"{len(bare_ceiling_reads)} callers of _stt_ceiling; the reason "
            "must come from _attempt_limit alone"
        )


class TestTheAssemblyLimitIsBoundedToo:
    """The other step with a limit of its own (#398, pass-1 finding 5).

    A flat limit larger than the time left is the same promise the run cannot
    keep, and the outer timeout then cancels the encode with the FFmpeg limit
    unreached. Driving `assemble_video` needs a rendered product, so the call
    site is read.
    """

    def test_the_assembler_reads_the_remaining_budget(self):
        source = Path("src/video/assembler/core.py").read_text()
        assert "remaining_pipeline_seconds()" in source, (
            "assemble_video must bound its FFmpeg limit by the render's "
            "remaining budget, as the STT limit is"
        )

    def test_the_bound_reaches_the_ffmpeg_call(self):
        tree = ast.parse(Path("src/video/assembler/core.py").read_text())

        ffmpeg_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "async_run_ffmpeg"
            and any(kw.arg == "timeout_sec" for kw in node.keywords)
        ]
        assert ffmpeg_calls, "no async_run_ffmpeg call carries a timeout_sec"

        final = [
            kw.value
            for call in ffmpeg_calls
            for kw in call.keywords
            if kw.arg == "timeout_sec"
            and isinstance(kw.value, ast.Name)
            and kw.value.id == "assembly_timeout"
        ]
        assert final, (
            "the final assembly must pass the bounded local, not "
            "final_assembly_timeout_sec straight from config"
        )

    def test_the_bound_never_raises_the_configured_value(self):
        """It is a min, so a large remaining budget changes nothing."""
        source = Path("src/video/assembler/core.py").read_text()
        assert (
            "remaining < assembly_timeout" in source
        ), "the bound must only ever lower the configured limit"


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
