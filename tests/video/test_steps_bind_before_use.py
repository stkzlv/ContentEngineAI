"""No step reads a local the compiler cannot prove is bound.

This bug class has bitten the project at least four times: twice in
`browser_functions.py` (a conditional import, then a variable used after an
`if/elif/else`), and twice here. Both of the ones here shipped:

- `stock_queries_issued` was declared inside `if profile_needs_stock_media(...)`
  while `save_visuals_info` reads it on every path. Ten of the eleven bundled
  profiles take that arm -- all nine a `--random-profile` run can draw -- so
  every fresh render failed at `gather_visuals` from v0.82.0 to v0.88.0.
- `ffmpeg_path` was bound inside `if audio_proc and ...silence_removal_enabled`
  while the duration read below needs it on every path, so setting that
  documented toggle to false, or omitting the `audio_processing:` section,
  crashed `create_voiceover` after the script and the TTS had been paid for.

The check reads CPython's own definite-assignment analysis rather than
reimplementing it: the compiler emits `LOAD_FAST_CHECK` where it cannot prove
a plain local is bound. It needs no maintenance as the syntax grows, and it
found the second bug above.

It is not total, and two gaps were measured rather than assumed. A name a
nested function closes over becomes a cell variable and is read with
`LOAD_DEREF`, so a conditionally bound name captured by an inner `def` is
missed though it raises `NameError`. A comprehension's iteration variable read
after the comprehension is compiled with `LOAD_FAST_AND_CLEAR` and a plain
`LOAD_FAST`, and is missed the same way. Neither shape exists in this module
today -- its only function with cell variables binds them unconditionally.

A hand-rolled AST walk was tried first and discarded. It reported names on
legitimate code across the producer package (a name bound in both arms of an
`if/else`, a walrus in an `if` test, a `global` declaration), and was blind to
bindings in `try` bodies, `except` handlers and `with ... as`. It did catch a
conditional `match` binding, by over-reporting every `match`.
"""

from __future__ import annotations

import dis
from pathlib import Path
from types import CodeType

import pytest

STEPS = Path("src/video/producer/steps.py")

# `step_assemble_video` reads these after an `async with` closes, so the
# compiler must assume the context manager might have suppressed an exception
# and skipped the binding. It cannot: `performance_monitor.measure_step`
# (`src/utils/performance.py`) re-raises in its `except` and its `finally`
# contains no `return`, which are the only two ways to suppress. Verified by
# reading it, not assumed -- and the entries are per-function, so the same
# name elsewhere is still checked.
CONSERVATIVE = {
    ("step_assemble_video", "final_video_path"),
    ("step_assemble_video", "results"),
}


def code_objects(code: CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, CodeType):
            yield from code_objects(const)


def unproven_locals(source: str, filename: str = "<probe>") -> set[tuple[str, str]]:
    """(function, name) pairs the compiler cannot prove are bound at use."""
    return {
        (code.co_name, instruction.argval)
        for code in code_objects(compile(source, filename, "exec"))
        for instruction in dis.get_instructions(code)
        if instruction.opname == "LOAD_FAST_CHECK"
    }


class TestTheCheckWorks:
    """A check that never fires is worse than none."""

    def test_it_catches_the_shape_that_shipped(self):
        planted = (
            "def f(ctx):\n"
            "    if ctx.needs:\n"
            "        queries = []\n"
            "    save(queries)\n"
        )

        assert ("f", "queries") in unproven_locals(planted)

    def test_it_clears_an_unconditional_binding(self):
        """The negative control: hoisted out, the same function is clean."""
        cleared = (
            "def f(ctx):\n"
            "    queries = []\n"
            "    if ctx.needs:\n"
            "        queries = build()\n"
            "    save(queries)\n"
        )

        assert not unproven_locals(cleared)

    @pytest.mark.parametrize(
        "body",
        [
            "    try:\n        x = f()\n    except E:\n        log()\n    use(x)\n",
            "    with open(p) as x:\n        pass\n" "    use(x)\n    use(y)\n",
            "    match ctx.kind:\n        case 1:\n            x = 1\n    use(x)\n",
        ],
    )
    def test_it_sees_every_binding_form(self, body):
        """`try`, `with ... as` and `match` all bind, and all hide this bug.

        The first two were measured blind to the discarded AST walk; `match`
        it caught, by over-reporting every `match`. All three are here because
        the guard has to be right about the form, not because the old one was
        wrong about each.
        """
        assert unproven_locals("def f(ctx, p, E, y=None):\n" + body) == {("f", "x")}

    def test_it_does_not_fire_on_a_plain_function(self):
        assert not unproven_locals("def f(a):\n    b = a + 1\n    return b\n")


class TestEveryStepBindsWhatItReads:
    def test_no_step_reads_an_unproven_local(self):
        found = unproven_locals(STEPS.read_text(encoding="utf-8"), str(STEPS)) - (
            CONSERVATIVE
        )

        assert not found, (
            f"{sorted(found)} may be read unbound, so a run taking the other "
            "branch raises UnboundLocalError mid-render"
        )

    @pytest.mark.parametrize(
        "function,name",
        [
            ("step_gather_visuals", "stock_queries_issued"),
            ("step_create_voiceover", "ffmpeg_path"),
        ],
    )
    def test_the_two_that_shipped_stay_fixed(self, function, name):
        assert (function, name) not in unproven_locals(
            STEPS.read_text(encoding="utf-8"), str(STEPS)
        )

    def test_the_conservative_set_is_still_conservative(self):
        """Each entry must still be reported, or it is stale and should go.

        Keeps the allowlist from outliving the compiler behaviour that
        justifies it, the way a shrink-only exemption list should.
        """
        reported = unproven_locals(STEPS.read_text(encoding="utf-8"), str(STEPS))

        assert reported >= CONSERVATIVE, (
            f"{sorted(CONSERVATIVE - reported)} are allowlisted but no longer "
            "reported; remove them"
        )


class TestTheMeasuredGaps:
    """The two shapes the opcode read does not see, pinned so they stay known.

    Recorded as tests rather than prose because a docstring claiming a gap is
    the same kind of thing as the comment that claimed this bug class was
    handled -- and that comment was wrong.
    """

    CLOSURE = (
        "def step(ctx):\n"
        "    if ctx.needs:\n"
        "        queries = []\n"
        "    def save():\n"
        "        return queries\n"
        "    return save()\n"
    )
    COMPREHENSION = "def f(ys):\n    [x for x in ys]\n    return x\n"

    @pytest.mark.parametrize("source", [CLOSURE, COMPREHENSION])
    def test_the_gap_is_where_it_is_documented(self, source):
        """A cell variable is read with LOAD_DEREF, an inlined comprehension
        target with a plain LOAD_FAST. Neither emits the opcode this scans for.
        """
        assert not unproven_locals(source)

    @pytest.mark.parametrize("source", [CLOSURE, COMPREHENSION])
    def test_the_gap_is_a_real_crash(self, source):
        """So the entry above records a limitation, not a harmless shape."""
        namespace: dict = {}
        exec(compile(source, "<probe>", "exec"), namespace)  # noqa: S102
        target = namespace.get("step") or namespace["f"]
        argument = type("C", (), {"needs": False})() if "step" in namespace else []

        with pytest.raises(NameError):
            target(argument)

    def test_this_module_has_no_such_shape(self):
        """The gap is acceptable only while the module stays clear of it."""
        module = compile(STEPS.read_text(encoding="utf-8"), str(STEPS), "exec")
        with_cells = [code.co_name for code in code_objects(module) if code.co_cellvars]

        assert with_cells == ["_check_existing_metadata"], (
            f"{with_cells} close over names, which the opcode scan cannot see; "
            "check each binds unconditionally or widen the guard"
        )
