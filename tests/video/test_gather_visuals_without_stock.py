"""`gather_visuals` must not need stock media to finish.

`stock_queries_issued` was declared inside `if profile_needs_stock_media(...)`
and read unconditionally by `save_visuals_info` at the end of the step, so any
profile drawing no stock media raised `UnboundLocalError` and failed the whole
render at `gather_visuals`.

Four of the eleven bundled profiles take that arm. The defect survived several
releases because the suite never drove the step for one of them and the
end-to-end checks happened to draw stock-using profiles; a batch run that
randomly picked `product_video_primary` is what surfaced it.

The previous fix for this exact symptom hoisted the name out of the inner
preloaded/fetch branch and left it inside the outer one, so the comment sitting
above it claimed the trap was handled while the trap was one level up. A
comment is not a check, which is what this file is for.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.video.producer.utils import profile_needs_stock_media

STEPS = Path("src/video/producer/steps.py")


def scoped_locals(node: ast.AST) -> set[str]:
    """Names bound in a nested scope of their own.

    A comprehension target and a lambda argument live in their own scope in
    Python 3, so they are never the unbound-local risk this file is about --
    but `ast.walk` reports them like any other Store, which had the check
    accusing `item` and `p` of the defect.
    """
    names: set[str] = set()
    for inner in ast.walk(node):
        if isinstance(
            inner, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
        ):
            for generator in inner.generators:
                names |= {
                    n.id for n in ast.walk(generator.target) if isinstance(n, ast.Name)
                }
        elif isinstance(inner, ast.Lambda):
            names |= {a.arg for a in inner.args.args}
    return names


def stores(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    } - scoped_locals(node)


def loads(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    } - scoped_locals(node)


def straight_line(body: list[ast.stmt]) -> list[ast.stmt]:
    """Flatten wrappers whose body runs unconditionally.

    `step_gather_visuals` puts its whole body inside one `async with`, so a
    walk over `step.body` alone sees a single statement and compares nothing.
    A `with` or a `try` body runs on the way through; an `if` or a loop body
    may not, and those stay nested so their bindings count as conditional.
    """
    flat: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, ast.With | ast.AsyncWith | ast.Try):
            flat.extend(straight_line(stmt.body))
        else:
            flat.append(stmt)
    return flat


def offenders_of(step: ast.AsyncFunctionDef | ast.FunctionDef) -> set[str]:
    """Names read on the straight-line path but bound only inside a branch.

    Dataflow over the body in order rather than an indentation heuristic: keep
    the names a plain assignment has already bound unconditionally, and report
    any name a later statement loads that has so far been bound only inside a
    conditional one.
    """
    unconditional = {arg.arg for arg in step.args.args}
    conditional: set[str] = set()
    offenders: set[str] = set()

    for stmt in straight_line(step.body):
        offenders |= (loads(stmt) & conditional) - unconditional
        if isinstance(stmt, ast.Assign | ast.AnnAssign | ast.AugAssign):
            unconditional |= stores(stmt)
        else:
            conditional |= stores(stmt) - unconditional

    return offenders


def gather_visuals() -> ast.AsyncFunctionDef | ast.FunctionDef:
    tree = ast.parse(STEPS.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
        and node.name == "step_gather_visuals"
    )


def test_some_bundled_profile_takes_the_empty_arm():
    """Every assertion below is about a path no bundled profile might take."""
    from src.video.config import config

    without_stock = [
        name
        for name, profile in config.video_profiles.items()
        if name != "base" and not profile_needs_stock_media(profile)
    ]
    assert without_stock, "no bundled profile skips stock media; the arm is dead"


def test_the_check_catches_the_shape_it_is_for():
    """Against a constructed function, so a green result means something."""
    planted = ast.parse(
        "def f(ctx):\n"
        "    if ctx.needs:\n"
        "        queries = []\n"
        "    save(queries)\n"
    )

    assert offenders_of(planted.body[0]) == {"queries"}


def test_the_check_sees_through_the_wrapper_the_real_step_uses():
    """The step's whole body is inside one `async with`.

    Without flattening that, the walk sees a single statement, compares
    nothing, and reports every function clean -- which is exactly what this
    check did on its first outing, passing against the reverted fix.
    """
    planted = ast.parse(
        "async def f(ctx):\n"
        "    async with ctx.thing():\n"
        "        if ctx.needs:\n"
        "            queries = []\n"
        "        save(queries)\n"
    )

    assert offenders_of(planted.body[0]) == {"queries"}


def test_the_check_clears_an_unconditional_binding():
    """The negative control: hoisted out, the same function is clean."""
    planted = ast.parse(
        "def f(ctx):\n"
        "    queries = []\n"
        "    if ctx.needs:\n"
        "        queries = build()\n"
        "    save(queries)\n"
    )

    assert offenders_of(planted.body[0]) == set()


def test_the_step_binds_every_name_it_reads():
    offenders = offenders_of(gather_visuals())

    assert not offenders, (
        f"{sorted(offenders)} are bound only inside a branch and read outside "
        "it, so a run that skips that branch raises UnboundLocalError and "
        "fails the whole render"
    )


def test_stock_queries_issued_specifically():
    """The name this was filed for, pinned by itself."""
    assert "stock_queries_issued" not in offenders_of(gather_visuals())
