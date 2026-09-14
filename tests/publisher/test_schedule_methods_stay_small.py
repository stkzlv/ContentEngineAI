"""No method in the scheduler is big enough to hide a defect in (#449).

`auto_schedule` was 744 lines with no nested definitions: occupancy, dedupe,
slot conflict, both publish branches, tracking, cleanup and all three
post-publish hooks, separated only by comments. The caption-clamping defect
(#403/#408) survived four review rounds inside those branches, and the fix
that finally held had to be a structural AST test, because reading a function
this size kept failing.

The bound counts code, not the docstring: `auto_schedule` documents eleven
public arguments and an example, which is not the complexity this is about.
"""

from __future__ import annotations

import ast

from src.utils.outputs_paths import get_project_root

REPO = get_project_root()
SCHEDULE = REPO / "src/publisher/schedule.py"
MAX_CODE_LINES = 150


def _code_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    total = (node.end_lineno or node.lineno) - node.lineno + 1
    docstring = node.body[0] if node.body else None
    if (
        isinstance(docstring, ast.Expr)
        and isinstance(docstring.value, ast.Constant)
        and isinstance(docstring.value.value, str)
    ):
        total -= (docstring.end_lineno or docstring.lineno) - docstring.lineno + 1
    return total


class TestTheSchedulerIsReadable:
    def test_no_method_is_oversized(self):
        tree = ast.parse(SCHEDULE.read_text(encoding="utf-8"))
        oversized = {
            node.name: _code_lines(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and _code_lines(node) > MAX_CODE_LINES
        }
        assert not oversized, (
            f"oversized methods in the scheduler: {oversized}. "
            "Extract, rather than adding another comment-separated section"
        )

    def test_the_publish_branches_are_their_own_methods(self):
        """The two that the caption defect hid between.

        Named rather than merely counted: the point is that each posting mode
        is a thing a reviewer can read start to finish, with its own target
        list in scope.
        """
        tree = ast.parse(SCHEDULE.read_text(encoding="utf-8"))
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert {"_post_unified", "_post_per_platform"} <= names
