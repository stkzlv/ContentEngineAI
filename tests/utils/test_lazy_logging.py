"""Every logging call's format string matches the arguments it is given.

Lazy `%` formatting is enforced by ruff (`G004`), but ruff does not count the
placeholders: `logger.info("a %s b %s", one)` passes every linter and raises
inside `logging` at runtime, where the handler swallows it to stderr and the
call logs nothing. That failure mode arrived with the sweep that converted 601
f-string call sites, so the count is checked here rather than trusted.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from src.utils.outputs_paths import get_project_root

REPO = get_project_root()

LOG_METHODS = {
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "critical",
    "exception",
    "fatal",
    "log",
}

# A %-conversion, in the shape `logging` will hand to `str.__mod__`.
CONVERSION = re.compile(
    r"%(?:\((?P<key>[^)]*)\))?[#0\- +]*(?P<width>\*|\d+)?"
    r"(?:\.(?P<precision>\*|\d+))?[hlL]?(?P<kind>.)"
)


def placeholders(template: str) -> int | None:
    """How many arguments the template consumes, or None if unknowable.

    `%%` is a literal and consumes nothing; a `*` width or precision consumes
    an argument of its own; the mapping form (`%(name)s`) takes a single dict,
    so it is reported as unknowable rather than counted.
    """
    count = 0
    for match in CONVERSION.finditer(template):
        if match.group("kind") == "%":
            continue
        if match.group("key") is not None:
            return None
        count += 1 + [match.group("width"), match.group("precision")].count("*")
    return count


def literal_of(node: ast.AST) -> str | None:
    """The static text of a message argument, if it has one.

    A concatenation or an f-string has no static text to count, and both are
    ruff's business anyway (G003, G004).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _receiver(node: ast.AST) -> str:
    """The dotted name a method was called on, as far as it is static.

    Looks through a call, so `logging.getLogger(__name__).info(...)` -- ten
    converted sites in `src/` -- resolves to `logging.getLogger` rather than
    to nothing and being skipped.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_receiver(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _receiver(node.func)
    return ""


def _calls(tree: ast.AST):
    """Logging calls only.

    The method names are shared: `warnings.warn` takes a category rather than
    format arguments, and `parser.error` does no %-formatting at all, so
    counting placeholders there would fail a correct call. The receiver has to
    name a logger, or be the instance holding one.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in LOG_METHODS:
            continue
        receiver = _receiver(func.value)
        if "log" not in receiver.lower() and receiver not in {"self", "cls"}:
            continue
        yield node


def mismatches(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for call in _calls(tree):
        # `logger.log` takes the level first, so its message is one along.
        index = 1 if getattr(call.func, "attr", "") == "log" else 0
        if len(call.args) <= index:
            continue
        template = literal_of(call.args[index])
        if template is None:
            continue
        if any(isinstance(arg, ast.Starred) for arg in call.args):
            continue
        expected = placeholders(template)
        if expected is None:
            continue
        supplied = len(call.args) - 1 - index
        if expected != supplied:
            found.append(
                f"{path.relative_to(REPO)}:{call.lineno}: "
                f"{expected} placeholders, {supplied} arguments"
            )
    return found


class TestFormatStringsMatchTheirArguments:
    def test_source_tree(self):
        found: list[str] = []
        # Tests included: the sweep converted a call site there too, and a
        # miscount is as silent in a test as it is in production.
        for directory in ("src", "tools", "tests"):
            for path in sorted((REPO / directory).rglob("*.py")):
                found.extend(mismatches(path))
        assert not found, "logging calls whose arguments do not match: " + "; ".join(
            found
        )


class TestTheCheckItself:
    """The checker has to see the defect it exists to catch."""

    def test_a_missing_argument_is_caught(self, tmp_path):
        module = tmp_path / "probe.py"
        module.write_text('logger.info("a %s b %s", one)\n', encoding="utf-8")
        tree = ast.parse(module.read_text(encoding="utf-8"))
        call = next(_calls(tree))
        assert placeholders(call.args[0].value) == 2
        assert len(call.args) - 1 == 1

    def test_an_escaped_percent_consumes_nothing(self):
        assert placeholders("100%% done: %s") == 1

    def test_a_width_and_precision_count_once(self):
        assert placeholders("%.2f and %10s and %-3d") == 3

    def test_the_mapping_form_is_not_counted(self):
        assert placeholders("%(name)s") is None

    def test_a_star_precision_consumes_its_own_argument(self):
        """`%.*f` takes the precision first, then the value."""
        assert placeholders("Duration: %.*f seconds") == 2
