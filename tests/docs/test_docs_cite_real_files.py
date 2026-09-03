"""Shipped prose cites files that exist, and never by line number.

Two classes of doc defect were found repeatedly by review rather than by a
check, and both are mechanical.

The first is a citation to a file that has moved or never existed:
`src/video/assembler.py` (a package), and `producer.py` and `video_config.py`
in the contributor guide. Each one sends a reader to a path they cannot open,
and the contributor-guide pair also contradicted a correct statement nine
lines above it.

A citation whose first segment is a directory of this repository is checked as
a whole path, so a module named under the wrong package is caught even though
its basename exists. Anything else is judged by basename: a dependency's path
in a traceback, or a bare name. That leaves bare names weakly checked, since
six basenames here also exist in site-packages, and it cannot decide a claim
about what a file *contains* -- `base/config.py` exists but holds neither of
the two names the contributor guide once sourced from it. Both stay review
concerns.

The second is a citation that carries a line number. Those are wrong within a
release: an audit found `media_extractor.py:1261-1286` pointing past the end of
a 371-line file, `config_models.py` given as "283 lines" when it was 320, and
a config comment naming "lines 801-850" of a file that is a directory. Nothing
keeps them true, so this file refuses them outright rather than asking a
reviewer to re-check the arithmetic. Cite a symbol instead: `core.py::_normalize_video_format`
survives every edit above it.

Both sweeps are derived. The filename check resolves against the repository
and then against the installed packages, so a reference to a dependency's
module (`torch/overrides.py` in a traceback, openai-whisper's `setup.py`,
Botasaurus's `solve_cloudflare_captcha.py`) passes without an exemption list
that would itself go stale.

What this cannot decide is whether a sentence about behaviour is true, which
stays a review concern. It only removes the two classes that a machine can
settle.
"""

from __future__ import annotations

import re
import sysconfig
from pathlib import Path, PurePath

import pytest

REPO = Path(__file__).resolve().parents[2]

# Directories whose modules a doc may legitimately name.
CODE_ROOTS = ("src", "tools", "tests", "deploy", "scripts")

# Top-level directories of this repository. A citation starting with one of
# these is a claim about a path here, so it has to resolve exactly.
REPO_DIRS = frozenset(CODE_ROOTS) | {"config", "docs", ".github"}

# An optional directory prefix, then `.py` not followed by a word character,
# so `subtitle_settings.pycaps` and `www.python.org` are not read as filenames.
# The prefix is captured because a path can be checked exactly, while a bare
# name can only be checked against the set of names that exist.
PY_NAME = re.compile(r"(?:[A-Za-z0-9_./-]*/)?[A-Za-z0-9_]+\.py(?![A-Za-z0-9_])")

# A source file cited with a line or a line range, in prose or in a comment.
LINE_CITATION = re.compile(
    r"[A-Za-z0-9_]+\.py:\d+(?:-\d+)?"  # module.py:120, module.py:120-140
    r"|\(~?\d+\s+lines?\)"  # "(~690 lines)", a hand-maintained size
)

# "lines 801-850" is only a source citation when the same line names a source
# file. On its own the phrase is ordinary prose about output, and the failure
# message ("cite a symbol instead") would be nonsense for it.
LINE_RANGE = re.compile(r"\blines?\s+\d+-\d+")
SOURCE_FILE_ON_LINE = re.compile(r"[A-Za-z0-9_]+\.(?:py|yaml|yml)(?![A-Za-z0-9_])")


def shipped_prose() -> list[Path]:
    """Docs and config a reader follows, minus the gitignored private overlay."""
    files = [
        p
        for p in sorted((REPO / "docs").glob("*.md"))
        if not p.name.endswith(".private.md")
    ]
    files += [REPO / "README.md", REPO / "CONTRIBUTING.md"]
    files += sorted((REPO / "config").glob("*.yaml"))
    return [p for p in files if p.exists()]


def known_module_names() -> set[str]:
    """Every module basename this repository or its dependencies provide."""
    names: set[str] = set()
    for root in CODE_ROOTS:
        directory = REPO / root
        if directory.is_dir():
            names |= {p.name for p in directory.rglob("*.py")}
    site = sysconfig.get_paths().get("purelib")
    if site and Path(site).is_dir():
        names |= {p.name for p in Path(site).rglob("*.py")}
    return names


def test_the_sweep_looks_at_something() -> None:
    """A sweep over no files passes for the wrong reason."""
    files = shipped_prose()

    assert len(files) > 20, f"only {len(files)} files swept"
    assert any(p.name == "architecture.md" for p in files)
    assert any(p.suffix == ".yaml" for p in files)


def unresolved(citation: str, known: set[str]) -> bool:
    """True when nothing in the repo or its dependencies matches the citation.

    A citation carrying a directory is checked as a whole path against the
    repository, so a name written under the wrong package is caught. A path
    that is not repo-relative (a dependency's module in a traceback) falls
    back to its basename, as does a bare name.
    """
    head = citation.split("/", 1)[0]
    if "/" in citation and head in REPO_DIRS:
        # A path into this repository is checked exactly, so a name written
        # under the wrong package is caught even though its basename exists.
        return not (REPO / citation).exists()
    # A dependency's path in a traceback, or a bare name: judge it by name.
    return PurePath(citation).name not in known


def test_every_python_file_named_in_shipped_prose_exists() -> None:
    known = known_module_names()
    offenders = [
        f"{p.relative_to(REPO)}: {name}"
        for p in shipped_prose()
        for name in sorted(set(PY_NAME.findall(p.read_text())))
        if unresolved(name, known)
    ]

    assert not offenders, (
        "prose names a Python file that does not exist:\n" + "\n".join(offenders)
    )


def test_no_shipped_prose_cites_a_source_line_number() -> None:
    offenders = []
    for p in shipped_prose():
        text = p.read_text()
        offenders += [
            f"{p.relative_to(REPO)}: {h}"
            for h in sorted(set(LINE_CITATION.findall(text)))
        ]
        offenders += [
            f"{p.relative_to(REPO)}: {line.strip()[:70]}"
            for line in text.splitlines()
            if LINE_RANGE.search(line) and SOURCE_FILE_ON_LINE.search(line)
        ]

    assert not offenders, (
        "line numbers go stale on the next edit above them; cite a symbol "
        "(`core.py::_normalize_video_format`) instead:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "text",
    [
        "see src/video/nonexistent_module.py for details",
        # The wrong-directory shape: the basename exists elsewhere in the
        # repository, but not at the path written.
        "see src/video/producer/keyword_pillars.py for the rotation",
    ],
)
def test_the_filename_rule_would_catch_a_new_one(text: str) -> None:
    known = known_module_names()
    found = PY_NAME.findall(text)

    assert found, "the pattern did not match the planted citation"
    assert any(unresolved(c, known) for c in found)


@pytest.mark.parametrize(
    "text",
    [
        "see visual_builder.py:1261-1286 for details",
        "- `core.py` - VideoAssembler orchestrator (~690 lines)",
        "# Used by: freesound_client.py (lines 349-418)",
    ],
)
def test_the_line_rule_would_catch_a_new_one(text: str) -> None:
    assert LINE_CITATION.findall(text) or (
        LINE_RANGE.search(text) and SOURCE_FILE_ON_LINE.search(text)
    )


@pytest.mark.parametrize(
    "text",
    [
        "See lines 3-4 of the output.",
        "The banner spans lines 10-12 of the log.",
    ],
)
def test_the_line_rule_leaves_prose_about_output_alone(text: str) -> None:
    """A range with no source file on the line is not a source citation."""
    assert not LINE_CITATION.findall(text)
    assert not (LINE_RANGE.search(text) and SOURCE_FILE_ON_LINE.search(text))
