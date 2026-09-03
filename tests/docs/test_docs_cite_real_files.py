"""Shipped prose cites files that exist, and never by line number.

Two classes of doc defect were found repeatedly by review rather than by a
check, and both are mechanical.

The first is a citation to a file that has moved or never existed:
`src/video/assembler.py` (a package), `producer.py` and `video_config.py` in
the contributor guide, `src/scraper/base/config.py` for two names that live in
`models.py`. Each one sends a reader to a path they cannot open, and the
contributor-guide pair also contradicted a correct statement nine lines above
it.

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
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Directories whose modules a doc may legitimately name.
CODE_ROOTS = ("src", "tools", "tests", "deploy", "scripts")

# `.py` not followed by a word character, so `subtitle_settings.pycaps` and
# `www.python.org` are not read as filenames.
PY_NAME = re.compile(r"[A-Za-z0-9_]+\.py(?![A-Za-z0-9_])")

# A source file cited with a line or a line range, in prose or in a comment.
LINE_CITATION = re.compile(
    r"[A-Za-z0-9_]+\.py:\d+(?:-\d+)?"  # module.py:120, module.py:120-140
    r"|\blines?\s+\d+-\d+"  # "lines 801-850"
    r"|\(~?\d+\s+lines?\)"  # "(~690 lines)", a hand-maintained size
)


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


def test_every_python_file_named_in_shipped_prose_exists() -> None:
    known = known_module_names()
    offenders = [
        f"{p.relative_to(REPO)}: {name}"
        for p in shipped_prose()
        for name in sorted(set(PY_NAME.findall(p.read_text())))
        if name not in known
    ]

    assert not offenders, (
        "prose names a Python file that does not exist:\n" + "\n".join(offenders)
    )


def test_no_shipped_prose_cites_a_source_line_number() -> None:
    offenders = [
        f"{p.relative_to(REPO)}: {hit}"
        for p in shipped_prose()
        for hit in sorted(set(LINE_CITATION.findall(p.read_text())))
    ]

    assert not offenders, (
        "line numbers go stale on the next edit above them; cite a symbol "
        "(`core.py::_normalize_video_format`) instead:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("see src/video/nonexistent_module.py for details", "filename"),
        ("see visual_builder.py:1261-1286 for details", "line"),
        ("configured above (lines 170-171)", "line"),
        ("- `core.py` - VideoAssembler orchestrator (~690 lines)", "line"),
    ],
)
def test_the_sweep_would_catch_a_new_one(text: str, rule: str, tmp_path: Path) -> None:
    """Each rule fires on a planted instance of what it is for."""
    if rule == "filename":
        assert PY_NAME.findall(text) == ["nonexistent_module.py"]
        assert "nonexistent_module.py" not in known_module_names()
    else:
        assert LINE_CITATION.findall(text)
