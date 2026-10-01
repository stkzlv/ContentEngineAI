"""Every relative link in the docs, and every `docs/` path the code cites, resolves.

The docs are being regrouped into folders by layer (`docs/README.md`), and each
move breaks the links into the moved page. Nothing else notices: a reader who
follows a dead link from a requirement or a code comment lands on a 404, and
the page they wanted is read by nobody.

Two sweeps cover the two ways a page is reached. Markdown links are resolved
relative to the file that holds them, the way GitHub renders them. A bare
`docs/...md` path is how code comments, config and `CLAUDE.md` cite a page, and
it is resolved from the repository root.

`CHANGELOG.md` is left out: it records pages as they were named at each
release, and rewriting history to follow a move would make it wrong. Tests are
left out of the second sweep because a test that reads a doc fails on its own
when the doc moves, and some tests plant a missing path on purpose. Private
overlays are gitignored, so a link to one can't be checked in CI.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# `[text](target)`; the target stops at whitespace or the closing parenthesis.
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")

# A repo-relative docs path, not preceded by a path character, so the tail of
# `../docs/x.md` or `src/docs/x.md` is not read as a citation of its own.
DOCS_PATH = re.compile(r"(?<![A-Za-z0-9_./-])(docs/[A-Za-z0-9_./-]+\.md)")

# Files that cite pages by bare path.
CITING_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".toml", ".sh", ".example"}

SKIPPED = {"CHANGELOG.md"}


def tracked() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return [REPO / p for p in out.splitlines() if p not in SKIPPED]


def is_external(target: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9+.-]*:", target)) or target.startswith("#")


def is_unresolvable_by_design(path: str) -> bool:
    """A private overlay, or a template placeholder such as `NNNN-title.md`."""
    return path.endswith(".private.md") or "NNNN" in path


def broken_links(source: Path, text: str) -> list[str]:
    broken = []
    for target in MD_LINK.findall(text):
        path = target.split("#", 1)[0]
        if not path or is_external(target) or is_unresolvable_by_design(path):
            continue
        base = REPO if path.startswith("/") else source.parent
        if not (base / path.lstrip("/")).exists():
            broken.append(target)
    return broken


def broken_citations(text: str) -> list[str]:
    return [
        path
        for path in sorted(set(DOCS_PATH.findall(text)))
        if not is_unresolvable_by_design(path) and not (REPO / path).exists()
    ]


def test_the_sweep_looks_at_something() -> None:
    """A sweep over no files passes for the wrong reason."""
    files = tracked()
    markdown = [p for p in files if p.suffix == ".md"]

    assert len(markdown) > 40, f"only {len(markdown)} markdown files swept"
    assert REPO / "docs" / "README.md" in markdown


def test_every_relative_markdown_link_resolves() -> None:
    offenders = [
        f"{p.relative_to(REPO)}: {target}"
        for p in tracked()
        if p.suffix == ".md" and p.exists()
        for target in broken_links(p, p.read_text(encoding="utf-8"))
    ]

    assert not offenders, "links to files that do not exist:\n" + "\n".join(offenders)


def test_every_cited_docs_path_exists() -> None:
    offenders = [
        f"{p.relative_to(REPO)}: {path}"
        for p in tracked()
        if p.exists()
        and p.parts[len(REPO.parts)] != "tests"
        and (p.suffix in CITING_SUFFIXES or p.name == "Makefile")
        for path in broken_citations(p.read_text(encoding="utf-8"))
    ]

    assert not offenders, (
        "docs paths that do not exist; a moved page needs its citations "
        "updated:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    ("source", "text"),
    [
        (REPO / "docs" / "README.md", "see [the map](missing-page.md)"),
        (REPO / "docs" / "notes" / "video.md", "see [it](../no-such-guide.md#x)"),
    ],
)
def test_the_link_rule_would_catch_a_new_one(source: Path, text: str) -> None:
    assert broken_links(source, text)


@pytest.mark.parametrize(
    "text",
    [
        "[site](https://example.com/docs/x.md)",
        "[section](#layers)",
        "[overlay](roadmap.private.md)",
        "[template](decisions/NNNN-title.md)",
        "[map](README.md#layers)",
    ],
)
def test_the_link_rule_leaves_valid_and_external_links_alone(text: str) -> None:
    assert not broken_links(REPO / "docs" / "README.md", text)


def test_the_citation_rule_would_catch_a_moved_page() -> None:
    assert broken_citations("# See docs/no-such-page.md for the reason.")
    assert not broken_citations("# See docs/README.md for the map.")
