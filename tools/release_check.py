"""Check that a branch carries its own release, as docs/versioning.md requires.

Every pull request bumps the version in `pyproject.toml` and moves its
CHANGELOG entries under a dated heading for that version; the merge is then
tagged. Six documentation PRs once merged with their entries left under
`[Unreleased]` and no tag, because the rule lived only in prose. This makes it
a check.

Without `--base`, it checks the files are consistent with each other: the first
version heading in the CHANGELOG is the `pyproject.toml` version, it carries a
date, and `[Unreleased]` holds no entries. With `--base <ref>`, it also checks
the version is the next one after the base's (sequential, per versioning.md)
and that a `**Breaking**:` entry comes with at least a minor bump.

Usage:
  python -m tools.release_check                     # consistency only
  python -m tools.release_check --base origin/main  # what CI runs on a PR
  python -m tools.release_check --version           # print the version
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# `## [0.127.2] - 2026-10-03`; the date is optional here so a missing one is
# reported as such rather than as a missing heading.
VERSION_HEADING = re.compile(
    r"^## \[(\d+\.\d+\.\d+)\](?:\s+-\s+(\d{4}-\d{2}-\d{2}))?\s*$", re.M
)
UNRELEASED = re.compile(r"^## \[Unreleased\]\s*$", re.M)
NEXT_HEADING = re.compile(r"^## \[", re.M)


@dataclass(frozen=True)
class Release:
    version: tuple[int, int, int]
    date: str | None
    body: str


def parse_version(text: str) -> tuple[int, int, int]:
    major, minor, patch = (int(p) for p in text.split("."))
    return major, minor, patch


def fmt(version: tuple[int, int, int]) -> str:
    return ".".join(str(p) for p in version)


def pyproject_version(text: str) -> tuple[int, int, int]:
    data = tomllib.loads(text)
    raw = data.get("tool", {}).get("poetry", {}).get("version") or data.get(
        "project", {}
    ).get("version")
    if not raw:
        raise ValueError("pyproject.toml declares no version")
    return parse_version(raw)


def top_release(changelog: str) -> Release | None:
    match = VERSION_HEADING.search(changelog)
    if not match:
        return None
    start = match.end()
    following = NEXT_HEADING.search(changelog, start)
    body = changelog[start : following.start() if following else len(changelog)]
    return Release(parse_version(match.group(1)), match.group(2), body)


def unreleased_entries(changelog: str) -> list[str]:
    match = UNRELEASED.search(changelog)
    if not match:
        return []
    following = NEXT_HEADING.search(changelog, match.end())
    section = changelog[match.end() : following.start() if following else None]
    return [line for line in section.splitlines() if line.startswith("- ")]


def next_versions(base: tuple[int, int, int]) -> dict[str, tuple[int, int, int]]:
    major, minor, patch = base
    return {
        "patch": (major, minor, patch + 1),
        "minor": (major, minor + 1, 0),
        "major": (major + 1, 0, 0),
    }


def check(
    pyproject: str,
    changelog: str,
    base_pyproject: str | None = None,
    today: dt.date | None = None,
) -> list[str]:
    """Return the problems found; an empty list means the release is in order."""
    problems: list[str] = []
    version = pyproject_version(pyproject)
    release = top_release(changelog)

    if entries := unreleased_entries(changelog):
        problems.append(
            f"[Unreleased] holds {len(entries)} entries; move them under "
            f"## [{fmt(version)}] - <date>"
        )
    if release is None:
        problems.append("CHANGELOG.md has no version heading")
        return problems
    if release.version != version:
        problems.append(
            f"pyproject.toml says {fmt(version)} but the first CHANGELOG "
            f"heading is {fmt(release.version)}"
        )
    if release.date is None:
        problems.append(f"## [{fmt(release.version)}] has no date")
    elif today and dt.date.fromisoformat(release.date) > today:
        problems.append(f"## [{fmt(release.version)}] is dated in the future")

    if base_pyproject is None:
        return problems

    base = pyproject_version(base_pyproject)
    allowed = next_versions(base)
    if version == base:
        problems.append(
            f"the version is still {fmt(base)}; every PR is a release "
            f"(docs/versioning.md): bump to {fmt(allowed['patch'])} for a "
            f"fix or docs, {fmt(allowed['minor'])} for a feature"
        )
    elif version not in allowed.values():
        problems.append(
            f"{fmt(version)} does not follow {fmt(base)}; versions are "
            f"sequential, so use one of "
            f"{', '.join(fmt(v) for v in allowed.values())}"
        )
    elif "**Breaking**" in release.body and version == allowed["patch"]:
        problems.append(
            "a **Breaking** entry needs at least a minor bump "
            f"({fmt(allowed['minor'])})"
        )
    return problems


def git_show(ref: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="git ref to compare the version against")
    parser.add_argument(
        "--version", action="store_true", help="print the version and exit"
    )
    args = parser.parse_args(argv)

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    if args.version:
        print(fmt(pyproject_version(pyproject)))
        return 0

    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    base = git_show(args.base, "pyproject.toml") if args.base else None
    problems = check(pyproject, changelog, base, dt.datetime.now(dt.UTC).date())
    for problem in problems:
        print(f"release check: {problem}", file=sys.stderr)
    if not problems:
        print(f"release check: {fmt(pyproject_version(pyproject))} is in order")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
