"""The release check that makes every PR a release.

docs/versioning.md asks each pull request to bump the version and date its
CHANGELOG heading, and the merge to be tagged. Six documentation PRs once
merged without either, because nothing checked. These pin what the check
refuses, and that the repository's own files pass it.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tools.release_check import REPO, check, main, pyproject_version

TODAY = dt.date(2026, 10, 3)


def pyproject(version: str) -> str:
    return f'[tool.poetry]\nname = "x"\nversion = "{version}"\n'


def changelog(
    version: str = "0.2.1", date: str | None = "2026-10-03", unreleased: str = ""
) -> str:
    heading = f"## [{version}] - {date}" if date else f"## [{version}]"
    return (
        "# Changelog\n\n## [Unreleased]\n\n"
        f"{unreleased}\n{heading}\n\n### Fixed\n- A fix.\n\n"
        "## [0.2.0] - 2026-09-01\n\n### Added\n- Something.\n"
    )


def test_a_bumped_and_dated_release_passes() -> None:
    assert check(pyproject("0.2.1"), changelog(), pyproject("0.2.0"), TODAY) == []


def test_an_unbumped_version_fails() -> None:
    problems = check(pyproject("0.2.0"), changelog("0.2.0"), pyproject("0.2.0"), TODAY)
    assert any("every PR is a release" in p for p in problems)


def test_entries_left_under_unreleased_fail() -> None:
    problems = check(pyproject("0.2.1"), changelog(unreleased="- Pending.\n"))
    assert any("[Unreleased] holds 1 entries" in p for p in problems)


def test_a_heading_that_disagrees_with_pyproject_fails() -> None:
    problems = check(pyproject("0.2.2"), changelog("0.2.1"))
    assert any("pyproject.toml says 0.2.2" in p for p in problems)


@pytest.mark.parametrize(
    ("date", "expected"), [(None, "has no date"), ("2026-10-09", "in the future")]
)
def test_a_missing_or_future_date_fails(date: str | None, expected: str) -> None:
    problems = check(pyproject("0.2.1"), changelog(date=date), today=TODAY)
    assert any(expected in p for p in problems)


@pytest.mark.parametrize("version", ["0.2.3", "0.4.0", "0.3.1", "2.0.0"])
def test_a_skipped_version_fails(version: str) -> None:
    problems = check(pyproject(version), changelog(version), pyproject("0.2.0"), TODAY)
    assert any("versions are sequential" in p for p in problems)


@pytest.mark.parametrize("version", ["0.2.1", "0.3.0", "1.0.0"])
def test_each_next_version_passes(version: str) -> None:
    assert check(pyproject(version), changelog(version), pyproject("0.2.0")) == []


def test_a_breaking_entry_needs_a_minor_bump() -> None:
    breaking = changelog().replace("- A fix.", "- **Breaking**: a key is gone.")
    problems = check(pyproject("0.2.1"), breaking, pyproject("0.2.0"))
    assert any("needs at least a minor bump" in p for p in problems)
    minor = breaking.replace("0.2.1", "0.3.0")
    assert check(pyproject("0.3.0"), minor, pyproject("0.2.0")) == []


def test_the_repository_is_consistent() -> None:
    """The version and the top CHANGELOG heading agree on every commit."""
    assert (
        check(
            (REPO / "pyproject.toml").read_text(encoding="utf-8"),
            (REPO / "CHANGELOG.md").read_text(encoding="utf-8"),
        )
        == []
    )


def test_the_cli_prints_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    expected = pyproject_version((REPO / "pyproject.toml").read_text())
    assert capsys.readouterr().out.strip() == ".".join(map(str, expected))
