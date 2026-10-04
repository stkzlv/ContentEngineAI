"""A merge skips the suite only when its exact tree already passed on the PR."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence

import pytest

from tools.merge_was_tested import main, merge_was_tested

REPO = "owner/repo"
MERGE = "m" * 40
HEAD = "h" * 40


def fake(
    pulls: list[dict] | None = None,
    head_tree: str = "tree1",
    merge_tree: str = "tree1",
    checks: list[tuple[str, str]] | None = None,
):
    if pulls is None:
        pulls = [{"number": 7, "merge_commit_sha": MERGE, "head": {"sha": HEAD}}]
    if checks is None:
        checks = [("test (3.12)", "success"), ("lint", "success")]
    calls: list[list[str]] = []

    def runner(args: Sequence[str]) -> str:
        calls.append(list(args))
        if args[:2] == ["gh", "api"] and args[2].endswith("/pulls"):
            return json.dumps(pulls)
        if args[:2] == ["gh", "api"] and "/check-runs" in args[2]:
            runs = [{"name": n, "conclusion": c} for n, c in checks]
            return json.dumps({"check_runs": runs})
        if args[:2] == ["git", "rev-parse"]:
            return (head_tree if args[2].startswith(HEAD) else merge_tree) + "\n"
        if args[:2] == ["git", "fetch"]:
            return ""
        raise AssertionError(f"unexpected call {args}")

    return runner, calls


@pytest.mark.req("REQ-OPS-103")
def test_the_same_tree_that_passed_is_not_tested_again() -> None:
    runner, calls = fake()

    assert merge_was_tested(REPO, MERGE, runner) is True
    assert ["git", "fetch", "--quiet", "origin", "pull/7/head"] in calls


@pytest.mark.req("REQ-OPS-103")
@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("no pull request", {"pulls": []}),
        (
            "a pull request merged as another commit",
            {"pulls": [{"number": 7, "merge_commit_sha": "x", "head": {"sha": HEAD}}]},
        ),
        ("the merge changed the tree", {"merge_tree": "tree2"}),
        ("no test job ran", {"checks": [("lint", "success")]}),
        ("a test job failed", {"checks": [("test (3.12)", "failure")]}),
        (
            "one of two test jobs did not pass",
            {"checks": [("test (3.12)", "success"), ("test (3.13)", "cancelled")]},
        ),
    ],
)
def test_anything_else_runs_the_suite(label: str, kwargs: dict) -> None:
    runner, _ = fake(**kwargs)

    assert merge_was_tested(REPO, MERGE, runner) is False, label


@pytest.mark.req("REQ-OPS-103")
def test_an_error_finding_out_runs_the_suite(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(args: Sequence[str]) -> str:
        raise subprocess.CalledProcessError(1, list(args), stderr="not found")

    monkeypatch.setattr("tools.merge_was_tested.run", broken)

    assert main(["--repo", REPO, "--sha", MERGE]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "false"
    assert "running the suite" in out.err
