"""Whether a push to main merges a tree its pull request already tested.

A squash merge of a branch that was current with main produces the same tree
the pull request's last CI run tested, so running the suite again on main
repeats a result already known; it cost about five minutes per release. This
answers `true` only when the merge commit came from a pull request, the pull
request head's tree is byte-identical to the merged tree, and every test job
on that head passed. Anything else, including any failure to find out, is
`false`, and CI runs the suite.

Usage:
  python -m tools.merge_was_tested --repo owner/name --sha <merge commit>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any

Runner = Callable[[Sequence[str]], str]


def run(args: Sequence[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def _api(runner: Runner, path: str) -> Any:
    return json.loads(runner(["gh", "api", path]))


def _tree(runner: Runner, ref: str) -> str:
    return runner(["git", "rev-parse", f"{ref}^{{tree}}"]).strip()


def merge_was_tested(repo: str, sha: str, runner: Runner = run) -> bool:
    pulls = _api(runner, f"repos/{repo}/commits/{sha}/pulls")
    merged = [p for p in pulls if p.get("merge_commit_sha") == sha]
    if len(merged) != 1:
        return False
    number = merged[0]["number"]
    head = merged[0]["head"]["sha"]

    # The branch is usually deleted on merge; the pull request ref is not.
    runner(["git", "fetch", "--quiet", "origin", f"pull/{number}/head"])
    if _tree(runner, head) != _tree(runner, sha):
        return False

    runs = _api(runner, f"repos/{repo}/commits/{head}/check-runs?per_page=100")
    tests = [r for r in runs["check_runs"] if r["name"].startswith("test")]
    return bool(tests) and all(r["conclusion"] == "success" for r in tests)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args(argv)
    try:
        tested = merge_was_tested(args.repo, args.sha, run)
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as exc:
        print(f"could not tell, running the suite: {exc}", file=sys.stderr)
        tested = False
    print("true" if tested else "false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
