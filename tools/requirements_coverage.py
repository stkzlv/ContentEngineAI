"""Report which requirements in docs/requirements/ a test cites.

A test cites a requirement with `@pytest.mark.req("REQ-PUB-012")`. This lists,
per area file, the ids no test cites, so coverage gaps are a report rather than
an audit. It reads the sources as text and imports nothing from the project,
so it runs without the pipeline's dependencies.

Usage: python -m tools.requirements_coverage [--uncited]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO / "docs" / "requirements"
TESTS = REPO / "tests"

STATUSES = ("shipped", "partial", "planned", "held", "deprecated")

# `- **REQ-PUB-012** `planned #567` The publisher ...`
ENTRY = re.compile(r"^- \*\*(REQ-[A-Z]{3}-\d{3})\*\* `([^`]+)`(.*)$")
ID = re.compile(r"\bREQ-[A-Z]{3}-\d{3}\b")
CITATION = re.compile(r"mark\.req\(([^)]*)\)")


@dataclass(frozen=True)
class Requirement:
    req_id: str
    status: str
    statement: str
    source: Path
    sub_bullets: tuple[str, ...]


def area_files() -> list[Path]:
    return sorted(p for p in REQUIREMENTS.glob("*.md") if p.name != "README.md")


def parse(path: Path) -> Iterator[Requirement]:
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        match = ENTRY.match(line)
        if not match:
            continue
        subs = []
        for follow in lines[i + 1 :]:
            if not follow.startswith("  - "):
                break
            subs.append(follow[4:])
        req_id, status, statement = match.groups()
        yield Requirement(req_id, status, statement.strip(), path, tuple(subs))


def requirements() -> list[Requirement]:
    return [r for path in area_files() for r in parse(path)]


def cited_ids() -> dict[str, list[Path]]:
    """Map each id a test cites to the test files citing it."""
    cited: dict[str, list[Path]] = {}
    for path in sorted(TESTS.rglob("*.py")):
        for args in CITATION.findall(path.read_text(encoding="utf-8")):
            for req_id in ID.findall(args):
                cited.setdefault(req_id, []).append(path)
    return cited


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--uncited", action="store_true", help="list every uncited id")
    args = parser.parse_args(argv)

    cited = cited_ids()
    for path in area_files():
        reqs = [r for r in parse(path) if not r.status.startswith("deprecated")]
        uncited = [r for r in reqs if r.req_id not in cited]
        print(f"{path.name}: {len(reqs) - len(uncited)} of {len(reqs)} cited")
        if args.uncited:
            for r in uncited:
                print(f"  {r.req_id} `{r.status}` {r.statement[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
