"""The requirement files keep the shape that makes them checkable.

`docs/requirements/` gives every requirement a stable id and a status so that a
test, a pull request or a report can cite it (`docs/requirements/README.md`).
That only holds while the format does: an id reused after a deletion points
old citations at a different requirement, a bullet with no id can't be cited
at all, and a `partial` with no stated gap tells the reader nothing about what
is missing. None of that shows up in review of a single edit, so it is pinned
here.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from tools.requirements_coverage import (
    ENTRY,
    REPO,
    area_files,
    cited_ids,
    parse,
    requirements,
)

STATUS = re.compile(
    r"^(shipped|partial|held|deprecated|planned #\d+|planned \(decision \d{4}\))$"
)
SUB_LABELS = ("Gap:", "On when:", "Why:", "Check:")
REQUIRED_SUB = {"partial": "Gap:", "held": "On when:"}


def test_the_sweep_looks_at_something() -> None:
    files = area_files()
    assert len(files) >= 7, [p.name for p in files]
    assert len(requirements()) > 300


def test_ids_are_unique_across_areas() -> None:
    counts = Counter(r.req_id for r in requirements())
    assert not [i for i, n in counts.items() if n > 1]


@pytest.mark.parametrize("path", area_files(), ids=lambda p: p.name)
def test_each_area_numbers_its_ids_without_gaps_under_one_prefix(path) -> None:
    """Ids are permanent, so a requirement added later takes the next free
    number in its file and sits in its own section: the numbers have no gaps,
    but they need not follow the file order.
    """
    ids = [r.req_id for r in parse(path)]
    prefixes = {i.rsplit("-", 1)[0] for i in ids}
    assert len(prefixes) == 1, prefixes
    numbers = sorted(int(i.rsplit("-", 1)[1]) for i in ids)
    assert numbers == list(range(1, len(ids) + 1))


@pytest.mark.parametrize("path", area_files(), ids=lambda p: p.name)
def test_every_line_under_a_section_is_a_requirement_or_its_sub_bullet(path) -> None:
    """A bullet without an id can't be cited, and a wrapped or `*` line is
    invisible to the parser, so under the first section heading only headings,
    requirements, their one-line sub-bullets and blank lines are allowed.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("## "))
    stray = [
        line
        for line in lines[first:]
        if line.strip()
        and not line.startswith("## ")
        and not ENTRY.match(line)
        and not line.startswith("  - ")
    ]
    assert not stray, stray


def test_every_status_is_one_of_the_documented_forms() -> None:
    bad = [(r.req_id, r.status) for r in requirements() if not STATUS.match(r.status)]
    assert not bad, bad


def test_partial_and_held_say_what_is_missing_or_what_turns_them_on() -> None:
    missing = [
        (r.req_id, label)
        for r in requirements()
        if (label := REQUIRED_SUB.get(r.status))
        and not any(s.startswith(label) for s in r.sub_bullets)
    ]
    assert not missing, missing


def test_sub_bullets_use_the_documented_labels() -> None:
    bad = [
        (r.req_id, s[:40])
        for r in requirements()
        for s in r.sub_bullets
        if not s.startswith(SUB_LABELS)
    ]
    assert not bad, bad


def test_a_decision_cited_as_a_status_exists() -> None:
    decisions = REPO / "docs" / "decisions"
    missing = [
        (r.req_id, n)
        for r in requirements()
        if (m := re.search(r"decision (\d{4})", r.status))
        and not list(decisions.glob(f"{(n := m.group(1))}-*.md"))
    ]
    assert not missing, missing


def test_every_id_a_test_cites_exists() -> None:
    known = {r.req_id for r in requirements()}
    unknown = {
        i: [str(p.relative_to(REPO)) for p in paths]
        for i, paths in cited_ids().items()
        if i not in known
    }
    assert not unknown, unknown
