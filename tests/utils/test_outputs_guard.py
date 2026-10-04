"""The suite's outputs/ guard reports every kind of change.

The guard in `tests/conftest.py` is the evidence that a test run leaves the
real outputs tree alone, so a comparison that silently stops seeing a kind of
change would let the suite go green while writing there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import _outputs_snapshot, _snapshot_changes


@pytest.mark.req("REQ-OPS-013")
@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (None, None, []),
        (None, {}, ["outputs/ (created)"]),
        ({}, None, ["outputs/ (removed)"]),
        ({"a.log": (1, 1)}, {}, ["a.log (removed)"]),
        ({}, {"d/": (0, 0)}, ["d/"]),
        ({"a.log": (1, 1)}, {"a.log": (2, 1)}, ["a.log"]),
        ({"a.log": (1, 1), "d/": (0, 0)}, {"a.log": (1, 1), "d/": (0, 0)}, []),
    ],
)
def test_every_change_is_reported(before, after, expected) -> None:
    assert _snapshot_changes(before, after) == expected


@pytest.mark.req("REQ-OPS-013")
def test_the_snapshot_holds_directories_and_files(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "a.log").write_text("x")
    snapshot = _outputs_snapshot(root)

    assert snapshot is not None
    assert set(snapshot) == {"logs/", "logs/a.log"}
    assert snapshot["logs/a.log"][1] == 1


def test_an_absent_tree_has_no_snapshot(tmp_path: Path) -> None:
    assert _outputs_snapshot(tmp_path / "outputs") is None
