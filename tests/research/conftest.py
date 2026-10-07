"""No test reaches Wikipedia or Stack Exchange unless it stubs them itself."""

from __future__ import annotations

import pytest

from src.research import demand as demand_mod

REAL_OPEN_DATA = demand_mod._open_data


@pytest.fixture
def real_open_data():
    """The unstubbed fetch, for the tests that drive it with a fake session."""
    return REAL_OPEN_DATA


@pytest.fixture(autouse=True)
def _no_open_data(monkeypatch):
    async def none(articles, stack, today):
        return {k: None for k in articles}, [], []

    monkeypatch.setattr(demand_mod, "_open_data", none)
