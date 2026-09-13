"""Swallowed navigation failures stay visible on a normal run (#446 review).

Both handlers catch everything and degrade -- the initial navigation returns
an empty list the caller records, and the back-navigation breaks the card
loop, truncating the page's collection. A mechanical re-leveling sweep once
demoted both to DEBUG, which a normal (INFO-level) run drops: the operator
saw a 0-product outcome with no cause line anywhere. The level is the whole
point here, so it is pinned.
"""

import logging
import re
from pathlib import Path

SOURCE = Path("src/scraper/amazon/browser_functions.py").read_text(encoding="utf-8")


def _level_of(message_fragment: str) -> str:
    """The logger method invoked for the call carrying the fragment."""
    idx = SOURCE.index(message_fragment)
    call = SOURCE.rindex("logger.", 0, idx)
    match = re.match(r"logger\.(\w+)", SOURCE[call:])
    assert match is not None
    return match.group(1)


def test_the_initial_navigation_failure_logs_at_error():
    assert _level_of("Navigation failed after") == "error"


def test_the_back_navigation_failure_logs_at_warning():
    assert _level_of("Back-navigation failed after") == "warning"


def test_both_levels_survive_a_normal_run_threshold():
    """The handlers are ungated, so their records must clear INFO."""
    for name in ("ERROR", "WARNING"):
        assert logging.getLevelName(name) >= logging.INFO
