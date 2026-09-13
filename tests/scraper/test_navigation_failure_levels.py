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


def _handler_slice(message_fragment: str) -> tuple[str, str]:
    """(level, source between the enclosing except and the call).

    The slice is what makes "ungated" assertable: a DEBUG_MODE gate wedged
    between the handler and the call would keep the level intact while
    silencing the record on every normal run -- the exact regression the
    pass that created this file caught.
    """
    idx = SOURCE.index(message_fragment)
    call = SOURCE.rindex("logger.", 0, idx)
    handler = SOURCE.rindex("except ", 0, call)
    match = re.match(r"logger\.(\w+)", SOURCE[call:])
    assert match is not None
    return match.group(1), SOURCE[handler:call]


def test_the_initial_navigation_failure_logs_at_error_ungated():
    level, between = _handler_slice("Navigation failed after")
    assert level == "error"
    assert "DEBUG_MODE" not in between, "the failure record was re-gated"


def test_the_back_navigation_failure_logs_at_warning_ungated():
    level, between = _handler_slice("Back-navigation failed after")
    assert level == "warning"
    assert "DEBUG_MODE" not in between, "the failure record was re-gated"


def test_the_no_cards_truncation_logs_at_warning_ungated():
    """Back-navigation SUCCEEDING onto a cardless page breaks the loop the
    same way a failed one does, and was as invisible (#466).
    """
    idx = SOURCE.index("No product cards after back-navigation")
    call = SOURCE.rindex("logger.", 0, idx)
    match = re.match(r"logger\.(\w+)", SOURCE[call:])
    assert match is not None and match.group(1) == "warning"
    gate = SOURCE.rindex("else:", 0, call)
    assert "DEBUG_MODE" not in SOURCE[gate:call]


def test_the_card_processing_swallow_logs_at_warning_ungated():
    level, between = _handler_slice("Error processing card")
    assert level == "warning"
    assert "DEBUG_MODE" not in between


def test_all_levels_survive_a_normal_run_threshold():
    """Derived from the source, so a demotion fails here too."""
    for fragment in (
        "Navigation failed after",
        "Back-navigation failed after",
        "Error processing card",
    ):
        level, _ = _handler_slice(fragment)
        assert logging.getLevelName(level.upper()) >= logging.INFO
