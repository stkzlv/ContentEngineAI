"""Scraper log lines describe the run as it happened.

Five lines in the 2026-09-19 batch log did not: a navigation duration of
1788936527 seconds (a monotonic start subtracted from the wall clock), a
product counter that counted search cards, a promise of a visible window on
a run that had just said it was on a virtual display, a per-input "no ASIN"
warning from an output callback that never received products, and a media
summary that counted URLs found rather than files downloaded.
"""

from __future__ import annotations

import re

from src.scraper.amazon import browser_functions
from src.scraper.amazon.botasaurus_output import get_browser_config_for_outputs
from src.scraper.amazon.browser_functions import (
    _build_browser_config,
    describe_debug_display,
)
from src.scraper.base.display import DisplayInfo
from src.utils.outputs_paths import get_project_root


class TestOneClockPerDuration:
    def test_no_monotonic_start_is_subtracted_from_the_wall_clock(self):
        source = (
            get_project_root() / "src/scraper/amazon/browser_functions.py"
        ).read_text()
        monotonic_starts = set(re.findall(r"(\w+)\s*=\s*time\.monotonic\(\)", source))
        assert monotonic_starts, "the module no longer takes monotonic starts"
        mixed = [
            name
            for name in monotonic_starts
            if re.search(rf"time\.time\(\)\s*-\s*{re.escape(name)}\b", source)
        ]
        assert mixed == [], f"wall clock minus a monotonic start: {mixed}"


class TestTheDebugLineReadsTheDisplay:
    def test_headless_chrome_has_no_window(self):
        assert "no window" in describe_debug_display({"headless": "new"})

    def test_a_virtual_display_has_no_visible_window(self):
        line = describe_debug_display(
            {"headless": False, "enable_xvfb_virtual_display": True}
        )
        assert "no visible window" in line

    def test_a_real_display_shows_the_window(self):
        line = describe_debug_display(
            {"headless": False, "enable_xvfb_virtual_display": False}
        )
        assert "visible" in line and "no " not in line

    def test_a_wayland_debug_run_does_not_promise_a_window(self, monkeypatch, caplog):
        monkeypatch.setattr(browser_functions, "detect_monitors", lambda: [])
        monkeypatch.setattr(
            browser_functions,
            "get_optimal_browser_position",
            lambda monitors: (0, 0, 1920, 1080),
        )
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setattr(browser_functions, "_BROWSER_CONFIG", {"headless": False})
        monkeypatch.setattr(
            browser_functions,
            "resolve_debug_display",
            lambda: DisplayInfo(":0", "/run/user/1000/.cookie", "xwayland"),
        )

        with caplog.at_level("INFO"):
            _build_browser_config(debug_mode=True)

        debug_lines = [
            r.message for r in caplog.records if r.message.startswith("Debug mode")
        ]
        assert debug_lines, "no debug-mode line was logged"
        assert all("no visible window" in line for line in debug_lines), debug_lines


class TestBotasaurusWritesNothing:
    def test_the_browser_output_is_disabled(self):
        """The browser function returns per-input envelopes; the save happens
        in `_save_products` after the downloads. A callback here only ever
        warned about envelopes with no ASIN.
        """
        assert get_browser_config_for_outputs() == {"output": None}
