"""Tests for src/scraper/base/display.py.

Covers display resolution for headful debug runs: an already-set DISPLAY wins, Wayland
falls back to parsing a running Xwayland process, and the no-display case returns
source="none" so the caller can switch to a virtual display.
"""

import pytest

from src.scraper.base import display
from src.scraper.base.display import (
    _normalize_display,
    _parse_xwayland_cmdline,
    resolve_debug_display,
)


class TestParseXwaylandCmdline:
    def test_extracts_display_and_auth(self):
        args = [
            "/usr/bin/Xwayland",
            ":0",
            "-rootless",
            "-auth",
            "/run/user/1000/.mutter-Xwaylandauth.ABC123",
            "-listenfd",
            "4",
        ]
        assert _parse_xwayland_cmdline(args) == (
            ":0",
            "/run/user/1000/.mutter-Xwaylandauth.ABC123",
        )

    def test_no_auth_returns_none_auth(self):
        assert _parse_xwayland_cmdline(["Xwayland", ":1", "-rootless"]) == (":1", None)

    def test_screen_suffix_normalized(self):
        assert _parse_xwayland_cmdline(["Xwayland", ":0.0"]) == (":0", None)


class TestNormalizeDisplay:
    @pytest.mark.parametrize(
        "value,expected",
        [(":0", ":0"), (":0.0", ":0"), (":12", ":12"), ("", None), (None, None)],
    )
    def test_normalize(self, value, expected):
        assert _normalize_display(value) == expected


class TestResolveDebugDisplay:
    def test_prefers_existing_display(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":1")
        monkeypatch.setenv("XAUTHORITY", "/home/u/.Xauthority")
        info = resolve_debug_display()
        assert info.source == "existing"
        assert info.display == ":1"
        assert info.xauthority == "/home/u/.Xauthority"

    def test_existing_display_normalized(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":0.0")
        monkeypatch.delenv("XAUTHORITY", raising=False)
        info = resolve_debug_display()
        assert info.display == ":0"
        assert info.xauthority is None

    def test_wayland_from_xwayland_process(self, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr(
            display,
            "_find_xwayland_cmdline",
            lambda: [
                "Xwayland",
                ":0",
                "-auth",
                "/run/user/1000/.mutter-Xwaylandauth.Z",
            ],
        )
        info = resolve_debug_display()
        assert info.source == "xwayland"
        assert info.display == ":0"
        assert info.xauthority == "/run/user/1000/.mutter-Xwaylandauth.Z"

    def test_wayland_auth_glob_fallback(self, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("XAUTHORITY", raising=False)
        monkeypatch.setattr(
            display, "_find_xwayland_cmdline", lambda: ["Xwayland", ":0", "-rootless"]
        )
        monkeypatch.setattr(
            display,
            "_newest_mutter_cookie",
            lambda: "/run/user/1000/.mutter-Xwaylandauth.G",
        )
        info = resolve_debug_display()
        assert info.source == "xwayland"
        assert info.display == ":0"
        assert info.xauthority == "/run/user/1000/.mutter-Xwaylandauth.G"

    def test_no_display_returns_none(self, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr(display, "_find_xwayland_cmdline", lambda: None)
        info = resolve_debug_display()
        assert info.source == "none"
        assert info.display is None
        assert info.xauthority is None
