"""Tests for _build_browser_config display/Xvfb wiring in browser_functions.py.

Normal runs use headed mode under Botasaurus's Xvfb virtual display because this
Botasaurus version raises StopIteration in headless mode. When headless="new" is
configured explicitly, no display is requested. Debug runs must resolve a real display
via resolve_debug_display and set DISPLAY/XAUTHORITY, falling back to the virtual
display when no real display exists.
"""

from src.scraper.amazon import browser_functions
from src.scraper.amazon.browser_functions import _build_browser_config
from src.scraper.base.display import DisplayInfo


class TestNormalMode:
    def test_headed_mode_uses_xvfb(self, monkeypatch):
        # Default config uses headed mode under Xvfb (Botasaurus headless bug).
        monkeypatch.delenv("DISPLAY", raising=False)
        cfg = _build_browser_config(debug_mode=False)
        assert cfg["headless"] is False
        assert cfg["enable_xvfb_virtual_display"] is True
        assert "DISPLAY" not in __import__("os").environ

    def test_modern_headless_skips_xvfb(self, monkeypatch):
        # When headless="new" is configured explicitly, no virtual display is needed.
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr(browser_functions, "_BROWSER_CONFIG", {"headless": "new"})
        cfg = _build_browser_config(debug_mode=False)
        assert cfg["headless"] == "new"
        assert "enable_xvfb_virtual_display" not in cfg
        assert "DISPLAY" not in __import__("os").environ

    def test_forces_x11_ozone(self, monkeypatch):
        # Chrome must use the X11 backend, or on Wayland it draws a real window even in
        # normal mode (libwayland defaults to the wayland-0 socket).
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.delenv("DISPLAY", raising=False)
        cfg = _build_browser_config(debug_mode=False)
        assert "--ozone-platform=x11" in cfg["add_arguments"]

    def test_warns_when_xvfb_missing(self, monkeypatch, caplog):
        monkeypatch.setattr(browser_functions, "_BROWSER_CONFIG", {"headless": False})
        monkeypatch.setattr(browser_functions.shutil, "which", lambda _: None)
        with caplog.at_level("WARNING"):
            cfg = _build_browser_config(debug_mode=False)
        assert cfg["enable_xvfb_virtual_display"] is True
        assert any("xvfb" in r.message.lower() for r in caplog.records)


class TestDebugMode:
    def _patch_monitors(self, monkeypatch):
        monkeypatch.setattr(browser_functions, "detect_monitors", lambda: [])
        monkeypatch.setattr(
            browser_functions,
            "get_optimal_browser_position",
            lambda monitors: (0, 0, 1920, 1080),
        )

    def test_sets_real_display(self, monkeypatch):
        # No live Wayland session (real X11 desktop, or the dedicated Xvfb that
        # `make scrape-watch` provides): debug uses the resolved real display.
        self._patch_monitors(monkeypatch)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setattr(browser_functions, "_BROWSER_CONFIG", {"headless": False})
        monkeypatch.setattr(
            browser_functions,
            "resolve_debug_display",
            lambda: DisplayInfo(":0", "/run/user/1000/.cookie", "existing"),
        )
        cfg = _build_browser_config(debug_mode=True)
        import os

        assert os.environ["DISPLAY"] == ":0"
        assert os.environ["XAUTHORITY"] == "/run/user/1000/.cookie"
        assert cfg["enable_xvfb_virtual_display"] is False

    def test_wayland_session_routes_to_xvfb(self, monkeypatch, caplog):
        # A headful window on a live Wayland session freezes Chromium's CDP, so
        # debug must fall back to a virtual Xvfb display instead of the live one.
        self._patch_monitors(monkeypatch)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setattr(browser_functions, "_BROWSER_CONFIG", {"headless": False})
        monkeypatch.setattr(
            browser_functions,
            "resolve_debug_display",
            lambda: DisplayInfo(":0", "/run/user/1000/.cookie", "xwayland"),
        )
        with caplog.at_level("WARNING"):
            cfg = _build_browser_config(debug_mode=True)
        assert cfg["enable_xvfb_virtual_display"] is True
        assert any("wayland" in r.message.lower() for r in caplog.records)

    def test_none_source_falls_back_to_virtual(self, monkeypatch, caplog):
        self._patch_monitors(monkeypatch)
        monkeypatch.setattr(browser_functions, "_BROWSER_CONFIG", {"headless": False})
        monkeypatch.setattr(
            browser_functions,
            "resolve_debug_display",
            lambda: DisplayInfo(None, None, "none"),
        )
        with caplog.at_level("WARNING"):
            cfg = _build_browser_config(debug_mode=True)
        assert cfg["enable_xvfb_virtual_display"] is True
        assert any("display" in r.message.lower() for r in caplog.records)

    def test_modern_headless_skips_display(self, monkeypatch):
        # Chrome's modern headless mode needs no display even in debug mode.
        self._patch_monitors(monkeypatch)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr(browser_functions, "_BROWSER_CONFIG", {"headless": "new"})
        cfg = _build_browser_config(debug_mode=True)
        assert cfg["headless"] == "new"
        assert "enable_xvfb_virtual_display" not in cfg
        assert "DISPLAY" not in __import__("os").environ
