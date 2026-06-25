"""Normal-mode scraper window size stays desktop-width.

`WindowSize.RANDOM` drew narrow/mobile widths, which make Amazon serve a
responsive layout the desktop product-card selectors don't match, silently
yielding 0 products in normal mode (#161). The browser config now randomizes
only among desktop-width sizes (all >= 1280 wide), so the desktop layout always
renders while keeping anti-detection variety.
"""

from src.scraper.amazon import config as scraper_config

MIN_DESKTOP_WIDTH = 1280


def _load_window_size():
    scraper_config.load_browser_config_from_yaml("config/scraper.yaml")
    return scraper_config._BROWSER_CONFIG["window_size"]


def test_window_size_is_desktop_width_across_runs():
    """Every load yields a desktop-width window, never a narrow/RANDOM draw."""
    seen = set()
    for _ in range(50):
        ws = _load_window_size()
        assert ws != "RANDOM", "window_size must not be the unbounded RANDOM draw"
        width, height = ws
        assert width >= MIN_DESKTOP_WIDTH, f"narrow width would trigger mobile: {ws}"
        assert height >= 720
        seen.add((width, height))
    # Randomization is preserved: more than one desktop size appears across runs.
    assert len(seen) > 1
