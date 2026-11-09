"""One returning browser across runs, behind `consistent_identity` (REQ-SCR-081).

Without it every scrape is a new visitor: no cookies, a random window size,
and a random user agent from the driver library's list, which stops at
Chrome 120 while the installed browser is newer and its client hints still
report the real version. With the switch on, the browser keeps its own user
agent, the window size is drawn once and stored, and cookies carry from one
run to the next. The files live under the outputs state directory, one
identity per installation.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from src.utils.outputs_paths import durable_state_path, resolve_outputs_dir

logger = logging.getLogger(__name__)

IDENTITY_FILE = "scraper_identity.json"
COOKIES_FILE = "scraper_cookies.json"

# Desktop widths only (REQ-SCR-041): a narrow window gets the mobile layout,
# which the card selectors do not match.
DESKTOP_WINDOW_SIZES: tuple[tuple[int, int], ...] = (
    (1280, 720),
    (1366, 768),
    (1440, 900),
    (1536, 864),
    (1600, 900),
    (1920, 1080),
)

# What the driver library reads as "send the browser's own user agent".
REAL_USER_AGENT = "REAL"

# Asked once per session when the switch is on, so the log shows the user
# agent and the client hints agree.
_USER_AGENT_JS = """
const brands = (navigator.userAgentData && navigator.userAgentData.brands) || [];
return {
    user_agent: navigator.userAgent,
    hints: brands.map(b => b.brand + '/' + b.version).join(', '),
};
"""


def _state_path(filename: str, outputs_dir: Path | None) -> Path:
    return durable_state_path(outputs_dir or resolve_outputs_dir(None), filename)


def window_size(outputs_dir: Path | None = None) -> tuple[int, int]:
    """The profile's window size, drawn on first use and kept after that."""
    path = _state_path(IDENTITY_FILE, outputs_dir)
    try:
        stored = json.loads(path.read_text())
        width, height = (int(v) for v in stored["window_size"])
        if width >= DESKTOP_WINDOW_SIZES[0][0]:
            return width, height
        logger.warning(
            "Stored window %sx%s is not desktop width; redrawing", width, height
        )
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Unreadable scraper identity %s (%s); redrawing", path, exc)
    size = random.choice(DESKTOP_WINDOW_SIZES)  # noqa: S311 - not security
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"window_size": list(size)}))
    logger.info("Drew the scraper's window size %sx%s, kept in %s", *size, path)
    return size


def apply_to_browser_config(
    config: dict[str, Any], outputs_dir: Path | None = None
) -> dict[str, Any]:
    """The browser options for the returning identity, over the run's own."""
    config = dict(config)
    config["user_agent"] = REAL_USER_AGENT
    config["window_size"] = list(window_size(outputs_dir))
    return config


def load_cookies(driver: Any, outputs_dir: Path | None = None) -> int:
    """Give the session the cookies the last one left; the count loaded."""
    path = _state_path(COOKIES_FILE, outputs_dir)
    try:
        cookies = json.loads(path.read_text())
    except FileNotFoundError:
        return 0
    except (OSError, ValueError) as exc:
        logger.warning(
            "Unreadable scraper cookies %s (%s); starting without", path, exc
        )
        return 0
    if not isinstance(cookies, list) or not cookies:
        return 0
    try:
        driver.add_cookies(cookies)
    except Exception as exc:  # noqa: BLE001 - a session without cookies still runs
        logger.warning("Could not load %d scraper cookies: %s", len(cookies), exc)
        return 0
    return len(cookies)


_forgotten_at = float("-inf")


def forget_cookies(outputs_dir: Path | None = None) -> None:
    """Drop the kept cookies after a captcha or bare-title page.

    The block travels with them: right after blocked sessions that had
    loaded the kept cookies, a session without them got through. The
    blocked session's own cookies are not saved either (`forgotten_since`).
    """
    global _forgotten_at
    _forgotten_at = time.monotonic()
    path = _state_path(COOKIES_FILE, outputs_dir)
    try:
        path.unlink()
        logger.warning("Dropped the kept scraper cookies after a block page")
    except FileNotFoundError:
        pass


def forgotten_since(started: float) -> bool:
    """Whether the cookies were dropped after `started` (monotonic)."""
    return _forgotten_at >= started


def save_cookies(driver: Any, outputs_dir: Path | None = None) -> int:
    """Keep the session's cookies for the next run; the count saved."""
    try:
        cookies = driver.get_cookies()
    except Exception as exc:  # noqa: BLE001 - a closed session keeps the old file
        logger.warning("Could not read the session's cookies: %s", exc)
        return 0
    if isinstance(cookies, dict):
        cookies = cookies.get("cookies")
    if not isinstance(cookies, list):
        return 0
    path = _state_path(COOKIES_FILE, outputs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cookies))
    tmp.replace(path)
    return len(cookies)


def log_user_agent(driver: Any) -> None:
    """Log the user agent beside the client hints, which should agree."""
    try:
        seen = driver.run_js(_USER_AGENT_JS)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        logger.debug("Could not read the user agent: %s", exc)
        return
    if isinstance(seen, dict):
        logger.info(
            "Browser identity: user agent %s; client hints %s",
            seen.get("user_agent"),
            seen.get("hints") or "none",
        )
