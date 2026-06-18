"""Display resolution for headful browser runs on Wayland and X11.

Botasaurus runs Chrome headful (its headless mode is detectable and crash-prone).
Headful Chrome needs an X display. On X11 the session exports ``DISPLAY``; on Wayland
it does not, even though an Xwayland server is usually running on ``:0`` behind the
compositor.

This module resolves a usable display for debug (visible-window) runs only. Normal runs
use Botasaurus's own Xvfb virtual display and never call in here. Everything is pure and
never raises: callers get a ``DisplayInfo`` and decide what to do, including the
``source="none"`` case where no real display exists.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Xwayland advertises ":0"; an optional screen suffix like ":0.0" is dropped.
_DISPLAY_RE = re.compile(r"^(:\d+)(?:\.\d+)?$")


@dataclass
class DisplayInfo:
    """A resolved display for a headful browser run.

    ``source`` is "existing" when ``$DISPLAY`` was already set (X11, SSH X11, exported
    sessions), "xwayland" when found from a running Xwayland process, or "none" when
    nothing usable exists.
    """

    display: str | None
    xauthority: str | None
    source: str


def _normalize_display(value: str | None) -> str | None:
    """Strip a screen suffix so ":0.0" becomes ":0"; return None if unparseable."""
    if not value:
        return None
    match = _DISPLAY_RE.match(value.strip())
    return match.group(1) if match else value.strip() or None


def _parse_xwayland_cmdline(args: list[str]) -> tuple[str | None, str | None]:
    """Extract (display, auth_path) from an Xwayland argv. Pure: no I/O.

    Xwayland launches as ``Xwayland :0 ... -auth <run-dir>/.mutter-Xwaylandauth.X``. The
    display is the first ``:N`` token; the auth file follows ``-auth``.
    """
    display: str | None = None
    xauthority: str | None = None
    for i, arg in enumerate(args):
        if display is None and _DISPLAY_RE.match(arg):
            display = _normalize_display(arg)
        elif arg == "-auth" and i + 1 < len(args):
            xauthority = args[i + 1]
    return display, xauthority


def _iter_proc_cmdlines() -> list[list[str]]:
    """Read every process argv from /proc. Returns argv lists; missing /proc -> []."""
    cmdlines: list[list[str]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return cmdlines
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        args = [part.decode("utf-8", "replace") for part in raw.split(b"\x00") if part]
        if args:
            cmdlines.append(args)
    return cmdlines


def _find_xwayland_cmdline() -> list[str] | None:
    """Return the first running Xwayland process argv, or None."""
    for args in _iter_proc_cmdlines():
        if os.path.basename(args[0]) == "Xwayland":
            return args
    return None


def _newest_mutter_cookie() -> str | None:
    """Fallback cookie: newest /run/user/<uid>/.mutter-Xwaylandauth.* file, or None."""
    run_dir = Path(f"/run/user/{os.getuid()}")
    cookies = sorted(
        run_dir.glob(".mutter-Xwaylandauth.*"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    return str(cookies[0]) if cookies else None


def resolve_debug_display() -> DisplayInfo:
    """Resolve a real display for a visible debug browser run.

    Order: an already-set ``$DISPLAY`` wins (X11, SSH X11, Ubuntu 22, or exported).
    Else parse a running Xwayland process for its display and auth cookie, falling back
    to the newest mutter cookie when the cmdline has no ``-auth``. Returns source "none"
    if neither yields a display, so the caller can switch to a virtual display.
    """
    existing = _normalize_display(os.environ.get("DISPLAY"))
    if existing:
        return DisplayInfo(
            display=existing,
            xauthority=os.environ.get("XAUTHORITY") or None,
            source="existing",
        )

    args = _find_xwayland_cmdline()
    if args:
        display, xauthority = _parse_xwayland_cmdline(args)
        if display:
            if not xauthority:
                xauthority = os.environ.get("XAUTHORITY") or _newest_mutter_cookie()
            return DisplayInfo(
                display=display, xauthority=xauthority, source="xwayland"
            )

    return DisplayInfo(display=None, xauthority=None, source="none")
