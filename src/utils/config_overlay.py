"""Private config overlays (design 0026).

`config/<name>.private.yaml`, when present, is merged over `config/<name>.yaml`
for every reader of that file, so an installation keeps its own settings out
of the tracked defaults. Git ignores `*.private.yaml`.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

OVERLAY_SUFFIX = ".private.yaml"
# Machine setting: `off` reads the shipped files alone (the test suite does).
OVERLAYS_ENV = "CONFIG_OVERLAYS"

_logged: set[Path] = set()
# Applied before any handler could show the line; logging setup flushes them.
_pending: list[Path] = []
_logged_lock = threading.Lock()


class ConfigOverlayError(ValueError):
    """An overlay that does not parse or is not a mapping."""


def overlay_path(path: Path) -> Path | None:
    """The overlay for `path`, or None for a file that is itself an overlay."""
    if path.name.endswith(OVERLAY_SUFFIX):
        return None
    return path.with_name(path.stem + OVERLAY_SUFFIX)


def overlays_enabled() -> bool:
    return os.environ.get(OVERLAYS_ENV, "").strip().lower() != "off"


def deep_merge(base: Any, overlay: Any) -> Any:
    """`overlay` over `base`: mappings merge key by key, anything else replaces."""
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = deep_merge(base[key], value) if key in base else value
        return merged
    return overlay


def read_config_yaml(path: Path | str) -> Any:
    """Parse the YAML at `path` and apply its overlay.

    Raises what reading and parsing `path` raises, so each caller keeps its
    own handling of a missing or broken shipped file. A broken overlay raises
    ConfigOverlayError naming it.
    """
    path = Path(path)
    # From a stream, so a parse error names the file.
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    overlay = overlay_path(path)
    if overlay is None or not overlays_enabled() or not overlay.is_file():
        return data
    try:
        with overlay.open(encoding="utf-8") as handle:
            extra = yaml.safe_load(handle)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigOverlayError(
            f"Config overlay {overlay} is unreadable: {exc}"
        ) from exc
    if extra is None:
        return data
    if not isinstance(extra, dict):
        raise ConfigOverlayError(
            f"Config overlay {overlay} is not a mapping "
            f"(parsed as {type(extra).__name__})"
        )
    _log_applied(overlay)
    return deep_merge(data if data is not None else {}, extra)


def _can_show() -> bool:
    return logger.isEnabledFor(logging.INFO) and logger.hasHandlers()


def _log_applied(overlay: Path) -> None:
    """Log an overlay once; before logging is set up, hold it for log_pending.

    A read at import time (the circuit breakers) may be the only read of its
    file in a run, so a line logged then, with no handler, would be lost.
    """
    key = overlay.resolve()
    with _logged_lock:
        if key in _logged:
            return
        if not _can_show():
            if overlay not in _pending:
                _pending.append(overlay)
            return
        _logged.add(key)
        if overlay in _pending:
            _pending.remove(overlay)
    logger.info("Config overlay applied: %s", overlay)


def log_pending() -> None:
    """Log the overlays applied before logging was set up. Call after setup."""
    if not _can_show():
        return
    with _logged_lock:
        held = [o for o in _pending if o.resolve() not in _logged]
        _pending.clear()
        _logged.update(o.resolve() for o in held)
    for overlay in held:
        logger.info("Config overlay applied: %s", overlay)
