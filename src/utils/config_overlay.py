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
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    overlay = overlay_path(path)
    if overlay is None or not overlays_enabled() or not overlay.is_file():
        return data
    try:
        extra = yaml.safe_load(overlay.read_text(encoding="utf-8"))
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
    with _logged_lock:
        first = overlay.resolve() not in _logged
        _logged.add(overlay.resolve())
    if first:
        logger.info("Config overlay applied: %s", overlay)
    return deep_merge(data if data is not None else {}, extra)
