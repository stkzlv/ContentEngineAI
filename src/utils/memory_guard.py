"""Wait for memory before a render, and report what the render used.

The lowpri scope caps a render at `MEM_LIMIT` with no swap, which stops the
render growing; it does nothing when the machine is already short. A render
started with 1.5 GB available and swap full tipped the machine into a kernel
OOM, which killed the user's browser rather than the render. So each render
first waits until the machine has its budget free, and gives up after a
bounded wait rather than starting into an OOM.

Linux only: without `/proc/meminfo` the check passes, since it cannot tell.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Skips the wait, for a run the operator knows will fit.
SKIP_ENV = "ALLOW_LOW_MEMORY"
# The producer's exit code when it stopped for memory (EX_TEMPFAIL).
EXIT_NO_MEMORY = 75
_GB = 1024**3


class InsufficientMemoryError(RuntimeError):
    """The machine did not free a render's budget within the wait."""


@dataclass(frozen=True)
class MemoryState:
    available_gb: float
    swap_total_gb: float
    swap_free_gb: float
    # The share of the last 10 s in which every task stalled on memory.
    pressure_full_avg10: float | None


def read_state(
    meminfo: Path = Path("/proc/meminfo"),
    pressure: Path = Path("/proc/pressure/memory"),
) -> MemoryState | None:
    """The machine's memory now; None where the kernel files are missing."""
    try:
        fields = {}
        for line in meminfo.read_text().splitlines():
            name, _, rest = line.partition(":")
            fields[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    if "MemAvailable" not in fields:
        return None
    full = None
    try:
        for line in pressure.read_text().splitlines():
            if line.startswith("full "):
                full = float(line.split("avg10=")[1].split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return MemoryState(
        available_gb=fields["MemAvailable"] / _GB,
        swap_total_gb=fields.get("SwapTotal", 0) / _GB,
        swap_free_gb=fields.get("SwapFree", 0) / _GB,
        pressure_full_avg10=full,
    )


def shortfall(state: MemoryState, settings: Any) -> str | None:
    """Why the machine cannot take a render now, or None when it can."""
    if state.available_gb < settings.min_available_gb:
        return (
            f"{state.available_gb:.1f} GB available, under "
            f"{settings.min_available_gb:g} GB"
        )
    # Swap matters only when RAM headroom is thin: the render cannot swap,
    # but other apps must have somewhere to go as it grows. Swapped pages stay
    # swapped after RAM frees, so full swap with plenty available is no risk.
    # No swap at all is a machine choice, not a full swap.
    need = settings.min_available_gb + settings.min_swap_free_gb
    if state.swap_total_gb > 0 and state.available_gb + state.swap_free_gb < need:
        return (
            f"{state.available_gb:.1f} GB available and "
            f"{state.swap_free_gb:.1f} GB swap free, under {need:g} GB together"
        )
    return None


async def wait_for_memory(settings: Any, label: str) -> None:
    """Return once the render's budget is free; raise after `wait_sec`.

    `label` names the render in the log (the product id).
    """
    if not settings.enabled or os.environ.get(SKIP_ENV) == "1":
        return
    deadline = time.monotonic() + settings.wait_sec
    warned = False
    while True:
        state = read_state()
        if state is None:
            return
        reason = shortfall(state, settings)
        if reason is None:
            if warned:
                logger.info("Memory freed for %s, starting the render", label)
            return
        if time.monotonic() >= deadline:
            raise InsufficientMemoryError(
                f"Not starting {label}: {reason} after waiting "
                f"{settings.wait_sec:g} s (set {SKIP_ENV}=1 to start anyway)"
            )
        if not warned:
            logger.warning(
                "Waiting up to %g s for memory before %s: %s",
                settings.wait_sec,
                label,
                reason,
            )
            warned = True
        await asyncio.sleep(settings.poll_sec)


def _own_cgroup(proc: Path = Path("/proc/self/cgroup")) -> Path | None:
    try:
        for line in proc.read_text().splitlines():
            if line.startswith("0::"):
                return Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    except OSError:
        return None
    return None


def _read_bytes(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def log_peak(label: str, cgroup: Path | None = None) -> None:
    """Log this process's cgroup peak memory, inside a capped scope only.

    The peak covers every child (FFmpeg, Whisper, the caption browser). It
    is a high-water mark: in a batch it is the largest so far, not this
    product's alone.
    """
    cgroup = cgroup or _own_cgroup()
    if cgroup is None:
        return
    peak = _read_bytes(cgroup / "memory.peak")
    if peak is None:
        return
    cap = _read_bytes(cgroup / "memory.max")  # "max" (no cap) reads as None
    if cap is None:
        # Outside a capped scope the cgroup is the terminal's or the
        # editor's, and its peak says nothing about the render.
        return
    logger.info(
        "Memory peak after %s: %.2f GB of a %.1f GB cap (the scope's high-water "
        "mark so far)",
        label,
        peak / _GB,
        cap / _GB,
    )
