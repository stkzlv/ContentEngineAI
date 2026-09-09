"""The remaining share of a render's total budget.

`pipeline_timeout_sec` bounds a whole render, and several steps derive their
own limits independently of it. Nothing reconciled the two, so an inner limit
could exceed the outer one: with the shipped settings a 59-second voiceover
earned Whisper 1007s inside a 900s pipeline (#398). The step then finished
inside its own budget, having consumed most of the run's, and the timeout
fired during assembly -- so the failure was attributed to the pipeline rather
than to the step that spent the time, and the retry schedule was widening a
limit that already had nothing behind it.

A `ContextVar` rather than a module global or a threaded parameter. The
caller that applies the pipeline timeout sets it, and `asyncio.wait_for`
copies the context into the task it creates, so the value reaches every step
without passing through the four signatures between here and the STT call --
the same signatures that, on the subtitle engine, already produced a
documented class of silent failure by disagreeing. Per-task rather than
per-process, so two products rendered concurrently do not share one deadline.
"""

from __future__ import annotations

import time
from contextvars import ContextVar

__all__ = [
    "clear_pipeline_deadline",
    "remaining_pipeline_seconds",
    "set_pipeline_deadline",
]

_DEADLINE: ContextVar[float | None] = ContextVar("pipeline_deadline", default=None)


def set_pipeline_deadline(budget_sec: float | None) -> None:
    """Start the clock for a render allowed ``budget_sec`` in total.

    A non-positive or absent budget clears the deadline, which reads as "no
    outer bound" rather than "no time left": a caller that does not apply a
    pipeline timeout must not thereby cap every step at zero.
    """
    if budget_sec is None or budget_sec <= 0:
        _DEADLINE.set(None)
        return
    _DEADLINE.set(time.monotonic() + budget_sec)


def clear_pipeline_deadline() -> None:
    """Forget the deadline, for a caller that applies no pipeline timeout."""
    _DEADLINE.set(None)


def remaining_pipeline_seconds() -> float | None:
    """Seconds left of the render's budget, or None when none was set.

    Never negative: a budget already spent returns 0.0, which a caller should
    read as "no room" rather than as "unset".
    """
    deadline = _DEADLINE.get()
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())
