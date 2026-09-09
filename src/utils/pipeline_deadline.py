"""The remaining share of a render's total budget.

`pipeline_timeout_sec` bounds a whole render, and several steps derive their
own limits independently of it. Nothing reconciled the two, so an inner limit
could exceed the outer one: with the shipped settings a 59-second voiceover
earned Whisper 1008s inside a 900s pipeline (#398). The step then finished
inside its own budget, having consumed most of the run's, and the timeout
fired during assembly -- so the failure was attributed to the pipeline rather
than to the step that spent the time, and the retry schedule was widening a
limit that already had nothing behind it.

A `ContextVar` rather than a module global or a threaded parameter, so the
value reaches every step without passing through the four signatures between
here and the STT call -- the same signatures that, on the subtitle engine,
already produced a documented class of silent failure by disagreeing.

Reads propagate downward through everything on this path: measured on the
pinned 3.12 interpreter, the value survives `asyncio.wait_for`, `gather`,
`create_task` and `to_thread`. It is lost only through
`loop.run_in_executor`, which the Whisper subprocess uses -- after the limit
has already been read, so that does not matter here.

The isolation between products comes from each caller setting it inside its
own per-product loop, not from `wait_for`: on 3.12 `wait_for` awaits the
coroutine in the current context rather than wrapping it in a task, so a set
performed inside would leak back out to the caller. Nothing here sets it
downward, so that is a caveat rather than a bug, but do not rely on
`wait_for` for isolation. `create_task` and `gather` do isolate.
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
