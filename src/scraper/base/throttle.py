"""Telling a rate limit apart from a query that never works.

Amazon answers both with the same page, so the scraper logged both with the
same line and retried both the same way:

    Amazon error page detected: Sorry! Something went wrong!

The two want opposite responses. Throttling clears on its own after a few
minutes, so the move is to wait. A query that consistently returns the error
page never clears -- one keyword was observed failing for the better part of
an hour while other keywords succeeded minutes apart in between -- so the move
is to name it and go on to the next input.

Nothing in a single failure separates them. What does is what happens to the
*other* inputs in the same run: a rate limit blocks the connection, so nothing
else gets through either, while a dead query is specific to itself and its
neighbours keep working. `ThrottleTracker` is that observation, kept as run
state: an input that keeps failing while some other input got through is dead,
and one that fails with nothing else succeeding is throttled and worth waiting
for.

The evidence is deliberately not ordered. Requiring the other input's success
to come *after* the failure reads better -- it rules out a rate limit that
began in between -- but it cannot rule on the last input in a run, which is
where a dead query is as likely to sit as anywhere else. So a success before
the failure counts too, and `dead_query_after` is the guard instead: the
default of 3 means two backoffs have already elapsed, several minutes, before
anything is called dead.

The residual case is a rate limit that begins partway through a run and blocks
only the tail. Its first tail input is waited on and then called dead. What it
costs is the label, not the run: every later input is throttled too, so the
summary reports a run that mostly failed rather than one dead keyword, which
is the reading an operator needs.

The tracker holds no browser and sleeps for nobody. It answers what to do and
for how long; the caller waits. Both scraping paths use it -- the standalone
scraper's per-input retry and the batch's single-session loop -- because they
re-implement each other and would otherwise drift.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# The one spelling of the sentinel. Two sites raise it and several match it;
# they were three separate string literals, so a change to Amazon's wording
# could have been applied to the raise and not to the match, which reads as
# the throttle handling having quietly stopped working.
AMAZON_ERROR_PAGE_TITLE = "Sorry! Something went wrong!"
AMAZON_ERROR_PAGE_MESSAGE = f"Amazon error page detected: {AMAZON_ERROR_PAGE_TITLE}"


def is_error_page_failure(error: BaseException | str) -> bool:
    """Whether this failure is Amazon's error page rather than anything else.

    Matches on the marker rather than the whole message so a caller may add
    context to the exception without the classification falling through to
    "some other error", which is not retried at all.
    """
    return "Amazon error page detected" in str(error)


class Verdict(Enum):
    """What to do about an input that just failed."""

    RETRY = "retry"
    """Throttled as far as anything can tell. Wait, then try the same input."""

    DEAD_QUERY = "dead_query"
    """Other inputs are getting through, so this one is the problem. Skip it."""

    EXHAUSTED = "exhausted"
    """Still failing after the whole retry budget. Give up on this input."""


@dataclass(frozen=True)
class ThrottleSettings:
    """How long to wait and how many times.

    Defaults come from what was measured against Amazon rather than from a
    round number: spacing runs roughly eight minutes apart cleared the block,
    so the ceiling sits above that and the schedule reaches it within the
    budget.
    """

    inter_input_delay_sec: tuple[float, float] = (2.0, 5.0)
    """Jittered pause between consecutive inputs, on the happy path too."""

    backoff_base_sec: float = 60.0
    backoff_max_sec: float = 600.0
    max_attempts: int = 5
    dead_query_after: int = 3
    """Error pages for one input, in a run where something else got through.

    3 rather than 1 because the two backoffs it implies are the wait that
    separates the cases: a rate limit affecting only this input would have had
    several minutes to clear before the verdict lands.
    """

    def __post_init__(self) -> None:
        low, high = self.inter_input_delay_sec
        if low < 0 or high < low:
            raise ValueError(
                "inter_input_delay_sec must be a non-negative [min, max] pair, "
                f"got {self.inter_input_delay_sec!r}"
            )
        if self.backoff_base_sec <= 0:
            raise ValueError("backoff_base_sec must be positive")
        if self.backoff_max_sec < self.backoff_base_sec:
            raise ValueError("backoff_max_sec must be at least backoff_base_sec")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.dead_query_after < 1:
            raise ValueError("dead_query_after must be at least 1")

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None) -> ThrottleSettings:
        """Build from the `rate_limiting` block, falling back per field.

        A malformed value is reported and the whole block is discarded rather
        than half-applied: a run that silently kept two of four settings is
        harder to diagnose than one that says it is using the defaults.
        """
        if not raw:
            return cls()
        try:
            delay = raw.get("inter_input_delay_sec")
            return cls(
                inter_input_delay_sec=(
                    (float(delay[0]), float(delay[1]))
                    if isinstance(delay, list | tuple) and len(delay) == 2
                    else cls.inter_input_delay_sec
                ),
                backoff_base_sec=float(
                    raw.get("throttle_backoff_base_sec", cls.backoff_base_sec)
                ),
                backoff_max_sec=float(
                    raw.get("throttle_backoff_max_sec", cls.backoff_max_sec)
                ),
                max_attempts=int(raw.get("throttle_max_attempts", cls.max_attempts)),
                dead_query_after=int(raw.get("dead_query_after", cls.dead_query_after)),
            )
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Invalid rate_limiting throttle settings (%s); using defaults", exc
            )
            return cls()


@dataclass
class _InputState:
    error_pages: int = 0
    """Error pages seen for this input in this run."""


@dataclass
class ThrottleTracker:
    """Run-level memory of which inputs failed and what succeeded in between.

    One tracker per run. It is deliberately not global: two runs sharing it
    would let the first run's successes mark the second run's first failure as
    a dead query, which is the misclassification this exists to prevent.
    """

    settings: ThrottleSettings = field(default_factory=ThrottleSettings)
    _states: dict[str, _InputState] = field(default_factory=dict, init=False)
    _succeeded: list[str] = field(default_factory=list, init=False)
    _dead: list[str] = field(default_factory=list, init=False)
    _throttled: list[str] = field(default_factory=list, init=False)

    # -- outcomes -----------------------------------------------------------

    def record_success(self, input_label: str) -> None:
        """Note that an input got through.

        The succeeding input's own failures are cleared: a run that was
        throttled and recovered should not carry them into a later decision
        about the same input.
        """
        self._states[input_label] = _InputState()
        if input_label not in self._succeeded:
            self._succeeded.append(input_label)

    def _another_input_got_through(self, input_label: str) -> bool:
        """Whether anything other than this input succeeded in this run.

        The connection is the thing being tested. One success anywhere in the
        run says it works, which leaves the failing input itself as the
        explanation.
        """
        return any(label != input_label for label in self._succeeded)

    def record_error_page(self, input_label: str) -> Verdict:
        """Note an error page and say what to do about it.

        The order of the two checks matters. A dead query is decided first, so
        an input the run has evidence against is skipped rather than spending
        the rest of its budget on waits that cannot help.
        """
        state = self._states.setdefault(input_label, _InputState())
        state.error_pages += 1

        if state.error_pages >= self.settings.dead_query_after and (
            self._another_input_got_through(input_label)
        ):
            if input_label not in self._dead:
                self._dead.append(input_label)
            logger.warning(
                "Query is dead, not throttled: %r has returned Amazon's error "
                "page %d times in a run where other inputs got through. "
                "Skipping it rather than waiting again.",
                input_label,
                state.error_pages,
            )
            return Verdict.DEAD_QUERY

        if state.error_pages >= self.settings.max_attempts:
            if input_label not in self._throttled:
                self._throttled.append(input_label)
            logger.warning(
                "Giving up on %r after %d error pages. Nothing else has "
                "succeeded either, so this reads as a rate limit that "
                "outlasted the retry budget rather than a bad query.",
                input_label,
                state.error_pages,
            )
            return Verdict.EXHAUSTED

        if input_label not in self._throttled:
            self._throttled.append(input_label)
        return Verdict.RETRY

    # -- waits --------------------------------------------------------------

    def backoff_sec(self, input_label: str) -> float:
        """How long to wait before retrying this input.

        Doubles per error page seen for the input and stops at the ceiling.
        The ceiling matters more than the growth: the observed block cleared
        after several minutes, so a schedule that tops out in seconds returns
        to a still-blocked Amazon and burns the budget without waiting.
        """
        seen = self._states.get(input_label, _InputState()).error_pages
        exponent = max(0, seen - 1)
        raw = self.settings.backoff_base_sec * (2.0**exponent)
        return min(raw, self.settings.backoff_max_sec)

    def inter_input_delay_sec(self) -> float:
        """A jittered pause to put between two consecutive inputs."""
        low, high = self.settings.inter_input_delay_sec
        return random.uniform(low, high)  # noqa: S311 - pacing, not cryptography

    # -- reporting ----------------------------------------------------------

    @property
    def dead_queries(self) -> list[str]:
        """Inputs the run has evidence against, in the order they were named."""
        return list(self._dead)

    @property
    def throttled_inputs(self) -> list[str]:
        """Inputs that hit an error page without being ruled dead.

        An input that later succeeded is dropped: it was throttled and the
        wait worked, which is the mechanism doing its job rather than a
        problem to report.
        """
        return [label for label in self._throttled if label not in self._succeeded]

    def summary_lines(self) -> list[str]:
        """Lines for the end-of-run summary, empty when nothing went wrong."""
        lines = []
        if self._dead:
            lines.append(
                "Dead queries (returned Amazon's error page while other "
                f"inputs succeeded): {', '.join(self._dead)}"
            )
        stuck = self.throttled_inputs
        if stuck:
            lines.append(
                "Rate-limited and not recovered within the retry budget: "
                f"{', '.join(stuck)}"
            )
        return lines
