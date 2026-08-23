"""Day-N views and a durability ratio per published post.

Two questions need two different windows, and answering both with one gives a
confidently wrong result. Launch performance is a short window, because most of
a short-form post's lifetime views arrive within a day or two. Durability is
whether a post keeps earning afterwards, which a seven-day window cannot see at
all: at day 7 a post that accumulated search traffic and one that spiked and
stopped look the same.

Any claim that a content format is evergreen rests on the second, so it is the
number that settles a format comparison rather than merely describing it.

The scheduler's timeline is **cumulative**: each row carries total views as of
that date, not that day's delta. So a day-N figure is a lookup rather than a
sum, and reading it as a delta understates every figure after the first day.

The rich analytics methods exist on the SDK's generated resource but are not
wired onto the client, so `timeline_resource` reaches them in one place instead
of every call site importing from a private module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Views after this many days count as durable rather than launch traffic.
DURABILITY_WINDOW_DAYS = 30

# The launch figures worth keeping. Day 2 is most of the curve; day 7 is where
# it has essentially finished.
LAUNCH_DAYS = (2, 7)


@dataclass
class PostMetrics:
    """Per-post view history, reduced to the figures a comparison needs."""

    post_id: str
    published_at: str = ""
    views_day_2: int | None = None
    views_day_7: int | None = None
    views_total: int | None = None
    # Views after the first 30 days over views within them. At or above 1.0 the
    # post earned more attention later than at launch. None when the post is
    # not yet old enough to say, which is not the same as 0.0 and must not be
    # ranked alongside it.
    durability_ratio: float | None = None
    # How far the timeline reached. A ratio measured one day past the window is
    # not comparable with one measured six months past it, and without this the
    # stored number cannot be told apart later.
    timeline_end: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def timeline_resource(client: Any) -> Any:
    """The SDK's analytics resource that actually carries the timeline calls.

    `client.analytics` exposes four methods; the timeline, per-platform
    analytics and content-decay calls live on the generated resource, which the
    client never attaches. Reaching into it is contained here so a future SDK
    release that wires them up is a one-line change.
    """
    from late.resources._generated.analytics import AnalyticsResource

    return AnalyticsResource(client)


def _parse_date(value: Any) -> datetime | None:
    """Parse an API timestamp, tolerating the shapes the scheduler returns."""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def _instant(value: Any) -> datetime | None:
    """Parse a timestamp as a point in time, keeping its offset.

    `_parse_date` deliberately drops the offset so timeline rows compare
    against a naive publication date. That is the wrong reading for ordering
    two legs against each other: 09:12+02:00 is earlier than 08:30Z, and
    dropping the offset reverses them.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def normalize_timeline(rows: Any) -> list[tuple[datetime, int]]:
    """Reduce an API timeline to sorted (date, cumulative views) pairs.

    Rows missing a usable date or view count are dropped rather than defaulted:
    a zero would be indistinguishable from a real zero and would drag a
    durability ratio toward nothing.
    """
    out: list[tuple[datetime, int]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        when = _parse_date(row.get("date") or row.get("timestamp"))
        views = row.get("views")
        if when is None or not isinstance(views, int | float):
            continue
        out.append((when, int(views)))
    return sorted(out, key=lambda pair: pair[0])


def views_at_day(
    timeline: list[tuple[datetime, int]], published_at: datetime, day: int
) -> int | None:
    """Cumulative views as of `day` days after publication.

    Takes the last row at or before the cutoff, because the timeline is
    cumulative and may skip days. Returns None when the post had not reached
    that age by the last row, so an unfinished window is never reported as a
    small number.
    """
    if not timeline:
        return None
    # Compared as datetimes, not timestamps: `.timestamp()` on a naive value
    # converts through local-time rules, so across a DST transition the two
    # conversions disagree by an hour and the cutoff moves past a row.
    cutoff = published_at + timedelta(days=day)
    if timeline[-1][0] < cutoff:
        return None
    latest: int | None = None
    for when, views in timeline:
        if when <= cutoff:
            latest = views
        else:
            break
    return latest


def durability_ratio(
    timeline: list[tuple[datetime, int]],
    published_at: datetime,
    window_days: int = DURABILITY_WINDOW_DAYS,
) -> float | None:
    """Views after the first `window_days` over views within them.

    None when the post is not yet old enough to have an "after", and None when
    it earned nothing in the window: a ratio against zero is undefined, and
    returning 0.0 would rank an unmeasurable post alongside a genuinely dead
    one.
    """
    if not timeline:
        return None
    cutoff = published_at + timedelta(days=window_days)
    if timeline[-1][0] <= cutoff:
        # Every row is inside the window, so "after" is empty. Returning 0.0
        # here would report a post measured too early as one that stopped
        # earning, which is the opposite conclusion.
        return None
    within = views_at_day(timeline, published_at, window_days)
    if within is None or within <= 0:
        return None
    total = timeline[-1][1]
    return (total - within) / within


def publish_time(post: dict[str, Any]) -> str:
    """When the post first went live, falling back to its scheduled slot.

    The day-N clock has to start when the content existed. `scheduledFor` is
    only that for a post whose legs all published on time; a leg that failed
    and was retried can go live days later, and measuring from the slot then
    starts the clock before the video existed and understates every figure.

    Retry rate is not random with respect to content format, so this biases a
    format comparison rather than adding noise.

    The earliest leg is used when legs published on different days, since that
    is when the content first reached anyone.
    """
    times = [
        leg.get("publishedAt")
        for leg in (post.get("platforms") or [])
        if isinstance(leg, dict) and leg.get("publishedAt")
    ]
    # Ordered by instant, not by the printed string. Two legs with different
    # UTC offsets sort the wrong way lexically, and the API returns whatever
    # offset the platform reported.
    dated = [(_instant(t), t) for t in times]
    usable = [(d, t) for d, t in dated if d is not None]
    if usable:
        return str(min(usable, key=lambda pair: pair[0])[1])
    if times:
        return str(times[0])
    return str(post.get("publishedAt") or post.get("scheduledFor") or "")


def summarize_post(post_id: str, published_at: Any, rows: Any) -> PostMetrics:
    """Reduce one post's raw timeline to the figures worth storing."""
    when = _parse_date(published_at)
    timeline = normalize_timeline(rows)
    if when is None or not timeline:
        return PostMetrics(post_id=post_id, published_at=str(published_at or ""))
    return PostMetrics(
        post_id=post_id,
        published_at=when.isoformat(),
        views_day_2=views_at_day(timeline, when, LAUNCH_DAYS[0]),
        views_day_7=views_at_day(timeline, when, LAUNCH_DAYS[1]),
        views_total=timeline[-1][1],
        durability_ratio=durability_ratio(timeline, when),
        timeline_end=timeline[-1][0].isoformat(),
    )


def rank_by_durability(metrics: list[PostMetrics]) -> list[PostMetrics]:
    """Most durable first, with unmeasurable posts last.

    A post too young to have a ratio is not a bad post, so it sorts after every
    measured one rather than being treated as a zero.
    """
    measured = [m for m in metrics if m.durability_ratio is not None]
    unmeasured = [m for m in metrics if m.durability_ratio is None]
    measured.sort(key=lambda m: m.durability_ratio or 0.0, reverse=True)
    return measured + unmeasured


METRICS_FILENAME = "post_metrics.json"


def metrics_path(outputs_dir: Path) -> Path:
    """Where the per-post figures live, beside the other tracking files."""
    return outputs_dir / METRICS_FILENAME


def load_metrics(outputs_dir: Path) -> list[PostMetrics]:
    """Read stored per-post figures. Missing or unreadable reads as empty."""
    path = metrics_path(outputs_dir)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [PostMetrics(**row) for row in raw]
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to load post metrics: %s", exc)
        return []


def save_metrics(metrics: list[PostMetrics], outputs_dir: Path) -> None:
    """Write per-post figures, merging each post's row field by field.

    Not "the later reading wins": past the provider's retention horizon a
    later reading has *less* history behind it, and its absent day-N figures
    would erase ones captured while they were still reachable.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_path(outputs_dir)
    if path.exists() and not _readable(path):
        # Merging onto an empty read would drop every post outside this sweep,
        # unrecoverably. The sibling tracking files guard their writes the same
        # way; this one refuses rather than repairing.
        raise OSError(f"Refusing to overwrite unreadable metrics file: {path}")

    merged = {m.post_id: m for m in load_metrics(outputs_dir)}
    for m in metrics:
        stored = merged.get(m.post_id)
        merged[m.post_id] = _combine(stored, m) if stored else m

    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps([m.to_dict() for m in merged.values()], indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def _combine(stored: PostMetrics, fresh: PostMetrics) -> PostMetrics:
    """Keep the best answer per field rather than the newest row.

    The provider's timeline does not reach back indefinitely: measured against
    the live API, every post older than about five weeks returns rows starting
    at the same recent date, whatever `from_date` is passed.

    Rows are lifetime-cumulative -- each carries total views as of its date,
    not views within the window -- so a truncated reading is never *wrong*,
    only incomplete. `views_total` stays correct, while any day-N earlier than
    the window start, and a ratio that would divide by one, come back absent.
    Taking the newer row wholesale was therefore the defect: it let absence
    overwrite figures captured while they were still reachable.

    A measured value is never replaced by an absent one. `timeline_end` moves
    with the ratio rather than independently, because it exists to say how
    mature that ratio is -- advancing it under an older ratio would misreport
    exactly the thing it was added to record.
    """
    ratio_from_fresh = fresh.durability_ratio is not None

    def kept(a: int | None, b: int | None) -> int | None:
        """The fresh figure when it has one, else what was already measured."""
        return b if b is not None else a

    return PostMetrics(
        post_id=fresh.post_id,
        published_at=fresh.published_at or stored.published_at,
        views_day_2=kept(stored.views_day_2, fresh.views_day_2),
        views_day_7=kept(stored.views_day_7, fresh.views_day_7),
        # Window-independent, so the fresh figure is simply the better one:
        # cumulative rows mean the last row is the lifetime total whether or
        # not the window reaches back to publication. Truncation cannot lower
        # it, so a smaller fresh figure is a real downward revision -- which
        # platforms do make, observed here as a day-7 count exceeding the
        # lifetime total -- and keeping the larger would freeze a stale number.
        views_total=kept(stored.views_total, fresh.views_total),
        durability_ratio=(
            fresh.durability_ratio if ratio_from_fresh else stored.durability_ratio
        ),
        # Pinned only while it is dating a preserved ratio. With no ratio on
        # either side it still means "how far the timeline reached", and
        # freezing it there would make it stale for every post through its
        # whole first month.
        timeline_end=(
            stored.timeline_end
            if (stored.durability_ratio is not None and not ratio_from_fresh)
            else (fresh.timeline_end or stored.timeline_end)
        ),
    )


def _readable(path: Path) -> bool:
    """Whether an existing metrics file parses into records."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        [PostMetrics(**row) for row in raw]
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False
    return True
