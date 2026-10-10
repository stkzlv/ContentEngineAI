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
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from src.utils.outputs_paths import durable_state_path

logger = logging.getLogger(__name__)

_V = TypeVar("_V", int, float)

# Views after this many days count as durable rather than launch traffic.
DURABILITY_WINDOW_DAYS = 30

# Per-platform quality metrics, from the provider's per-post analytics
# (REQ-PUB-084). The provider returns 0 for a field a platform does not
# expose, so a 0 is only trusted for the fields listed as exposed; every other
# field is stored as None, unknown. Exposed fields follow each platform's own
# public post metrics. The first-seconds metrics the platforms rank on
# (YouTube engaged views and viewed-vs-swiped-away, TikTok watch time and
# completion) are not exposed at all and are listed so the gap stays visible.
QUALITY_METRICS: dict[str, dict[str, bool]] = {
    "youtube": {
        "views": True,
        "likes": True,
        "comments": True,
        "engagedViews": False,
        "viewedVsSwipedAway": False,
        "averageViewDuration": False,
    },
    "tiktok": {
        "views": True,
        "likes": True,
        "comments": True,
        "shares": True,
        "averageWatchTime": False,
        "completionRate": False,
    },
    "instagram": {
        "views": True,
        "reach": True,
        "likes": True,
        "comments": True,
        "shares": True,
        "saves": True,
        "igReelsAvgWatchTime": True,
        "igReelsVideoViewTotalTime": True,
        "reelsSkipRate": True,
    },
}

# The launch figures worth keeping. Day 2 is most of the curve; day 7 is where
# it has essentially finished.
LAUNCH_DAYS = (2, 7)


@dataclass
class PostMetrics:
    """Per-post view history, reduced to the figures a comparison needs."""

    post_id: str
    published_at: str = ""
    # None means not comparable, never zero. Three causes, and `timeline_end`
    # does not separate them: the window had not closed when the sweep ran,
    # the retained rows begin after the cutoff, or a leg had not started
    # reporting by it. `lagged_cutoff_days` names the third.
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
    # Day cutoffs this post is known to have straddled: some legs reporting
    # by then, others not. Recorded rather than recomputed, because the sweep
    # that first measures a young post usually cannot see the lag -- the slow
    # platform has no rows at all yet, so one leg looks like the whole post
    # and the biased figure is stored. A later sweep sees both and marks the
    # cutoff, which is what lets the merge withdraw a number it already kept.
    #
    # Marked only when legs disagree. A sweep whose rows all begin late is a
    # truncated timeline, not a lagging leg: nothing is biased relative to
    # anything, and `views_at_day` already reports those cutoffs unknown.
    lagged_cutoff_days: list[int] = field(default_factory=list)
    # Whether the timeline this row was measured from still reached back to
    # publication. Past the retention horizon it does not, and a ratio
    # computed from what remains divides by a partial "within" -- a different
    # quantity, not a newer reading of the same one. Recorded so the merge can
    # keep the figure taken while the record was whole.
    #
    # None means a row written before this was recorded. Read as "whole", not
    # as False: defaulting to False would tell the merge every stored ratio
    # came from a truncated record, so the first sweep after the upgrade would
    # overwrite all of them -- the outcome the field exists to prevent, applied
    # to exactly the figures that cannot be recomputed.
    covers_publication: bool | None = None
    # Set once a sweep finds this post had a view count and returned none.
    # Without it the check cannot fire on a transition: the merge keeps the
    # stored figure, so the same post satisfies "had a count, returned none"
    # on every later sweep and an account whose posts have aged out would
    # warn daily forever -- the outcome the rule above was chosen to avoid.
    # Cleared as soon as a figure comes back, so a later break still reports.
    stopped_reporting: bool = False
    # Per platform, per field: the latest reading, or None when the platform
    # does not expose the field or the sweep could not read it (REQ-PUB-084).
    platform_metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)
    # Per platform, its own day-N views ("day_2", "day_7"), so a readout can
    # count only the platforms a protocol names (REQ-PUB-149). None when that
    # platform had no row by the cutoff.
    views_by_platform: dict[str, dict[str, int | None]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quality_metrics(
    platform_analytics: Any,
) -> dict[str, dict[str, float | None]]:
    """The exposed quality fields per platform from one post's analytics.

    `platform_analytics` is the provider's `platformAnalytics` list. A field
    the platform does not expose is None however the provider fills it.
    """
    out: dict[str, dict[str, float | None]] = {}
    if not isinstance(platform_analytics, list):
        return out
    for entry in platform_analytics:
        if not isinstance(entry, dict):
            continue
        platform = str(entry.get("platform") or "").lower()
        exposed = QUALITY_METRICS.get(platform)
        if exposed is None:
            continue
        # A leg not synced yet, or whose post failed, is no reading at all:
        # its zero-filled figures would overwrite stored counts.
        if entry.get("syncStatus") not in (None, "synced"):
            continue
        if entry.get("status") == "failed":
            continue
        figures = entry.get("analytics") or {}
        row: dict[str, float | None] = {}
        for name, is_exposed in exposed.items():
            value = figures.get(name) if isinstance(figures, dict) else None
            is_number = isinstance(value, int | float) and not isinstance(value, bool)
            row[name] = value if is_exposed and is_number else None
        out[platform] = row
    return out


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
    """Parse an API timestamp, tolerating the shapes the scheduler returns.

    An already-parsed `datetime` is normalised the same way a string is, so
    the function cannot return an aware value down one path and a naive one
    down the other. Mixing the two raises on the first comparison, and every
    caller here compares a row date against a publication date.
    """
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
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

    **One row per date, summed across platforms.** The API returns a row per
    platform per date, so a post published to three platforms has three rows
    for each day. Taking them as-is made every figure the last-listed
    platform's number wearing the post's name: one post reads 357 views this
    way against 1187 actually earned, because the rows for that date are
    Instagram 15, TikTok 815, YouTube 357 and YouTube sorts last.

    Summing is the right reduction for a reach question, which asks how many
    people a post reached, not how many it reached on the platform whose row
    happened to come last.

    A platform absent from a date carries its last known figure forward rather
    than contributing zero, because platforms report on their own lag and the
    newest date frequently holds only some of them.

    Rows missing a usable date or view count are dropped rather than
    defaulted: a zero would be indistinguishable from a real zero and would
    drag a durability ratio toward nothing.
    """
    by_date: dict[datetime, dict[str, int]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        when = _parse_date(row.get("date") or row.get("timestamp"))
        views = row.get("views")
        if when is None or not isinstance(views, int | float):
            continue
        platform = str(row.get("platform") or "")
        by_date.setdefault(when, {})[platform] = int(views)

    # Carry each platform's last known figure forward. Platforms report on
    # their own lag, so a date -- the newest one especially -- often carries
    # only some of them, and summing just what is present collapses the total:
    # one post reads 359 on the day it has all three and 19 on the next, where
    # only YouTube has reported. Each platform's own series is cumulative, so
    # a missing row means "unchanged since it last reported", not zero.
    out: list[tuple[datetime, int]] = []
    latest: dict[str, int] = {}
    for when in sorted(by_date):
        latest.update(by_date[when])
        out.append((when, sum(latest.values())))
    return out


def first_report_dates(rows: Any) -> dict[str, datetime]:
    """The earliest date each platform appears in a raw timeline.

    A platform's first row carries its **lifetime** total to that date, not
    that day's increment, so a leg that starts reporting late drops its whole
    accumulated figure into the middle of the series. Knowing when each leg
    began is what lets a day-N figure say whether it covers the post or only
    part of it.
    """
    first: dict[str, datetime] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        when = _parse_date(row.get("date") or row.get("timestamp"))
        views = row.get("views")
        if when is None or not isinstance(views, int | float):
            continue
        platform = str(row.get("platform") or "")
        if platform not in first or when < first[platform]:
            first[platform] = when
    return first


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

    A leg that started reporting after the window closed would inflate this,
    landing its whole lifetime figure in the "after" half while contributing
    nothing to "within". `summarize_post` withholds the ratio in that case;
    the check lives there because it needs the raw rows.
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
    if total < within:
        # Reachable, though not observed: a per-platform downward revision
        # straddling the window puts the total below the day-30 figure. The
        # dips that first prompted this guard turned out to be the partial-date
        # artifact fixed in 0.71.4 rather than revisions, and a scan of 287
        # live rows found no per-platform decrease at all -- so treat this as
        # defence against a case the API allows rather than one it exhibits.
        # Either way "views earned after the window" is not worth reporting
        # negative, and the next sweep recomputes it.
        return None
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


def platform_day_views(
    rows: Any, published_at: datetime
) -> dict[str, dict[str, int | None]]:
    """Each platform's own day-N views, from its rows alone.

    A platform's series needs no lag guard: a leg that had not reported by a
    cutoff has no row at or before it and reads None, never a partial sum.
    """
    by_platform: dict[str, list[dict[str, Any]]] = {}
    for row in rows or []:
        if isinstance(row, dict) and row.get("platform"):
            by_platform.setdefault(str(row["platform"]).lower(), []).append(row)
    return {
        platform: {
            f"day_{day}": views_at_day(
                normalize_timeline(platform_rows), published_at, day
            )
            for day in LAUNCH_DAYS
        }
        for platform, platform_rows in by_platform.items()
    }


def summarize_post(post_id: str, published_at: Any, rows: Any) -> PostMetrics:
    """Reduce one post's raw timeline to the figures worth storing."""
    when = _parse_date(published_at)
    timeline = normalize_timeline(rows)
    if when is None or not timeline:
        return PostMetrics(post_id=post_id, published_at=str(published_at or ""))
    # A day-N figure counts only the legs that were reporting by the cutoff.
    # A leg that started later contributes nothing to it while contributing
    # everything to `views_total`, so the two are not measured over the same
    # post. Reported as unknown rather than as a small number, which is the
    # rule `views_at_day` already applies to a window the timeline has not
    # reached: a comparison that ranks by median day-7 views would rank a
    # post understated by reporting lag below an identical one, for a reason
    # that is not reach.
    first_seen = first_report_dates(rows)

    def straddles(day: int) -> bool:
        """Whether some leg had reported by this cutoff and another had not."""
        cutoff = when + timedelta(days=day)
        started = [f <= cutoff for f in first_seen.values()]
        return any(started) and not all(started)

    # A leg's first retained row is evidence of when it *started* reporting
    # only while the record still reaches back to publication. Past the
    # retention horizon every leg's rows begin at the window edge, and a leg
    # that happens to be absent from that first date looks identical to one
    # that started late -- so marking there would withdraw a ratio an earlier,
    # fuller sweep had measured correctly, and no later sweep could recompute
    # it. Nothing is lost by staying silent: a truncated reading already
    # reports those cutoffs unknown on its own.
    covers_publication = timeline[0][0] <= when + timedelta(days=1)
    cutoffs = (*LAUNCH_DAYS, DURABILITY_WINDOW_DAYS)

    # Two different questions, and conflating them let a truncated sweep
    # store a one-leg figure unmarked. Marking is persisted and withdraws a
    # figure other sweeps stored, so it needs a record reaching publication
    # to tell a late start from a truncated one. Withholding only governs
    # this reading, and is safe wherever the retained window actually covers
    # the cutoff -- there the legs demonstrably disagree.
    marked = [d for d in cutoffs if covers_publication and straddles(d)]
    withheld = {
        d
        for d in cutoffs
        if straddles(d) and timeline[0][0] <= when + timedelta(days=d)
    }

    def at_day(day: int) -> int | None:
        if day in withheld:
            return None
        return views_at_day(timeline, when, day)

    return PostMetrics(
        post_id=post_id,
        published_at=when.isoformat(),
        views_day_2=at_day(LAUNCH_DAYS[0]),
        views_day_7=at_day(LAUNCH_DAYS[1]),
        views_total=timeline[-1][1],
        durability_ratio=(
            None
            if DURABILITY_WINDOW_DAYS in withheld
            else durability_ratio(timeline, when)
        ),
        timeline_end=timeline[-1][0].isoformat(),
        lagged_cutoff_days=marked,
        covers_publication=covers_publication,
        views_by_platform=platform_day_views(rows, when),
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
    return durable_state_path(outputs_dir, METRICS_FILENAME)


def load_metrics(outputs_dir: Path) -> list[PostMetrics]:
    """Read stored per-post figures. Missing or unreadable reads as empty."""
    path = metrics_path(outputs_dir)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Unknown keys are dropped rather than raising, the way the published
        # registry already does it. Without this a file written by a newer
        # release is unreadable to an older one, which then refuses to write
        # at all and the whole history is stuck behind a rollback.
        known = {f.name for f in fields(PostMetrics)}
        return [
            PostMetrics(**{k: v for k, v in row.items() if k in known}) for row in raw
        ]
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to load post metrics: %s", exc)
        return []


def _warn_regressed(regressed: list[str], measured: int) -> None:
    """Report posts that had a view count and came back with none.

    This is the one signal that separates a broken reader from an account
    whose posts are simply too young to have rows. Both look identical in a
    single sweep -- every timeline empty, every call successful -- so the
    sweep cannot fail on it without failing a new account daily until its
    first rows land. Across sweeps they diverge: a young post gains a figure
    within days, while a post that *had* one and stopped reporting it did not
    get younger. A renamed timeline key regresses every mature post at once.

    A warning rather than an error on purpose. The figures already stored are
    safe -- the merge keeps them per field -- so nothing is lost at the moment
    this fires; what is at risk is the capture continuing to look healthy
    while it collects nothing.

    Fires on the transition, not on the state. `stopped_reporting` marks a
    post whose figure went away, and a marked post stops counting, so an
    account dormant long enough for its whole measured window to age out
    warns once rather than daily. Without that marker the merge's own
    keep-per-field behaviour makes the condition permanent: the stored figure
    survives the empty reading, so the same posts satisfy it forever.
    """
    sample = ", ".join(regressed[:5])
    more = f" (+{len(regressed) - 5} more)" if len(regressed) > 5 else ""
    logger.warning(
        "Every post with a stored view count returned none this sweep: "
        "%d of %d measured, including %s%s. Posts age out of the provider's "
        "timeline one at a time, never all at once, so check that the "
        "response still carries the rows this reads.",
        len(regressed),
        measured,
        sample,
        more,
    )


def save_metrics(metrics: list[PostMetrics], outputs_dir: Path) -> list[str]:
    """Write per-post figures, merging each post's row field by field.

    Not "the later reading wins": past the provider's retention horizon a
    later reading has *less* history behind it, and its absent day-N figures
    would erase ones captured while they were still reachable.

    Returns post ids only when *every* re-measured post that still counted --
    one with a stored view count, not already marked quiet -- came back
    without one, and an empty list otherwise. A partial regression is ageing
    and deliberately reports nothing. The caller routes the result where an
    operator will see it, because a warning alone reaches no surface anyone
    is told to check.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_path(outputs_dir)
    if path.exists() and not _readable(path):
        # Merging onto an empty read would drop every post outside this sweep,
        # unrecoverably. The sibling tracking files guard their writes the same
        # way; this one refuses rather than repairing.
        raise OSError(f"Refusing to overwrite unreadable metrics file: {path}")

    merged = {m.post_id: m for m in load_metrics(outputs_dir)}
    # Posts this sweep re-measured that already had a view count, and those of
    # them that came back without one.
    had_a_count: list[str] = []
    regressed: list[str] = []
    for m in metrics:
        stored = merged.get(m.post_id)
        # A post already known to be quiet does not count again. The merge
        # keeps its stored figure, so without this it satisfies "had a count,
        # returned none" on every later sweep and the warning repeats daily.
        if (
            stored is not None
            and stored.views_total is not None
            and not stored.stopped_reporting
        ):
            had_a_count.append(m.post_id)
            if m.views_total is None:
                regressed.append(m.post_id)
        merged[m.post_id] = _combine(stored, m) if stored else m

    # Only when *every* such post went quiet at once. A single post losing its
    # figures is ordinary: past the retention horizon the provider stops
    # returning a post's rows entirely, so an aged-out post regresses on a
    # completely healthy install. Those age out one at a time. A reader that
    # stopped understanding the response takes every post with it in the same
    # sweep, which no amount of ageing does.
    #
    # An earlier version tried to tell the two apart per post, by asking when
    # each had last produced rows. There is no field that answers that:
    # `timeline_end` is pinned to whichever reading a preserved durability
    # ratio came from (see `_combine`), so it freezes on live posts and gating
    # on it silenced the check on exactly the mature posts that make the
    # signal unambiguous.
    if regressed and len(regressed) == len(had_a_count):
        _warn_regressed(regressed, len(metrics))
    else:
        regressed = []

    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps([m.to_dict() for m in merged.values()], indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    return regressed


def _withdraw_lagged(metrics: PostMetrics) -> PostMetrics:
    """Blank every figure whose cutoff a leg had not started reporting by.

    Applied to the merged row, because the sweep that stored a figure is
    usually not the one that can tell it was biased. This is the single case
    where an absence must beat a stored number: the per-field merge keeps a
    measured value over a missing one, which is right for a truncated
    timeline -- the figure was true when taken -- and wrong here, where it
    was never true.
    """
    if not metrics.lagged_cutoff_days:
        return metrics
    blanked = replace(metrics)
    if LAUNCH_DAYS[0] in metrics.lagged_cutoff_days:
        blanked.views_day_2 = None
    if LAUNCH_DAYS[1] in metrics.lagged_cutoff_days:
        blanked.views_day_7 = None
    if DURABILITY_WINDOW_DAYS in metrics.lagged_cutoff_days:
        blanked.durability_ratio = None
    return blanked


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

    A measured value is never replaced by an absent one, with one exception:
    a cutoff recorded in `lagged_cutoff_days` withdraws the figure taken for
    it, because that figure counted only part of the post and was never true.
    `timeline_end` moves
    with the ratio rather than independently, because it exists to say how
    mature that ratio is -- advancing it under an older ratio would misreport
    exactly the thing it was added to record.
    """
    lagged = sorted(set(stored.lagged_cutoff_days) | set(fresh.lagged_cutoff_days))
    # A ratio measured from a truncated record is not a newer reading of the
    # same quantity: its "within" half is missing the days the window needs,
    # so it can read higher than the figure taken while the record was whole.
    # Prefer the one measured from the fuller timeline.
    # None means a row written before this was recorded; read as whole, so an
    # existing ratio is not discarded on the strength of a missing key.
    stored_whole = stored.covers_publication is not False
    ratio_from_fresh = fresh.durability_ratio is not None and not (
        stored.durability_ratio is not None
        and stored_whole
        # Positively truncated, not merely unrecorded: with both provenances
        # unknown the newer reading is still the better one, and rejecting it
        # would freeze whatever happened to be stored first.
        and fresh.covers_publication is False
    )
    # A ratio about to be withdrawn is not a preserved one, so `timeline_end`
    # must not be pinned to date it. Deciding that after the withdrawal would
    # leave the field frozen against a ratio that no longer exists.
    ratio_preserved = (
        stored.durability_ratio is not None
        and not ratio_from_fresh
        and DURABILITY_WINDOW_DAYS not in lagged
    )

    def kept(a: int | None, b: int | None) -> int | None:
        """The fresh figure when it has one, else what was already measured."""
        return b if b is not None else a

    merged = PostMetrics(
        post_id=fresh.post_id,
        published_at=fresh.published_at or stored.published_at,
        # Sticky while the post stays quiet, cleared the moment a figure comes
        # back. What makes the regression check fire on the transition rather
        # than on every sweep that follows it.
        stopped_reporting=(
            fresh.views_total is None
            and (stored.views_total is not None or stored.stopped_reporting)
        ),
        views_day_2=kept(stored.views_day_2, fresh.views_day_2),
        views_day_7=kept(stored.views_day_7, fresh.views_day_7),
        # Window-independent, so the fresh figure is simply the more recent
        # measurement of the same quantity: summed across platforms, the last
        # row is the lifetime total whether or not the window reaches back to
        # publication, and truncation cannot lower it. Keeping the larger of
        # the two would freeze a figure the platform had since corrected.
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
            if ratio_preserved
            else (fresh.timeline_end or stored.timeline_end)
        ),
        # Union, and never unmarked: a sweep that no longer sees the lag is
        # one whose rows all begin after it, which says nothing about whether
        # the figure was biased when it was taken.
        lagged_cutoff_days=lagged,
        # Follows the ratio that survived, rather than unioning. Its only
        # reader asks where the stored ratio came from, and a union let a
        # young sweep that carried no ratio mark the row whole, after which
        # every later reading was rejected as truncated and the first
        # truncated ratio froze permanently.
        covers_publication=(
            fresh.covers_publication if ratio_from_fresh else stored.covers_publication
        ),
        # Per field, like the views: a fresh reading wins, and a sweep whose
        # analytics call failed (an empty dict) loses nothing already stored.
        platform_metrics=_combine_quality(
            stored.platform_metrics, fresh.platform_metrics
        ),
        # Same rule as the summed day-N figures: a reading past the retention
        # horizon comes back None and must not erase one taken earlier.
        views_by_platform=_combine_quality(
            stored.views_by_platform, fresh.views_by_platform
        ),
    )
    return _withdraw_lagged(merged)


def _combine_quality(
    stored: dict[str, dict[str, _V | None]],
    fresh: dict[str, dict[str, _V | None]],
) -> dict[str, dict[str, _V | None]]:
    merged = {platform: dict(row) for platform, row in stored.items()}
    for platform, row in fresh.items():
        target = merged.setdefault(platform, {})
        for name, value in row.items():
            if value is not None or name not in target:
                target[name] = value
    return merged


def _readable(path: Path) -> bool:
    """Whether an existing metrics file parses into records."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        [PostMetrics(**row) for row in raw]
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False
    return True


# The render choices a quality report groups by, beside the content format.
SEGMENT_CHOICES = (
    "profile",
    "script_template",
    "voice_profile",
    "caption_template",
    "cold_open_variant",
    "duration_band",
)


def duration_band(seconds: Any, edges: list[float]) -> str:
    """The band a video's length falls in, by upper edges in seconds.

    `[20, 30]` gives "<20s", "20-30s" and ">30s"; a missing or unreadable
    length is "unknown", so posts without one stay visible in the report.
    """
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    low = 0.0
    for edge in edges:
        if value < edge:
            return f"<{edge:g}s" if low == 0 else f"{low:g}-{edge:g}s"
        low = edge
    return f">{low:g}s"


def segment_quality(
    metrics: list[PostMetrics],
    product_by_post: dict[str, str],
    labels_by_product: dict[str, dict[str, str]],
) -> list[str]:
    """Mean of each exposed quality metric per segment (REQ-PUB-084).

    A segment is a value of `content_format` or of a recorded render choice.
    Unknown readings are left out of the mean rather than counted as zero, and
    each mean carries the number of posts it rests on.
    """
    dimensions = ("content_format", *SEGMENT_CHOICES)
    lines = []
    for dim in dimensions:
        groups: dict[str, list[PostMetrics]] = {}
        for m in metrics:
            label = labels_by_product.get(product_by_post.get(m.post_id, ""), {})
            value = label.get(dim)
            if value:
                groups.setdefault(str(value), []).append(m)
        for value, posts in sorted(groups.items()):
            parts = []
            for platform, exposed in QUALITY_METRICS.items():
                for name, is_exposed in exposed.items():
                    if not is_exposed:
                        continue
                    readings = [
                        reading
                        for p in posts
                        if (reading := p.platform_metrics.get(platform, {}).get(name))
                        is not None
                    ]
                    if readings:
                        mean = sum(readings) / len(readings)
                        parts.append(f"{platform}.{name}={mean:.1f}(n={len(readings)})")
            if parts:
                lines.append(f"{dim}={value} [{len(posts)} post(s)]: {' '.join(parts)}")
    return lines


# The reach test's decision rule: the topic arm's median day-7 views over the
# product arm's, read against these bands (REQ-PUB-150).
READOUT_ARMS = ("topic", "product")
PROCEED_RATIO = 0.7
REFORECAST_RATIO = 0.4


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def _views_on(m: PostMetrics, platforms: list[str], day: int) -> int | None:
    """A post's day-N views on the named platforms, or None if any is missing.

    A post missing one platform's figure is left out rather than counted on
    the others: a partial sum ranks it below an identical post for a reason
    that is not reach.
    """
    total = 0
    for platform in platforms:
        value = m.views_by_platform.get(platform, {}).get(f"day_{day}")
        if value is None:
            return None
        total += value
    return total


def _verdict(ratio: float) -> str:
    if ratio >= PROCEED_RATIO:
        return "reach holds: proceed"
    if ratio >= REFORECAST_RATIO:
        return "proceed, re-forecast revenue down by the ratio"
    return "reach premise fails: revisit niche or format"


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


def readout_by_arm(
    metrics: list[PostMetrics],
    arm_by_post: dict[str, str],
    *,
    since: datetime | None,
    platforms: list[str],
    breakout_multiple: float,
) -> list[str]:
    """The reach-test readout: day-N medians by format arm and their ratio.

    Medians, not means: one spike would swamp an average. The gate counts
    only `platforms`; per-platform medians and the breakout rate are
    secondary lines that do not change the verdict (REQ-PUB-152). Posts with
    no arm are counted and named rather than silently dropped; posts before
    `since` are counted in the header.
    """
    in_window = [
        m
        for m in metrics
        if since is None
        or ((when := _parse_date(m.published_at)) is not None and when >= since)
    ]
    excluded = len(metrics) - len(in_window)
    unplaced = [m.post_id for m in in_window if arm_by_post.get(m.post_id) is None]
    placed = {
        arm: [m for m in in_window if arm_by_post.get(m.post_id) == arm]
        for arm in READOUT_ARMS
    }
    lines = [
        f"Reach readout by arm on {'+'.join(platforms)}"
        + (f" since {since.date().isoformat()}" if since else "")
        + f": {len(in_window)} post(s), {len(unplaced)} with no arm"
        + (f", {excluded} earlier left out" if excluded else "")
    ]
    if unplaced:
        lines.append(f"  no arm: {', '.join(unplaced)}")

    medians: dict[tuple[str, int], float | None] = {}
    for arm, posts in placed.items():
        parts = []
        for day in LAUNCH_DAYS:
            values = [
                v for m in posts if (v := _views_on(m, platforms, day)) is not None
            ]
            medians[(arm, day)] = _median(values)
            parts.append(
                f"day {day} median {_fmt(medians[(arm, day)])} (n={len(values)})"
            )
        # Summed over every platform: a ratio of views after day 30 to views
        # within it means the same on any platform subset.
        ratios = [m.durability_ratio for m in posts if m.durability_ratio is not None]
        median_ratio = _median(ratios)
        parts.append(
            "durability median "
            f"{_fmt(None if median_ratio is None else round(median_ratio, 3))}"
            f" (n={len(ratios)})"
        )
        lines.append(f"  {arm} [{len(posts)} post(s)]: {', '.join(parts)}")

    for day in LAUNCH_DAYS:
        topic, product = medians[("topic", day)], medians[("product", day)]
        if topic is None or not product:
            lines.append(f"  ratio day {day}: not measurable")
            continue
        # The verdict reads the printed figure, so a ratio shown as 0.70 is
        # never placed in the band below 0.7.
        ratio = round(topic / product, 2)
        verdict = f": {_verdict(ratio)}" if day == LAUNCH_DAYS[-1] else ""
        lines.append(f"  ratio day {day}: {ratio:.2f}{verdict}")

    lines.append("Secondary (does not change the verdict):")
    all_platforms = sorted({p for m in in_window for p in m.views_by_platform})
    day = LAUNCH_DAYS[-1]
    for platform in all_platforms:
        parts = []
        for arm, posts in placed.items():
            values = [
                v for m in posts if (v := _views_on(m, [platform], day)) is not None
            ]
            parts.append(f"{arm} {_fmt(_median(values))} (n={len(values)})")
        lines.append(f"  {platform} day {day} median: {', '.join(parts)}")
    pooled = [
        v
        for posts in placed.values()
        for m in posts
        if (v := _views_on(m, platforms, day)) is not None
    ]
    pooled_median = _median(pooled)
    if pooled_median:
        threshold = pooled_median * breakout_multiple
        parts = []
        for arm, posts in placed.items():
            values = [
                v for m in posts if (v := _views_on(m, platforms, day)) is not None
            ]
            hits = sum(v >= threshold for v in values)
            parts.append(f"{arm} {hits} of {len(values)}")
        lines.append(
            f"  breakouts (day {day} at {breakout_multiple:g}x the pooled median "
            f"{pooled_median:g} or more): {', '.join(parts)}"
        )
    return lines
