"""Account-level audience figures for judging a recurring format over weeks.

The analytics sweep's per-post figures say how each video did. These say what
the audience did across a date range: whether viewers followed, how much of a
YouTube video they watched, and whether Instagram viewers looked up the
profile (REQ-PUB-162 to REQ-PUB-165). They are read per account and per day,
so they need no post-by-post attribution.

Unknown is never 0 here: a day the provider returned nothing for stays absent,
and a rate whose inputs are missing is None.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial
from pathlib import Path
from typing import Any

from src.utils.outputs_paths import durable_state_path

logger = logging.getLogger(__name__)

AUDIENCE_FILENAME = "audience_metrics.json"

# How far back each sweep re-reads. Re-reading overwrites days a platform
# revised later (Instagram reports up to 48 hours late), and fills days missed
# while the sweep did not run.
FOLLOWER_LOOKBACK_DAYS = 30
DAILY_LOOKBACK_DAYS = 30
INSTAGRAM_PROFILE_LOOKBACK_DAYS = 7

# Per-post YouTube fields added to `platform_metrics["youtube"]` from the
# per-video daily stats (REQ-PUB-163). The per-post analytics' own
# `averageViewDuration` comes back as 0, so it is not used.
YOUTUBE_RETENTION_FIELDS = (
    "avgViewPercentage",
    "avgViewDurationSec",
    "subscribersGained",
)

INSTAGRAM_PROFILE_METRICS = ("profile_views", "profile_links_taps")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def youtube_leg(platform_analytics: Any) -> tuple[str, str] | None:
    """The YouTube video id and account id from a post's `platformAnalytics`."""
    if not isinstance(platform_analytics, list):
        return None
    for entry in platform_analytics:
        if not isinstance(entry, dict) or entry.get("platform") != "youtube":
            continue
        video_id = entry.get("platformPostId")
        account = entry.get("accountId")
        if isinstance(account, dict):
            account = account.get("_id") or account.get("id")
        if isinstance(video_id, str) and video_id and isinstance(account, str):
            return video_id, account
    return None


def youtube_retention(daily_views: Any) -> dict[str, float | None]:
    """View-weighted watch figures and subscribers gained from daily rows.

    Each field is None when the rows carry no views to weigh by, so a video
    nobody watched yet does not read as watched to 0%.
    """
    rows = [r for r in daily_views or [] if isinstance(r, dict)]
    views = sum(_number(r.get("views")) or 0.0 for r in rows)
    if not views:
        return dict.fromkeys(YOUTUBE_RETENTION_FIELDS)

    def weighted(name: str) -> float | None:
        total = 0.0
        for r in rows:
            v, x = _number(r.get("views")), _number(r.get(name))
            if v and x is None:
                return None
            total += (v or 0.0) * (x or 0.0)
        return round(total / views, 2)

    gained = [_number(r.get("subscribersGained")) for r in rows]
    return {
        "avgViewPercentage": weighted("averageViewPercentage"),
        "avgViewDurationSec": weighted("averageViewDuration"),
        "subscribersGained": (
            None if any(g is None for g in gained) else sum(g or 0.0 for g in gained)
        ),
    }


def follower_history(raw: Any) -> dict[str, dict[str, int]]:
    """Followers per platform per day from the follower-stats response."""
    if not isinstance(raw, dict):
        return {}
    platform_of = {
        a.get("_id"): str(a.get("platform") or "").lower()
        for a in raw.get("accounts") or []
        if isinstance(a, dict)
    }
    out: dict[str, dict[str, int]] = {}
    for account_id, points in (raw.get("stats") or {}).items():
        platform = platform_of.get(account_id)
        if not platform:
            continue
        for point in points or []:
            if not isinstance(point, dict):
                continue
            count = _number(point.get("followers"))
            day = point.get("date")
            if count is not None and isinstance(day, str):
                out.setdefault(platform, {})[day[:10]] = int(count)
    return out


def daily_totals(raw: Any) -> dict[str, dict[str, dict[str, float]]]:
    """Views and interactions per platform per publish date.

    The provider groups these by the day a post was published, with each
    post's figures as of now, not by the day the views happened.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, dict[str, float]]] = {}
    for day_row in raw.get("dailyData") or []:
        if not isinstance(day_row, dict) or not isinstance(day_row.get("date"), str):
            continue
        day = day_row["date"][:10]
        for platform, figures in (day_row.get("platformMetrics") or {}).items():
            if not isinstance(figures, dict):
                continue
            row = {
                name: value
                for name in ("views", "comments", "shares", "saves", "postCount")
                if (value := _number(figures.get(name))) is not None
            }
            if row:
                out.setdefault(str(platform).lower(), {})[day] = row
    return out


def instagram_profile_day(raw: Any) -> dict[str, float]:
    """Profile views and link taps from a one-day account-insights window."""
    metrics = raw.get("metrics") if isinstance(raw, dict) else None
    if not isinstance(metrics, dict):
        return {}
    out: dict[str, float] = {}
    for name in INSTAGRAM_PROFILE_METRICS:
        entry = metrics.get(name)
        value = _number(entry.get("total")) if isinstance(entry, dict) else None
        if value is not None:
            out[name] = value
    return out


def merge_audience(stored: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """Overlay fresh readings on the stored ones, never dropping a stored day."""
    merged: dict[str, Any] = json.loads(json.dumps(stored))

    def overlay(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                overlay(target[key], value)
            elif value is not None:
                target[key] = value

    overlay(merged, fresh)
    return merged


def audience_path(outputs_dir: Path) -> Path:
    return durable_state_path(outputs_dir, AUDIENCE_FILENAME)


def load_audience(outputs_dir: Path) -> dict[str, Any]:
    """The stored audience figures; missing reads as empty."""
    path = audience_path(outputs_dir)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def save_audience(fresh: dict[str, Any], outputs_dir: Path) -> None:
    """Merge ``fresh`` into the stored figures and write them atomically.

    An unreadable store is refused rather than overwritten, as the per-post
    metrics file is: merging onto an empty read would drop the history.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    path = audience_path(outputs_dir)
    try:
        stored = load_audience(outputs_dir)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise OSError(
            f"Refusing to overwrite unreadable audience file: {path}"
        ) from exc
    merged = merge_audience(stored, fresh)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def sweep_audience(
    accounts: Any,
    analytics: Any,
    today: date,
    guard: Callable[[str, Callable[[], Any]], Any],
) -> dict[str, Any]:
    """Read the account-level figures. Each call goes through ``guard``.

    ``guard(label, call)`` returns the call's result, or None after logging
    a failure, so one endpoint failing keeps the others' readings.
    """
    fresh: dict[str, Any] = {}
    start = str(today - timedelta(days=FOLLOWER_LOOKBACK_DAYS))
    raw = guard(
        "follower stats",
        lambda: accounts.get_follower_stats(from_date=start, to_date=str(today)),
    )
    if raw is not None:
        fresh["followers"] = follower_history(raw)
        ig_accounts = [
            a.get("_id")
            for a in raw.get("accounts") or []
            if isinstance(a, dict) and a.get("platform") == "instagram"
        ]
    else:
        ig_accounts = []

    start = str(today - timedelta(days=DAILY_LOOKBACK_DAYS))
    raw = guard(
        "daily metrics",
        lambda: analytics.get_daily_metrics(from_date=start, to_date=str(today)),
    )
    if raw is not None:
        fresh["daily"] = daily_totals(raw)

    profile: dict[str, dict[str, float]] = {}
    for account_id in ig_accounts[:1]:
        for back in range(INSTAGRAM_PROFILE_LOOKBACK_DAYS, 0, -1):
            day = today - timedelta(days=back)
            raw = guard(
                f"Instagram insights {day}",
                partial(
                    analytics.get_instagram_account_insights,
                    account_id=account_id,
                    metrics=",".join(INSTAGRAM_PROFILE_METRICS),
                    since=str(day),
                    until=str(day + timedelta(days=1)),
                ),
            )
            figures = instagram_profile_day(raw)
            if figures:
                profile[str(day)] = figures
    if profile:
        fresh["instagram_profile"] = profile
    return fresh


@dataclass
class PlatformRates:
    """One platform's audience rates over a date range; None means unknown."""

    platform: str
    views: float | None
    net_follows: int | None
    follows_per_1k: float | None
    shares_saves_per_1k: float | None
    comments_per_1k: float | None


def _per_1k(count: float | None, views: float | None) -> float | None:
    if count is None or not views:
        return None
    return round(count * 1000 / views, 2)


def _days(since: date, until: date) -> Iterable[str]:
    day = since
    while day <= until:
        yield str(day)
        day += timedelta(days=1)


def net_follows(history: dict[str, int], since: date, until: date) -> int | None:
    """Followers at ``until`` minus followers the day before ``since``."""
    before = history.get(str(since - timedelta(days=1)))
    after = history.get(str(until))
    if before is None or after is None:
        return None
    return after - before


def platform_rates(
    store: dict[str, Any], platform: str, since: date, until: date
) -> PlatformRates:
    daily = (store.get("daily") or {}).get(platform) or {}
    rows = [daily[d] for d in _days(since, until) if d in daily]
    views = sum(r.get("views", 0.0) for r in rows) if rows else None
    follows = net_follows(
        (store.get("followers") or {}).get(platform) or {}, since, until
    )
    shares_saves = (
        sum(r.get("shares", 0.0) + r.get("saves", 0.0) for r in rows) if rows else None
    )
    comments = sum(r.get("comments", 0.0) for r in rows) if rows else None
    return PlatformRates(
        platform=platform,
        views=views,
        net_follows=follows,
        follows_per_1k=_per_1k(follows, views),
        shares_saves_per_1k=_per_1k(shares_saves, views),
        comments_per_1k=_per_1k(comments, views),
    )


def _sum_known(values: list[float | None]) -> float | None:
    if not values or any(v is None for v in values):
        return None
    return sum(v or 0.0 for v in values)


def summed_rates(
    store: dict[str, Any], platforms: list[str], since: date, until: date
) -> PlatformRates:
    """All platforms together; unknown when any platform's input is unknown."""
    per = [platform_rates(store, p, since, until) for p in platforms]
    views = _sum_known([r.views for r in per])
    follows = _sum_known([r.net_follows for r in per])
    shares_saves: list[float | None] = []
    comments: list[float | None] = []
    for p in platforms:
        daily = (store.get("daily") or {}).get(p) or {}
        rows = [daily[d] for d in _days(since, until) if d in daily]
        shares_saves.append(
            sum(r.get("shares", 0.0) + r.get("saves", 0.0) for r in rows)
            if rows
            else None
        )
        comments.append(sum(r.get("comments", 0.0) for r in rows) if rows else None)
    return PlatformRates(
        platform="all",
        views=views,
        net_follows=None if follows is None else int(follows),
        follows_per_1k=_per_1k(follows, views),
        shares_saves_per_1k=_per_1k(_sum_known(shares_saves), views),
        comments_per_1k=_per_1k(_sum_known(comments), views),
    )


def median_view_percentage(
    post_metrics: Iterable[Any], since: date, until: date
) -> float | None:
    """Median YouTube average view percentage of posts published in the range."""
    values = []
    for m in post_metrics:
        published = str(getattr(m, "published_at", "") or "")[:10]
        if not (str(since) <= published <= str(until)):
            continue
        row = (getattr(m, "platform_metrics", {}) or {}).get("youtube") or {}
        value = _number(row.get("avgViewPercentage"))
        if value is not None:
            values.append(value)
    return round(statistics.median(values), 2) if values else None


def instagram_profile_views_per_1k(
    store: dict[str, Any], since: date, until: date
) -> float | None:
    profile = store.get("instagram_profile") or {}
    days = [profile[d] for d in _days(since, until) if d in profile]
    if not days:
        return None
    views = platform_rates(store, "instagram", since, until).views
    return _per_1k(sum(d.get("profile_views", 0.0) for d in days), views)
