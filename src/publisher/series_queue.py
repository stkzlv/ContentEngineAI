"""Keep a series claim queue and grouping in step with publishing.

A series render leaves `series.json` beside its video (the claim, the queue
file and the grouping, written by the producer), so the publisher needs no
video config. The claim row goes to `scheduled` with its post id when the
publisher records the post, and to `live` when the delivery sweep sees the
post live on any platform (REQ-PUB-168). A YouTube leg joins the series
playlist on upload; the groupings the provider has no call for are logged as
manual steps with the post's URL once it is live (REQ-PUB-169).
Bookkeeping: nothing here raises into a publish.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.claim_queue import set_published
from src.utils.outputs_paths import durable_state_path

logger = logging.getLogger(__name__)

SERIES_FILE = "series.json"
# Post ids whose manual grouping steps were logged, so each is logged once.
GROUPING_LOG = "series_grouping_logged.json"


def _read(product_id: str, outputs_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads((outputs_dir / product_id / SERIES_FILE).read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def series_groups(product_id: str, outputs_dir: Path) -> dict[str, Any]:
    """The render's series grouping, or {} for none."""
    groups = _read(product_id, outputs_dir).get("groups")
    return groups if isinstance(groups, dict) else {}


def with_playlist(
    platform_contents: dict[str, dict[str, Any]] | None, groups: dict[str, Any]
) -> dict[str, dict[str, Any]] | None:
    """`platform_contents` with the YouTube playlist id added, when set."""
    playlist = groups.get("youtube_playlist_id")
    if not playlist:
        return platform_contents
    contents = dict(platform_contents or {})
    contents["youtube"] = {**contents.get("youtube", {}), "playlist_id": playlist}
    return contents


def series_record(product_id: str, outputs_dir: Path) -> tuple[Path, str] | None:
    """The render's queue file and claim, or None for a non-series render."""
    data = _read(product_id, outputs_dir)
    queue, claim = data.get("queue_file"), data.get("claim")
    if not isinstance(queue, str) or not isinstance(claim, str) or not claim:
        return None
    return Path(queue), claim


def mark_scheduled(found: tuple[Path, str], post_id: str) -> None:
    """Set a series render's claim row to `scheduled` with its post id."""
    set_published(found[0], found[1], "scheduled", post_id)


# A partial post is live on at least one platform.
LIVE_STATUSES = frozenset({"published", "partial"})


def _history_series(outputs_dir: Path) -> dict[str, dict[str, Any]]:
    """Post id -> its series data (claim, queue_file, groups) from history."""
    from src.publisher.tracking import load_tracking

    found: dict[str, dict[str, Any]] = {}
    for record in load_tracking(outputs_dir).get("posts", {}).values():
        series = record.get("series") if isinstance(record, dict) else None
        if isinstance(series, dict) and (
            (series.get("claim") and series.get("queue_file")) or series.get("groups")
        ):
            found[str(record.get("post_id"))] = series
    return found


def has_series_posts(outputs_dir: Path) -> bool:
    """Whether publish history holds any series post."""
    return bool(_history_series(outputs_dir))


def manual_steps(groups: dict[str, Any], urls: dict[str, str]) -> list[str]:
    """The groupings the provider has no call for, as steps to do by hand."""
    steps = []
    tiktok = groups.get("tiktok_playlist")
    if tiktok:
        steps.append(
            f"add the TikTok post {urls.get('tiktok') or '(no URL yet)'} "
            f"to the playlist {tiktok!r}"
        )
    if groups.get("instagram_link"):
        steps.append(
            f"link the Instagram post {urls.get('instagram') or '(no URL yet)'} "
            "to the previous episode"
        )
    return steps


def _logged(outputs_dir: Path) -> set[str]:
    try:
        data = json.loads(durable_state_path(outputs_dir, GROUPING_LOG).read_text())
    except (OSError, ValueError):
        return set()
    return {str(x) for x in data} if isinstance(data, list) else set()


def _save_logged(outputs_dir: Path, ids: set[str]) -> None:
    path = durable_state_path(outputs_dir, GROUPING_LOG)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(ids)))
    except OSError as e:
        logger.warning("Series grouping log not saved: %s", e)


def mark_live(posts: list[dict[str, Any]], outputs_dir: Path) -> int:
    """Set each live series post's claim row to `live`, and log its manual
    grouping steps once; returns how many rows changed.

    Read from publish history, not the render directory, which cleanup
    removes after publishing.
    """
    series_by_post = _history_series(outputs_dir)
    logged = _logged(outputs_dir)
    newly_logged = set()
    marked = 0
    for post in posts:
        post_id = str(post.get("id"))
        series = series_by_post.get(post_id)
        if not series or str(post.get("status") or "").lower() not in LIVE_STATUSES:
            continue
        if (
            series.get("claim")
            and series.get("queue_file")
            and set_published(Path(series["queue_file"]), series["claim"], "live")
        ):
            marked += 1
        groups = series.get("groups")
        if isinstance(groups, dict) and post_id not in logged:
            urls = {
                str(leg.get("platform")): str(leg.get("url"))
                for leg in post.get("platforms") or []
                if isinstance(leg, dict) and leg.get("url")
            }
            for step in manual_steps(groups, urls):
                logger.info(
                    "Series grouping, manual step for post %s: %s", post_id, step
                )
            newly_logged.add(post_id)
    if newly_logged:
        _save_logged(outputs_dir, logged | newly_logged)
    return marked
