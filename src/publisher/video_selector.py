"""Per-platform video file selection for the publisher.

Phase 1.3 introduces per-platform video profile routing. The producer
already writes one MP4 per profile under the per-ASIN output directory
(`outputs/<asin>/video_<asin>_<profile>.mp4`). The publisher resolves
which file to upload for a given platform from `PublisherConfig.profiles`.

When `profiles` is empty or the routed profile has no matching render on
disk, the helper falls back to the first `video_<asin>_*.mp4` match — the
pre-1.3 behaviour. Logged at INFO when fallback fires so a missing
per-platform render is visible in `outputs/logs/publisher.log`.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def select_video_for_platform(
    product_dir: Path,
    asin: str,
    platform: str,
    profiles: dict[str, str] | None,
) -> Path | None:
    """Pick the right rendered video file for `platform`.

    Args:
    ----
        product_dir: `outputs/<asin>/`.
        asin: Product ID.
        platform: Lowercased platform name (`tiktok`/`youtube`/`instagram`).
        profiles: `PublisherConfig.profiles` mapping or `None`.

    Returns:
    -------
        Path to the chosen `video_<asin>_<profile>.mp4`, or the first
        `video_<asin>_*.mp4` if no per-platform routing applies. `None`
        when no rendered video exists at all.

    """
    if profiles:
        profile_name = profiles.get(platform.lower())
        if profile_name:
            candidate = product_dir / f"video_{asin}_{profile_name}.mp4"
            if candidate.exists():
                return candidate
            logger.info(
                "Routed profile %r for %s has no render at %s; "
                "falling back to first available video.",
                profile_name,
                platform,
                candidate.name,
            )

    matches = sorted(product_dir.glob(f"video_{asin}_*.mp4"))
    if matches:
        return matches[0]
    return None
