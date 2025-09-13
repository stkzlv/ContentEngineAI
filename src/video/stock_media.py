"""Stock Media Fetcher Module

This module handles the fetching and management of stock media (images and videos) from
external providers like Pexels. It provides functionality for searching, downloading,
and tracking attribution information for media used in video production.

Key features:
- API integration with Pexels for high-quality stock media
- Query caching to reduce redundant API calls
- Concurrent download management with rate limiting
- Media metadata tracking for proper attribution
- Support for both images and videos with format validation

The module implements resilient error handling and fallback mechanisms to ensure
reliable media acquisition even when API issues occur.
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from pexelsapi.pexels import Pexels

from src.utils import download_file, ensure_dirs_exist, get_filename_from_url
from src.utils.circuit_breaker import pexels_circuit_breaker
from src.video.video_config import MediaSettings, StockMediaSettings

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class StockMediaInfo:
    """Container for stock media metadata and attribution information.

    This class stores all the necessary information about a stock media item,
    including its source, type, author, and local path. It's used for both
    tracking downloaded media and generating proper attribution.

    Attributes
    ----------
        source: The provider name (e.g., "Pexels")
        type: Media type ("image" or "video")
        url: Original URL of the media
        author: Creator/photographer name for attribution
        path: Local filesystem path where the media is stored
        duration: Length in seconds (for videos only, None for images)

    """

    source: str
    type: str
    url: str
    author: str
    path: Path
    duration: float | None = None


class StockMediaFetcher:
    """Fetches and manages stock media from external providers.

    This class handles the searching, downloading, and tracking of stock media
    (images and videos) from providers like Pexels. It implements caching to reduce
    API calls and manages concurrent downloads with rate limiting.

    The fetcher supports searching by keywords, downloading media to local storage,
    and tracking attribution information for proper crediting in videos.
    """

    def __init__(
        self,
        settings: StockMediaSettings,
        secrets: dict[str, str],
        media_settings: MediaSettings,
        api_settings=None,
    ):
        """Initialize the stock media fetcher with configuration and credentials.

        Args:
        ----
            settings: Configuration for stock media providers and search parameters
            secrets: Dictionary containing API keys and credentials
            media_settings: Media format and quality settings
            api_settings: API configuration for timeouts and concurrency limits

        """
        self.settings = settings
        self.secrets = secrets
        self.media_settings = media_settings
        self.api_settings = api_settings
        self.api_key = secrets.get(settings.pexels_api_key_env_var)
        self.pexels_client = None
        # Cache for API query results
        self._query_cache: dict[
            tuple[str, str, int, str, str], list[dict[str, Any]]
        ] = {}

        # Use configurable concurrent download limit
        concurrent_downloads = (
            api_settings.stock_media_concurrent_downloads if api_settings else 5
        )
        self._semaphore = asyncio.Semaphore(concurrent_downloads)

        # Initialize Pexels client if API key is available
        if not self.api_key:
            logger.warning(
                f"Pexels API key not found in {settings.pexels_api_key_env_var}. "
                "Stock media fetching disabled."
            )
        else:
            try:
                self.pexels_client = Pexels(self.api_key)
                logger.debug("Pexels API client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Pexels API client: {e}")
                self.pexels_client = None

    async def _search_and_select_pexels(
        self,
        search_query: str,
        item_type: str,
        count: int,
        orientation: str = "portrait",
        size: str = "large",
    ) -> list[dict[str, Any]]:
        if not self.pexels_client or count <= 0 or not search_query:
            return []

        cache_key = (search_query, item_type, count, orientation, size)
        if cache_key in self._query_cache:
            logger.debug(f"Using cached Pexels results for {cache_key}")
            return self._query_cache[cache_key]

        # Use configurable search multiplier and max per page
        multiplier = (
            self.api_settings.stock_media_search_multiplier if self.api_settings else 2
        )
        max_per_page = (
            self.api_settings.stock_media_max_per_page if self.api_settings else 80
        )
        per_page = min(count * multiplier, max_per_page)
        logger.debug(
            f"Searching Pexels for {item_type} with query '{search_query}' "
            f"(per_page={per_page})..."
        )

        fetched_items_raw: list[dict] = []
        try:
            if item_type == "photos":
                result = await asyncio.to_thread(
                    self.pexels_client.search_photos,
                    query=search_query,
                    per_page=per_page,
                    orientation=orientation,
                    size=size,
                )
                fetched_items_raw = (
                    result.get("photos", []) if isinstance(result, dict) else []
                )
            elif item_type == "videos":
                result = await asyncio.to_thread(
                    self.pexels_client.search_videos,
                    query=search_query,
                    per_page=per_page,
                    orientation=orientation,
                    min_duration=self.media_settings.stock_video_min_duration_sec,
                    max_duration=self.media_settings.stock_video_max_duration_sec,
                )
                fetched_items_raw = (
                    result.get("videos", []) if isinstance(result, dict) else []
                )

            logger.debug(
                f"Found {len(fetched_items_raw)} potential {item_type} matching "
                f"query '{search_query}'."
            )

        except Exception as e:
            logger.error(
                f"Error searching Pexels API for {item_type} with query "
                f"'{search_query}': {e}",
                exc_info=True,
            )
            return []

        processed_items: list[dict[str, Any]] = []
        if item_type == "photos":
            for photo in fetched_items_raw:
                original_url = photo.get("src", {}).get("original")
                if (
                    original_url
                    and isinstance(original_url, str)
                    and original_url.lower().startswith("http")
                ):
                    processed_items.append(
                        {
                            "id": photo.get("id"),
                            "url": original_url,
                            "photographer": photo.get("photographer") or "Unknown",
                            "type": "image",
                            "duration": None,
                            "source": self.settings.source,
                        }
                    )
        elif item_type == "videos":
            for video in fetched_items_raw:
                best_file_url = None
                files = sorted(
                    video.get("video_files", []),
                    key=lambda x: x.get("width", 0) * x.get("height", 0),
                    reverse=True,
                )
                for vf in files:
                    if vf.get("link") and vf.get("file_type") == "video/mp4":
                        best_file_url = vf["link"]
                        break
                if (
                    best_file_url
                    and isinstance(best_file_url, str)
                    and best_file_url.lower().startswith("http")
                ):
                    processed_items.append(
                        {
                            "id": video.get("id"),
                            "url": best_file_url,
                            "photographer": video.get("user", {}).get("name")
                            or "Unknown",
                            "type": "video",
                            "duration": video.get("duration"),
                            "source": self.settings.source,
                        }
                    )

        selected_items = random.sample(  # noqa: S311
            processed_items, min(count, len(processed_items))
        )
        self._query_cache[cache_key] = selected_items  # Cache results
        logger.debug(f"Selected {len(selected_items)} {item_type} for download.")
        return selected_items

    @pexels_circuit_breaker
    async def fetch_and_download_stock(
        self,
        keywords: list[str],
        image_count: int,
        video_count: int,
        download_dir: Path,
        session: aiohttp.ClientSession,
    ) -> list[StockMediaInfo]:
        all_downloaded_info: list[StockMediaInfo] = []
        ensure_dirs_exist(download_dir)
        search_query = " ".join(keywords)
        download_tasks = []

        async def download_with_semaphore(url: str, save_path: Path) -> bool:
            async with self._semaphore:
                api_config = self.api_settings or {}
                return await download_file(
                    url,
                    save_path,
                    session,
                    timeout_sec=getattr(api_config, "download_timeout_sec", 30),
                    retry_attempts=getattr(api_config, "download_retry_attempts", 3),
                    retry_min_wait_sec=getattr(
                        api_config, "download_retry_min_wait_sec", 2
                    ),
                    retry_max_wait_sec=getattr(
                        api_config, "download_retry_max_wait_sec", 10
                    ),
                )

        if image_count > 0 and self.pexels_client:
            selected_images = await self._search_and_select_pexels(
                search_query=search_query,
                item_type="photos",
                count=image_count,
                orientation="portrait",
                size="large",
            )
            logger.info(f"Queuing {len(selected_images)} stock images for download.")
            for i, item in enumerate(selected_images):
                img_url = item.get("url")
                author = item.get("photographer") or "Unknown"
                item_id = item.get("id")
                source = item.get("source", self.settings.source)
                if not img_url:
                    logger.warning(
                        f"Skipping stock image {item_id or i} due to missing URL."
                    )
                    continue
                save_filename = get_filename_from_url(
                    img_url, item_id or i, "image/jpeg"
                )
                save_path = download_dir / save_filename
                task = asyncio.create_task(download_with_semaphore(img_url, save_path))
                download_tasks.append(
                    (task, "image", img_url, author, save_path, source, None)
                )

        if video_count > 0 and self.pexels_client:
            selected_videos = await self._search_and_select_pexels(
                search_query=search_query,
                item_type="videos",
                count=video_count,
                orientation="portrait",
            )
            logger.info(f"Queuing {len(selected_videos)} stock videos for download.")
            for i, item in enumerate(selected_videos):
                vid_url = item.get("url")
                author = item.get("photographer") or "Unknown"
                item_id = item.get("id")
                duration = item.get("duration")
                source = item.get("source", self.settings.source)
                if not vid_url:
                    logger.warning(
                        f"Skipping stock video {item_id or i} due to missing URL."
                    )
                    continue
                save_filename = get_filename_from_url(
                    vid_url, item_id or i, "video/mp4"
                )
                save_path = download_dir / save_filename
                task = asyncio.create_task(download_with_semaphore(vid_url, save_path))
                download_tasks.append(
                    (task, "video", vid_url, author, save_path, source, duration)
                )

        if not download_tasks:
            logger.warning(
                f"No stock media queued for download for keywords: {search_query}."
            )
            return []

        logger.info(f"Starting {len(download_tasks)} stock media downloads...")
        download_results = await asyncio.gather(
            *[task for task, *_ in download_tasks], return_exceptions=True
        )

        for i, result in enumerate(download_results):
            task_info = download_tasks[i]
            media_type, media_url, author, save_path, source, duration = task_info[1:]
            if isinstance(result, Exception):
                logger.error(
                    f"Stock media download failed for {media_type} from {media_url}: "
                    f"{result}"
                )
            elif result is True:
                if save_path.exists() and save_path.stat().st_size > 0:
                    all_downloaded_info.append(
                        StockMediaInfo(
                            source=source,
                            type=media_type,
                            url=media_url,
                            author=author,
                            path=save_path,
                            duration=duration,
                        )
                    )
                else:
                    logger.error(
                        f"Stock media download reported success but file "
                        f"missing/empty: {save_path}"
                    )

        logger.info(
            f"Successfully downloaded {len(all_downloaded_info)} stock media items."
        )
        return all_downloaded_info
