"""Background Processing Framework

This module provides utilities for running resource-intensive operations
in the background
while other pipeline steps are executing. This includes pre-fetching stock media,
warming TTS models, and pre-loading common resources to reduce cold-start overhead.

Key features:
- Async background task management with proper lifecycle
- Resource pre-loading based on product data
- TTS model warming with configurable models
- Stock media pre-fetching with intelligent caching
- Safe resource cleanup and error handling
"""

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.scraper.amazon.scraper import ProductData
from src.video.video_config import VideoConfig, VideoProfile

logger = logging.getLogger(__name__)


@dataclass
class BackgroundTask:
    """Container for a background task with metadata."""

    task_id: str
    name: str
    task: asyncio.Task
    start_time: float
    priority: int = 1  # Lower number = higher priority
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get the current duration of the task."""
        return time.time() - self.start_time

    @property
    def is_completed(self) -> bool:
        """Check if the task is completed."""
        return self.task.done()

    @property
    def is_successful(self) -> bool:
        """Check if the task completed successfully."""
        return (
            self.task.done()
            and not self.task.cancelled()
            and self.task.exception() is None
        )


class BackgroundProcessor:
    """Manages background processing tasks for pipeline optimization."""

    def __init__(
        self,
        max_concurrent_tasks: int = 3,
        thread_pool_workers: int = 2,
        max_recent_completed: int = 5,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_recent_completed = max_recent_completed
        self.active_tasks: dict[str, BackgroundTask] = {}
        self.completed_tasks: list[BackgroundTask] = []
        self.thread_pool = ThreadPoolExecutor(
            max_workers=thread_pool_workers, thread_name_prefix="bg_proc"
        )
        self._shutdown_event = asyncio.Event()

    async def start_task(
        self,
        task_id: str,
        name: str,
        coro_func: Callable,
        *args,
        priority: int = 1,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> BackgroundTask | None:
        """Start a background task if resources are available."""
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            logger.debug(f"Background task queue full, skipping task: {name}")
            return None

        if task_id in self.active_tasks:
            logger.debug(f"Background task already running: {task_id}")
            return self.active_tasks[task_id]

        try:
            task = asyncio.create_task(coro_func(*args, **kwargs))
            bg_task = BackgroundTask(
                task_id=task_id,
                name=name,
                task=task,
                start_time=time.time(),
                priority=priority,
                metadata=metadata or {},
            )

            self.active_tasks[task_id] = bg_task
            logger.debug(f"Started background task: {name} (ID: {task_id})")

            # Add completion callback
            task.add_done_callback(lambda t: self._on_task_complete(task_id))

            return bg_task

        except Exception as e:
            logger.error(f"Failed to start background task {name}: {e}")
            return None

    def _on_task_complete(self, task_id: str) -> None:
        """Handle task completion."""
        if task_id in self.active_tasks:
            bg_task = self.active_tasks.pop(task_id)
            self.completed_tasks.append(bg_task)

            if bg_task.is_successful:
                logger.debug(
                    f"Background task completed successfully: {bg_task.name} "
                    f"({bg_task.duration:.2f}s)"
                )
            else:
                if bg_task.task.cancelled():
                    logger.debug(f"Background task cancelled: {bg_task.name}")
                else:
                    error = bg_task.task.exception()
                    logger.warning(f"Background task failed: {bg_task.name} - {error}")

    async def wait_for_task(
        self, task_id: str, timeout: float | None = None
    ) -> Any | None:
        """Wait for a specific background task to complete and return its result."""
        if task_id in self.active_tasks:
            bg_task = self.active_tasks[task_id]
            try:
                if timeout:
                    result = await asyncio.wait_for(bg_task.task, timeout=timeout)
                else:
                    result = await bg_task.task
                logger.debug(f"Background task result retrieved: {bg_task.name}")
                return result
            except TimeoutError:
                logger.warning(f"Background task timed out: {bg_task.name}")
                return None
            except Exception as e:
                logger.error(f"Background task error: {bg_task.name} - {e}")
                return None
        else:
            # Check completed tasks
            for bg_task in self.completed_tasks:
                if bg_task.task_id == task_id and bg_task.is_successful:
                    try:
                        return bg_task.task.result()
                    except Exception as e:
                        logger.error(f"Error retrieving completed task result: {e}")
                        return None
        return None

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get the status of a background task."""
        if task_id in self.active_tasks:
            bg_task = self.active_tasks[task_id]
            return {
                "status": "running",
                "name": bg_task.name,
                "duration": bg_task.duration,
                "priority": bg_task.priority,
                "metadata": bg_task.metadata,
            }

        for bg_task in self.completed_tasks:
            if bg_task.task_id == task_id:
                return {
                    "status": "completed" if bg_task.is_successful else "failed",
                    "name": bg_task.name,
                    "duration": bg_task.duration,
                    "priority": bg_task.priority,
                    "metadata": bg_task.metadata,
                }

        return {"status": "not_found"}

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a background task."""
        if task_id in self.active_tasks:
            bg_task = self.active_tasks[task_id]
            bg_task.task.cancel()
            logger.debug(f"Cancelled background task: {bg_task.name}")
            return True
        return False

    async def cleanup(self, timeout_sec: float = 5.0) -> None:
        """Clean up all background tasks and resources."""
        logger.debug("Cleaning up background processor...")

        # Cancel all active tasks
        for task_id in list(self.active_tasks.keys()):
            self.cancel_task(task_id)

        # Wait for tasks to complete or timeout
        if self.active_tasks:
            tasks = [bg_task.task for bg_task in self.active_tasks.values()]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_sec
                )
            except TimeoutError:
                logger.warning(
                    "Some background tasks did not complete within cleanup timeout"
                )

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        self.active_tasks.clear()
        logger.debug("Background processor cleanup completed")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of background processing activity."""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "max_concurrent": self.max_concurrent_tasks,
            "active_task_names": [
                bg_task.name for bg_task in self.active_tasks.values()
            ],
            "recent_completed": [
                {
                    "name": bg_task.name,
                    "duration": bg_task.duration,
                    "successful": bg_task.is_successful,
                }
                for bg_task in self.completed_tasks[-self.max_recent_completed :]
            ],
        }


class ResourcePreloader:
    """Handles pre-loading of common resources based on product data."""

    def __init__(self, background_processor: BackgroundProcessor):
        self.bg_processor = background_processor
        self._preloaded_resources: dict[str, Any] = {}

    async def preload_for_product(
        self, product: ProductData, config: VideoConfig, profile: VideoProfile
    ) -> list[str]:
        """Start preloading resources based on product data."""
        task_ids = []

        # Extract keywords for stock media pre-fetching
        keywords = self._extract_keywords(product)

        if profile.use_stock_images or profile.use_stock_videos:
            task_id = f"stock_media_{product.asin or 'unknown'}"
            task = await self.bg_processor.start_task(
                task_id=task_id,
                name=f"Stock Media Prefetch ({product.title[:30]}...)",
                coro_func=self._prefetch_stock_media,
                keywords=keywords,
                profile=profile,
                config=config,
                priority=2,
                metadata={"product_id": product.asin, "keywords": keywords},
            )
            if task:
                task_ids.append(task_id)

        return task_ids

    def _extract_keywords(self, product: ProductData) -> list[str]:
        """Extract relevant keywords from product data for stock media search."""
        keywords = []

        # Extract from title (remove brand names and model numbers)
        if product.title:
            title_words = product.title.lower().split()
            # Filter out common e-commerce words and keep descriptive terms
            descriptive_words = [
                word
                for word in title_words
                if len(word) > 3
                and word
                not in {
                    "with",
                    "pack",
                    "set",
                    "piece",
                    "black",
                    "white",
                    "blue",
                    "red",
                    "large",
                    "small",
                    "medium",
                    "size",
                    "inch",
                    "foot",
                    "yard",
                }
            ]
            keywords.extend(descriptive_words[:3])  # Top 3 descriptive words

        # Extract from category or features if available
        if hasattr(product, "category") and product.category:
            keywords.append(product.category.lower())

        # Add generic terms based on common product types
        if any(
            word in product.title.lower() for word in ["phone", "mobile", "smartphone"]
        ):
            keywords.extend(["technology", "communication"])
        elif any(
            word in product.title.lower() for word in ["headphone", "speaker", "audio"]
        ):
            keywords.extend(["music", "sound"])
        elif any(
            word in product.title.lower() for word in ["watch", "fitness", "sport"]
        ):
            keywords.extend(["lifestyle", "fitness"])
        else:
            keywords.append("product")  # Generic fallback

        return list(set(keywords))[:5]  # Return unique keywords, max 5

    async def _prefetch_stock_media(
        self, keywords: list[str], profile: VideoProfile, config: VideoConfig
    ) -> dict[str, Any]:
        """Pre-fetch stock media based on keywords."""
        try:
            from src.video.stock_media import StockMediaFetcher

            fetcher = StockMediaFetcher(
                config.stock_media_settings,
                secrets={},  # Empty secrets for pre-fetch
                media_settings=config.media_settings,
                api_settings=None,
            )

            # Prepare search terms
            search_query = " ".join(keywords[:2])  # Use top 2 keywords

            # Pre-fetch a small amount of media for caching
            prefetch_counts = {
                "images": (
                    min(profile.stock_image_count, 3) if profile.use_stock_images else 0
                ),
                "videos": (
                    min(profile.stock_video_count, 2) if profile.use_stock_videos else 0
                ),
            }

            results = {}

            if prefetch_counts["images"] > 0 or prefetch_counts["videos"] > 0:
                logger.debug(f"Pre-fetching stock media for query: {search_query}")
                # Create session for downloading (would need real session in practice)
                from aiohttp import ClientSession

                async with ClientSession() as session:
                    media_items = await fetcher.fetch_and_download_stock(
                        keywords=keywords[:2],
                        image_count=prefetch_counts["images"],
                        video_count=prefetch_counts["videos"],
                        download_dir=Path.cwd() / "outputs" / "temp" / "stock_prefetch",
                        session=session,
                    )
                results["media_items"] = media_items
                logger.debug(f"Pre-fetched {len(media_items)} stock media items")

            # Cache the results for later use
            cache_key = f"stock_media_{hash(search_query)}"
            self._preloaded_resources[cache_key] = {
                "query": search_query,
                "results": results,
                "timestamp": time.time(),
            }

            return results

        except Exception as e:
            logger.warning(f"Stock media pre-fetching failed: {e}")
            return {}

    def get_preloaded_stock_media(self, keywords: list[str]) -> dict[str, Any] | None:
        """Retrieve pre-loaded stock media if available."""
        search_query = " ".join(keywords[:2])
        cache_key = f"stock_media_{hash(search_query)}"

        if cache_key in self._preloaded_resources:
            cached = self._preloaded_resources[cache_key]
            # Check if cache is still fresh (within 10 minutes)
            if time.time() - cached["timestamp"] < 600:
                logger.debug(f"Using pre-loaded stock media for query: {search_query}")
                results: dict[str, Any] = cached["results"]
                return results
            else:
                # Remove stale cache
                del self._preloaded_resources[cache_key]

        return None


class TTSWarmer:
    """Handles TTS model warming and pre-loading."""

    def __init__(self, background_processor: BackgroundProcessor):
        self.bg_processor = background_processor
        self._warmed_models: set[str] = set()

    async def warm_tts_models(self, config: VideoConfig) -> list[str]:
        """Start warming TTS models in the background."""
        task_ids = []

        # Get TTS providers from config
        tts_config = config.tts_config

        # Check each provider individually
        providers_to_check = [
            ("coqui", tts_config.coqui),
            ("google_cloud", tts_config.google_cloud),
        ]

        for provider_name, provider_config in providers_to_check:
            if provider_config and getattr(provider_config, "enabled", True):
                task_id = f"tts_warm_{provider_name}"

                if provider_name == "coqui":
                    task = await self.bg_processor.start_task(
                        task_id=task_id,
                        name=f"TTS Model Warming ({provider_name})",
                        coro_func=self._warm_coqui_model,
                        provider_config=provider_config,
                        priority=3,
                        metadata={"provider": provider_name},
                    )
                elif provider_name == "google_cloud":
                    task = await self.bg_processor.start_task(
                        task_id=task_id,
                        name=f"TTS Client Warming ({provider_name})",
                        coro_func=self._warm_google_cloud_client,
                        provider_config=provider_config,
                        priority=3,
                        metadata={"provider": provider_name},
                    )

                if task:
                    task_ids.append(task_id)

        return task_ids

    async def _warm_coqui_model(self, provider_config) -> bool:
        """Warm up Coqui TTS model by loading it in a background thread."""
        try:
            loop = asyncio.get_event_loop()

            # Run model loading in thread pool to avoid blocking
            def load_model():
                try:
                    from src.video.tts import _initialize_coqui_tts_model

                    model = _initialize_coqui_tts_model(provider_config)
                    return model is not None
                except Exception as e:
                    logger.debug(f"Coqui model warming failed: {e}")
                    return False

            success = await loop.run_in_executor(
                self.bg_processor.thread_pool, load_model
            )

            if success:
                self._warmed_models.add("coqui")
                logger.debug("Coqui TTS model warmed successfully")

            result: bool = success
            return result

        except Exception as e:
            logger.warning(f"Failed to warm Coqui TTS model: {e}")
            return False

    async def _warm_google_cloud_client(self, provider_config) -> bool:
        """Warm up Google Cloud TTS client."""
        try:
            # Import here to avoid circular imports

            # Skip TTS manager initialization during warming to avoid complex config
            # requirements
            logger.debug("TTS client warming skipped during background processing")

            # Try to initialize Google Cloud TTS (this will cache the client)
            try:
                from google.cloud import texttospeech

                client = texttospeech.TextToSpeechClient()

                # Test with a minimal request to warm up the connection
                voices_request = texttospeech.ListVoicesRequest()
                voices = client.list_voices(request=voices_request)

                self._warmed_models.add("google_cloud")
                logger.debug(
                    f"Google Cloud TTS client warmed successfully "
                    f"({len(voices.voices)} voices available)"
                )
                return True

            except Exception as e:
                logger.debug(f"Google Cloud TTS warming failed: {e}")
                return False

        except Exception as e:
            logger.warning(f"Failed to warm Google Cloud TTS client: {e}")
            return False

    def is_model_warmed(self, provider_name: str) -> bool:
        """Check if a TTS model has been warmed."""
        return provider_name in self._warmed_models


# Global background processor instance
_global_background_processor: BackgroundProcessor | None = None


@asynccontextmanager
async def get_background_processor(
    max_concurrent_tasks: int = 3,
    thread_pool_workers: int = 2,
    max_recent_completed: int = 5,
):
    """Get a global background processor instance with proper lifecycle management."""
    global _global_background_processor

    if _global_background_processor is None:
        _global_background_processor = BackgroundProcessor(
            max_concurrent_tasks, thread_pool_workers, max_recent_completed
        )

    try:
        yield _global_background_processor
    finally:
        # Note: We don't automatically cleanup here since other parts of the pipeline
        # might still be using the processor. Cleanup should be called explicitly
        # at the end of the pipeline execution.
        pass


async def cleanup_global_background_processor():
    """Clean up the global background processor."""
    global _global_background_processor

    if _global_background_processor:
        await _global_background_processor.cleanup()
        _global_background_processor = None
