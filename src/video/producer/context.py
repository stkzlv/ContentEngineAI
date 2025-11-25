# src/video/producer/context.py
"""Pipeline context and exception classes."""

import asyncio
from pathlib import Path
from typing import Any

import aiohttp

from src.scraper.amazon.scraper import ProductData
from src.video.config import VideoConfig, VideoProfile


class PipelineError(Exception):
    """Custom exception for pipeline failures."""

    pass


class InsufficientMediaError(PipelineError):
    """Exception raised when product has insufficient media for video creation."""

    pass


class PipelineContext:
    """Container for pipeline state and artifacts."""

    def __init__(
        self,
        product: ProductData,
        profile: VideoProfile,
        profile_name: str,
        config: VideoConfig,
        secrets: dict,
        session: aiohttp.ClientSession,
        run_paths: dict,
        debug_mode: bool,
        cli_overrides: dict[str, Any] | None = None,
    ):
        self.product = product
        self.profile = profile
        self.profile_name = profile_name
        self.config = config
        self.secrets = secrets
        self.session = session
        self.run_paths = run_paths
        self.debug_mode = debug_mode
        self.cli_overrides = cli_overrides or {}
        self.visuals: list[Path] | None = None
        self.script: str | None = None
        self.description: str | None = None
        self.voiceover_duration: float | None = None
        self.state: dict[str, Any] = {}

        # Background processing support
        self.background_processor: Any | None = None
        self.resource_preloader: Any | None = None
        self.tts_warmer: Any | None = None
        self.preload_task_ids: list[str] = []

        # Pipeline state synchronization
        self._state_lock = asyncio.Lock()

        # Additional attributes for pipeline steps
        self.scraped_images: list[Path] = []
        self.scraped_videos: list[Path] = []
        self.stock_media: list[Any] = []
