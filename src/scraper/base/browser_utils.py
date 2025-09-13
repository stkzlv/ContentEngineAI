"""Shared browser utilities for the multi-platform scraper architecture.

This module provides common browser automation utilities that can be used across
all platform scrapers, including browser configuration, detection, and common
interaction patterns.
"""

import logging
import os
import platform
from typing import Any

from .config import get_config_manager
from .models import Platform
from .utils import detect_monitors, get_optimal_browser_position


class BaseBrowserConfig:
    """Base browser configuration class for all platforms."""

    def __init__(self, platform: Platform, debug_mode: bool = False):
        """Initialize browser configuration.

        Args:
        ----
            platform: The platform this configuration is for
            debug_mode: Whether to enable debug mode

        """
        self.platform = platform
        self.debug_mode = debug_mode
        self.config_manager = get_config_manager()
        self.logger = logging.getLogger(__name__)

    def get_browser_options(self) -> dict[str, Any]:
        """Get browser options for the platform.

        Returns
        -------
            Dictionary of browser configuration options

        """
        # Start with base configuration
        options = {
            "headless": not self.debug_mode,
            "close_on_crash": not self.debug_mode,
            "reuse_driver": False,
            "keep_alive": True,
        }

        # Add platform-specific options
        try:
            platform_config = self.config_manager.get_platform_config(self.platform)
            browser_config = platform_config.get("browser_config", {})
            options.update(browser_config)
        except Exception as e:
            self.logger.debug(f"Could not load platform browser config: {e}")

        # Apply debug mode overrides
        if self.debug_mode:
            options.update(  # type: ignore[dict-item]
                {
                    "headless": False,
                    "close_on_crash": False,
                    "window_size": self._get_debug_window_size(),
                    "window_position": self._get_debug_window_position(),
                }
            )

        return options

    def get_chrome_options(self) -> list[str]:
        """Get Chrome-specific command line options.

        Returns
        -------
            List of Chrome command line arguments

        """
        chrome_options = [
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--disable-notifications",
            "--disable-popup-blocking",
            "--disable-blink-features=AutomationControlled",
        ]

        # Environment-specific options
        if self._is_docker_environment():
            chrome_options.extend(
                [
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-extensions",
                ]
            )

        if self._is_ci_environment():
            chrome_options.extend(
                [
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ]
            )

        # Debug mode options
        if self.debug_mode:
            chrome_options.extend(
                [
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                ]
            )

            # Set window position and size for debug mode
            window_x, window_y, window_width, window_height = (
                self._get_debug_window_geometry()
            )
            chrome_options.append(f"--window-position={window_x},{window_y}")
            chrome_options.append(f"--window-size={window_width},{window_height}")

        return chrome_options

    def get_user_agent(self) -> str:
        """Get user agent string for the platform.

        Returns
        -------
            User agent string

        """
        try:
            platform_config = self.config_manager.get_platform_config(self.platform)
            headers = platform_config.get("http_headers", {})

            # Try different header configurations
            for config_key in ["browser", "media_download", "video_validation"]:
                config_headers = headers.get(config_key, {})
                if "User-Agent" in config_headers:
                    return config_headers["User-Agent"]

        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(
                f"Could not load user agent from config: {e}"
            )

        # Default user agent
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    def _get_debug_window_size(self) -> tuple[int, int]:
        """Get debug window size.

        Returns
        -------
            Tuple of (width, height)

        """
        _, _, width, height = self._get_debug_window_geometry()
        return (width, height)

    def _get_debug_window_position(self) -> tuple[int, int]:
        """Get debug window position.

        Returns
        -------
            Tuple of (x, y)

        """
        x, y, _, _ = self._get_debug_window_geometry()
        return (x, y)

    def _get_debug_window_geometry(self) -> tuple[int, int, int, int]:
        """Get complete debug window geometry.

        Returns
        -------
            Tuple of (x, y, width, height)

        """
        try:
            monitors = detect_monitors()
            return get_optimal_browser_position(monitors)
        except Exception:
            # Fallback geometry
            return (100, 100, 1200, 800)

    def _is_docker_environment(self) -> bool:
        """Check if running in Docker environment.

        Returns
        -------
            True if in Docker, False otherwise

        """
        return (
            os.path.exists("/.dockerenv")
            or os.environ.get("DOCKER", "").lower() == "true"
        )

    def _is_ci_environment(self) -> bool:
        """Check if running in CI environment.

        Returns
        -------
            True if in CI, False otherwise

        """
        ci_vars = ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL"]
        return any(os.environ.get(var, "").lower() == "true" for var in ci_vars)

    def _has_display(self) -> bool:
        """Check if display is available.

        Returns
        -------
            True if display available, False otherwise

        """
        if platform.system() == "Linux":
            return os.environ.get("DISPLAY") is not None
        return True  # Assume display available on Windows/macOS


class BrowserDetection:
    """Browser and environment detection utilities."""

    @staticmethod
    def detect_captcha_challenge(page_source: str) -> bool:
        """Detect if page contains CAPTCHA challenge.

        Args:
        ----
            page_source: HTML source of the page

        Returns:
        -------
            True if CAPTCHA detected, False otherwise

        """
        captcha_indicators = [
            "captcha",
            "robot",
            "automated",
            "unusual traffic",
            "verify you are human",
            "security check",
            "please complete",
        ]

        page_lower = page_source.lower()
        return any(indicator in page_lower for indicator in captcha_indicators)

    @staticmethod
    def detect_bot_detection(page_source: str) -> bool:
        """Detect if page contains bot detection mechanisms.

        Args:
        ----
            page_source: HTML source of the page

        Returns:
        -------
            True if bot detection found, False otherwise

        """
        bot_detection_indicators = [
            "blocked",
            "access denied",
            "forbidden",
            "rate limit",
            "too many requests",
            "automated queries",
            "blocked by",
        ]

        page_lower = page_source.lower()
        return any(indicator in page_lower for indicator in bot_detection_indicators)

    @staticmethod
    def detect_regional_redirect(
        current_url: str, expected_domain: str
    ) -> tuple[bool, str | None]:
        """Detect if page was redirected to a different region.

        Args:
        ----
            current_url: Current page URL
            expected_domain: Expected domain (e.g., amazon.com)

        Returns:
        -------
            Tuple of (is_redirected, redirect_info)

        """
        if expected_domain not in current_url:
            # Extract actual domain
            from urllib.parse import urlparse

            parsed = urlparse(current_url)
            actual_domain = parsed.netloc

            return True, f"Redirected from {expected_domain} to {actual_domain}"

        return False, None

    @staticmethod
    def get_environment_info() -> dict[str, Any]:
        """Get comprehensive environment information for debugging.

        Returns
        -------
            Dictionary of environment information

        """
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python_version": platform.python_version(),
            "is_docker": (
                os.path.exists("/.dockerenv")
                or os.environ.get("DOCKER", "").lower() == "true"
            ),
            "is_ci": any(
                os.environ.get(var, "").lower() == "true"
                for var in ["CI", "GITHUB_ACTIONS", "GITLAB_CI"]
            ),
            "has_display": os.environ.get("DISPLAY") is not None,
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }


def create_browser_config(
    platform: Platform, debug_mode: bool = False
) -> BaseBrowserConfig:
    """Factory function to create browser configuration for a platform.

    Args:
    ----
        platform: The platform to create configuration for
        debug_mode: Whether to enable debug mode

    Returns:
    -------
        BaseBrowserConfig instance for the platform

    """
    return BaseBrowserConfig(platform, debug_mode)


def get_default_chrome_options(debug_mode: bool = False) -> list[str]:
    """Get default Chrome options that work across platforms.

    Args:
    ----
        debug_mode: Whether to enable debug mode

    Returns:
    -------
        List of Chrome command line arguments

    """
    options = [
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--disable-notifications",
        "--disable-popup-blocking",
        "--disable-blink-features=AutomationControlled",
    ]

    # Add production-specific options
    if not debug_mode:
        options.extend(
            [
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-extensions",
            ]
        )

    return options
