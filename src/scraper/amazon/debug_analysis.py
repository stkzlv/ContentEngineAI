"""Debug analysis utilities for Amazon scraper.

This module contains debug-only functions for analyzing page content,
images, and URLs during scraper development and troubleshooting.
"""

import json
import logging
from pathlib import Path

from botasaurus.browser import Driver

from .config import CONFIG

logger = logging.getLogger(__name__)


def _perform_advanced_debug_analysis(driver: Driver, debug_options: dict, logger):
    """Perform advanced debug analysis of the page"""
    try:
        # Create debug output directory
        from ...utils.outputs_paths import get_temp_directory

        debug_dir = get_temp_directory() / "debug" / "image_analysis"
        debug_dir.mkdir(parents=True, exist_ok=True)

        current_url = driver.current_url
        asin = (
            current_url.split("/dp/")[1].split("/")[0].split("?")[0]
            if "/dp/" in current_url
            else "unknown"
        )

        logger.info("[ADVANCED DEBUG] Starting advanced analysis for ASIN: %s", asin)

        # 1. Save page source if requested
        if debug_options.get("save_page_source"):
            try:
                page_source = driver.page_source
                source_file = debug_dir / f"{asin}_page_source.html"
                with open(source_file, "w", encoding="utf-8") as f:
                    f.write(page_source)
                logger.info("Saved page source to: %s", source_file)
            except Exception as e:
                logger.warning("Failed to save page source: %s", e)

        # 2. Take screenshots if requested (controlled by config)
        try:
            save_screenshots = (
                CONFIG.get("global_settings", {})
                .get("debug_settings", {})
                .get("save_screenshots", False)
            )
        except Exception:
            save_screenshots = False

        if debug_options.get("save_screenshots") or save_screenshots:
            try:
                screenshot_file = debug_dir / f"{asin}_screenshot.png"
                driver.save_screenshot(str(screenshot_file))
                logger.info("Saved screenshot to: %s", screenshot_file)
            except Exception as e:
                logger.warning("Failed to save screenshot: %s", e)

        # 3. Deep image analysis if requested
        if debug_options.get("analyze_images"):
            _perform_deep_image_analysis(driver, asin, debug_dir, logger)

        # 4. Dump all image URLs if requested
        if debug_options.get("dump_image_urls"):
            _dump_all_image_urls(driver, asin, debug_dir, logger)

    except Exception as e:
        logger.error("Advanced debug analysis failed: %s", e)


def _perform_deep_image_analysis(driver: Driver, asin: str, debug_dir: Path, logger):
    """Perform deep analysis of all images on the page"""
    try:
        logger.info("🔍 [DEEP ANALYSIS] Analyzing all images on page...")

        # Find all image elements
        all_imgs = driver.select_all("img")
        analysis_results = []

        for i, img in enumerate(all_imgs):
            try:
                img_data = {
                    "index": i,
                    "src": img.get_attribute("src"),
                    "data_old_hires": img.get_attribute("data-old-hires"),
                    "data_src": img.get_attribute("data-src"),
                    "alt": img.get_attribute("alt"),
                    "class": img.get_attribute("class"),
                    "id": img.get_attribute("id"),
                    "width": img.get_attribute("width"),
                    "height": img.get_attribute("height"),
                }

                # Check if it's Amazon media
                src = img_data["src"]
                if src and any(
                    domain in src
                    for domain in ["media-amazon.com", "images-amazon.com"]
                ):
                    img_data["is_amazon_media"] = True
                    img_data["resolution_indicators"] = {
                        "SL1500": "_SL1500_" in src,
                        "SL2000": "_SL2000_" in src,
                        "SL1600": "_SL1600_" in src,
                        "AC_SL": "_AC_SL" in src,
                    }
                else:
                    img_data["is_amazon_media"] = False

                analysis_results.append(img_data)

            except Exception as e:
                logger.warning("Error analyzing image %d: %s", i, e)

        # Save analysis results
        analysis_file = debug_dir / f"{asin}_image_analysis.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        # Log summary
        amazon_images = [img for img in analysis_results if img.get("is_amazon_media")]
        high_res_images = [
            img
            for img in amazon_images
            if any(img.get("resolution_indicators", {}).values())
        ]

        logger.info("[DEEP ANALYSIS] Found %d total images", len(all_imgs))
        logger.info("[DEEP ANALYSIS] Found %d Amazon media images", len(amazon_images))
        logger.info(
            "[DEEP ANALYSIS] Found %d potential high-res images", len(high_res_images)
        )
        logger.info("[DEEP ANALYSIS] Detailed analysis saved to: %s", analysis_file)

        # Log high-res image candidates
        for img in high_res_images:
            src = img.get("src", "")
            indicators = [
                k for k, v in img.get("resolution_indicators", {}).items() if v
            ]
            logger.info(
                "[HIGH-RES CANDIDATE] %s... (indicators: %s)",
                src[:80],
                ", ".join(indicators),
            )

    except Exception as e:
        logger.error("Deep image analysis failed: %s", e)


def _dump_all_image_urls(driver: Driver, asin: str, debug_dir: Path, logger):
    """Dump all discovered image URLs to a file"""
    try:
        logger.info("📝 [URL DUMP] Collecting all image URLs...")

        all_urls = set()

        # Method 1: From img src attributes
        imgs = driver.select_all("img")
        for img in imgs:
            src = img.get_attribute("src")
            if src:
                all_urls.add(("img_src", src))

        # Method 2: From data-old-hires attributes
        for img in imgs:
            data_old_hires = img.get_attribute("data-old-hires")
            if data_old_hires:
                all_urls.add(("data_old_hires", data_old_hires))

        # Method 3: From JavaScript variables (if accessible)
        try:
            # Try to get image data from JavaScript
            js_result = driver.run_js(
                """
                var imageData = [];
                if (window.ImageBlockATF) {
                    imageData.push([
                        'ImageBlockATF', JSON.stringify(window.ImageBlockATF)
                    ]);
                }
                if (window.P && window.P.imageBlockATF) {
                    imageData.push([
                        'P.imageBlockATF', JSON.stringify(window.P.imageBlockATF)
                    ]);
                }
                return imageData;
            """
            )

            if js_result:
                for source_type, data_str in js_result:
                    all_urls.add((f"js_{source_type}", data_str))

        except Exception as e:
            logger.debug("JS image data extraction failed: %s", e)

        # Save all URLs
        urls_file = debug_dir / f"{asin}_all_image_urls.txt"
        with open(urls_file, "w", encoding="utf-8") as f:
            f.write(f"# All discovered image URLs for ASIN: {asin}\n")
            f.write(f"# Total URLs found: {len(all_urls)}\n\n")

            for source_type, url in sorted(all_urls):
                f.write(f"[{source_type}] {url}\n")

        logger.info("[URL DUMP] Saved %d URLs to: %s", len(all_urls), urls_file)

        # Filter and log Amazon high-res URLs
        amazon_urls = [
            url
            for source_type, url in all_urls
            if any(
                domain in url for domain in ["media-amazon.com", "images-amazon.com"]
            )
            and any(pattern in url for pattern in ["_SL1500_", "_SL2000_", "_SL1600_"])
        ]

        logger.info(
            "[URL DUMP] Found %d potential high-res Amazon URLs:", len(amazon_urls)
        )
        for url in amazon_urls:
            logger.info("   - %s", url)

    except Exception as e:
        logger.error("URL dump failed: %s", e)
