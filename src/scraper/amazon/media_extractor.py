"""Media extraction utilities for Amazon scraper.

This module handles extraction of high-resolution images and videos from Amazon
product pages using Botasaurus browser automation.
"""

import logging

from botasaurus.browser import Driver

from .config import CONFIG, get_config_value
from .constants import DEFAULT_MAX_IMAGES_PER_PRODUCT, HIGH_RES_DIMENSION
from .debug_analysis import _perform_advanced_debug_analysis  # noqa: F401
from .image_utils import (  # noqa: F401
    _is_irrelevant_image,
    check_amazon_high_res_pattern,
    filter_amazon_fallback_image,
    is_amazon_product_image,
    is_placeholder_image,
    is_valid_high_res_image,
    is_valid_video_url,
    modify_amazon_image_for_high_res,
    validate_video_url_accessibility,
)
from .video_extractor import (  # noqa: F401
    capture_m3u8_urls_from_network,
    extract_functional_videos_with_validation,
)

logger = logging.getLogger(__name__)


def extract_high_res_images_botasaurus(
    driver: Driver, max_images: int = None, debug_options: dict = None
) -> list[str]:
    """Extract high-resolution images using advanced Botasaurus methods

    This function uses a combination of:
    1. JavaScript extraction of dynamic image data from page JSON
    2. Strategic clicking on thumbnails to reveal high-res versions
    3. Direct element selection for immediate high-res sources

    This approach mimics human behavior to access the high-quality images
    that are dynamically loaded when users interact with the image gallery.

    Args:
    ----
        driver: Botasaurus driver instance
        max_images: Maximum number of images to extract (uses config if None)
        debug_options: Dictionary with debug options for detailed analysis

    Returns:
    -------
        List of high-resolution image URLs

    """
    logger = logging.getLogger(__name__)
    debug_options = debug_options or {}

    # Get max images from config if not provided
    if max_images is None:
        try:
            max_images = (
                CONFIG.get("global_settings", {})
                .get("image_config", {})
                .get("max_images_per_product", DEFAULT_MAX_IMAGES_PER_PRODUCT)
            )
        except Exception:
            max_images = DEFAULT_MAX_IMAGES_PER_PRODUCT

    # Get high-res threshold from config
    high_res_threshold = get_config_value(
        "global_settings",
        "image_config",
        "min_high_res_dimension",
        default=HIGH_RES_DIMENSION,
    )

    DEBUG_MODE = debug_options.get("debug_mode", False)

    if DEBUG_MODE:
        logger.info(
            "🔍 Using fast Botasaurus extraction methods " "(max: %d, threshold: %dpx)",
            max_images,
            high_res_threshold,
        )

        # Page analysis using Botasaurus built-in methods
        logger.info("🔬 [FAST EXTRACTION] Starting page analysis...")
        try:
            page_title = driver.title
            logger.info("📄 Page title: %s", page_title)
        except Exception as e:
            logger.warning("⚠️ Page title extraction failed: %s", e)

    # Advanced debug functionality
    if DEBUG_MODE and debug_options:
        _perform_advanced_debug_analysis(driver, debug_options, logger)

    if DEBUG_MODE:
        logger.info("⚡ Using fast direct extraction methods")

    image_urls: list[str] = []

    try:
        # Method 1: Advanced JavaScript extraction of dynamic image data
        if DEBUG_MODE:
            logger.info(
                "🎯 Method 1: Advanced JavaScript extraction of dynamic image data"
            )

        try:
            js_result = driver.run_js(
                """
                // Advanced extraction targeting Amazon's dynamic image system
                const imageUrls = new Set();
                const highResUrls = new Set();

                // 1. Extract from main image JSON data (landingImage
                // data-a-dynamic-image)
                const mainImg = document.querySelector('#landingImage');
                if (mainImg && mainImg.getAttribute('data-a-dynamic-image')) {
                    try {
                        const dynamicImages = JSON.parse(
                            mainImg.getAttribute('data-a-dynamic-image')
                        );
                        Object.entries(dynamicImages).forEach(([url, dimensions]) => {
                            if (Array.isArray(dimensions) && dimensions.length >= 2) {
                                const maxDim = Math.max(dimensions[0], dimensions[1]);
                                if (maxDim >= """
                + str(high_res_threshold)
                + """) { // High-res threshold from config
                                    highResUrls.add(url);
                                }
                            }
                        });
                    } catch (e) {
                        console.log('Failed to parse dynamic image data:', e);
                    }
                }

                // 2. Extract from all script tags containing image data
                document.querySelectorAll('script:not([src])').forEach(script => {
                    const content = script.textContent;
                    const hasImageContent = (
                        content.includes('ImageBlockATF') ||
                        content.includes('imageBlock') ||
                        content.includes('landingImage') ||
                        content.includes('colorImages')
                    );
                    if (hasImageContent) {

                        // Look for colorImages object which contains high-res variants
                        const colorImagesMatch = content.match(
                            /"colorImages"\\s*:\\s*({[^}]+})/
                        );
                        if (colorImagesMatch) {
                            try {
                                // Extract image data from colorImages
                                const urlPattern = new RegExp(
                                    'https?://[^\"\\\\s]*(?:media-amazon|' +
                                    'images-amazon)' +
                                    '[^\"\\\\s]*\\\\._[A-Z]*S[LX]' +
                                    '(1[5-9][0-9][0-9]|[2-9][0-9][0-9][0-9])' +
                                    '[^\"\\\\s]*\\\\.(jpg|jpeg|png|webp)',
                                    'gi'
                                );
                                const matches = content.match(urlPattern);
                                if (matches) {
                                    matches.forEach(url => highResUrls.add(url));
                                }
                            } catch (e) {
                                console.log('Failed to parse colorImages:', e);
                            }
                        }

                        // Look for imageBlock data
                        if (content.includes('imageBlock')) {
                            const imageBlockPattern = /"hiRes"\\s*:\\s*"([^"]+)"/g;
                            let match;
                            while ((match = imageBlockPattern.exec(content)) !== null) {
                                if (match[1] && match[1] !== 'null') {
                                    highResUrls.add(match[1]);
                                }
                            }
                        }
                    }
                });

                // 3. Look for thumbnail elements that might trigger high-res loading
                const thumbnailSelector = '#altImages .imageThumbnail img, ' +
                    '#altImages li img';
                const thumbnails = document.querySelectorAll(thumbnailSelector);
                thumbnails.forEach(thumb => {
                    const hiresAttrs = [
                        'data-old-hires', 'data-a-hires', 'data-zoom-hires'
                    ];
                    hiresAttrs.forEach(attr => {
                        const hiresUrl = thumb.getAttribute(attr);
                        if (hiresUrl && hiresUrl !== 'null') {
                            highResUrls.add(hiresUrl);
                        }
                    });
                });

                return {
                    high_res: Array.from(highResUrls),
                    thumbnails: Array.from(
                        document.querySelectorAll(
                            '#altImages .imageThumbnail, #altImages li'
                        )
                    ).length
                };
            """
            )

            if js_result and isinstance(js_result, dict):
                high_res_images = js_result.get("high_res", [])
                thumbnail_count = js_result.get("thumbnails", 0)

                if DEBUG_MODE:
                    logger.info(
                        "📊 Found %d high-res images in JSON data", len(high_res_images)
                    )
                    logger.info(
                        "📊 Found %d thumbnails for potential clicking", thumbnail_count
                    )

                # Add high-res images from JavaScript extraction
                for url in high_res_images:
                    if len(image_urls) >= max_images:
                        break
                    if url not in image_urls and url != "null":
                        image_urls.append(url)
                        if DEBUG_MODE:
                            logger.info(
                                "✅ Method 1 found high-res image: %s...", url[:80]
                            )

        except Exception as e:
            if DEBUG_MODE:
                logger.warning("⚠️ Method 1 failed: %s", e)

        # Method 2: Strategic thumbnail clicking for dynamic image loading
        if len(image_urls) < max_images:
            if DEBUG_MODE:
                logger.info(
                    "🖱️ Method 2: Strategic thumbnail clicking for dynamic loading"
                )

            try:
                # Find clickable thumbnails
                thumbnails = driver.select_all(
                    "#altImages .imageThumbnail, #altImages li"
                )
                if DEBUG_MODE:
                    logger.info("🖱️ Found %d clickable thumbnails", len(thumbnails))

                for i, thumb in enumerate(thumbnails[:max_images]):
                    if len(image_urls) >= max_images:
                        break

                    try:
                        # Click the thumbnail to potentially load high-res version
                        if DEBUG_MODE:
                            logger.info(
                                "🖱️ Clicking thumbnail %d/%d", i + 1, len(thumbnails)
                            )

                        # Use Botasaurus click with short wait
                        thumb.click()
                        driver.short_random_sleep()  # Short pause for image to load

                        # Extract any newly loaded high-res image
                        new_image_url = driver.run_js("""
                            const mainImg = document.querySelector('#landingImage');
                            if (mainImg) {
                                const src = mainImg.getAttribute('src');
                                const dataSrc = (
                                    mainImg.getAttribute('data-old-hires') ||
                                    mainImg.getAttribute('data-a-hires')
                                );

                                // Return the highest quality URL available
                                if (dataSrc && dataSrc !== 'null') return dataSrc;
                                if (src && src.includes('media-amazon.com')) return src;
                            }
                            return null;
                        """)

                        if new_image_url and new_image_url not in image_urls:
                            # Enhance to highest quality
                            enhanced_url = modify_amazon_image_for_high_res(
                                new_image_url
                            )
                            if filter_amazon_fallback_image(
                                enhanced_url, high_res_threshold
                            ):
                                image_urls.append(enhanced_url)
                                if DEBUG_MODE:
                                    logger.info(
                                        "✅ Method 2 found clicked image: %s...",
                                        enhanced_url[:80],
                                    )

                    except Exception as e:
                        if DEBUG_MODE:
                            logger.warning(
                                "⚠️ Error clicking thumbnail %d: %s", i + 1, e
                            )
                        continue

            except Exception as e:
                if DEBUG_MODE:
                    logger.warning("⚠️ Method 2 failed: %s", e)

        # Method 3: Fallback to enhanced direct extraction
        if len(image_urls) < max_images:
            if DEBUG_MODE:
                logger.info("📋 Method 3: Enhanced direct extraction fallback")

            try:
                # Use get_all_image_links but with better filtering
                all_links = driver.get_all_image_links()
                amazon_links = [
                    url
                    for url in all_links
                    if any(
                        domain in url
                        for domain in ["media-amazon.com", "images-amazon.com"]
                    )
                    and not _is_irrelevant_image(url)
                ]

                for url in amazon_links:
                    if len(image_urls) >= max_images:
                        break
                    if url not in image_urls:
                        enhanced_url = modify_amazon_image_for_high_res(url)
                        if filter_amazon_fallback_image(
                            enhanced_url, high_res_threshold
                        ):
                            image_urls.append(enhanced_url)
                            if DEBUG_MODE:
                                logger.info(
                                    "✅ Method 3 found fallback image: %s...",
                                    enhanced_url[:80],
                                )

            except Exception as e:
                if DEBUG_MODE:
                    logger.warning("⚠️ Method 3 failed: %s", e)

    except Exception as e:
        if DEBUG_MODE:
            logger.error("❌ Error in advanced image extraction: %s", e)

    # Remove duplicates while preserving order and limit results
    unique_urls = []
    for url in image_urls:
        if url not in unique_urls:
            unique_urls.append(url)
        if len(unique_urls) >= max_images:
            break  # Final limit enforcement

    if DEBUG_MODE:
        logger.info(
            "🎯 Extracted %d high-res images using Botasaurus (limit: %d)",
            len(unique_urls),
            max_images,
        )

    return unique_urls
