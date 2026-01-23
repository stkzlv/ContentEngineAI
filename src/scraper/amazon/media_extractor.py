"""Media extraction utilities for Amazon scraper.

This module handles extraction of high-resolution images and videos from Amazon
product pages using Botasaurus browser automation.
"""

import json
import logging
import re
import time
from pathlib import Path

from botasaurus.browser import Driver

from .config import CONFIG, get_config_value
from .constants import (
    DEFAULT_MAX_IMAGES_PER_PRODUCT,
    HIGH_RES_DIMENSION,
    HIGH_RES_UPGRADE_DIMENSION,
    VERY_HIGH_RES_DIMENSION,
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
            f"🔍 Using fast Botasaurus extraction methods "
            f"(max: {max_images}, threshold: {high_res_threshold}px)"
        )

        # Page analysis using Botasaurus built-in methods
        logger.info("🔬 [FAST EXTRACTION] Starting page analysis...")
        try:
            page_title = driver.title
            logger.info(f"📄 Page title: {page_title}")
        except Exception as e:
            logger.warning(f"⚠️ Page title extraction failed: {e}")

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
                        f"📊 Found {len(high_res_images)} high-res images in JSON data"
                    )
                    logger.info(
                        f"📊 Found {thumbnail_count} thumbnails for potential clicking"
                    )

                # Add high-res images from JavaScript extraction
                for url in high_res_images:
                    if len(image_urls) >= max_images:
                        break
                    if url not in image_urls and url != "null":
                        image_urls.append(url)
                        if DEBUG_MODE:
                            logger.info(
                                f"✅ Method 1 found high-res image: {url[:80]}..."
                            )

        except Exception as e:
            if DEBUG_MODE:
                logger.warning(f"⚠️ Method 1 failed: {e}")

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
                    logger.info(f"🖱️ Found {len(thumbnails)} clickable thumbnails")

                for i, thumb in enumerate(thumbnails[:max_images]):
                    if len(image_urls) >= max_images:
                        break

                    try:
                        # Click the thumbnail to potentially load high-res version
                        if DEBUG_MODE:
                            logger.info(f"🖱️ Clicking thumbnail {i+1}/{len(thumbnails)}")

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
                                        f"✅ Method 2 found clicked image: "
                                        f"{enhanced_url[:80]}..."
                                    )

                    except Exception as e:
                        if DEBUG_MODE:
                            logger.warning(f"⚠️ Error clicking thumbnail {i+1}: {e}")
                        continue

            except Exception as e:
                if DEBUG_MODE:
                    logger.warning(f"⚠️ Method 2 failed: {e}")

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
                                    f"✅ Method 3 found fallback image: "
                                    f"{enhanced_url[:80]}..."
                                )

            except Exception as e:
                if DEBUG_MODE:
                    logger.warning(f"⚠️ Method 3 failed: {e}")

    except Exception as e:
        if DEBUG_MODE:
            logger.error(f"❌ Error in advanced image extraction: {e}")

    # Remove duplicates while preserving order and limit results
    unique_urls = []
    for url in image_urls:
        if url not in unique_urls:
            unique_urls.append(url)
        if len(unique_urls) >= max_images:
            break  # Final limit enforcement

    if DEBUG_MODE:
        logger.info(
            f"🎯 Extracted {len(unique_urls)} high-res images using Botasaurus "
            f"(limit: {max_images})"
        )

    return unique_urls


def capture_m3u8_urls_from_network(
    driver: Driver, timeout: int = 20, debug: bool = False
) -> list[str]:
    """Capture m3u8 video URLs from browser network traffic.

    Uses Botasaurus network response monitoring to capture m3u8 HLS streaming
    URLs from Amazon's media servers.

    Args:
    ----
        driver: Botasaurus driver instance
        timeout: How long to monitor network traffic (seconds)
        debug: Enable debug logging

    Returns:
    -------
        List of m3u8 URLs found in network traffic

    """
    logger = logging.getLogger(__name__)
    m3u8_urls = []
    seen_urls = set()

    try:
        # Define response handler to capture m3u8 URLs
        def capture_m3u8_handler(_request_id, response, _event):  # noqa: ARG001
            """Handler function to capture m3u8 URLs from network responses."""
            try:
                url = response.url

                # Filter for m3u8 URLs from Amazon media servers
                if (
                    url
                    and ".m3u8" in url
                    and "media-amazon.com" in url
                    and "blob" not in url.lower()
                    and url not in seen_urls
                ):
                    m3u8_urls.append(url)
                    seen_urls.add(url)

                    if debug:
                        logger.info(f"✅ Found m3u8 URL: {url[:100]}...")
            except Exception as e:
                if debug:
                    logger.debug(f"Error in response handler: {e}")

        # Register response handler
        driver.after_response_received(capture_m3u8_handler)

        if debug:
            logger.info(f"🔍 Monitoring network traffic for {timeout} seconds...")

        # Wait for video to load and network requests to fire
        time.sleep(timeout)

        if debug:
            logger.info(f"🎯 Captured {len(m3u8_urls)} m3u8 URLs from network traffic")

        return m3u8_urls

    except AttributeError as e:
        # Botasaurus might not have after_response_received method
        if debug:
            logger.warning(f"Network monitoring not available: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to capture network traffic: {e}")
        return []


def extract_functional_videos_with_validation(
    driver: Driver, debug_mode: bool = False
) -> list[str]:
    """Extract product videos using the same systematic approach as images

    This function now uses the same 3-method approach as image extraction:
    1. **JavaScript extraction** from page data and scripts
    2. **Strategic thumbnail clicking** to load video players
    3. **Direct element extraction** as fallback

    Focus on official Amazon product videos and customer review videos only.

    Args:
    ----
        driver: Botasaurus driver instance
        debug_mode: Enable debug logging and detailed output

    Returns:
    -------
        List of validated video URLs (official product + customer review videos)

    """
    logger = logging.getLogger(__name__)
    DEBUG_MODE = debug_mode

    # Get max videos from config
    try:
        max_videos = (
            CONFIG.get("global_settings", {})
            .get("video_config", {})
            .get("max_videos_per_product", 10)
        )
    except Exception:
        max_videos = 10

    # Always log to verify function is called
    logger.info(
        f"🎥 extract_functional_videos_with_validation called "
        f"(max: {max_videos}, DEBUG={DEBUG_MODE})"
    )

    if DEBUG_MODE:
        logger.info(f"🎥 Using systematic video extraction (max: {max_videos} videos)")

    video_urls: list[str] = []

    try:
        # Get current ASIN and product title for matching
        page_info = driver.run_js(r"""
            const url = window.location.href;
            const asinMatch = url.match(/\/dp\/([A-Z0-9]{10})/);
            const asin = asinMatch ? asinMatch[1] : null;

            // Extract product title and create search keywords
            const titleElement =
                document.querySelector(
                    '#productTitle, h1.a-size-large, .product-title'
                ) ||
                document.querySelector('h1') ||
                document.querySelector('[data-feature-name="productTitle"] h1');

            const productTitle = titleElement ?
                titleElement.textContent.trim() : document.title;

            // Extract key brand/product keywords from title
            const titleWords = productTitle.toLowerCase()
                .split(/[^a-zA-Z0-9]+/).filter(word =>
                    word.length > 2 &&
                    !['the', 'and', 'for', 'with', 'true', 'wireless',
                     'bluetooth'].includes(word)
                );

            // Get brand name (usually first word or after "by")
            const brandMatch = productTitle.match(/(?:^|by )([A-Za-z]+)/i);
            const brand = brandMatch ? brandMatch[1].toLowerCase() : '';

            // Get model/product name keywords
            // P20i, etc.
            const modelMatch = productTitle.match(/([A-Z]+[0-9]+[A-Za-z]*)/);
            const model = modelMatch ? modelMatch[1].toLowerCase() : '';

            return {
                asin: asin,
                title: productTitle,
                brand: brand,
                model: model,
                keywords: titleWords.slice(0, 5) // Top 5 keywords
            };
        """)

        if not page_info or not page_info.get("asin"):
            if DEBUG_MODE:
                logger.warning("❌ Could not extract ASIN from current page")
            return []

        current_asin = page_info["asin"]
        product_brand = page_info.get("brand", "")
        product_model = page_info.get("model", "")
        product_keywords = page_info.get("keywords", [])

        if not current_asin:
            if DEBUG_MODE:
                logger.warning("❌ Could not extract ASIN from current page")
            return []

        if DEBUG_MODE:
            logger.info(f"🔍 Current product ASIN: {current_asin}")
            logger.info(f"🏷️ Product brand: {product_brand}")
            logger.info(f"🔤 Product model: {product_model}")
            logger.info(f"🔑 Product keywords: {product_keywords}")

        # Method 1: JavaScript extraction from page data (same as images)
        if DEBUG_MODE:
            logger.info("🎯 Method 1: JavaScript extraction from page data")

        try:
            js_result = driver.run_js(  # noqa: E501
                f"""
                const currentAsin = '{current_asin}';
                const productBrand = '{product_brand}';
                const productModel = '{product_model}';
                const productKeywords = {product_keywords};
                const videoUrls = new Set();
                const vdpLinks = new Set();

                // Exclusion: DOM sections with related/competitor products
                const excludedSelectors = [
                    '[id*="comparison"]',           // Comparison tables
                    '[class*="comparison"]',
                    '[id*="similar"]',              // Similar items sections
                    '[class*="similar"]',
                    '[class*="compare"]',           // Compare sections
                    '[id*="related"]',              // Related products
                    '[class*="related"]',
                    '[data-a-carousel-options]',    // Product carousels
                    '.a-carousel',                  // Amazon carousels
                    '[id*="sims-fbt"]',             // Frequently bought together
                    '#HLCXComparisonWidget',        // Comparison widget
                    '#comparison_table',
                    '#aplus',                       // A+ content (has related products)
                    '.aplus-module',                // A+ module variations
                    '[id*="aplus"]',                // Any A+ content
                    '[class*="aplus"]',
                    '.a-plus',
                    '#feature-bullets-btf',         // Below the fold content
                    '#btfContent',                  // Below the fold
                ];

                // Function to check if element is in excluded section
                function isInExcludedSection(element) {{
                    return excludedSelectors.some(selector => {{
                        try {{
                            return element.closest(selector) !== null;
                        }} catch (e) {{
                            return false;
                        }}
                    }});
                }}

                // Function to check if container has different ASIN
                function hasDifferentAsin(element) {{
                    const container = element.closest('[data-asin]');
                    if (container) {{
                        const asin = container.getAttribute('data-asin');
                        return asin && asin !== currentAsin && asin !== '' && asin !== 'null';
                    }}
                    return false;
                }}

                // Strict product validation - only official product videos
                function isValidProductVideo(element) {{
                    // Exclude if in related products section
                    if (isInExcludedSection(element)) {{
                        return false;
                    }}

                    // Exclude if container has different ASIN
                    if (hasDifferentAsin(element)) {{
                        return false;
                    }}

                    // ONLY accept from main image/video gallery areas
                    const inTrustedArea = element.closest('#imageBlock') ||
                                         element.closest('#altImages') ||
                                         element.closest('#ivTitle') ||
                                         element.closest('#main-image-container');

                    // Also accept if in ASIN container that's NOT in A+ content
                    const inAsinContainer = element.closest('[data-asin="' + currentAsin + '"]');
                    const notInAplus = inAsinContainer && !element.closest('#aplus, .aplus-module, [id*="aplus"], [class*="aplus"]');

                    return inTrustedArea || notInAplus;
                }}

                // 1. Extract from script tags containing video data
                document.querySelectorAll('script:not([src])').forEach(script => {{
                    const content = script.textContent;

                    // Only process scripts that mention current ASIN to avoid related products
                    if (!content.includes(currentAsin)) {{
                        return;
                    }}

                    const hasVideoContent = (
                        content.includes('videoUrl') ||
                        content.includes('productVideo') ||
                        content.includes('customerVideo') ||
                        content.includes('videoMimeType') ||
                        content.includes('vse-vms')
                    );

                    if (hasVideoContent) {{
                        // Extract direct MP4 and M3U8 URLs
                        const videoPattern = new RegExp(
                            'https?://[^\"\\\\\\s]*media-amazon\\\\.com[^\"\\\\\\s]*\\\\.(mp4|m3u8)[^\"\\\\\\s]*',
                            'gi'
                        );
                        const mp4Matches = content.match(videoPattern);
                        if (mp4Matches) {{
                            mp4Matches.forEach(url => {{
                                // Check if video URL is near current ASIN in script
                                const urlIndex = content.indexOf(url);
                                const nearbyText = content.substring(
                                    Math.max(0, urlIndex - 500),
                                    urlIndex + 500
                                );

                                // Only add if ASIN is mentioned near the video URL
                                if (nearbyText.includes(currentAsin)) {{
                                    videoUrls.add(url);
                                }}
                            }});
                        }}

                        // Extract JSON video properties (only if ASIN context exists)
                        const jsonVideoPattern =
                            /"(?:videoUrl|video_url|src)"\\s*:\\s*"([^"]*\\.(mp4|m3u8)[^"]*)"/gi;
                        let jsonMatch;
                        while ((jsonMatch = jsonVideoPattern.exec(content)) !== null) {{
                            const url = jsonMatch[1];
                            if (url.includes('media-amazon.com')) {{
                                const urlIndex = content.indexOf(url);
                                const nearbyText = content.substring(
                                    Math.max(0, urlIndex - 500),
                                    urlIndex + 500
                                );
                                if (nearbyText.includes(currentAsin)) {{
                                    videoUrls.add(url);
                                }}
                            }}
                        }}
                    }}
                }});

                // 2. Extract VDP (Video Detail Page) links for official videos
                document.querySelectorAll('a[href*="/vdp/"]').forEach(link => {{
                    const href = link.href;

                    // Use strict validation: must be in valid product area
                    if (isValidProductVideo(link)) {{
                        // Double check: VDP link must contain current ASIN or be in ASIN container
                        const inAsinContainer = link.closest('[data-asin="' + currentAsin + '"]');
                        const urlHasAsin = href.includes(currentAsin);

                        if (inAsinContainer || urlHasAsin) {{
                            vdpLinks.add(href);
                        }}
                    }}
                }});

                // 3. Extract video sources from loaded video elements and
                //    their network requests
                document.querySelectorAll('video').forEach(video => {{
                    // Use strict validation to exclude related products
                    if (isValidProductVideo(video)) {{
                        // Extract direct video sources
                        if (video.src &&
                            video.src.includes('media-amazon.com') &&
                            video.src.includes('.mp4')) {{
                            videoUrls.add(video.src);
                        }}

                        // Extract from source elements
                        video.querySelectorAll('source').forEach(source => {{
                            if (source.src &&
                                source.src.includes('media-amazon.com') &&
                                source.src.includes('.mp4')) {{
                                videoUrls.add(source.src);
                            }}
                        }});

                        // For blob URLs, look for data attributes that might contain
                        // original URL
                        if (video.src && video.src.startsWith('blob:')) {{
                            const dataAttrs = [
                                'data-video-url', 'data-src',
                                'data-original-src', 'data-video-source'
                            ];
                            dataAttrs.forEach(attr => {{
                                const attrValue = video.getAttribute(attr);
                                if (attrValue &&
                                    attrValue.includes('media-amazon.com') &&
                                    attrValue.includes('.mp4')) {{
                                    videoUrls.add(attrValue);
                                }}
                            }});

                            // Check parent elements for video URL data
                            let parent = video.parentElement;
                            while (parent && parent !== document.body) {{
                                dataAttrs.forEach(attr => {{
                                    const attrValue = parent.getAttribute(attr);
                                    if (attrValue &&
                                        attrValue.includes('media-amazon.com') &&
                                        attrValue.includes('.mp4')) {{
                                        videoUrls.add(attrValue);
                                    }}
                                }});
                                parent = parent.parentElement;
                            }}
                        }}
                    }}
                }});

                // 4. Look for video thumbnail elements for clicking
                const videoThumbnails = [];
                const thumbnailSelectors = [
                    '.videoThumbnail',
                    '[class*="video-thumb"]',
                    '[data-video]',
                    '[data-video-url]',
                    '.video-player',
                    '#altImages .videoThumbnail',
                    '#imageBlock .videoThumbnail'
                ];

                thumbnailSelectors.forEach(selector => {{
                    document.querySelectorAll(selector).forEach(thumb => {{
                        // Use strict validation to exclude related products
                        if (isValidProductVideo(thumb)) {{
                            videoThumbnails.push({{
                                element: selector,
                                dataVideo: thumb.getAttribute('data-video'),
                                dataVideoUrl: thumb.getAttribute('data-video-url'),
                                onclick: thumb.getAttribute('onclick')
                            }});
                        }}
                    }});
                }});

                return {{
                    direct_videos: Array.from(videoUrls),
                    vdp_links: Array.from(vdpLinks),
                    thumbnails: videoThumbnails
                }};
            """
            )  # noqa: E501

            if js_result and isinstance(js_result, dict):
                direct_videos = js_result.get("direct_videos", [])
                vdp_links = js_result.get("vdp_links", [])
                video_thumbnails = js_result.get("thumbnails", [])

                if DEBUG_MODE:
                    logger.info(
                        f"📊 Method 1 found: {len(direct_videos)} direct videos, "
                        f"{len(vdp_links)} VDP links, "
                        f"{len(video_thumbnails)} thumbnails"
                    )

                # Method 1 direct video extraction with isValidProductVideo() filtering
                # Re-enabled with DOM context validation to exclude competitor videos
                for url in direct_videos:
                    if len(video_urls) >= max_videos:
                        if DEBUG_MODE:
                            logger.info(
                                f"🛑 Method 1: Reached video limit "
                                f"({len(video_urls)}/{max_videos}), stopping extraction"
                            )
                        break
                    if (
                        url not in video_urls
                        and url != "null"
                        and is_valid_video_url(url)
                    ):
                        video_urls.append(url)
                        if DEBUG_MODE:
                            logger.info(
                                f"✅ Method 1 found direct video: {url[:80]}..."
                            )

                # DISABLED: VDP links also pick up comparison videos
                # for url in vdp_links:
                #     if len(video_urls) >= max_videos:
                #         break
                #     if url not in video_urls:
                #         video_urls.append(url)
                #         if DEBUG_MODE:
                #             logger.info(f"✅ Method 1 found VDP link: {url[:80]}...")

        except Exception as e:
            if DEBUG_MODE:
                logger.warning(f"⚠️ Method 1 failed: {e}")

        # Method 2: Strategic thumbnail clicking (same approach as images)
        logger.info(
            f"🖱️ Method 2 check: {len(video_urls)} videos so far, " f"max={max_videos}"
        )
        if len(video_urls) < max_videos:
            logger.info(
                "🖱️ Method 2: Strategic thumbnail clicking for videos - STARTING"
            )

            try:
                # PRAGMATIC SOLUTION: Amazon mixes competitor videos everywhere
                # - Main gallery contains competitors
                # - "Videos for this product" widget not reliably accessible
                # - A+ content has related products
                #
                # STRATEGY: Only extract FIRST video from main gallery
                # - First video is usually (but not always) the official product video
                # - Minimizes competitor exposure
                # - Set max_videos=1 in config for best results

                # Check if M3U8 network monitoring is enabled
                global_settings = CONFIG.get("global_settings", {})
                video_config = global_settings.get("video_config", {})
                enable_m3u8_monitoring = video_config.get(
                    "enable_m3u8_monitoring", False
                )

                # NEW APPROACH: Network monitoring for M3U8 streams (optional)
                # Amazon serves videos as HLS (m3u8) streams, not direct MP4 URLs.
                # Instead of clicking thumbnails, we monitor network traffic to capture
                # the actual m3u8 URLs that load when the video player initializes.
                #
                # This approach:
                # - Captures the official hero video (main product video)
                # - Avoids competitor videos from gallery/carousel
                # - Works with Amazon's bot detection (doesn't require carousel access)

                if enable_m3u8_monitoring:
                    logger.info("🎬 Using network monitoring to capture video streams")
                    logger.info("   Method: Capture m3u8 HLS URLs from network traffic")

                    # Find and click the first video thumbnail to trigger video loading
                    try:
                        video_thumbnail = driver.select(
                            "#imageBlock .videoThumbnail, #altImages .videoThumbnail"
                        )
                        if video_thumbnail:
                            if DEBUG_MODE:
                                logger.info("🖱️ Clicking video thumbnail to load player")

                            # Mute videos before clicking
                            driver.run_js("""
                                document.querySelectorAll('video').forEach(video => {
                                    video.muted = true;
                                    video.volume = 0;
                                });
                            """)

                            # Click to trigger video player and network requests
                            video_thumbnail.click()
                            driver.short_random_sleep()

                            if DEBUG_MODE:
                                logger.info(
                                    "✅ Video player triggered, waiting for network requests"
                                )
                        else:
                            logger.warning("⚠️ No video thumbnail found")
                    except Exception as e:
                        logger.warning(f"Failed to trigger video player: {e}")

                    # Capture m3u8 URLs from network traffic
                    try:
                        network_timeout = video_config.get(
                            "network_capture_timeout", 20
                        )

                        # Call network capture function
                        m3u8_urls = capture_m3u8_urls_from_network(
                            driver, timeout=network_timeout, debug=DEBUG_MODE
                        )

                        if m3u8_urls:
                            # Take only the first m3u8 URL (hero video)
                            for m3u8_url in m3u8_urls[:max_videos]:
                                if len(video_urls) >= max_videos:
                                    if DEBUG_MODE:
                                        logger.info(
                                            f"🛑 Method 2: Reached video limit "
                                            f"({len(video_urls)}/{max_videos}), "
                                            f"stopping network capture"
                                        )
                                    break

                                if m3u8_url and m3u8_url not in video_urls:
                                    video_urls.append(m3u8_url)
                                    if DEBUG_MODE:
                                        logger.info(
                                            f"✅ Captured m3u8 URL from network: "
                                            f"{m3u8_url[:80]}..."
                                        )
                        else:
                            logger.warning(
                                "⚠️ No m3u8 URLs captured from network traffic"
                            )

                    except Exception as e:
                        if DEBUG_MODE:
                            logger.warning(f"⚠️ Method 2 network capture failed: {e}")

            except Exception as e:
                if DEBUG_MODE:
                    logger.warning(f"⚠️ Method 2 failed: {e}")

        # Method 3: Direct element extraction with DOM context filtering
        if len(video_urls) < max_videos:
            if DEBUG_MODE:
                logger.info(
                    "📋 Method 3: Direct element extraction with context filtering"
                )

            try:
                # Define valid product gallery selectors (where product videos should be)
                valid_gallery_selectors = [
                    "#imageBlock",  # Main product image block
                    "#altImages",  # Alternative images carousel
                    "#main-image-container",  # Main image container
                    "#imageBlockThumbs",  # Image thumbnails
                    ".imageBlockContainer",  # Image block container
                    "#immersive-view-front-image",  # Immersive view front image
                    "#ivTitle",  # Immersive view title/video section
                    "#immersive-view-main-content",  # Main immersive view content
                    "[data-a-modal-name='immersive-view']",  # Immersive view modal
                ]

                # Define excluded sections (where videos should NOT be extracted from)
                excluded_selectors = [
                    "#ask-dp-search_feature_div",  # Customer Q&A
                    "#cm-cr-dp-review-list",  # Customer reviews
                    "#HLCXComparisonWidget",  # Comparison widget
                    "#similarities_feature_div",  # Similar items
                    "#sp_detail",  # Sponsored products
                    "#sims-fbt",  # Frequently bought together
                    ".a-carousel-card",  # Carousel cards (related products)
                ]

                def is_in_product_gallery(element) -> bool:
                    """Check if video element is within the main product gallery.

                    Uses JavaScript to check element containment by finding the element
                    in the DOM using its src attribute, since Python Element objects
                    cannot be passed to JavaScript.
                    """
                    try:
                        # Get the video source URL to identify the element in JavaScript
                        video_src = element.get_attribute("src")
                        if not video_src:
                            return False

                        # Build JavaScript that checks containment without passing Element object
                        js_code = f"""
                        (function() {{
                            // Find the video element by its src attribute
                            const videoElements = document.querySelectorAll('video[src], video source[src]');
                            let targetElement = null;

                            for (const elem of videoElements) {{
                                if (elem.src === '{video_src}' ||
                                    (elem.tagName === 'SOURCE' && elem.src === '{video_src}')) {{
                                    targetElement = elem.tagName === 'SOURCE' ? elem.parentElement : elem;
                                    break;
                                }}
                            }}

                            if (!targetElement) return false;

                            // Check if element is in any EXCLUDED section (priority check)
                            const excludedSelectors = {excluded_selectors};
                            for (const selector of excludedSelectors) {{
                                const excluded = document.querySelector(selector);
                                if (excluded && excluded.contains(targetElement)) {{
                                    return false; // In excluded section
                                }}
                            }}

                            // Check if element is within any valid gallery section
                            const validSelectors = {valid_gallery_selectors};
                            for (const selector of validSelectors) {{
                                const gallery = document.querySelector(selector);
                                if (gallery && gallery.contains(targetElement)) {{
                                    return true; // In valid gallery
                                }}
                            }}

                            return false; // Not in valid gallery
                        }})();
                        """

                        result = driver.run_js(js_code)

                        if not result and DEBUG_MODE:
                            logger.debug(
                                f"❌ Video rejected (not in product gallery): {video_src[:80]}..."
                            )

                        return bool(result)
                    except Exception as e:
                        if DEBUG_MODE:
                            logger.debug(f"Context check failed: {e}")
                        return False

                # Find video elements directly
                video_elements = driver.select_all("video[src], video source[src]")
                for video_elem in video_elements:
                    if len(video_urls) >= max_videos:
                        if DEBUG_MODE:
                            logger.info(
                                f"🛑 Method 3a: Reached video limit "
                                f"({len(video_urls)}/{max_videos}), stopping video element extraction"
                            )
                        break

                    src = video_elem.get_attribute("src")
                    if (
                        src
                        and src not in video_urls
                        and is_valid_video_url(src)
                        and "media-amazon.com" in src
                        and current_asin in driver.current_url
                    ):
                        # Check DOM context - only accept videos from product gallery
                        if is_in_product_gallery(video_elem):
                            video_urls.append(src)
                            if DEBUG_MODE:
                                logger.info(
                                    f"✅ Method 3 found product gallery video: {src[:80]}..."
                                )
                        elif DEBUG_MODE:
                            logger.debug(
                                f"❌ Video rejected (not in product gallery): {src[:80]}..."
                            )

                # Check for embedded video URLs in visible elements
                video_containers = driver.select_all(
                    "[data-video-url], [data-src*='.mp4']"
                )
                for container in video_containers:
                    if len(video_urls) >= max_videos:
                        if DEBUG_MODE:
                            logger.info(
                                f"🛑 Method 3b: Reached video limit "
                                f"({len(video_urls)}/{max_videos}), stopping container extraction"
                            )
                        break

                    video_url = container.get_attribute(
                        "data-video-url"
                    ) or container.get_attribute("data-src")
                    if (
                        video_url
                        and video_url not in video_urls
                        and is_valid_video_url(video_url)
                        and "media-amazon.com" in video_url
                    ):
                        # Check DOM context for container elements too
                        if is_in_product_gallery(container):
                            video_urls.append(video_url)
                            if DEBUG_MODE:
                                logger.info(
                                    f"✅ Method 3 found product gallery container video: "
                                    f"{video_url[:80]}..."
                                )
                        elif DEBUG_MODE:
                            logger.debug(
                                f"❌ Video rejected (container not in product gallery): "
                                f"{video_url[:80]}..."
                            )

            except Exception as e:
                if DEBUG_MODE:
                    logger.warning(f"⚠️ Method 3 failed: {e}")

    except Exception as e:
        if DEBUG_MODE:
            logger.error(f"❌ Error in systematic video extraction: {e}")

    # Remove duplicates while preserving order and limit results
    unique_urls = []
    for url in video_urls:
        if url not in unique_urls:
            unique_urls.append(url)
            if len(unique_urls) >= max_videos:
                if DEBUG_MODE:
                    logger.info(
                        f"🛑 Deduplication: Reached video limit "
                        f"({len(unique_urls)}/{max_videos})"
                    )
                break

    # Final summary logging
    if DEBUG_MODE:
        if len(unique_urls) >= max_videos:
            logger.info(
                f"🎯 Extracted {len(unique_urls)} videos (hit configured limit) "
                f"for {current_asin}"
            )
        elif len(unique_urls) > 0:
            logger.info(
                f"🎯 Extracted {len(unique_urls)} videos (found all available) "
                f"for {current_asin}"
            )
        else:
            logger.warning(
                f"⚠️ No videos found for {current_asin} (limit was {max_videos})"
            )

    return unique_urls


def modify_amazon_image_for_high_res(url: str) -> str:
    """Convert Amazon image URL to high-resolution version"""
    if not url or not isinstance(url, str):
        return url

    # Replace small size indicators with large ones
    # Pattern for _SL{size}_, _SX{size}_, _SY{size}_, etc.
    size_pattern = r"\._(?:AC_)?(SL|SX|SY)(\d+)_\."
    match = re.search(size_pattern, url)

    if match:
        # Replace with high-res version from config
        try:
            high_res_dimension = (
                CONFIG.get("global_settings", {})
                .get("media_config", {})
                .get("high_res_upgrade_dimension", HIGH_RES_UPGRADE_DIMENSION)
            )
        except Exception:
            high_res_dimension = HIGH_RES_UPGRADE_DIMENSION

        new_url = re.sub(
            size_pattern, f"._AC_{match.group(1)}{high_res_dimension}_.", url
        )
        return new_url

    # If no size pattern found, try adding high-res suffix
    # Get high-res suffix from config
    high_res_suffix = (
        CONFIG.get("global_settings", {})
        .get("media_config", {})
        .get("amazon_high_res_suffix", "._AC_SL2000_.jpg")
    )
    default_ext = (
        CONFIG.get("global_settings", {})
        .get("media_config", {})
        .get("default_image_extension", ".jpg")
    )

    if "._" in url and url.endswith(default_ext):
        return url.replace(default_ext, high_res_suffix)

    return url


def is_amazon_product_image(url: str) -> bool:
    """Check if URL appears to be an Amazon product image"""
    if not url or not isinstance(url, str):
        return False

    # Amazon image domain patterns
    # Get Amazon domains from config
    amazon_domains = (
        CONFIG.get("global_settings", {})
        .get("media_config", {})
        .get("amazon_media_domains", ["images-amazon.com", "m.media-amazon.com"])
    )

    # Check for Amazon image domains
    return any(domain in url for domain in amazon_domains)


def is_valid_high_res_image(url: str) -> bool:
    """Check if image URL is valid and high-resolution"""
    if not url or not isinstance(url, str):
        return False

    # Must be HTTP URL
    if not url.startswith("http"):
        return False

    # Get high-res threshold from config
    high_res_threshold = int(
        get_config_value(
            "global_settings",
            "image_config",
            "min_high_res_dimension",
            default=HIGH_RES_DIMENSION,
        )
    )

    # Check for Amazon high-res patterns
    # Look for size indicators
    size_match = re.search(r"\._(?:AC_)?(SL|SX|SY)(\d+)_", url)
    if size_match:
        size = int(size_match.group(2))
        return size >= high_res_threshold

    # If no specific size pattern, assume it's valid if from Amazon
    return is_amazon_product_image(url)


def is_valid_video_url(url: str) -> bool:
    """Check if URL appears to be a valid video URL"""
    if not url or not isinstance(url, str):
        return False

    # Must be HTTP URL
    if not url.startswith("http"):
        return False

    # Exclude VDP (Video Detail Page) links as they are navigation URLs, not direct
    # video files
    if "/vdp/" in url:
        return False

    # Check for video file extensions or streaming formats
    video_patterns = [
        r"\.mp4(\?|$)",
        r"\.m3u8(\?|$)",
        r"\.webm(\?|$)",
        r"\.mov(\?|$)",
        r"default\.mp4",
        r"media-amazon\.com.*\/.*\.mp4",
        r"vse-vms-transcoding",
    ]

    return any(re.search(pattern, url, re.IGNORECASE) for pattern in video_patterns)


def validate_video_url_accessibility(url: str) -> bool:
    """Validate that a video URL is accessible with proper rate limiting and
    error handling
    """
    if not url or not isinstance(url, str):
        return False

    # Skip problematic URL types early
    if url.startswith(("blob:", "data:")) or not url.startswith("http"):
        logger.debug(f"Skipping invalid URL type: {url[:50]}...")
        return False

    try:
        import random
        import time

        import requests

        # Get config values for delays and headers
        try:
            rate_config = CONFIG.get("global_settings", {}).get("rate_limiting", {})
            download_config = CONFIG.get("global_settings", {}).get(
                "download_config", {}
            )
            amazon_config = CONFIG.get("scrapers", {}).get("amazon", {})
            video_headers = amazon_config.get("http_headers", {}).get(
                "video_validation", {}
            )

            delay_range = rate_config.get("video_validation_delay", [0.5, 1.5])
            validation_timeout = download_config.get(
                "validation_timeout",
                CONFIG.get("global_settings", {})
                .get("system_timeouts", {})
                .get("head_request_timeout", 10),
            )
            range_bytes = download_config.get("validation_range_bytes", "0-1023")
        except Exception:
            # Fallback values
            delay_range = [0.5, 1.5]
            validation_timeout = (
                CONFIG.get("global_settings", {})
                .get("system_timeouts", {})
                .get("head_request_timeout", 10)
            )
            range_bytes = "0-1023"
            # Use fallback headers from config
            video_headers = (
                CONFIG.get("scrapers", {})
                .get("amazon", {})
                .get("http_headers", {})
                .get(
                    "video_validation",
                    {
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                        "Accept": (
                            "video/webm,video/ogg,video/*;q=0.9,"
                            "application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5"
                        ),
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "identity",
                        "Referer": "https://www.amazon.com/",
                    },
                )
            )

        # Add random delay to prevent rate limiting
        delay = random.uniform(delay_range[0], delay_range[1])  # noqa: S311
        time.sleep(delay)

        # Enhanced headers to appear more like a real browser
        headers = video_headers.copy()
        headers["Range"] = (
            f"bytes={range_bytes}"  # Only request first portion to minimize bandwidth
        )

        # Make HEAD request with enhanced error handling
        response = requests.head(
            url, timeout=validation_timeout, allow_redirects=True, headers=headers
        )
        is_accessible = response.status_code < 400

        status_msg = f"HTTP {response.status_code}"
        content_type = response.headers.get("content-type", "unknown")
        if is_accessible:
            logger.debug(
                f"Video URL accessible ({status_msg}, {content_type}): "
                f"{url[:60]}..."
            )
        else:
            logger.debug(f"Video URL failed ({status_msg}): {url[:60]}...")

        return is_accessible

    except requests.exceptions.Timeout:
        logger.debug(f"Video URL timeout: {url[:60]}...")
        return False
    except requests.exceptions.RequestException as e:
        logger.debug(f"Video URL request failed ({type(e).__name__}): {url[:60]}...")
        return False
    except Exception as e:
        logger.debug(f"Video URL validation error ({e}): {url[:60]}...")
        return False


def check_amazon_high_res_pattern(url: str, min_sl_size: int = None) -> bool:
    """Check if an image URL matches Amazon's high-resolution pattern"""
    if not url or not isinstance(url, str):
        return False

    # Get min_sl_size from config if not provided
    if min_sl_size is None:
        min_sl_size = get_config_value(
            "global_settings",
            "image_config",
            "min_high_res_dimension",
            default=HIGH_RES_DIMENSION,
        )

    # Check for Amazon high-res pattern ._SL{size}_
    match = re.search(r"\._(?:AC_)?(SL|SX|SY)([1-9]\d{2,})_", url)
    if match:
        try:
            size = int(match.group(2))
            return size >= min_sl_size
        except (ValueError, IndexError):
            logger.debug(f"Regex matched but size extraction failed for URL: {url}")
    return False


def filter_amazon_fallback_image(url: str, min_sl_size: int = None) -> bool:
    """Filter Amazon fallback images based on quality checks (matching old scraper)

    Args:
    ----
        url: Image URL to check
        min_sl_size: Minimum size threshold

    Returns:
    -------
        True if the URL should be kept, False if it should be filtered out

    """
    if not url or not isinstance(url, str) or not url.lower().startswith("http"):
        return False

    # Get min_sl_size from config if not provided
    if min_sl_size is None:
        min_sl_size = get_config_value(
            "global_settings",
            "image_config",
            "min_high_res_dimension",
            default=HIGH_RES_DIMENSION,
        )

    if is_placeholder_image(url):
        return False

    if check_amazon_high_res_pattern(url, min_sl_size):
        return True

    # Exclude low-res patterns
    if re.search(r"\._(?:S[XYR]|UX|US|AC)\d{1,3}[_,.]", url):
        logger.debug(f"Excluding low-res pattern image: {url}")
        return False

    # Must be a valid image extension
    if not re.search(r"\.(jpg|jpeg|png|webp)$", url, re.IGNORECASE):
        logger.debug(f"Excluding non-image file extension: {url}")
        return False

    logger.debug(f"Keeping filtered fallback image: {url}")
    return True


def is_placeholder_image(url: str) -> bool:
    """Check if an image URL is a placeholder or low-quality image

    Args:
    ----
        url: Image URL to check

    Returns:
    -------
        True if the URL appears to be a placeholder

    """
    if not url or not isinstance(url, str):
        return True

    url_lower = url.lower()

    # Check for common placeholder patterns (matching old scraper exactly)
    placeholder_patterns = [
        "pixel",
        "spinner",
        "loading",
        "grey-pixel",
        "adsystem",
        "transparent",
        "csgid=",
    ]

    if any(pattern in url_lower for pattern in placeholder_patterns):
        return True

    if "placeholder.com" in url_lower or "placehold.it" in url_lower:
        return True

    # Check for small thumbnail patterns
    if re.search(r"\._(?:SS|SR|SX|SY)\d{2,3}_", url):
        return True

    # Check for 1x1 pixel images (note: double backslash in old scraper)
    return bool(re.search("1x1\\.(png|gif|jpg)", url_lower))


def _validate_image_dimensions(
    url: str, min_dimension: int, debug_mode: bool = False, logger=None
) -> bool:
    """Validate that an image meets minimum dimension requirements.

    Since Amazon URL patterns like _SL1500_ don't guarantee actual dimensions,
    this function performs actual HTTP requests to verify image size.

    Args:
    ----
        url: Image URL to validate
        min_dimension: Minimum required dimension (width or height)
        debug_mode: Whether to log debug information
        logger: Logger instance for debug output

    Returns:
    -------
        True if image meets dimension requirements, False otherwise

    """
    import io
    import re

    import requests
    from PIL import Image

    try:
        # Quick URL pattern check first for obvious cases
        size_match = re.search(r"\._(?:AC_)?(SL|SX|SY)(\d+)_", url)
        if size_match:
            size = int(size_match.group(2))
            # If URL pattern indicates very high resolution, trust it
            try:
                very_high_res_threshold = (
                    CONFIG.get("global_settings", {})
                    .get("image_config", {})
                    .get("very_high_res_dimension", VERY_HIGH_RES_DIMENSION)
                )
            except Exception:
                very_high_res_threshold = VERY_HIGH_RES_DIMENSION

            if size >= very_high_res_threshold:
                if debug_mode and logger:
                    logger.debug(
                        f"URL pattern indicates very high-res ({size}px), trusting: "
                        f"{url[:80]}..."
                    )
                return True

        # For other cases, check actual dimensions by downloading image headers
        if debug_mode and logger:
            logger.debug(f"Checking actual dimensions for: {url[:80]}...")

        # Get headers from config for realistic requests
        headers = (
            CONFIG.get("scrapers", {})
            .get("amazon", {})
            .get("http_headers", {})
            .get(
                "media_download",
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                    ),
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                },
            )
        )

        timeout = (
            CONFIG.get("global_settings", {})
            .get("system_timeouts", {})
            .get("head_request_timeout", 10)
        )

        # Download first 4KB to get image headers and determine dimensions
        range_headers = headers.copy()
        range_headers["Range"] = "bytes=0-4095"

        response = requests.get(url, headers=range_headers, timeout=timeout)
        if response.status_code in [200, 206]:  # 206 = Partial Content
            try:
                # Try to get dimensions from partial image data
                img = Image.open(io.BytesIO(response.content))
                width, height = img.size
                max_dimension = max(width, height)

                result = max_dimension >= min_dimension
                if debug_mode and logger:
                    logger.debug(
                        f"Actual dimensions: {width}x{height}, max: {max_dimension}px, "
                        f"required: {min_dimension}px -> {'PASS' if result else 'FAIL'}"
                    )

                return result

            except Exception as e:
                # If we can't determine dimensions from partial data,
                # fall back to URL pattern if available
                if size_match:
                    size = int(size_match.group(2))
                    result = size >= min_dimension
                    if debug_mode and logger:
                        logger.debug(
                            f"Fallback to URL pattern {size}px: "
                            f"{'PASS' if result else 'FAIL'}"
                        )
                    return result
                else:
                    if debug_mode and logger:
                        logger.debug(f"Cannot determine dimensions, rejecting: {e}")
                    return False
        else:
            if debug_mode and logger:
                logger.debug(f"HTTP error {response.status_code}, rejecting")
            return False

    except Exception as e:
        if debug_mode and logger:
            logger.warning(f"Error validating image dimensions: {e}")
        return False


def _is_irrelevant_image(url: str) -> bool:
    """Check if image URL appears to be irrelevant (sprites, icons, ads, etc.)"""
    if not url or not isinstance(url, str):
        return True

    url_lower = url.lower()

    # Skip obvious non-product images
    irrelevant_patterns = [
        # UI elements
        "sprite",
        "nav-sprite",
        "icon",
        "button",
        "arrow",
        "logo",
        # Navigation/UI
        "gno/sprites",
        "navbar",
        "header",
        "footer",
        "ui-",
        # Ads and tracking
        "adsystem",
        "adnxs",
        "doubleclick",
        "amazon-adsystem",
        # Small/placeholder images
        "1x1",
        "pixel",
        "transparent",
        "loading",
        "spinner",
        # Size indicators for very small images
        "_sx38_",
        "_sy38_",
        "_sx50_",
        "_sy50_",
        "_ac_ux",
        "_sx75_",
        "_sy75_",
        # Amazon UI elements
        "homecustomproduct/360_icon",
        "g/01/gno",
        "g/01/ui",
    ]

    return any(pattern in url_lower for pattern in irrelevant_patterns)


def _meets_dimension_requirements(url: str, min_dimension: int) -> bool:
    """Quick check if URL pattern indicates it meets dimension requirements"""
    if not url or not isinstance(url, str):
        return False

    import re

    # Check URL pattern for size indicators
    size_match = re.search(r"_(?:AC_)?(SL|SX|SY)(\d+)_", url)
    if size_match:
        try:
            size = int(size_match.group(2))
            # If URL indicates size >= minimum, likely meets requirements
            return size >= min_dimension
        except (ValueError, IndexError):
            pass

    # If no size pattern, check if it's a main product image (usually high-res)
    # Main product images on Amazon typically have these patterns
    main_image_patterns = [
        r"/I/[A-Z0-9]+\._AC_",  # Main product image pattern
        r"/I/[A-Z0-9]+\.jpg$",  # Direct product image
    ]

    return any(re.search(pattern, url) for pattern in main_image_patterns)


def _is_product_related_video(url: str, page_title: str = "") -> bool:
    """Strict filtering for actual product videos only"""
    if not url or not isinstance(url, str):
        return False

    url_lower = url.lower()

    # STRICT: Must be Amazon product video domain
    if not any(domain in url_lower for domain in ["media-amazon.com"]):
        return False

    # STRICT: Must be MP4 video file
    if not (".mp4" in url_lower or "video" in url_lower):
        return False

    # CRITICAL: Must be official Amazon product video (al-na namespace), NOT
    # influencer videos
    if not ("al-na-" in url_lower and "productvideooptimized" in url_lower):
        return False

    # EXCLUDE: Any ad-related, promotional, or influencer content
    exclude_patterns = [
        # Ad services and tracking
        "adsystem",
        "adnxs",
        "doubleclick",
        "amazon-adsystem",
        "ads/",
        # CRITICAL: Influencer/VSE video services (main problem source)
        "vse-vms-transcoding",
        "vse-vms-closed-captions",
        "videopreview.jobtemplate",
        "default.jobtemplate.hls",
        "default.vertical.jobtemplate",
        "gandalf_preview",
        # Promotional/marketing videos
        "promo",
        "advertisement",
        "commercial",
        "marketing",
        "brand-video",
        # Generic content
        "howto",
        "tutorial",
        "generic",
        "demo",
        "training",
        # Amazon internal/UI videos
        "amazon-internal",
        "ui-video",
        "template",
        "widget",
        # Third-party content
        "youtube",
        "vimeo",
        "facebook",
        "instagram",
        "tiktok",
        # Non-product specific
        "category",
        "browse",
        "search",
        "recommendation",
    ]

    if any(pattern in url_lower for pattern in exclude_patterns):
        return False

    # REQUIRE: Official Amazon product video indicators ONLY
    required_patterns = [
        # Amazon product video paths (very specific) - EXCLUDE VSE
        "/al-na-",  # Amazon Labs video namespace (official product videos)
        "productvideooptimized",  # Optimized product videos
        # NOTE: Removed "/vse-vms/" as it's for influencer videos
    ]

    has_required = any(pattern in url_lower for pattern in required_patterns)

    # ADDITIONAL: Check for product-specific video patterns (OFFICIAL ONLY)
    product_video_patterns = [
        "/s/al-na-",  # Amazon product video storage (official)
        "item-video",  # Item-specific videos (official)
        "product-video",  # Product demonstration videos (official)
        # NOTE: Must still have al-na- pattern to be considered official
    ]

    has_product_pattern = any(
        pattern in url_lower for pattern in product_video_patterns
    )

    # Must have either required pattern OR product pattern
    return has_required or has_product_pattern


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

        logger.info(f"🔬 [ADVANCED DEBUG] Starting advanced analysis for ASIN: {asin}")

        # 1. Save page source if requested
        if debug_options.get("save_page_source"):
            try:
                page_source = driver.page_source
                source_file = debug_dir / f"{asin}_page_source.html"
                with open(source_file, "w", encoding="utf-8") as f:
                    f.write(page_source)
                logger.info(f"📄 Saved page source to: {source_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save page source: {e}")

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
                logger.info(f"📸 Saved screenshot to: {screenshot_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save screenshot: {e}")

        # 3. Deep image analysis if requested
        if debug_options.get("analyze_images"):
            _perform_deep_image_analysis(driver, asin, debug_dir, logger)

        # 4. Dump all image URLs if requested
        if debug_options.get("dump_image_urls"):
            _dump_all_image_urls(driver, asin, debug_dir, logger)

    except Exception as e:
        logger.error(f"❌ Advanced debug analysis failed: {e}")


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
                logger.warning(f"⚠️ Error analyzing image {i}: {e}")

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

        logger.info(f"📊 [DEEP ANALYSIS] Found {len(all_imgs)} total images")
        logger.info(
            f"📊 [DEEP ANALYSIS] Found {len(amazon_images)} Amazon media images"
        )
        logger.info(
            f"📊 [DEEP ANALYSIS] Found {len(high_res_images)} potential high-res images"
        )
        logger.info(f"📄 [DEEP ANALYSIS] Detailed analysis saved to: {analysis_file}")

        # Log high-res image candidates
        for img in high_res_images:
            src = img.get("src", "")
            indicators = [
                k for k, v in img.get("resolution_indicators", {}).items() if v
            ]
            logger.info(
                f"🎯 [HIGH-RES CANDIDATE] {src[:80]}... "
                f"(indicators: {', '.join(indicators)})"
            )

    except Exception as e:
        logger.error(f"❌ Deep image analysis failed: {e}")


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
            logger.debug(f"JS image data extraction failed: {e}")

        # Save all URLs
        urls_file = debug_dir / f"{asin}_all_image_urls.txt"
        with open(urls_file, "w", encoding="utf-8") as f:
            f.write(f"# All discovered image URLs for ASIN: {asin}\\n")
            f.write(f"# Total URLs found: {len(all_urls)}\\n\\n")

            for source_type, url in sorted(all_urls):
                f.write(f"[{source_type}] {url}\\n")

        logger.info(f"📝 [URL DUMP] Saved {len(all_urls)} URLs to: {urls_file}")

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
            f"🎯 [URL DUMP] Found {len(amazon_urls)} potential high-res Amazon URLs:"
        )
        for url in amazon_urls:
            logger.info(f"   • {url}")

    except Exception as e:
        logger.error(f"❌ URL dump failed: {e}")
