"""Video extraction utilities for Amazon scraper.

This module handles extraction of product videos from Amazon product pages
using Botasaurus browser automation, including m3u8 network capture and
systematic DOM-based video discovery.
"""

import logging
import time

from botasaurus.browser import Driver

from .config import CONFIG
from .image_utils import is_valid_video_url

logger = logging.getLogger(__name__)


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
                        logger.info("Found m3u8 URL: %s...", url[:100])
            except Exception as e:
                if debug:
                    logger.debug("Error in response handler: %s", e)

        # Register response handler
        driver.after_response_received(capture_m3u8_handler)

        if debug:
            logger.info("Monitoring network traffic for %d seconds...", timeout)

        # Wait for video to load and network requests to fire
        time.sleep(timeout)

        if debug:
            logger.info("Captured %d m3u8 URLs from network traffic", len(m3u8_urls))

        return m3u8_urls

    except AttributeError as e:
        # Botasaurus might not have after_response_received method
        if debug:
            logger.warning("Network monitoring not available: %s", e)
        return []
    except Exception as e:
        logger.error("Failed to capture network traffic: %s", e)
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
        "extract_functional_videos_with_validation called (max: %d, DEBUG=%s)",
        max_videos,
        DEBUG_MODE,
    )

    if DEBUG_MODE:
        logger.info("Using systematic video extraction (max: %d videos)", max_videos)

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
            logger.info("Current product ASIN: %s", current_asin)
            logger.info("Product brand: %s", product_brand)
            logger.info("Product model: %s", product_model)
            logger.info("Product keywords: %s", product_keywords)

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
                        return asin && asin !== currentAsin && asin !== '' &&
                               asin !== 'null';
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
                    const inTrustedArea = (
                        element.closest('#imageBlock') ||
                        element.closest('#altImages') ||
                        element.closest('#ivTitle') ||
                        element.closest('#main-image-container')
                    );

                    // Also accept if in ASIN container that's NOT in A+ content
                    const asinSel = '[data-asin="' + currentAsin + '"]';
                    const inAsinContainer = element.closest(asinSel);
                    const notInAplus = (
                        inAsinContainer &&
                        !element.closest(
                            '#aplus, .aplus-module, [id*="aplus"], [class*="aplus"]'
                        )
                    );

                    return inTrustedArea || notInAplus;
                }}

                // 1. Extract from script tags containing video data
                document.querySelectorAll('script:not([src])').forEach(script => {{
                    const content = script.textContent;

                    // Only process scripts that mention current ASIN to avoid related
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
                        // Double check: VDP link must contain current ASIN or be in
                        // ASIN container
                        const vdpAsinSel = '[data-asin="' + currentAsin + '"]';
                        const inAsinContainer = link.closest(vdpAsinSel);
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
                        "Method 1 found: %d direct videos, %d VDP links, %d thumbnails",
                        len(direct_videos),
                        len(vdp_links),
                        len(video_thumbnails),
                    )

                # Method 1 direct video extraction with isValidProductVideo() filtering
                # Re-enabled with DOM context validation to exclude competitor videos
                for url in direct_videos:
                    if len(video_urls) >= max_videos:
                        if DEBUG_MODE:
                            logger.info(
                                "Method 1: Reached video limit (%d/%d),"
                                " stopping extraction",
                                len(video_urls),
                                max_videos,
                            )
                        break
                    if (
                        url not in video_urls
                        and url != "null"
                        and is_valid_video_url(url)
                    ):
                        video_urls.append(url)
                        if DEBUG_MODE:
                            logger.info("Method 1 found direct video: %s...", url[:80])

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
                logger.warning("Method 1 failed: %s", e)

        # Method 2: Strategic thumbnail clicking (same approach as images)
        logger.info(
            "Method 2 check: %d videos so far, max=%d", len(video_urls), max_videos
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
                                    "✅ Video player triggered, waiting for "
                                    "network requests"
                                )
                        else:
                            logger.warning("⚠️ No video thumbnail found")
                    except Exception as e:
                        logger.warning("Failed to trigger video player: %s", e)

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
                                            "Method 2: Reached video limit (%d/%d), "
                                            "stopping network capture",
                                            len(video_urls),
                                            max_videos,
                                        )
                                    break

                                if m3u8_url and m3u8_url not in video_urls:
                                    video_urls.append(m3u8_url)
                                    if DEBUG_MODE:
                                        logger.info(
                                            "Captured m3u8 URL from network: %s...",
                                            m3u8_url[:80],
                                        )
                        else:
                            logger.warning(
                                "⚠️ No m3u8 URLs captured from network traffic"
                            )

                    except Exception as e:
                        if DEBUG_MODE:
                            logger.warning("Method 2 network capture failed: %s", e)

            except Exception as e:
                if DEBUG_MODE:
                    logger.warning("Method 2 failed: %s", e)

        # Method 3: Direct element extraction with DOM context filtering
        if len(video_urls) < max_videos:
            if DEBUG_MODE:
                logger.info(
                    "📋 Method 3: Direct element extraction with context filtering"
                )

            try:
                # Define valid product gallery selectors (product videos location)
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

                        # Build JS that checks containment without passing Element
                        js_code = f"""
                        (function() {{
                            // Find the video element by its src attribute
                            const sel = 'video[src], video source[src]';
                            const videoElements = (
                                document.querySelectorAll(sel)
                            );
                            let targetElement = null;

                            for (const elem of videoElements) {{
                                if (elem.src === '{video_src}' ||
                                    (elem.tagName === 'SOURCE' &&
                                     elem.src === '{video_src}')) {{
                                    targetElement = (
                                        elem.tagName === 'SOURCE' ?
                                        elem.parentElement : elem
                                    );
                                    break;
                                }}
                            }}

                            if (!targetElement) return false;

                            // Check if element is in EXCLUDED section (priority)
                            const excl = {excluded_selectors};
                            for (const selector of excl) {{
                                const excluded = document.querySelector(selector);
                                if (excluded && excluded.contains(targetElement)) {{
                                    return false;
                                }}
                            }}

                            // Check if element is in valid gallery section
                            const valid = {valid_gallery_selectors};
                            for (const selector of valid) {{
                                const gallery = document.querySelector(selector);
                                if (gallery && gallery.contains(targetElement)) {{
                                    return true;
                                }}
                            }}

                            return false; // Not in valid gallery
                        }})();
                        """

                        result = driver.run_js(js_code)

                        if not result and DEBUG_MODE:
                            logger.debug(
                                "Video rejected (not in product gallery): %s...",
                                video_src[:80],
                            )

                        return bool(result)
                    except Exception as e:
                        if DEBUG_MODE:
                            logger.debug("Context check failed: %s", e)
                        return False

                # Find video elements directly
                video_elements = driver.select_all("video[src], video source[src]")
                for video_elem in video_elements:
                    if len(video_urls) >= max_videos:
                        if DEBUG_MODE:
                            logger.info(
                                "Method 3a: Reached video limit (%d/%d), stopping "
                                "video element extraction",
                                len(video_urls),
                                max_videos,
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
                        # Check DOM context - only accept videos from gallery
                        if is_in_product_gallery(video_elem):
                            video_urls.append(src)
                            if DEBUG_MODE:
                                logger.info(
                                    "Method 3 found product gallery video: %s...",
                                    src[:80],
                                )
                        elif DEBUG_MODE:
                            logger.debug(
                                "Video rejected (not in product gallery): %s...",
                                src[:80],
                            )

                # Check for embedded video URLs in visible elements
                video_containers = driver.select_all(
                    "[data-video-url], [data-src*='.mp4']"
                )
                for container in video_containers:
                    if len(video_urls) >= max_videos:
                        if DEBUG_MODE:
                            logger.info(
                                "Method 3b: Reached video limit (%d/%d), stopping "
                                "container extraction",
                                len(video_urls),
                                max_videos,
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
                                    "Method 3 found product gallery"
                                    " container video: %s...",
                                    video_url[:80],
                                )
                        elif DEBUG_MODE:
                            logger.debug(
                                "Video rejected (container not in gallery): %s...",
                                video_url[:80],
                            )

            except Exception as e:
                if DEBUG_MODE:
                    logger.warning("Method 3 failed: %s", e)

    except Exception as e:
        if DEBUG_MODE:
            logger.error("Error in systematic video extraction: %s", e)

    # Remove duplicates while preserving order and limit results
    unique_urls = []
    for url in video_urls:
        if url not in unique_urls:
            unique_urls.append(url)
            if len(unique_urls) >= max_videos:
                if DEBUG_MODE:
                    logger.info(
                        "Deduplication: Reached video limit (%d/%d)",
                        len(unique_urls),
                        max_videos,
                    )
                break

    # Final summary logging
    if DEBUG_MODE:
        if len(unique_urls) >= max_videos:
            logger.info(
                "Extracted %d videos (hit configured limit) for %s",
                len(unique_urls),
                current_asin,
            )
        elif len(unique_urls) > 0:
            logger.info(
                "Extracted %d videos (found all available) for %s",
                len(unique_urls),
                current_asin,
            )
        else:
            logger.warning(
                "No videos found for %s (limit was %d)", current_asin, max_videos
            )

    return unique_urls
