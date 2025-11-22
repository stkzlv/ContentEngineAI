# Requirements Document

## Introduction

The scraper-video-detection feature enhances ContentEngineAI's Amazon scraper to reliably detect, validate, and download high-quality product videos. Currently, the video extraction infrastructure exists but requires validation and enhancement to ensure it consistently captures the best available video content from Amazon product pages. This feature enables video-first promotional content creation by providing high-quality product videos as source material for the video assembly pipeline.

## Alignment with Product Vision

This feature directly supports ContentEngineAI's core principle of "Quality Through Intelligence" by using smart extraction and validation logic to automatically select the highest quality product videos. It extends the "End-to-End Automation" capability to include video content, enabling richer, more engaging promotional videos. By ensuring reliable video acquisition, this feature enables the product to scale content production with video-rich outputs that meet social media quality standards.

## Requirements

### Requirement 1: High-Quality Video Detection

**User Story:** As an e-commerce marketer, I want the scraper to automatically detect and extract high-quality product videos from Amazon pages, so that my promotional videos can showcase products in motion and detail.

#### Acceptance Criteria

1. WHEN the scraper processes an Amazon product page THEN it SHALL extract all available MP4 video URLs from page scripts and video elements
2. WHEN multiple video sources exist THEN the scraper SHALL filter videos to prioritize product-specific content using ASIN matching
3. WHEN video URLs are found THEN the scraper SHALL validate video accessibility using HEAD requests before download
4. IF a product has VDP (Video Detail Page) links THEN the scraper SHALL navigate to VDP pages and extract high-resolution video streams
5. WHEN extracting videos THEN the scraper SHALL select highest quality versions available (highest resolution and bitrate)

### Requirement 2: Video Metadata Extraction

**User Story:** As a video producer, I want the scraper to capture video metadata (duration, resolution, codec), so that the video assembly pipeline can make informed decisions about clip selection and processing.

#### Acceptance Criteria

1. WHEN a video is downloaded THEN the scraper SHALL extract and store video duration using FFprobe
2. WHEN a video is downloaded THEN the scraper SHALL capture video resolution (width x height)
3. WHEN a video is downloaded THEN the scraper SHALL identify video codec and format information
4. WHEN metadata extraction fails THEN the scraper SHALL log a warning and continue with the download

### Requirement 3: Video Validation and Quality Filtering

**User Story:** As an e-commerce marketer, I want the scraper to validate downloaded videos for quality and accessibility, so that only usable videos are passed to the video production pipeline.

#### Acceptance Criteria

1. WHEN videos are extracted THEN the scraper SHALL filter out low-quality videos below minimum resolution thresholds
2. WHEN a video URL is identified THEN the scraper SHALL validate file accessibility with HEAD request before attempting full download
3. IF a video file is corrupted or unreadable THEN the scraper SHALL skip it and log an error
4. WHEN validation is complete THEN the scraper SHALL track downloaded video paths in product data structure

### Requirement 4: Organized Video Storage

**User Story:** As a system operator, I want product videos stored in organized directories with clear naming, so that the video assembly pipeline can easily locate and process video files.

#### Acceptance Criteria

1. WHEN videos are downloaded THEN the scraper SHALL store them in `outputs/{product_id}/videos/` directory
2. WHEN multiple videos exist for a product THEN the scraper SHALL name files using pattern `video_{index}.mp4` with sequential indexing
3. WHEN video download completes THEN the scraper SHALL update product data with list of downloaded video file paths
4. IF the videos directory doesn't exist THEN the scraper SHALL create it before downloading

### Requirement 5: Robust Error Handling

**User Story:** As a batch processing user, I want the scraper to handle video extraction failures gracefully, so that a single video failure doesn't halt processing of an entire product batch.

#### Acceptance Criteria

1. WHEN video extraction fails THEN the scraper SHALL log the error and continue processing remaining videos
2. IF all videos fail to download THEN the scraper SHALL complete product processing with images only
3. WHEN network errors occur THEN the scraper SHALL retry video downloads with exponential backoff (max 2 retries)
4. WHEN video processing errors occur THEN the scraper SHALL provide clear, actionable error messages in logs

### Requirement 6: Product Data Integration

**User Story:** As a video producer, I want video information integrated into product data structures, so that the video assembly pipeline has all necessary information to process videos.

#### Acceptance Criteria

1. WHEN videos are discovered THEN the scraper SHALL populate `videos: list[str]` field with video URLs
2. WHEN videos are downloaded THEN the scraper SHALL populate `downloaded_videos: list[str]` field with file paths
3. WHEN product data is saved THEN the scraper SHALL include video metadata in `data.json`
4. WHEN no videos are found THEN the scraper SHALL set video fields to empty lists without raising errors

## Non-Functional Requirements

### Code Architecture and Modularity

- **Single Responsibility Principle**: Video extraction logic must remain in `media_extractor.py`, download logic in `downloader.py`, validation in `media_validator.py`
- **Modular Design**: Video detection components must be reusable across different e-commerce platforms (not Amazon-specific where possible)
- **Dependency Management**: Minimize dependencies between scraper modules and video producer modules
- **Clear Interfaces**: `BaseProductData` model provides clean contract for video data exchange between scraper and producer

### Performance

- **Extraction Speed**: Video URL extraction must complete in <5 seconds per product page
- **Download Efficiency**: Use async downloads to fetch multiple videos concurrently (max 3 concurrent downloads)
- **Resource Usage**: Video downloads must use streaming to avoid loading entire files into memory
- **Timeout Management**: Video downloads must timeout after 300 seconds to prevent hanging

### Security

- **URL Validation**: All video URLs must be validated against expected Amazon CDN domains
- **File Type Validation**: Downloaded files must be verified as valid MP4 format before storage
- **Path Security**: File paths must be sanitized to prevent directory traversal attacks

### Reliability

- **Retry Logic**: Failed video downloads must retry with exponential backoff (2 attempts max)
- **Graceful Degradation**: Product processing must succeed even if all videos fail to download
- **Data Consistency**: Product data must remain valid even if video extraction partially fails
- **Validation Coverage**: All downloaded videos must pass accessibility validation before use

### Usability

- **Debug Logging**: Detailed logging must be available with `--debug` flag for troubleshooting video extraction
- **Progress Visibility**: Video download progress must be visible in console output
- **Clear Error Messages**: Video extraction errors must include actionable information (URL, error type, suggested fixes)
- **Configuration**: Video quality thresholds and download settings must be configurable via YAML
