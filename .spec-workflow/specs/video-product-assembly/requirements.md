# Requirements Document

## Introduction

The video-product-assembly feature enables ContentEngineAI to create engaging promotional videos using product videos as primary content. Currently, the video producer supports image-based slideshows with voiceover, subtitles, and background music. This feature extends the producer to assemble product videos with flexible modes (sequential, single-best, mixed-media, video-first-fallback), configurable aspect ratio handling (letterbox, crop-to-fit, smart-scale), and audio normalization options. This unlocks video-rich content creation that showcases products in motion for maximum engagement on social platforms.

## Alignment with Product Vision

This feature directly supports ContentEngineAI's "Quality Through Intelligence" principle by implementing smart video assembly algorithms that optimize for engagement and platform requirements. It extends "End-to-End Automation" to include video-first content creation, enabling users to leverage rich product videos without manual editing. By providing multiple assembly modes and intelligent aspect ratio handling, this feature maintains the "Modular Flexibility" core principle while scaling content production quality.

## Requirements

### Requirement 1: Video Assembly Modes

**User Story:** As an e-commerce marketer, I want to choose how product videos are assembled (sequential, single-best, mixed-media, video-first-fallback), so that I can create different video styles optimized for various social platforms and campaign objectives.

#### Acceptance Criteria

1. WHEN profile is configured with `video_assembly_mode: "sequential"` THEN the producer SHALL concatenate all product videos end-to-end with crossfade transitions
2. WHEN profile is configured with `video_assembly_mode: "single_best"` THEN the producer SHALL select the longest video and loop it seamlessly to match voiceover duration
3. WHEN profile is configured with `video_assembly_mode: "mixed_media"` THEN the producer SHALL interleave product videos and images throughout the timeline using duration-based algorithm
4. WHEN profile is configured with `video_assembly_mode: "video_first_fallback"` THEN the producer SHALL use all videos first, then add images for remaining duration
5. WHEN video duration is insufficient THEN the producer SHALL loop videos or add images to match voiceover length ±1 second
6. WHEN video duration exceeds voiceover THEN the producer SHALL trim the last video with fade-out effect

### Requirement 2: Aspect Ratio Handling

**User Story:** As a content creator, I want configurable aspect ratio handling (letterbox, crop-to-fit, smart-scale), so that product videos display optimally in 9:16 vertical format without distortion or loss of key content.

#### Acceptance Criteria

1. WHEN profile is configured with `video_aspect_mode: "letterbox"` THEN the producer SHALL maintain original aspect ratio with black padding to fill 9:16 frame
2. WHEN profile is configured with `video_aspect_mode: "crop_to_fit"` THEN the producer SHALL scale video to fill frame and crop edges, centering the crop region
3. WHEN profile is configured with `video_aspect_mode: "smart_scale"` THEN the producer SHALL automatically choose crop or letterbox based on aspect ratio similarity (within 10% → crop, else letterbox)
4. WHEN applying letterbox THEN the producer SHALL center video both vertically and horizontally
5. WHEN applying crop THEN the producer SHALL center crop to preserve main subject area

### Requirement 3: Audio Normalization

**User Story:** As a video producer, I want configurable audio handling for product videos (complete removal or mixed at low volume), so that the final output has clean audio with only voiceover and background music or includes ambient product sounds.

#### Acceptance Criteria

1. WHEN profile is configured with `video_audio_handling: "remove"` THEN the producer SHALL strip all original audio from product videos
2. WHEN profile is configured with `video_audio_handling: "mixed"` THEN the producer SHALL preserve original video audio at configured volume level (default -30dB)
3. WHEN mixing audio THEN the producer SHALL combine original video audio, voiceover, and background music without clipping
4. WHEN video has no audio track THEN the producer SHALL handle gracefully without errors
5. WHEN `video_original_volume` is specified THEN the producer SHALL apply that volume adjustment (range: -60 to 0 dB)

### Requirement 4: Format Normalization

**User Story:** As a system operator, I want all product videos automatically normalized to consistent format (H.264, 30fps, yuv420p), so that video assembly is reliable and compatible across all platforms and devices.

#### Acceptance Criteria

1. WHEN product videos are processed THEN the producer SHALL detect video codec, frame rate, and pixel format
2. WHEN video has non-H.264 codec THEN the producer SHALL transcode to H.264 (libx264)
3. WHEN video has non-30fps frame rate THEN the producer SHALL normalize to 30fps
4. WHEN video has non-yuv420p pixel format THEN the producer SHALL convert to yuv420p
5. WHEN transcoding is required THEN the producer SHALL cache normalized videos to avoid re-processing
6. WHEN videos are already in correct format THEN the producer SHALL skip transcoding for efficiency

### Requirement 5: Duration Matching Algorithm

**User Story:** As a video producer, I want product videos to precisely match voiceover duration (±1 second), so that the final video is perfectly synchronized without awkward silence or cut-off audio.

#### Acceptance Criteria

1. WHEN assembling videos THEN the producer SHALL calculate required duration from voiceover audio file
2. WHEN total video duration is less than required THEN the producer SHALL loop videos with crossfade or add images to fill remaining time
3. WHEN total video duration exceeds required THEN the producer SHALL trim the last video with fade-out effect to match exactly
4. WHEN applying loops THEN the producer SHALL add crossfade transitions at loop points (configurable duration, default 0.5s)
5. WHEN final video is assembled THEN the duration SHALL match voiceover length within ±1 second tolerance

### Requirement 6: Video Transition System

**User Story:** As a content creator, I want smooth crossfade transitions between video clips and between videos and images, so that the final video flows naturally without jarring cuts.

#### Acceptance Criteria

1. WHEN concatenating video clips THEN the producer SHALL apply crossfade transition between consecutive clips
2. WHEN transitioning from video to image THEN the producer SHALL apply crossfade with same duration as video-to-video transitions
3. WHEN transition duration is configured THEN the producer SHALL use that value (default 0.5s, configurable via `video_transition_duration`)
4. WHEN creating loop transitions THEN the producer SHALL apply crossfade at loop point to create seamless infinite effect
5. WHEN transitions are applied THEN the producer SHALL ensure no visible discontinuity or black frames

### Requirement 7: Profile Configuration Integration

**User Story:** As a video producer, I want all video assembly settings configurable per profile, so that I can create different video styles (product_video_sequential, product_video_single, etc.) without changing global settings.

#### Acceptance Criteria

1. WHEN profile is defined in YAML THEN it SHALL support all video assembly settings (mode, aspect, audio, transitions)
2. WHEN profile settings are specified THEN they SHALL override global defaults through profile merging system
3. WHEN profile has `use_scraped_videos: true` THEN the producer SHALL use product videos as primary content
4. WHEN profile has `use_scraped_videos: false` THEN the producer SHALL ignore product videos and use images only
5. WHEN CLI arguments are provided THEN they SHALL override profile settings (highest precedence)

### Requirement 8: Media Validation and Error Handling

**User Story:** As a batch processing user, I want the producer to validate video availability and handle failures gracefully, so that a single video issue doesn't halt processing of an entire batch.

#### Acceptance Criteria

1. WHEN product has ≥1 video THEN the producer SHALL enable video-first profiles
2. WHEN product has no videos THEN the producer SHALL fall back to image-only profiles without error
3. WHEN video file is corrupted or unreadable THEN the producer SHALL skip it and continue with remaining videos
4. WHEN all videos fail THEN the producer SHALL complete processing with images only
5. WHEN video processing errors occur THEN the producer SHALL log clear error messages with file paths and error types

### Requirement 9: FFmpeg Filter Chain Construction

**User Story:** As a video engineer, I want efficient FFmpeg filter chains for video assembly, so that video processing completes quickly without excessive memory usage or rendering artifacts.

#### Acceptance Criteria

1. WHEN building filter chains THEN the producer SHALL construct efficient FFmpeg filtergraphs with minimal complexity
2. WHEN scaling videos THEN the producer SHALL use `scale` filter with `force_original_aspect_ratio` parameter
3. WHEN padding videos THEN the producer SHALL use `pad` filter with centered positioning calculation
4. WHEN cropping videos THEN the producer SHALL use `crop` filter with centered region selection
5. WHEN concatenating clips THEN the producer SHALL use `concat` filter with proper segment counting

### Requirement 10: Configuration System Extension

**User Story:** As a system administrator, I want video assembly settings in YAML configuration with clear documentation, so that I can customize video production behavior without modifying code.

#### Acceptance Criteria

1. WHEN configuration is loaded THEN it SHALL include all video assembly parameters with sensible defaults
2. WHEN invalid configuration is detected THEN the producer SHALL fail startup with clear error message indicating the problem
3. WHEN video profiles are defined THEN each SHALL include all required settings (assembly_mode, aspect_mode, audio_handling)
4. WHEN configuration is documented THEN it SHALL include inline comments explaining each video parameter and its effect
5. WHEN configuration follows schema THEN it SHALL be validated using Pydantic models with type checking

## Non-Functional Requirements

### Code Architecture and Modularity

- **Single Responsibility Principle**: Video assembly logic in `assembler.py`, profile management in `video_config.py`, producer orchestration in `producer.py`
- **Modular Design**: Video processing components reusable and testable independently from producer pipeline
- **Dependency Management**: Minimize coupling between video assembly and other pipeline steps
- **Clear Interfaces**: Well-defined data contracts between producer, assembler, and configuration systems

### Performance

- **Assembly Speed**: Video assembly (excluding voiceover generation) must complete in <60 seconds for typical 30-second output
- **Transcoding Efficiency**: Video format normalization must use efficient FFmpeg presets (medium or faster)
- **Memory Usage**: Video processing must use streaming and avoid loading entire videos into memory
- **Caching Strategy**: Normalized videos must be cached to avoid redundant transcoding

### Security

- **Path Validation**: All video file paths must be validated to prevent directory traversal
- **Command Injection**: FFmpeg commands must use parameterized arguments to prevent shell injection
- **Resource Limits**: Video processing must enforce maximum file sizes and durations to prevent DoS

### Reliability

- **Error Recovery**: Video processing failures must not crash the entire pipeline
- **Graceful Degradation**: System must fall back to image-only mode when videos unavailable or corrupted
- **Validation**: All video files must be validated before processing to avoid mid-pipeline failures
- **Retry Logic**: Transient FFmpeg errors must trigger retry with exponential backoff (max 2 attempts)

### Usability

- **Debug Logging**: Detailed video processing logs available with `--debug` flag
- **Progress Visibility**: Video assembly progress must be visible in console output
- **Clear Error Messages**: Video processing errors must include file paths, FFmpeg error details, and suggested fixes
- **Configuration Examples**: All video profiles must have documented examples in YAML with explanatory comments
