# Video Producer Module

Complete guide to video production with ContentEngineAI. Transform scraped product data into polished promotional videos with voiceovers, subtitles, and background music.

## Quick Start

### Single Product Video

```bash
# Basic production with a profile
poetry run python -m src.video.producer outputs/B0ASIN123/data.json slideshow_images1 --debug

# With ASS animated subtitles
poetry run python -m src.video.producer outputs/B0ASIN123/data.json slideshow_images1 \
  --subtitle-format ass --preset animated --debug
```

### Batch Processing

```bash
# Fixed profile for all products
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug

# Random profile per product (deterministic by product ID)
poetry run python -m src.video.producer --batch --random-profile --debug

# Random from specific pool
poetry run python -m src.video.producer --batch --random-profile \
  --profile-pool slideshow_images1 video_sequential mixed_media --debug
```

## CLI Reference

### Core Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `products_file` | Path to product data.json (positional) | `outputs/B0.../data.json` |
| `profile` | Video profile name from config (positional) | `slideshow_images1` |
| `--batch` | Process all products in outputs directory | `--batch` |
| `--batch-profile` | Profile for all batch products | `--batch-profile slideshow_images1` |
| `--random-profile` | Random profile per product | `--random-profile` |
| `--profile-pool` | Profiles for random selection | `--profile-pool prof1 prof2` |
| `--outputs-dir` | Custom outputs directory | `--outputs-dir custom_outputs` |
| `--fail-fast` | Stop batch on first failure | `--fail-fast` |

### Production Control

| Argument | Description | Example |
|----------|-------------|---------|
| `--debug` | Enable debug mode with verbose logging | `--debug` |
| `--clean` | Delete existing output before starting | `--clean` |
| `--step` | Run single pipeline step | `--step generate_script` |
| `--product-index` | Index for multi-product JSON files | `--product-index 0` |
| `--output-format` | Batch summary format (text/json) | `--output-format json` |

### Subtitle Configuration

| Argument | Description | Example |
|----------|-------------|---------|
| `--subtitle-engine` | Rendering engine: `ffmpeg` (default) or `pycaps` | `--subtitle-engine pycaps` |
| `--subtitle-format` | Format: `srt` (default) or `ass` (ffmpeg engine only) | `--subtitle-format ass` |
| `--preset` | Style preset (ffmpeg engine only) | `--preset animated` |
| `--subtitle-anchor` | Vertical position | `--subtitle-anchor bottom` |
| `--subtitle-margin` | Margin as frame fraction (0.0-0.5) | `--subtitle-margin 0.05` |
| `--content-aware` | Enable content-aware positioning | `--content-aware` |
| `--no-content-aware` | Disable content-aware positioning | `--no-content-aware` |

### Pycaps Engine Options

Only consumed when `--subtitle-engine pycaps`. See [docs/pycaps-subtitles.md](pycaps-subtitles.md)
for install and config details.

| Argument | Description | Example |
|----------|-------------|---------|
| `--pycaps-template` | Fixed template name | `--pycaps-template hype` |
| `--pycaps-template-pool` | Pool for deterministic per-product selection | `--pycaps-template-pool word-focus hype vibrant` |
| `--pycaps-renderer` | Renderer backend: `css` (default, Chromium) or `pictex` (browserless) | `--pycaps-renderer pictex` |

### ASS Effect Options

| Argument | Description | Example |
|----------|-------------|---------|
| `--ass-karaoke` | Enable karaoke word highlighting | `--ass-karaoke` |
| `--ass-fade` | Enable fade in/out effects | `--ass-fade` |

### Subtitle Styling

| Argument | Description | Example |
|----------|-------------|---------|
| `--subtitle-style-preset` | Style preset (minimal, modern, bold, animated, random) | `--subtitle-style-preset bold` |
| `--font-size-scale` | Font size multiplier (0.5-2.0) | `--font-size-scale 1.2` |
| `--subtitle-alignment` | Horizontal alignment | `--subtitle-alignment center` |
| `--max-line-length` | Max characters per line | `--max-line-length 25` |
| `--max-words-per-line` | Max words per line (0 to disable) | `--max-words-per-line 4` |
| `--max-duration` | Max subtitle duration (seconds) | `--max-duration 5.0` |
| `--min-duration` | Min subtitle duration (seconds) | `--min-duration 0.8` |

### Randomization

| Argument | Description |
|----------|-------------|
| `--randomize-fonts` / `--no-randomize-fonts` | Toggle font randomization |
| `--randomize-colors` / `--no-randomize-colors` | Toggle color randomization |
| `--randomize-effects` / `--no-randomize-effects` | Toggle effect randomization |

### Visual Layout

| Argument | Description | Example |
|----------|-------------|---------|
| `--image-width-percent` | Image width as frame fraction | `--image-width-percent 0.75` |
| `--image-top-position-percent` | Image top position | `--image-top-position-percent 0.2` |

### Platform & Metadata

| Argument | Description | Example |
|----------|-------------|---------|
| `--target-platform` | Target platform | `--target-platform youtube` |
| `--metadata-mode` | Metadata style | `--metadata-mode optimized` |

**Platforms**: `youtube`, `tiktok`, `instagram`, `multi`
**Metadata modes**: `unified` (same for all), `optimized` (platform-specific SEO)

---

## Video Profiles

Profiles define the complete video style including assembly mode, subtitle settings, and visual layout. Profiles are configured in `config/video_production.yaml`.

### Profile Structure

```yaml
video_profiles:
  slideshow_images1:
    description: "Image slideshow with animated subtitles"
    use_scraped_videos: false
    video_assembly_mode: null  # Images only
    subtitle_format: "ass"
    subtitle_preset: "animated"
    font_size_scale: 1.0

  video_sequential:
    description: "Sequential video clips"
    use_scraped_videos: true
    video_assembly_mode: "sequential"
    video_audio_handling: "remove"
    subtitle_format: "srt"
```

### Configuration Precedence

Settings are merged in this order (highest to lowest priority):

1. **CLI arguments** - Override everything
2. **Profile settings** - Per-profile customization
3. **Global config** - Default values from YAML

---

## Video Assembly Modes

Control how scraped videos are assembled into the final output.

| Mode | Description | Use Case |
|------|-------------|----------|
| `sequential` | Play all videos in order | Multiple product videos |
| `single_best` | Use longest/highest-quality video | Feature single hero video |
| `mixed_media` | Interleave videos with images | Dynamic variety |
| `video_first_fallback` | Try videos, fallback to images | Best of both worlds |

### Mode Selection

```bash
# In profile (video_production.yaml)
video_assembly_mode: "sequential"

# Or via profile selection
poetry run python -m src.video.producer data.json video_sequential
```

**Note**: If `use_scraped_videos: false`, assembly mode is ignored and only images are used.

---

## Subtitle Formats

### SRT (SubRip)

Simple text-based format with timing. Compatible with all players.

```bash
poetry run python -m src.video.producer data.json profile --subtitle-format srt
```

### ASS (Advanced SubStation Alpha)

Rich format supporting:
- **Styling**: Fonts, colors, outlines, shadows
- **Effects**: Karaoke, fade, typewriter, movement
- **Positioning**: Precise pixel placement, animations

```bash
# Basic ASS
poetry run python -m src.video.producer data.json profile --subtitle-format ass

# With effects
poetry run python -m src.video.producer data.json profile \
  --subtitle-format ass --preset animated --ass-karaoke --ass-fade
```

### Style Presets

| Preset | Description | Effects |
|--------|-------------|---------|
| `minimal` | Clean, no animations | None |
| `modern` | Subtle styling | Fade only |
| `bold` | Strong visual presence | Glow, fade |
| `animated` | Full animation suite | Karaoke, movement, pulse |
| `random` | Random effect per video | Varies |

---

## ASS Effects Reference

### Karaoke (`--ass-karaoke`)

Word-by-word highlighting synchronized with speech. Uses `\kf` tags for smooth fill transitions.

```
{\kf50}Hello {\kf40}world
```

### Fade (`--ass-fade`)

Smooth fade in/out transitions. Uses `\fad(in,out)` timing in milliseconds.

```
{\fad(200,200)}Subtitle text
```

### Typewriter

Character-by-character reveal effect. Uses alpha transparency transitions.

### Movement

Subtitles slide into position. Uses `\move(x1,y1,x2,y2)` for animated positioning.

### Scale Pulse

Text subtly grows and shrinks. Uses `\t(\fscx,\fscy)` transforms.

### Glow

Color transitions create pulsing glow effect. Uses `\t(\3c&H...)` color animation.

---

## Pipeline Steps

The producer runs through these steps in order:

1. **gather_visuals** - Collect images and videos from scraped data
2. **generate_script** - Create voiceover script via LLM
3. **generate_description** - Create platform metadata
4. **create_voiceover** - Generate speech via TTS
5. **generate_subtitles** - Create synchronized subtitles
6. **download_music** - Fetch background music (Freesound)
7. **assemble_video** - Combine all elements into final video

### Running Single Steps

```bash
# Resume from a specific step
poetry run python -m src.video.producer data.json profile --step generate_subtitles --debug

# Debug a failing step
poetry run python -m src.video.producer data.json profile --step assemble_video --debug
```

---

## Batch Processing

See [Batch Processing Guide](batch-processing.md#producer-batch-mode) for complete batch documentation.

### Quick Reference

```bash
# Fixed profile
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1

# Random profiles (deterministic per product)
poetry run python -m src.video.producer --batch --random-profile

# JSON output for automation
poetry run python -m src.video.producer --batch --batch-profile profile1 --output-format json
```

---

## Troubleshooting

### Common Issues

**Issue**: "Profile not found: xyz"
- **Cause**: Invalid profile name
- **Solution**: Check `config/video_production.yaml` for available profiles under `video_profiles:`

**Issue**: "Insufficient media for production"
- **Cause**: Product has fewer than minimum required images/videos
- **Solution**: Verify scraping completed successfully; check `min_images_if_no_video` setting (default: 5)

**Issue**: "TTS generation failed"
- **Cause**: Google Cloud credentials missing or invalid
- **Solution**: Set `GOOGLE_APPLICATION_CREDENTIALS` to your service account JSON path; verify Text-to-Speech API is enabled

**Issue**: "Freesound download failed"
- **Cause**: Freesound API credentials missing or circuit breaker tripped
- **Solution**: Set `FREESOUND_API_KEY`; local fallback will be used if configured

**Issue**: "FFmpeg not found"
- **Cause**: FFmpeg not installed or not in PATH
- **Solution**: Install FFmpeg (`apt install ffmpeg` or `brew install ffmpeg`)

**Issue**: "ASS subtitle effects not rendering"
- **Cause**: Player doesn't support ASS format
- **Solution**: Use VLC, mpv, or similar player with ASS support; or use `--subtitle-format srt`

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
poetry run python -m src.video.producer data.json profile --debug
```

Logs are written to `logs/producer.log`.

### Clean Run

Force fresh start by deleting cached artifacts:

```bash
poetry run python -m src.video.producer data.json profile --clean --debug
```

---

## Related Documentation

- **[Batch Processing](batch-processing.md)** - Multi-product workflows
- **[Configuration](configuration.md)** - YAML configuration reference
- **[Architecture](architecture.md)** - Technical architecture
- **[Troubleshooting](troubleshooting.md)** - Additional debugging
