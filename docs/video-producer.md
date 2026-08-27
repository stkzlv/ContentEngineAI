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

### Topic Video (no scraped product)

Renders from a subject rather than a listing, using a profile that sources every
visual from stock. Output lands in `outputs/topic-<slug>/`.

```bash
# One topic
poetry run python -m src.video.producer slideshow_stock \
  --topic "Why your wifi keeps dropping" \
  --topic-description "Router placement, channel congestion, 2.4 vs 5GHz." \
  --topic-keywords "wifi router, home network"

# Several, from a YAML file
poetry run python -m src.video.producer slideshow_stock --topics-file topics.yaml
```

```yaml
# topics.yaml
- title: "Why your wifi keeps dropping"
  description: "Router placement, channel congestion, 2.4 vs 5GHz."
  keywords: ["wifi router", "home network"]
- title: "Laptop fan always loud"
  description: "Dust, thermal paste, background CPU load."
```

A topic render draws from its own template pool, written to answer a question
rather than pitch a product, and uses a narrator profile whose calls to action
offer nothing to buy. Length follows the script, so a short description yields a
short video.

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
| `--strict` | Exit non-zero when any product was lost, to a failure or a skip | `--strict` |
| `--topic` | Render a subject instead of a product; replaces `products_file` | `--topic "Why wifi drops"` |
| `--topic-description` | Source material the script is written from | `--topic-description "Router placement."` |
| `--topic-keywords` | Comma-separated stock search terms for the topic | `--topic-keywords "wifi router, home network"` |
| `--topics-file` | YAML list of topics to render in turn | `--topics-file topics.yaml` |

### Production Control

| Argument | Description | Example |
|----------|-------------|---------|
| `--debug` | Enable debug mode with verbose logging | `--debug` |
| `--clean` | Delete existing output before starting | `--clean` |
| `--step` | Run single pipeline step | `--step generate_script` |
| `--product-index` | Index for multi-product JSON files | `--product-index 0` |
| `--output-format` | Batch summary format (text/json) | `--output-format json` |

### Script & Content

| Argument | Description | Example |
|----------|-------------|---------|
| `--script-template` | Force a specific script template (filename without `.md`) | `--script-template curiosity_hook` |
| `--voice-profile` | Force a specific TTS voice profile | `--voice-profile calm_confident` |
| `--pillar` | Content pillar for the run (filters templates, prepends pillar preamble, picks pillar audience) | `--pillar value` |

**Pillars** (default): `value` (mass-appeal staples), `novelty` (lesser-known finds), `utility` (problem/solution framing). Configured in `config/ai_services.yaml::script_templates.pillars`. Without `--pillar`, the product record's own pillar applies when it has one — the scraper attaches the source keyword's group. With neither, all templates are eligible and the global `target_audience` applies. `--pillar` works with `--topic` too: the preambles and audiences have topic counterparts (`pillar_preambles_topic`, `pillar_audiences_topic`) using the same keys, because the product versions are written about a thing being shown and would put a purchase in a script that recommends nothing. Template narrowing does not apply on a topic, since `pillars` maps to product templates and a topic uses the topic family; the pillar still shapes the preamble and the audience. See [Requirements](requirements.md) "Content Pillars" for the full system.

### Subtitle Configuration

| Argument | Description | Example |
|----------|-------------|---------|
| `--subtitle-engine` | Rendering engine: `ffmpeg` or `pycaps` (bundled YAML default is `pycaps`) | `--subtitle-engine ffmpeg` |
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
    subtitle_settings:
      style_preset: "animated"
      font_size_scale: 1.0

  slideshow_short_20s:
    description: "Short 15-30s slideshow tuned for hook iteration"
    use_scraped_images: true
    image_top_position_percent: 0.15
    first_frame_pre_motion: true   # Ken Burns settle-zoom on segment 0
    pre_motion_peak_zoom: 1.10

  video_sequential:
    description: "Sequential video clips"
    use_scraped_videos: true
    video_assembly_mode: "sequential"
    video_audio_handling: "remove"
```

Unknown keys are rejected at config load, so a typo here fails immediately
rather than being dropped and leaving the profile to render with the global
value. Subtitle settings go in the nested `subtitle_settings` block; the flat
`subtitle_*` keys still load but warn.

`subtitle_format` is not settable per profile, in either the flat or the
nested spelling, and a profile that sets it fails at config load. The subtitle
file's extension is derived from the global `config/subtitles.yaml` value, so a
profile-level format would be honoured by the merged settings and ignored by
the path, which mismatches the file against the filter that reads it. Set it
globally, or per run with `--subtitle-format`.

### Hook Overlay and Pre-Motion

Three visual-layer knobs live on `video_settings` and the per-profile partial override:

- `first_frame_pre_motion` / `pre_motion_peak_zoom` — when enabled, the first image segment starts at `pre_motion_peak_zoom` and settles to 1.0 over the segment, so frame 0 is mid-motion rather than static. Default off on the existing 30-45s profiles, on for `slideshow_short_20s`.
- `hook_overlay` — burns a short headline as centre-upper static drawtext on the first `duration_sec` seconds (default 1.5), at `size_factor` times narration size, with no per-word reveal. The text is an authored headline generated separately from the spoken script, so the hook doesn't repeat the first caption line; when no headline is available it falls back to the script's first sentence. Long text wraps to at most `max_lines` lines, each held within `max_width_fraction` of the frame width, and the font shrinks when wrapping alone can't fit. Drawn after subtitles and before the disclosure rewrite so `#ad` stays on top. The headline lands in `pipeline_state.json::hook_headline`. A topic render uses a
separate headline prompt: the product one requires a product category noun,
which on a topic with no device makes the model invent one. The topic prompt
asks for the symptom or the fix and forbids naming anything the script does
not cover.
- `cold_open_variant_pool` — list of named cold-open variants rotated deterministically per product (salted MD5). The chosen variant name lands in `pipeline_state.json::assemble_video.cold_open_variant` for downstream analytics.

See `config/video_production.yaml::video_settings` for the canonical defaults and inline notes.

### Stock-only profiles run script-first

`slideshow_stock` sets `use_scraped_images: false` and `use_scraped_videos:
false`, so nothing on screen comes from a scraped product. That profile
generates the script before gathering visuals, and searches the stock library
on phrases taken from the narration.

Every other bundled profile shows product photography and keeps the default
order, gathering visuals first. That also rejects a product with too few images
before an LLM call is paid for, which is why the order is not simply reversed
for everything.

The phrases are configured under `llm_settings.visual_search_terms` in
`config/ai_services.yaml`, and each one is a separate library search. Set
`enabled: false` there to search the topic title and profile keywords instead.
A failure to derive phrases leaves the existing search terms in place rather
than failing the render.

One consequence worth knowing when reading `--step` output: the step order
depends on the profile, so on `slideshow_stock`, `generate_script` runs first
and `gather_visuals` second.

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
8. **burn_pycaps_subtitles** - Burn animated captions onto the assembled
   video, when the resolved subtitle engine is pycaps

Steps 1 and 2 are swapped on a profile that draws no scraped media, so on
`slideshow_stock` the script is written before the footage is chosen. See
"Stock-only profiles run script-first" earlier. `--step` follows the profile's
real order, so on that profile `--step gather_visuals` requires a completed
`generate_script` rather than the other way round.

`--step` requires the chosen step's declared dependencies, not everything
listed above it. `--step create_voiceover` needs the script, and runs whether
or not `generate_description` has: the description feeds it nothing. Use this
to iterate on one part of a render — a voice profile, the music — without
re-running what comes before it.

Re-running one step also forgets every recorded step that reads its output,
and deletes the files those steps would otherwise short-circuit on — the
voiceover, the platform metadata. That is what makes the next
full run redo them against the new input rather than pair fresh narration
with stale captions, but it does mean `--step generate_script` discards the
voiceover you already have.

One caveat: `burn_pycaps_subtitles` replaces the assembled video with the
burned one, so it will not burn a second time over its own output. Re-run
`--step assemble_video` first to iterate on caption styling, which also drops
the recorded burn so the next run redoes it.

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
