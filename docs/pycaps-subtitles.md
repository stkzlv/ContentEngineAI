# Pycaps Subtitle Engine

ContentEngineAI ships two subtitle rendering engines. The default `ffmpeg`
engine generates SRT or ASS files and burns them via `libass`/`drawtext`.
The optional `pycaps` engine runs the [pycaps](https://github.com/francozanardi/pycaps)
library as a post-assembly step to produce animated, CSS-styled captions in
the TikTok/Reels style — word-by-word highlights, pop/slide/zoom animations,
emoji overlays, gradient fills.

The pycaps engine is opt-in. The default stays `ffmpeg`.

## When to use pycaps

Pick the pycaps engine when you want captions that look like what Submagic,
Opus Clip, or Captions.ai produce. Stick with ffmpeg when you need:

- The two-part layout (static product URL on top + voiceover captions on
  bottom) — pycaps is single-line only in v1.
- Minimal install footprint — pycaps adds ~1 GB of dependencies (Whisper
  already installed, plus Playwright + Chromium for the CSS renderer).
- Deterministic ASS output you can post-process with other tools.

## Install

The pycaps group is optional. The main `poetry install` does not pull it in.

```bash
# One-time setup
poetry install --with pycaps

# Only needed for the CSS renderer (default). Downloads ~110 MB.
poetry run playwright install chromium
```

The pycaps library is pinned to a validated git commit in `pyproject.toml`.
Upstream is alpha (0.2.1), so we lock to a specific SHA and bump it deliberately.

## Quick start

The fastest way to try pycaps is to override the engine on any existing
profile via CLI:

```bash
# Single product
poetry run python -m src.video.producer outputs/B0ABCD1234/data.json \
    slideshow_images1 --subtitle-engine pycaps --debug

# Batch with a specific template pool
make produce-lowpri ARGS="--batch --batch-profile slideshow_images1 \
    --subtitle-engine pycaps --pycaps-template-pool word-focus hype vibrant \
    --debug" MEM_LIMIT=10G

# Global batch pipeline (scrape + produce + publish)
make batch-lowpri ARGS="--product-ids B0ABCD1234 --profile slideshow_images1 \
    --subtitle-engine pycaps --pycaps-renderer css --debug"
```

No YAML or profile edits needed — the CLI overrides flow through the same
3-level merge as every other subtitle setting.

## Configuration reference

### Engine selector

| Level | Key | Example |
|---|---|---|
| YAML (`config/subtitles.yaml`) | `subtitle_settings.subtitle_engine` | `subtitle_engine: pycaps` |
| Profile override (`config/video_production.yaml`) | `profiles.<name>.subtitle_engine` | `subtitle_engine: pycaps` |
| Producer CLI | `--subtitle-engine` | `--subtitle-engine pycaps` |
| Global batch CLI | `--subtitle-engine` | `--subtitle-engine pycaps` |

CLI beats profile. Profile beats YAML. Default stays `ffmpeg`.

### Pycaps sub-settings

All fields live under `subtitle_settings.pycaps` (YAML + merged runtime).
Per-profile overrides use a flat prefix (`pycaps_template`, `pycaps_renderer`,
etc.) to mirror the other profile override naming.

| Field | Type | Default | Description |
|---|---|---|---|
| `template_name` | str | `word-focus` | Fixed template name. Used when `template_pool` is empty or single-entry. |
| `template_pool` | list[str] | `[word-focus, hype, minimalist, vibrant]` | Pool for deterministic per-product selection (md5 hash of product_id). |
| `renderer` | `css` \| `pictex` | `css` | `css` = Playwright + Chromium. `pictex` = browserless Skia path. |
| `max_width_ratio` | float | 0.85 | Max caption width as a fraction of frame width. |
| `max_number_of_lines` | int | 2 | Max lines per caption segment. |
| `vertical_align` | `top` \| `center` \| `bottom` | `bottom` | Base anchor. Runtime offset is derived from VisualBounds. |
| `vertical_align_offset` | float \| null | null | Manual override for the derived offset. Range: -1.0 to 1.0. |
| `fallback_policy` | `warn_and_skip` \| `raise` | `warn_and_skip` | On render failure: keep the FFmpeg video (warn) or abort the pipeline (raise). |

### CLI override dotted keys

If you pass `--subtitle-engine pycaps` or `--pycaps-template hype` at the
command line, the producer/global-batch builds the following dotted override
keys internally:

```
subtitle_settings.subtitle_engine
subtitle_settings.pycaps.template_name
subtitle_settings.pycaps.template_pool
subtitle_settings.pycaps.renderer
```

These are the same keys `VideoConfig.get_profile_merged_settings()` consumes,
so they also work if you construct overrides programmatically.

## Built-in templates

Pycaps ships with these preset templates. The default pool is a curated
subset chosen for e-commerce product videos.

| Template | Look |
|---|---|
| `word-focus` | Word-by-word reveal, white text, soft shadow. Default pick. |
| `hype` | High-energy pop animations, bold colors. |
| `minimalist` | Clean sans-serif, no animation fluff. |
| `vibrant` | Gradient fills, color cycling. |
| `explosive` | Heavy pop + scale, loud look. |
| `line-focus` | Line-level reveal instead of word-level. |
| `classic` | Static caption, no animation. |
| `default` | pycaps default — similar to `word-focus`. |
| `fast` | Quick fade, small text. |
| `neo-minimal` | Modern minimalist variant. |
| `retro-gaming` | Pixel font, arcade vibes. |

Preview any template against a clip via the pycaps CLI if you have the group
installed:

```bash
poetry run pycaps render --input my.mp4 --template hype --transcript my.json
```

Drop custom templates under `pycaps-templates/` in your project dir and
reference them by path.

## Content-aware positioning

Pycaps captions land in the whitespace below the product image because the
burn step computes the vertical offset from the same `VisualBounds` used by
the existing FFmpeg path:

```
offset = (visual_bottom + margin) - 0.95, clamped to [-0.9, 0]
```

Where `visual_bottom = bounds.y + bounds.height` and the 0.95 base matches
pycaps' internal `LayoutUtils.get_vertical_alignment_position` formula for
bottom-anchored blocks. Default profiles position the product image at y=0.10
with height=0.75, so the computed offset is roughly `-0.08` — captions sit
near the bottom of the frame with a small upward nudge to stay clear of the
image.

You can override the derived offset with `pycaps.vertical_align_offset`
(or use a CLI flag if/when we add one), but the default behaviour handles
all existing profiles automatically.

## Renderer tradeoffs

### `css` (default)

Playwright + Chromium. Full CSS fidelity — gradients, drop shadows, blur,
per-word keyframe animations, `@font-face` loading.

- Install: pip package + `playwright install chromium` (~110 MB)
- Peak RSS: ~400-500 MB per render (process ceiling, varies with template)
- Speed: ~0.7x realtime on a 30s portrait clip (benchmark winner)
- Template coverage: all 10+ built-in templates work.

### `pictex`

Browserless Skia path via the `pictex` package (same renderer engine as
Chrome, just without the browser shell).

- Install: `html2pic` + `skia-python` wheels (already in the pycaps group).
- Peak RSS: similar to css, with a slower startup curve on small clips.
- Speed: roughly on par with css in steady state.
- Template coverage: supports most templates but some CSS features fall
  back. Use this path if you need to avoid Chromium (e.g. CI, restricted
  sandboxes, headless servers without display libs).

Switch with `--pycaps-renderer pictex`.

## Limitations (v1)

- **Single-line captions only.** Two-part (upper URL + lower voiceover) is
  FFmpeg-only. When a profile has `two_part_subtitles_enabled: true` and you
  flip the engine to pycaps, the producer logs a warning and disables the
  two-part system for that run. The upper URL is not rendered.
- **Font randomization doesn't apply.** The `subtitle_randomize_fonts` setting
  only affects the FFmpeg path. Pycaps templates ship their own `@font-face`
  declarations via their `resources/` directory.
- **Upstream is alpha.** Pycaps 0.2.1 is the first public release. Breaking
  changes are possible. We pin to an exact git commit in `pyproject.toml`
  and upgrade deliberately.
- **No AI word tagging yet.** Pycaps supports LLM-driven word tagging
  (e.g. highlight action verbs via GPT). Not wired to ContentEngineAI's
  LLM config in v1. Follow-up work.

## Failure handling

Two policies, configured via `subtitle_settings.pycaps.fallback_policy`:

- `warn_and_skip` (default): pycaps errors are caught, logged as warnings,
  and the FFmpeg-assembled video is kept as the final output. The pipeline
  continues normally. Good for production batch runs where you want the
  pipeline to keep producing videos even if one template breaks.
- `raise`: any pycaps error aborts the pipeline with `PipelineError`. Good
  for development and CI when you want test runs to surface failures.

Common failure modes:

- **`PycapsUnavailableError`** — the optional group is not installed.
  Install with `poetry install --with pycaps`.
- **Missing Chromium** — you're using `renderer: css` but never ran
  `playwright install chromium`. Switch to `pictex` or install Chromium.
- **Template not found** — a value in `template_pool` doesn't match any
  built-in or project-local template name.

## Smoke test

The included integration test exercises the full render path against a
30-second fixture and verifies the output is a valid h264+aac mp4. Run it
after installing the group:

```bash
poetry install --with pycaps
poetry run pytest tests/video/test_pycaps_integration.py -v
```

It's marked `integration` and skipped automatically when pycaps isn't
installed, so it stays out of the default CI lane.

## Benchmarks (reference)

Measured during the initial reality check on a blue-background 30-second
1080x1920 portrait clip with a 40-word fixture transcript:

| Template | Wall time (30s video) | Realtime ratio | Peak RSS |
|---|---|---|---|
| word-focus | 20.96s | 0.70x | 411 MB |
| hype | 23.55s | 0.79x | 426 MB |
| minimalist | 20.04s | 0.67x | 427 MB |

All three rendered through the CSS/Chromium path. Your mileage depends on
transcript density and CPU. A 60-second clip should render in roughly 40-50
seconds on this hardware.

## Related documentation

- [Video Producer CLI reference](./video-producer.md)
- [Configuration system overview](./configuration.md)
- [Architecture: subtitle pipeline](./architecture.md)
- [Development workflow](./development.md)
- [Pycaps follow-up work](./pycaps-followups.md) — AI word tagging, two-part hybrid, CI integration test
