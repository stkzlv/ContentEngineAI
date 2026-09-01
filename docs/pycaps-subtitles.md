# Pycaps Subtitle Engine

ContentEngineAI ships two subtitle rendering engines. The `ffmpeg` engine
generates SRT or ASS files and burns them via `libass`/`drawtext`. The
`pycaps` engine runs the [pycaps](https://github.com/francozanardi/pycaps)
library as a post-assembly step to produce animated, CSS-styled captions in
the TikTok/Reels style — word-by-word highlights, pop/slide/zoom animations,
emoji overlays, gradient fills.

The bundled `config/subtitles.yaml` selects `pycaps` by default. Forks
without the optional pycaps group fall back to FFmpeg, with a warning naming
the install command, because the bundled `pycaps.fallback_policy` is
`fallback_ffmpeg`. Set
`subtitle_engine: "ffmpeg"` in YAML or pass `--subtitle-engine ffmpeg` to
opt out per-run.

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

On Ubuntu 26.04 the chromium install fails with `does not support
chromium on ubuntu26.04-x64` (Playwright has no 26.04 build yet, through
1.60). Prefix the install to force the binary-compatible 24.04 build:

```bash
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 poetry run playwright install chromium
```

At render time the producer sets that override itself, so runs don't need
the prefix. See `docs/troubleshooting.md` for the full writeup, including the
`xvfb-run` wrapper the CSS renderer needs on Wayland desktops.

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
| Profile override (`config/video_production.yaml`) | `video_profiles.<name>.subtitle_settings.subtitle_engine` | `subtitle_engine: pycaps` |
| Producer CLI | `--subtitle-engine` | `--subtitle-engine pycaps` |
| Global batch CLI | `--subtitle-engine` | `--subtitle-engine pycaps` |

`--subtitle-format` is available at the same two CLI levels and the same two config levels; the pycaps engine ignores it.

CLI beats profile. Profile beats YAML. Bundled YAML default is `pycaps`;
the `SubtitleSettings` Pydantic field default (used when constructing
without YAML) stays `ffmpeg`.

### Pycaps sub-settings

All fields live under `subtitle_settings.pycaps` (YAML + merged runtime).
Per-profile overrides go under `subtitle_settings.pycaps` and use the field
names in the table below (`template_name`, `renderer`, ...). The flat
`pycaps_*` prefix is refused at config load.

The "Default" column lists the `PycapsSettings` Pydantic field defaults
(used for programmatic construction without YAML). The bundled
`config/subtitles.yaml` overrides several of them — see the inline
comments in that file for the active values shipped to users.

| Field | Type | Default | Description |
|---|---|---|---|
| `template_name` | str | `explosive` | Fixed template name. Used when `template_pool` is empty or single-entry, which is also what `--pycaps-template NAME` triggers (the flag clears the pool). |
| `template_pool` | list[str] | `[word-focus, hype, minimalist, vibrant]` | Pool for deterministic per-product selection (md5 hash of product_id). Bundled YAML ships a 2-entry recipe-fit override. |
| `renderer` | `css` \| `pictex` | `css` | `css` = Playwright + Chromium, the only production-safe option. `pictex` = browserless Skia path, **preview only**: it drops the gaps between words (issue #174). |
| `max_width_ratio` | float | 0.85 | Max caption width as a fraction of frame width. |
| `max_number_of_lines` | int | 2 | Max lines per caption segment. |
| `vertical_align` | `top` \| `center` \| `bottom` | `bottom` | Base anchor. Runtime offset is derived from VisualBounds. |
| `vertical_align_offset` | float \| null | null | Manual override for the derived offset. Range: -1.0 to 1.0. |
| `fallback_policy` | `raise` \| `fallback_ffmpeg` \| `warn_and_skip` | `raise` | `raise` = abort if pycaps unavailable or the burn fails. `fallback_ffmpeg` = switch to FFmpeg when pycaps is *unavailable*, and burn captions with FFmpeg when a pycaps render fails; a missing transcript or assembled video still aborts. `warn_and_skip` = no subtitles (not recommended). |
| `enable_ai_tagging` | bool | `false` | Opt into AI word tagging via Gemini. See [AI word tagging](#ai-word-tagging). |
| `llm_model` | str | `gemini-2.5-flash` | Gemini model used when `enable_ai_tagging` is true. |
| `ai_tagging_on_error` | `skip` \| `raise` | `skip` | Per-call AI failure handling. `skip` swallows the error and drops the tag for that segment; `raise` propagates to `fallback_policy`. |

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
- Needs a real X display for the per-word screenshots. On Wayland desktops
  the screenshots hang (`Page.screenshot` timeout); wrap the run in
  `xvfb-run -a`. `pictex` avoids this, but see the warning under `pictex`
  before reaching for it: it is not a usable substitute for published work.

### `pictex` (preview only, not production-safe)

> **Do not use for published output.** pictex renders multi-word captions
> with no gaps between words, so `like my phone went from` comes out as
> `Likemyphonewentfrom`. Reproduced on the bundled `word-focus` and
> `explosive` templates. The output is unreadable but renders without any
> error, so nothing warns you. Use `css` for anything you intend to publish.

Browserless Skia path via the `pictex` package (same renderer engine as
Chrome, just without the browser shell).

- Install: `html2pic` + `skia-python` wheels (already in the pycaps group).
- Peak RSS: similar to css, with a slower startup curve on small clips.
- Speed: roughly on par with css in steady state.
- Needs no X display, which is its one genuine advantage over `css`.

**Why the spacing breaks.** Both bundled templates space words with CSS
padding on the word element (`word-focus` uses `padding: 4px 4px`,
`explosive` uses `padding: 5px 8px`); neither sets a word-spacing property.
The two renderers then disagree on what a word's measured width includes.
The css renderer measures each letter plus a `NON_CONTENT_WIDTH` sentinel
specifically so the padding is counted. The pictex renderer instead renders
the word and crops with `CropMode.CONTENT_BOX`, which by definition excludes
padding, so it reports a glyph-only width and the layout butts each word
against the next. At 1080x1920 the render scale is 3.0, so `word-focus`
loses roughly 24px of gap between adjacent words: a total collapse rather
than tight kerning.

This is an upstream defect in pycaps, not in this project's wiring, which
only instantiates the renderer class. Tracked in issue #174.

Switch with `--pycaps-renderer pictex` if you want to preview the path, and
check a frame before trusting the output.

## Limitations (v1)

- **Single-line captions only.** Two-part (upper URL + lower voiceover) is
  FFmpeg-only. When a profile has `subtitle_settings.two_part_subtitles.enabled:
  true` and you flip the engine to pycaps, the producer logs a warning and
  disables the two-part system for that run. The upper URL is not rendered.
- **Font randomization doesn't apply.** The `subtitle_settings.randomize_fonts`
  setting only affects the FFmpeg path. Pycaps templates ship their own `@font-face`
  declarations via their `resources/` directory.
- **Upstream is alpha.** Pycaps 0.2.1 is the first public release. Breaking
  changes are possible. We pin to an exact git commit in `pyproject.toml`
  and upgrade deliberately.
## AI word tagging

Pycaps templates can carry `tagger_rules` of `type: ai`. When applied, the
named CSS class lands on the LLM-selected words so the template's CSS picks
them out (e.g. background highlight, scale-pop animation). ContentEngineAI
wires the existing Gemini key into pycaps' `LlmProvider` so these rules
work without an OpenAI key.

### Enable

```yaml
# config/subtitles.yaml
subtitle_settings:
  subtitle_engine: pycaps
  pycaps:
    enable_ai_tagging: true
    template_name: neo-minimal      # built-in, ships an `ai` rule out of the box
    llm_model: gemini-2.5-flash     # default
    ai_tagging_on_error: skip       # default — degrade silently per call
```

The `enable_ai_tagging` flag defaults to `false`. Installing pycaps and
turning AI tagging on are deliberately separate flips so a default install
behaves predictably.

### Built-in templates with AI rules

Two presets ship with `tagger_rules: [{type: ai, ...}]` already wired:

| Template | What gets tagged | Visual effect |
|---|---|---|
| `neo-minimal` | "Most relevant and impactful phrases (around 4-5 words)" | Background highlight on the picked phrase |
| `explosive` | "The most important phrase or word in all the script" | Scale-pop animation on the picked words |

Switching `template_name` to either of those plus `enable_ai_tagging: true`
is the demo path. Other built-in templates (`word-focus`, `hype`, etc.) have
no AI rules — they ignore the adapter even when it's wired.

### Cost and latency

Gemini Flash is cheap (about $0.30 per 1M input tokens at the time of
writing) and fast. Pycaps' tagger calls the LLM once per caption segment,
so a 30-second video at typical pacing makes 5-10 calls and adds 1-3
seconds to wall time. The render summary log line includes an `ai_calls=N`
counter:

```
Replaced sample.mp4 with pycaps-burned video
  (template=neo-minimal, renderer=css, wall=22.4s, peak=412 MB, ai_calls=7)
```

### Failure modes

- **Key missing.** If `enable_ai_tagging` is true but the Gemini key
  (`GEMINI_API_KEY` in `.env` by default — controlled by
  `llm_settings.api_key_env_var`) is missing, the step logs a warning and
  proceeds without AI tagging. Pycaps' AI rules silently no-op for that
  run.
- **Per-call API error.** When `ai_tagging_on_error: skip` (default), a
  Gemini error logs a warning and returns an empty response so pycaps
  drops the tag for that segment. The render still completes. Set to
  `raise` to surface the error and let `fallback_policy` decide.
- **Template has no AI rule.** Most built-in templates (`word-focus`,
  `hype`, `vibrant`, etc.) carry no `type: ai` rules and ignore the
  adapter regardless of `enable_ai_tagging`. The flag is harmless but
  inert in those cases.

### CLI override

The dotted keys also work from `cli_overrides`:

```
subtitle_settings.pycaps.enable_ai_tagging
subtitle_settings.pycaps.llm_model
subtitle_settings.pycaps.ai_tagging_on_error
```

No dedicated CLI flags in v1. Toggle in YAML or pass an override dict
programmatically.

## Failure handling

Three policies, configured via `subtitle_settings.pycaps.fallback_policy`:

- `raise` (default): abort the pipeline if pycaps is unavailable or the burn
  fails. Prevents silently producing videos without subtitles.
- `fallback_ffmpeg`: fall back to FFmpeg captions rather than aborting, in
  two distinct situations.

  If pycaps isn't installed, the run switches to the FFmpeg subtitle engine
  outright. That check runs early in `step_generate_subtitles`, before
  committing to the pycaps-only code path, so the assembler burns the
  captions itself. This is what forks without `--with pycaps` hit.

  If pycaps is installed but its *render* fails — the CSS renderer with no
  display being the common case — the captions are burned onto the assembled
  video with a separate FFmpeg pass, built from the Whisper transcript the
  burn step already required. Nothing is re-transcribed.

  The other two burn failures still abort, because neither can degrade: a
  missing transcript leaves nothing to build captions from, and a missing
  assembled video leaves nothing to burn them onto. A fallback that itself
  fails also aborts rather than ship a caption-less video.
- `warn_and_skip`: log a warning and keep the video without subtitles. Not
  recommended for production since it silently produces subtitleless output.

Common failure modes:

- **`PycapsUnavailableError`** — the optional group is not installed.
  Install with `poetry install --with pycaps`.
- **Missing Chromium** — you're using `renderer: css` but never ran
  `playwright install chromium`. Install Chromium. Do not switch to
  `pictex` as a workaround for published output: it renders words with no
  spacing between them (issue #174).
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
- Pycaps follow-up work tracked as GitHub Issues with the `pycaps` label: AI word tagging, two-part hybrid, CSS-renderer CI test, and more
