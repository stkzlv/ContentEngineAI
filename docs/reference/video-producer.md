# Video producer reference

The command-line interface of `src.video.producer`, the profile keys, the assembly modes, the subtitle formats and the pipeline steps. For the task walkthroughs, see [Producing videos](../guides/producing-videos.md); for why the pipeline is shaped this way, see [The video pipeline](../explanation/video-pipeline.md). Global settings are in [Configuration](configuration.md).

## Synopsis

```bash
poetry run python -m src.video.producer <products_file> <profile> [options]
poetry run python -m src.video.producer <profile> --topic <title> [options]
poetry run python -m src.video.producer <profile> --topics-file <file.yaml> [options]
poetry run python -m src.video.producer --batch (--batch-profile <profile> | --random-profile) [options]
```

The parser is `create_argument_parser` in `src/video/producer/cli.py`. The flags marked *shared* are declared once in `src/video/producer/shared_cli.py` and are accepted with the same names and choices by the global batch (`src.pipeline.global_batch`).

## Arguments

### Positional arguments

| Argument | Description | Example |
|---|---|---|
| `products_file` | Path to the product `data.json`. Not used with `--batch`, `--topic` or `--topics-file`. | `outputs/B0.../data.json` |
| `profile` | Video profile name from `video_profiles`. Required except with `--batch`. | `slideshow_images1` |

In topic mode a lone positional is read as the profile, so `<profile> --topic ...` works without a `products_file`.

### Input and batch

| Argument | Description | Example |
|---|---|---|
| `--batch` | Process every product found in the outputs directory. | `--batch` |
| `--batch-profile` | Profile for every batch product. | `--batch-profile slideshow_images1` |
| `--random-profile` | Pick a profile per product, deterministic by product id. | `--random-profile` |
| `--profile-pool` | Profiles `--random-profile` draws from. Without it, the YAML `batch.profile_pool` applies, then every profile except `base` and `slideshow_stock`. | `--profile-pool prof1 prof2` |
| `--product-ids` | Limit `--batch` to these product ids. | `--product-ids B0ASIN1 B0ASIN2` |
| `--outputs-dir` | The outputs directory: batch mode finds products there, and every render writes its video, state and temp files there, with the producer's logs and performance history beside them (default `global_output_directory` from `config/core.yaml`, resolved against the repository root when relative). | `--outputs-dir custom_outputs` |
| `--fail-fast` | Stop the batch on the first failure. | `--fail-fast` |
| `--strict` | Exit non-zero when any product was lost, to a failure or a skip. By default only a run where nothing succeeded exits non-zero. A run that stopped because memory stayed short (`memory_guard`) exits 75, whatever succeeded before it, with or without `--strict`. | `--strict` |
| `--output-format` | Batch summary format: `text` (default) or `json`. | `--output-format json` |
| `--product-index` | 0-based index of the product in a `data.json` that holds a list. | `--product-index 0` |
| `--topic` | Render a subject instead of a product; replaces `products_file`. | `--topic "Why wifi drops"` |
| `--topic-description` | Source material the script is written from. The script generator reads only the title and this description. | `--topic-description "Router placement."` |
| `--topic-keywords` | Comma-separated stock search terms for the topic. Comma-separated rather than repeated, because a multi-value flag before a positional swallows it. | `--topic-keywords "wifi router, home network"` |
| `--topics-file` | YAML list of topics to render in turn, each with `title`, optional `description` and optional `keywords`. | `--topics-file topics.yaml` |

### Run control

| Argument | Description | Example |
|---|---|---|
| `--debug` | Verbose logging; keeps the intermediate files in `temp/`. | `--debug` |
| `--clean` | Delete the existing output directory before starting. | `--clean` |
| `--step` | Run one pipeline step. Choices are the names in [Pipeline steps](#pipeline-steps). | `--step generate_script` |

### Script and content

| Argument | Shared | Description | Example |
|---|---|---|---|
| `--script-template` | yes | Force a script template (filename without `.md`). | `--script-template curiosity_hook` |
| `--voice-profile` | yes | Force a TTS voice profile. | `--voice-profile calm_confident` |
| `--cta` | yes | Force the closing call to action. It must be one of the configured options; otherwise selection proceeds normally. | `--cta "Follow for more finds like this."` |
| `--pillar` | yes | Content pillar for the run: filters templates, prepends the pillar preamble, picks the pillar audience. | `--pillar value` |

**Pillars** (default): `value` (mass-appeal staples), `novelty` (lesser-known finds), `utility` (problem/solution framing). They are configured in `config/ai_services.yaml::script_templates.pillars`. Without `--pillar`, the product record's own pillar applies when it has one; the scraper attaches the source keyword's group. With neither, all templates are eligible and the global `target_audience` applies.

`--pillar` works with `--topic` too. The preambles and audiences have topic counterparts (`pillar_preambles_topic`, `pillar_audiences_topic`) under the same keys, because the product versions are written about a thing being shown and would put a purchase in a script that recommends nothing. Template narrowing does not apply on a topic, since `pillars` maps to product templates and a topic uses the topic family; the pillar still shapes the preamble and the audience. See [the content requirements](../requirements/content.md#content-pillars) for the full system.

Voice profiles are described in [TTS voice profiles](../explanation/tts-voice-profiles.md).

### Subtitle engine and format

| Argument | Shared | Description | Example |
|---|---|---|---|
| `--subtitle-engine` | yes | `ffmpeg` or `pycaps`. The bundled YAML selects `pycaps`. `ffmpeg` burns SRT or ASS through libass during assembly; `pycaps` burns animated captions after assembly and needs `poetry install --with pycaps`. | `--subtitle-engine ffmpeg` |
| `--subtitle-format` | yes | `srt` or `ass`. The pycaps engine ignores it, so pair it with `--subtitle-engine ffmpeg`. | `--subtitle-format ass` |

### Pycaps options

Read only when the resolved engine is `pycaps`. Install and configuration are in [Pycaps subtitles](../explanation/pycaps-subtitles.md).

| Argument | Shared | Description | Example |
|---|---|---|---|
| `--pycaps-template` | yes | Force one template for every product. Clears the template pool, so the per-product selector falls through to this name. | `--pycaps-template hype` |
| `--pycaps-template-pool` | yes | Pool for deterministic per-product selection. Wins over the clear when passed with `--pycaps-template`. | `--pycaps-template-pool word-focus hype vibrant` |
| `--pycaps-renderer` | yes | `css` (default, Playwright and Chromium, the only production-safe option) or `pictex` (browserless Skia; matches `css` on `word-focus`, renders glows and soft shadows differently). | `--pycaps-renderer pictex` |

### FFmpeg caption style

Read only by the FFmpeg engine.

| Argument | Shared | Description | Example |
|---|---|---|---|
| `--preset` | yes | Style preset: `minimal`, `modern`, `bold`, `animated`, `random`. See [Style presets](#style-presets). | `--preset bold` |
| `--font-size-scale` | yes | Font size multiplier (0.5-2.0). | `--font-size-scale 1.2` |

### Caption position

| Argument | Shared | Description | Example |
|---|---|---|---|
| `--subtitle-anchor` | yes | `top`, `center`, `bottom`, `above_content` or `below_content`. | `--subtitle-anchor bottom` |
| `--subtitle-margin` | yes | Margin from the anchor as a fraction of frame height (0.0-0.5). | `--subtitle-margin 0.05` |
| `--subtitle-alignment` | yes | Horizontal alignment: `left`, `center` or `right`. | `--subtitle-alignment center` |
| `--max-subtitle-width-fraction` | yes | Maximum caption width as a fraction of frame width (0.0-1.0). | `--max-subtitle-width-fraction 0.8` |
| `--content-aware` | yes | Position captions against the media's actual bounds. | `--content-aware` |
| `--no-content-aware` | yes | Turn content-aware positioning off. | `--no-content-aware` |

### Caption text segmentation

| Argument | Shared | Description | Example |
|---|---|---|---|
| `--max-line-length` | yes | Maximum characters per line. | `--max-line-length 25` |
| `--max-words-per-line` | yes | Maximum words per line (0 disables the limit). | `--max-words-per-line 4` |
| `--max-duration` | yes | Maximum caption duration in seconds. | `--max-duration 5.0` |
| `--min-duration` | yes | Minimum caption duration in seconds. | `--min-duration 0.8` |

### Randomization

| Argument | Shared | Description |
|---|---|---|
| `--randomize-fonts` / `--no-randomize-fonts` | yes | Draw the caption font per product from `font_pool`, or don't. |
| `--randomize-colors` / `--no-randomize-colors` | yes | Draw the caption colour per product from `color_pool`, or don't. |
| `--randomize-effects` / `--no-randomize-effects` | yes | Draw the caption effect per product from the preset's effects, or don't. |

### Image layout

| Argument | Shared | Description | Example |
|---|---|---|---|
| `--image-width-percent` | yes | Image width as a fraction of the frame (0.0-1.0). | `--image-width-percent 0.75` |
| `--image-top-position-percent` | yes | Image top edge as a fraction of frame height (0.0-1.0). | `--image-top-position-percent 0.2` |

### Platform and metadata

| Argument | Shared | Description | Example |
|---|---|---|---|
| `--metadata-mode` | yes | `unified` (one title, description and hashtag set for all platforms, the default) or `optimized` (platform-specific SEO). | `--metadata-mode optimized` |

### Argument rules

The parser refuses these combinations with an error:

- `--batch` needs exactly one of `--batch-profile` and `--random-profile`, and takes no positionals, `--topic` or `--topics-file`.
- `--topic` and `--topics-file` exclude each other and `products_file`, and need a profile.
- Without `--batch`: `--batch-profile`, `--fail-fast` and `--random-profile` are refused. Outside topic mode both positionals are required and `--profile-pool` is refused too.

## Profiles

A profile is a named block under `video_profiles` in `config/video_production.yaml`, validated by `VideoProfile` in `src/video/config/visual_models.py`. Each run renders with one profile.

### Bundled profiles

| Profile | Media | Assembly mode | Notes |
|---|---|---|---|
| `base` | scraped images | none | Template other profiles extend; never drawn by `--random-profile`. |
| `slideshow_short_20s` | scraped images | none | 15-30 s slideshow with pre-motion on the first image. |
| `slideshow_stock` | stock images only | none | Topic renders; script-first step order; never drawn by `--random-profile`. |
| `slideshow_images1` | scraped images | none | Image count follows the voiceover length. |
| `slideshow_images2` | scraped images | none | Alternative styling. |
| `slideshow_images3` | scraped images | none | Two-part captions: product URL above, voiceover below. |
| `slideshow_images4` | scraped images | none | Two-part captions with the URL shown only during the call to action. |
| `product_video_sequential` | scraped videos and images | `sequential` | Every product video in order with crossfades. |
| `product_video_single` | scraped videos and images | `single_best` | Longest video, looped to the voiceover length. |
| `product_video_mixed` | scraped videos and images | `mixed_media` | Videos and images interleaved. |
| `product_video_primary` | scraped videos and images | `video_first_fallback` | All videos first, then images. |

### Profile keys

Every key except `description` is optional. A key left unset inherits the global value from `video_settings` or `subtitle_settings`.

| Key | Type | Meaning |
|---|---|---|
| `description` | string | Required. A one-line summary of the profile. |
| `use_scraped_images`, `use_scraped_videos` | bool | Draw the product's own images or videos (default `false`). A profile with both off renders script-first. |
| `use_stock_images`, `use_stock_videos` | bool | Draw stock media (default `false`). |
| `stock_image_count`, `stock_video_count` | int >= 0 | Stock items to fetch (default 0). |
| `use_dynamic_image_count` | bool | Match the image count to the voiceover length (default `false`). |
| `stock_media_keywords` | list of strings | Stock search terms. Unset inherits `media_settings.stock_media_keywords`; an empty list searches on the product title alone. |
| `image_background_fill` | `color` or `blur` | Frame fill around an image. |
| `image_background_blur_sigma` | float 1.0-100.0 | Blur strength for the image backdrop. |
| `image_background_blur_darken`, `video_background_blur_darken` | float 0.1-1.0 | Darkening multiplier for the image or video backdrop. |
| `image_width_percent`, `image_top_position_percent` | float 0.0-1.0 | Image width and top edge as fractions of the frame. |
| `image_vertical_align` | `top` or `center` | Image vertical alignment. |
| `video_assembly_mode` | see [Assembly modes](#assembly-modes) | How scraped videos are combined. |
| `video_aspect_mode` | `letterbox`, `crop-to-fit`, `smart-scale` or `blur-fill` | Fit for a video whose aspect differs from the frame. |
| `video_background_blur_sigma` | float 1.0-100.0 | Blur strength for `blur-fill`. |
| `video_transition_duration` | float | Transition length in seconds. |
| `enable_format_normalization` | bool | Normalize input video formats before assembly. |
| `video_cache_dir` | string | Video cache directory. |
| `video_top_position_percent`, `video_content_height_percent` | float 0.0-1.0 | Video top edge and band height as fractions of the frame. |
| `video_vertical_align` | `top` or `center` | Video vertical alignment. |
| `subtitle_positioning` | mapping | Profile-specific caption positioning overrides. |
| `first_frame_pre_motion` | bool | See [Opening overlays and pre-motion](#opening-overlays-and-pre-motion). |
| `pre_motion_peak_zoom` | float 1.0-1.5 | See [Opening overlays and pre-motion](#opening-overlays-and-pre-motion). |
| `still_motion` | mapping | Replaces `video_settings.still_motion` as a whole block. See [Opening overlays and pre-motion](#opening-overlays-and-pre-motion). |
| `ending`, `peak_margin_sec` | `outro`, `peak` or `loop`; float 0.05-1.0 | See [Ending](#ending). |
| `upper_line` | mapping | Partial override of `video_settings.upper_line`, deep-merged. |
| `subtitle_settings` | mapping | Partial override of the global `subtitle_settings`, deep-merged, including the nested `pycaps`, `two_part_subtitles` and `safe_zone` blocks. |

Unknown keys are rejected at config load, inside the nested `pycaps`, `safe_zone` and `two_part_subtitles` blocks too. The flat `subtitle_*` keys are refused, with the nested `subtitle_settings` field to move each one to named in the error. The per-profile overrides are shown at length in [Configuration](configuration.md#12-video-profiles-with-per-profile-settings).

`subtitle_format` is settable per profile in the nested spelling only (`subtitle_settings.subtitle_format`), and per run with `--subtitle-format` on both the producer and the global batch. `--subtitle-format` wins over the profile and the global value. The subtitle file's extension follows the merged value.

### Example

```yaml
video_profiles:
  slideshow_images1:
    description: "Dynamically uses scraped product images to match voiceover duration."
    use_scraped_images: true
    use_scraped_videos: false   # no video_assembly_mode: images only
    subtitle_settings:
      style_preset: "bold"
      font_size_scale: 1.0

  slideshow_short_20s:
    description: "Short 15-30s slideshow tuned for hook iteration"
    use_scraped_images: true
    image_top_position_percent: 0.15
    first_frame_pre_motion: true   # Ken Burns settle-zoom on segment 0
    pre_motion_peak_zoom: 1.10

  product_video_sequential:
    description: "Sequential video clips"
    use_scraped_videos: true
    video_assembly_mode: "sequential"
```

### Precedence

Highest first, as the code resolves it:

1. CLI arguments, when passed.
2. Machine environment: `CONTENT_ENGINE_OUTPUT`, `OUTPUTS_DIR` and `FFMPEG_THREADS`, besides the secrets.
3. Profile settings.
4. Global values from the YAML files.

[Decision 0003](../decisions/0003-config-precedence.md) sets the order. No machine setting is one a profile can set, so the environment stays above the profile although it is applied when the YAML loads.

## Opening overlays and pre-motion

Five visual-layer keys live on `video_settings`. `first_frame_pre_motion`, `pre_motion_peak_zoom`, `still_motion` and `upper_line` are also settable per profile. `VideoProfile` declares neither `hook_overlay` nor `cold_open_variant_pool`, so writing either under a profile aborts the config load. Canonical defaults and inline notes are in `config/video_production.yaml::video_settings`.

| Key | Effect |
|---|---|
| `first_frame_pre_motion`, `pre_motion_peak_zoom` | When on, the first image segment starts at `pre_motion_peak_zoom` (default 1.10) and settles to 1.0 over the segment, so frame 0 is mid-motion rather than static. No effect when the first segment is a video clip. Off on the 30-45 s profiles, on for `slideshow_short_20s`. |
| `still_motion` | Off by default. When `enabled`, every still image moves inside its own image box, so the image band and the caption zone don't change: a move is drawn from `moves` (`push_in`, `pull_out`, `pan_left`, `pan_right`, `pan_up`) per product and per image, the same on every run, and two consecutive stills never share one unless the pool holds a single move (a repeated entry counts once). Zooms run between `min_zoom` (default 1.0) and `max_zoom` (default 1.15); pans travel across the `max_zoom` margin. The first image keeps its settle-zoom where `first_frame_pre_motion` is on. |
| `hook_overlay` | Burns a short headline as centre-upper static text on the first `duration_sec` seconds (default 1.5), with no per-word reveal. The headline is stored in `pipeline_state.json::hook_headline`. Every field, the wrapping and the fallback are in [Configuration](configuration.md#31-overlay-settings). |
| `upper_line` | A static line held above the visual for the whole clip: the affiliate link, the public link-in-bio page, or fixed text. Off by default; see [Configuration](configuration.md#31-overlay-settings). |
| `cold_open_variant_pool` | Named cold-open variants (`mid_zoom_title_card`, `static_title_card`, `pre_motion_only`), one picked per product by salted MD5. The choice is stored in `pipeline_state.json::assemble_video.cold_open_variant`. An empty list turns rotation off. |

## Beat snapping

`video_settings.beat_snap` (off by default) moves each visual cut, the middle of a crossfade, to the nearest beat of the render's music within `window_ms` (default 150). The neighbouring segments trade the difference, so the video's length, the voiceover and the captions don't move. A move is skipped when it would shorten a segment below `min_visual_segment_duration_sec`, reach the end of the video, or lengthen a video clip. Beats come from `librosa.beat.beat_track`, once per track, cached beside it as `<track>.beats.json` keyed to the file's size and modification time. librosa is in the optional `beats` dependency group (`poetry install --with beats`); without it the setting logs a warning and the cuts stay where they were.

## Image curation

`video_settings.image_curation` (off by default) prefers clean product images over text-heavy seller infographics.

| Key | Default | Effect |
|---|---|---|
| `enabled` | `false` | Judge and trim the scraped images after media validation. |
| `max_text_share` | `0.15` | An image whose seller-added text and graphics cover more than this share of the frame is text-heavy. |
| `min_clean_images` | `3` | Text-heavy images are dropped while at least this many images remain, and never below what media validation asks for: the image minimum (`min_images_if_no_video`, or `min_images_with_video` when the render has clips), and the share of `min_total_media` the clips and stock media don't cover; otherwise the least text-heavy are kept. |
| `model` | `gemini-2.5-flash` | The multimodal model that judges each image. On a ten-image listing it scored plain product shots 0.0 and marketing images 0.2-0.3; `gemini-2.5-flash-lite` counted a watch's own screen as text and could not tell them apart. |

Each image is judged once and the score is cached beside it as `<image>.text_score.json`, keyed to the file's size and modification time. A failed judgement is unknown and never removes an image. The scores and the kept and dropped images are recorded in `pipeline_state.json` under `image_curation`. The judge uses the `llm_settings.stock_relevance` concurrency and timeout and the LLM API key; without the key, curation is skipped with a warning.

## Ending

`video_settings.ending` decides what follows the last spoken word; a profile can set it and `peak_margin_sec`.

| Value | Effect |
|---|---|
| `outro` (default) | The video runs `outro_duration_sec` (`config/core.yaml`, 1.0 s) past the end of the voiceover file, and the music fades over `music_fade_out_duration`. |
| `peak` | The video ends `peak_margin_sec` (default 0.25 s) after the last spoken word, found by `silencedetect` as the start of the silence that runs to the end of the voiceover. The music fades within the margin. If the measurement fails, the whole voiceover file counts as speech. |
| `loop` | `peak`, plus a closing segment of the first image that replays its opening motion backwards (the settle-zoom where `first_frame_pre_motion` is on, else the reverse of its still motion), so the last frame matches the first apart from the captions and the hook overlay. Stills are dropped from the end of the timeline when needed to keep every segment at `min_visual_segment_duration_sec`. A render with video clips ends as `peak`, with a warning. |

With `peak` or `loop`, a sting at position `end` finishes at the measured end of the speech rather than of the voiceover file. Each render records its ending in `render_choices.jsonl`.

## Assembly modes

`video_assembly_mode` controls how scraped videos are combined into the output.

| Mode | Behaviour | Bundled profile |
|---|---|---|
| `sequential` | Plays every video in order. | `product_video_sequential` |
| `single_best` | Uses the longest video, looped to fill. | `product_video_single` |
| `mixed_media` | Interleaves videos with images. | `product_video_mixed` |
| `video_first_fallback` | Uses the videos first, then images. | `product_video_primary` |

With `use_scraped_videos: false` the mode is ignored and only images are used. Single-video handling per mode is in [Configuration](configuration.md#12-video-profiles-with-per-profile-settings).

## Aspect modes

`video_aspect_mode` fits a video whose aspect differs from the 9:16 frame. The fit is built in `src/video/assembler/visual_builder.py`.

| Mode | Result | Caption geometry |
|---|---|---|
| `letterbox` | The video is centred with black bars around it. | Reports the content band, so content-aware captions sit below it. |
| `crop-to-fit` | The video fills the frame and the edges are cropped. A 16:9 clip keeps the centre 31% of its width. | Reports none, so captions fall back to a full-frame band and sit over the content. |
| `blur-fill` | Placed as in letterbox, with a scaled, blurred and darkened copy of the same frame behind it instead of black. | Same as letterbox. |
| `smart-scale` | `crop-to-fit` when the aspect difference is within `smart_scale_tolerance` (default 0.10), `blur-fill` otherwise. | Follows the branch taken. |

A landscape source always takes the `blur-fill` branch of `smart-scale`: the aspect difference for 16:9 into 9:16 is 2.16, far above the tolerance, so the crop branch only separates near-vertical sources. A profile names `letterbox` to get black bars.

The backdrop is darkened by `video_background_blur_darken`, and the image backdrop by `image_background_blur_darken` (both default 0.6; 1.0 turns darkening off). The multiplier applies to the blurred copy only; the content band is composited on top afterwards. The filter is `colorlevels`, which scales rather than subtracts, so a dark backdrop keeps its detail where `eq=brightness` would flatten it to black. White caption fill measured 2.5:1 against a bright 165/255 backdrop, which is why the backdrop is darkened even though the caption stroke keeps the text legible.

## Subtitle formats

Read by the FFmpeg engine only.

| Format | Description |
|---|---|
| `srt` | SubRip: plain text with timing, readable by every player. |
| `ass` | Advanced SubStation Alpha: fonts, colours, outlines and shadows; karaoke, fade and typewriter effects; pixel positioning and animation. |

## Style presets

Defined under `style_presets` in `config/subtitles.yaml`. `modern` is the default.

| Preset | Description | Effects |
|---|---|---|
| `minimal` | Clean, no animation. | none |
| `modern` | Bold sans-serif. | karaoke |
| `bold` | Strong outline. | fade |
| `animated` | Karaoke for playful tones. | karaoke |
| `random` | Font, colour and effect drawn per product. | one of karaoke, fade, typewriter |

A preset other than `random` carries exactly one effect, and `minimal` none. Why these defaults: [Captions](../explanation/captions.md).

## ASS effects

The tags `src/video/unified_subtitle_generator.py` writes for each effect.

| Effect | Tag | Description |
|---|---|---|
| Karaoke | `{\kf50}Hello {\kf40}world` | Word-by-word fill in time with speech. `\kf` fills smoothly; `\k` marks timing only. |
| Fade | `{\fad(200,200)}Subtitle text` | Fade in and out, in milliseconds. The last caption has no fade-out. |
| Typewriter | alpha transitions | Character-by-character reveal. |
| Scale pulse | `\t(\fscx,\fscy)` | Text grows and shrinks. No bundled preset uses it. |
| Glow | `\t(\3c&H...)` | Outline colour pulses. No bundled preset uses it. |

## Pipeline steps

The default order. `--step` takes these names.

1. `gather_visuals`: collect images and videos from the scraped data and stock.
2. `generate_script`: write the voiceover script with the LLM.
3. `generate_description`: write the platform metadata.
4. `create_voiceover`: synthesize speech.
5. `generate_subtitles`: build synchronized captions.
6. `download_music`: fetch background music (Jamendo, then Freesound, then `background_music_paths`).
7. `assemble_video`: combine everything into the final video.
8. `burn_pycaps_subtitles`: burn animated captions onto the assembled video when the resolved engine is `pycaps`.

On a profile that draws no scraped media, such as `slideshow_stock`, steps 1 and 2 swap: `generate_script` runs first. `--step` follows the profile's real order. Each step's prerequisites are declared in `step_dependencies` in `src/video/producer/orchestration.py`; [The video pipeline](../explanation/video-pipeline.md#how-the-steps-fit) explains them.

Files a render leaves for inspection:

| File | Contents |
|---|---|
| `pipeline_state.json` | Completed steps, `hook_headline`, `assemble_video.cold_open_variant`. |
| `temp/gathered_visuals.json` | Each stock item's search phrase and `relevance_score`. |
| `temp/script_fact_check.json` | The script fact-check outcome. |
| `temp/step_list.json` | A topic's sourced step list, with refused steps, when step lists are on. |

A successful run without `--debug` deletes `temp/`.

## Render choices

Each finished render, not a `--step` run, appends one row to `state/render_choices.jsonl` under the outputs root: product id, profile, `script_template`, pillar, CTA, `hook_headline`, `voice_profile` and `voice_name`, whether the voice chain was on, the stock ids it gathered, the caption engine and pycaps template, the music track, `cold_open_variant`, the assembly mode, pre-motion, the transition duration, the ending, the still-motion move on each still, how many cuts beat snapping moved, each sound effect as `kind:file` (each left empty when its feature is off), where the search phrase appears (the product keyword or the topic title, checked in the first spoken sentence, the hook headline and the first 60 characters of each platform's caption text, before the publisher adds its disclosure), and the script. The report counts each move and effect file on its own, and ends with the share of renders carrying the search phrase in each of those places. A failed write is logged and the render still succeeds. The file is durable state, so cleanup leaves it. When `generate_script` writes a new script, it logs a warning for each of the last 14 recorded scripts it closely repeats; it never blocks the script.

```bash
python -m src.video.render_choices [--last N] [--dominance SHARE] [--similarity RATIO] [--outputs-dir PATH]
```

| Option | Default | Effect |
|---|---|---|
| `--last N` | 14 | How many of the most recent renders to read. |
| `--dominance SHARE` | 0.6 | Alert when one value of a dimension holds more than this share. Needs at least 5 products; a dimension with a single value in the window is a fixed setting and doesn't alert. |
| `--similarity RATIO` | 0.5 | Alert for each pair of scripts whose character 5-grams overlap at least this much (Jaccard similarity, case and spacing ignored); the generation-time warning uses the same measure and threshold. |
| `--outputs-dir PATH` | the outputs root (`OUTPUTS_DIR`, else `CONTENT_ENGINE_OUTPUT`, else the repository's `outputs/`) | Where `state/render_choices.jsonl` is read from. |

The report keeps each product's newest row, since a rerun of a finished product appends another, and prints each dimension's distribution and the alerts. The hook headline and script are recorded but not counted: each is written for its product. It measures only; nothing in it changes what a render picks.

