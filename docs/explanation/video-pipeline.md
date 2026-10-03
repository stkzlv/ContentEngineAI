# The video pipeline

This page explains how the producer's steps fit together, why their order depends on the profile, what a partial run keeps and drops, and why profiles and assembly modes exist. The flags and keys themselves are in [the video producer reference](../reference/video-producer.md); the requirements are in [the video requirements](../requirements/video.md); the defects behind the current design are in [the video module notes](../notes/video.md). Captions and voices have their own pages: [Captions](captions.md), [Pycaps subtitles](pycaps-subtitles.md) and [TTS voice profiles](tts-voice-profiles.md).

## How the steps fit

A render is eight steps declared once, as a dependency graph, in `step_dependencies` in `src/video/producer/orchestration.py`. The parallel executor and the `--step` prerequisite check both read that graph, so a partial run is never refused for a step the graph does not need. The voiceover drives everything after it: the render's duration is the voiceover's (`REQ-VID-001`), and captions are timed against it, so subtitles and music both wait on `create_voiceover` and then run beside each other. [Architecture](../architecture.md) shows the graph.

## Why stock-only profiles write the script first

`slideshow_stock` sets `use_scraped_images: false` and `use_scraped_videos: false`, so nothing on screen comes from a scraped product. Its stock search is the whole visual layer, and the narration is the only description of what the video is about. That profile therefore generates the script before gathering visuals and searches the stock library on phrases taken from the narration (`REQ-VID-101`).

Every other bundled profile shows product photography and keeps the default order, gathering visuals first. That order also rejects a product with too few images before an LLM call is paid for, which is why the order is not simply reversed for everything. Under the script-first order, `gather_visuals` is made a prerequisite of the voiceover and the description too, so a render that is going to be skipped for a stock shortfall does not pay for either first.

The phrases are configured under `llm_settings.visual_search_terms` in `config/ai_services.yaml`, each one a separate library search (`REQ-VID-102`). Set `enabled: false` there to search the topic title and profile keywords instead. A failure to derive phrases leaves the existing search terms in place rather than failing the render (`REQ-VID-104`). [Tutorials](tutorials.md) covers the same order from the topic side.

## What checks a stock render

Two model calls guard a script-first render, and neither can cost it. Each stock candidate's thumbnail is scored 0-3 against the script and the best are downloaded (`REQ-VID-108`), and a failed judgement keeps the random sample (`REQ-VID-109`). The script itself is fact-checked inside `generate_script`, before the visuals are searched for and before anything else consumes it: a topic script against grounded search, a product script against its own scraped listing. Both mechanisms, their guards and the files they leave are described in [Configuration](../reference/configuration.md#8-stock-media-settings).

## What a partial run keeps

`--step` requires the chosen step's declared dependencies, not everything listed above it. `--step create_voiceover` needs the script and runs whether or not `generate_description` has, because the description feeds it nothing. That is what makes it possible to iterate on one part of a render, a voice profile or the music, without re-running what comes before it.

Re-running one step also forgets every recorded step that reads its output, and deletes the files those steps would otherwise short-circuit on, such as the voiceover and the platform metadata. That makes the next full run redo them against the changed input rather than pair fresh narration with stale captions. The cost is that `--step generate_script` discards the voiceover you already have.

`burn_pycaps_subtitles` replaces the assembled video with the burned one, so it never burns a second time over its own output. Re-running `assemble_video` drops the recorded burn, which is why caption styling is iterated from that step.

## Why profiles exist

A profile bundles every choice that makes one render look different from another: which media to use, how to lay it out, which assembly mode, and how to caption it (`REQ-VID-091`). A batch picks one per product, by name or deterministically from a pool, so one run can produce varied output without per-product configuration. Precedence is CLI over profile over global, so a profile states only what differs from the global defaults and a run can still override it.

Profiles are strict about unknown keys (`REQ-VID-093`). Pydantic's default is to drop them, and a dropped key in a profile block is invisible: the render succeeds with the global value, so the profile appears to work and its override does nothing. Caption overrides live in one nested `subtitle_settings` block (`REQ-VID-092`), and the flat keys it replaced are refused with an error naming where each one moved (`REQ-VID-094`). The subtitle file's extension follows the merged `subtitle_format`, so the file the generator writes and the path the assembler reads always agree.

## Why assembly modes exist

Product listings carry anything from no video to several, of varying length. The assembly mode decides how that footage fills a voiceover-length timeline (`REQ-VID-025`): every clip in turn, the best clip looped, clips interleaved with stills, or clips first and stills after. A slideshow profile sets no mode and uses only images.

## The hook headline

The hook overlay text is an authored headline generated separately from the spoken script, so the hook doesn't repeat the first caption line (`REQ-VID-083`). A topic render uses a separate headline prompt (`REQ-VID-085`): the product prompt requires a product category noun (`REQ-VID-084`), which on a topic with no device makes the model invent one. The topic prompt asks for the symptom or the fix and forbids naming anything the script does not cover. The overlay is drawn after the FFmpeg captions and before the disclosure, so `#ad` stays on top of it (`REQ-VID-078`, `REQ-VID-079`). Why a hook at all: [Promotional videos](promotional-videos.md).
