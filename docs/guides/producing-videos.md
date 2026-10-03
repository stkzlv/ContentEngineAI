# Producing videos

How to turn a scraped product, or a topic, into a finished video with a voiceover, captions and background music. Every flag is listed in [the video producer reference](../reference/video-producer.md); how the steps fit together is in [The video pipeline](../explanation/video-pipeline.md).

Before you start, finish [Installation](installation.md) and scrape at least one product (see [Batch processing](batch-processing.md#scraper-batch-mode)). A render holds 2-2.5 GB of memory for several minutes; on a machine you are also working on, run it through `make produce-lowpri ARGS="..."` (see [Low-priority batch mode](batch-processing.md#low-priority-batch-mode)).

## Render one product

1. Pick a profile from [the bundled profiles](../reference/video-producer.md#bundled-profiles).
2. Run the producer on the product's `data.json`:

   ```bash
   poetry run python -m src.video.producer outputs/B0ASIN123/data.json slideshow_images1 --debug
   ```

3. Find the video under `outputs/B0ASIN123/`. The log is `outputs/logs/producer-<date>.log`.

To render ASS captions with the FFmpeg engine instead of the bundled pycaps engine:

```bash
poetry run python -m src.video.producer outputs/B0ASIN123/data.json slideshow_images1 \
  --subtitle-engine ffmpeg --subtitle-format ass --preset animated --debug
```

To render plain SRT captions, pass `--subtitle-format srt` with `--subtitle-engine ffmpeg`.

## Render a topic

A topic render works from a subject rather than a listing, with a profile that sources every visual from stock. Output lands in `outputs/topic-<slug>-<digest>/`.

1. Render one topic:

   ```bash
   poetry run python -m src.video.producer slideshow_stock \
     --topic "Why your wifi keeps dropping" \
     --topic-description "Router placement, channel congestion, 2.4 vs 5GHz." \
     --topic-keywords "wifi router, home network"
   ```

2. To render several, write them to a YAML file:

   ```yaml
   # topics.yaml
   - title: "Why your wifi keeps dropping"
     description: "Router placement, channel congestion, 2.4 vs 5GHz."
     keywords: ["wifi router", "home network"]
   - title: "Laptop fan always loud"
     description: "Dust, thermal paste, background CPU load."
   ```

3. Render the file:

   ```bash
   poetry run python -m src.video.producer slideshow_stock --topics-file topics.yaml
   ```

A topic render draws from its own template pool, written to answer a question rather than pitch a product, and uses a narrator profile whose calls to action offer nothing to buy. Length follows the script, so a short description yields a short video. For a long list, `make topics-batch` runs each step in its own process; see [Rendering a batch of topics](batch-processing.md#rendering-a-batch-of-topics).

## Render every scraped product

Pass `--batch` with either a fixed profile or a per-product random one:

```bash
# Fixed profile for all products
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug

# Random profile per product (deterministic by product id)
poetry run python -m src.video.producer --batch --random-profile --debug

# Random from a specific pool
poetry run python -m src.video.producer --batch --random-profile \
  --profile-pool slideshow_images1 product_video_sequential product_video_mixed --debug

# JSON summary for automation
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --output-format json
```

Profile pools, the summary and the global pipeline are in [Producer batch mode](batch-processing.md#producer-batch-mode).

## Choose how the video looks

The profile sets the media, layout, assembly mode and caption style. To change the look, pick another profile:

```bash
# Play every product video in sequence instead of a slideshow
poetry run python -m src.video.producer data.json product_video_sequential
```

To change one thing for a single run, pass the flag; a CLI flag wins over the profile, and the profile wins over the global YAML:

```bash
poetry run python -m src.video.producer data.json slideshow_images1 \
  --voice-profile calm_confident --pillar utility --subtitle-anchor bottom
```

To change a profile for good, edit its block under `video_profiles` in `config/video_production.yaml`. Put caption keys in the nested `subtitle_settings` block; the [profile keys](../reference/video-producer.md#profile-keys) table lists the rest.

## Run one step

Use `--step` to iterate on one part of a render, such as the voice or the music, without re-running what comes before it:

```bash
# Redo the captions
poetry run python -m src.video.producer data.json profile --step generate_subtitles --debug

# Debug a failing assembly
poetry run python -m src.video.producer data.json profile --step assemble_video --debug
```

Before you run a step, note the following; [What a partial run keeps](../explanation/video-pipeline.md#what-a-partial-run-keeps) explains why.

- Re-running a step drops the later steps that read its output, so `--step generate_script` discards the voiceover you already have.
- To iterate on pycaps caption styling, re-run `--step assemble_video` first; `burn_pycaps_subtitles` does not burn over its own output.
- On `slideshow_stock`, `generate_script` runs before `gather_visuals`, so `--step gather_visuals` needs a completed `generate_script`.

## Start a render over

Delete the cached artifacts and render from scratch:

```bash
poetry run python -m src.video.producer data.json profile --clean --debug
```

If a render fails, see [Video producer issues](troubleshooting.md#video-producer-issues).
