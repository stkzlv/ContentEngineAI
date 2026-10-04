# Video requirements

Ids use the prefix `REQ-VID`. The format and the statuses are described in [the requirements index](README.md).

## Video assembly

- **REQ-VID-001** `shipped` The producer sets a render's duration from the voiceover length.
- **REQ-VID-002** `shipped` After assembly, the producer logs a warning when the final video's duration differs from the voiceover by more than `video_duration_tolerance_sec` (default 1 second); the check never fails the render.
- **REQ-VID-003** `shipped` When a render shows still images, it divides the voiceover duration evenly across them, so each image's time on screen follows the image count rather than a fixed length.
- **REQ-VID-004** `shipped` When the visuals are shorter than the voiceover, the render stretches the images and loops the videos; it never repeats an image.
- **REQ-VID-005** `shipped` The render joins consecutive media elements with a crossfade.
- **REQ-VID-006** `shipped` When an image doesn't fill the 9:16 frame, the render surrounds it with a blurred, darkened copy of the same image by default (`image_background_fill: blur`); a profile can select a solid colour instead.
- **REQ-VID-007** `shipped` When an input image or video is larger than `max_image_input_edge` on its longest edge (default 2560 px, 0 disables the bound), the render downscales it before assembly.
  - Why: a full-resolution photo can exceed the render's memory cap in decoding alone.
- **REQ-VID-008** `shipped` If a render exceeds its total time budget (`pipeline_timeout_sec`), the producer stops it and reports the product as timed out.
- **REQ-VID-009** `shipped` Final assembly has its own timeout (`final_assembly_timeout_sec`), which sits inside the render's total time budget.
- **REQ-VID-010** `held` Where still motion is on, every still image carries slow, jitter-free motion whose direction varies per image and per product, so no render contains a static still.
  - On when: `video_settings.still_motion.enabled` is set after the reach-test readout (#540), once swipe-away and completion (#551) on a batch with motion are no worse than without.
- **REQ-VID-011** `held` Where a profile's ending is `peak`, a render ends on its last spoken word with no silent or fading tail; with `loop`, an image-only render's last frame also matches its first.
  - On when: a profile sets `ending` to `peak` or `loop` after the reach-test readout (#540), once average percentage viewed rises and the last spoken word stays intact.
- **REQ-VID-012** `planned #544` Where sound effects are on, sparse effects mark a few beats (a transition, the reveal, the call to action), drawn per product from a pool, capped per second and mastered with the rest of the mix.
  - On when: `audio_settings.sound_effects.enabled` is set after the reach-test readout (#540), once completion on an A/B batch is no worse with effects.
- **REQ-VID-013** `planned #546` Where beat snapping is on, visual cuts move to the nearest music beat within a small window without changing caption timing.
  - On when: `video_settings.beat_snap.enabled` is set once a blind listening comparison prefers it or completion improves.
- **REQ-VID-014** `planned #552` Every render produces a cover image (the hero visual plus the hook headline inside the centred 3:4 area), set on each platform that accepts one.
  - Why: it ships without a switch, since it doesn't change the feed video, once the publish payload change is verified on one post.
- **REQ-VID-015** `planned #554` Where image curation is on, a product render prefers clean product images over text-heavy seller infographics, using text-heavy ones only when too few clean images exist.
  - On when: `video_settings.image_curation.enabled` is set after the reach-test readout (#540), once a side-by-side review prefers the curated set and swipe-away is no worse.
- **REQ-VID-016** `planned #554` Where image curation is on and the profile accepts video, a product render prefers the listing's product video over stills.
  - On when: `video_settings.image_curation.enabled` is set, on the same condition as REQ-VID-015.
- **REQ-VID-127** `shipped` The final video is a 1080x1920 H.264 MP4 in yuv420p at 30 fps, with AAC audio at 192 kbps and 48 kHz.
- **REQ-VID-128** `shipped` Where a profile's `enable_format_normalization` is on (the default), the producer converts each input video clip to H.264, 30 fps and yuv420p before assembly.
- **REQ-VID-129** `shipped` After assembly, the producer probes the final video and logs a warning when the video or audio stream is missing or the captions match the script below `subtitle_similarity_threshold`; the check never fails the render.
- **REQ-VID-146** `shipped` A render applies no colour grade or stylistic filter to product or stock visuals; the only colour change darkens the blurred backdrop behind a visual that doesn't fill the frame.

## Image positioning

- **REQ-VID-017** `shipped` The render scales each image to a configurable share of the frame width (`image_width_percent`, default 100%).
- **REQ-VID-018** `shipped` The render centres each image horizontally.
- **REQ-VID-019** `shipped` The render centres each image vertically by default; a profile can align it to the top with a configurable offset.
- **REQ-VID-020** `shipped` The render preserves each image's aspect ratio by default.
- **REQ-VID-021** `shipped` The render places a product image so it never overlaps the caption block, wherever the caption engine puts the block.

## Video positioning

- **REQ-VID-022** `held` The render centres video content vertically by default; a profile can align it to the top with an offset (default 10%).
  - On when: `video_settings.video_vertical_align: top` is removed from `config/video_production.yaml` after the reach-test readout (#540), so the profiles that don't set it centre.
- **REQ-VID-023** `shipped` Video content takes a configurable share of the frame height (`video_content_height_percent`, default 75%).
- **REQ-VID-024** `shipped` A profile selects how a video whose aspect differs from the frame is fitted: letterbox, crop-to-fit, blur-fill, or smart-scale (crop when the aspect ratios are within 10%, blur-fill otherwise).
- **REQ-VID-025** `shipped` A profile selects the assembly mode: sequential, single-best, mixed-media or video-first-fallback.

## Render audio

- **REQ-VID-026** `shipped` The render drops the source videos' audio; the mix is the voiceover and the background music.
- **REQ-VID-027** `held` Where a signature sting is configured, the render plays it at the start or the end of the video, mastered with the rest of the mix.
  - On when: `audio_settings.signature_sting` is set to a local file after the reach-test readout (#540), turned on in stages.

## Caption step

- **REQ-VID-028** `shipped` If the subtitle step leaves no caption source, the render fails and the error names each path it looked for.
- **REQ-VID-029** `shipped` On the pycaps engine, the subtitle step accepts a transcript only when the current run wrote it.
  - Why: the transcript path is the same across runs, so a leftover file would caption a script that is no longer being narrated.
- **REQ-VID-130** `shipped` The producer takes caption word timings from a local Whisper transcription of the voiceover.
- **REQ-VID-131** `shipped` The Whisper time limit is `whisper_settings.base_timeout_sec` plus the audio length times `duration_multiplier`, capped at `max_timeout_sec` and at what remains of the render's time budget (`pipeline_timeout_sec`).
  - Why: without the outer cap a long transcription spends the render's budget and the timeout is reported against a later step.
- **REQ-VID-132** `shipped` If Whisper times out, the producer retries it with the limit widened by `timeout_retry_multiplier`, at most `timeout_retry_attempts` times, and stops retrying once the cap keeps the limit from widening.
- **REQ-VID-133** `shipped` Where Google Cloud STT is enabled with valid credentials, it supplies the word timings when Whisper is unavailable or returns none.
- **REQ-VID-030** `shipped` When a run produces no captions on purpose (captions disabled, or the engine unavailable under a skip policy), the subtitle step records why, so a resume can tell it from a step that produced nothing by accident.

## Caption engines

- **REQ-VID-031** `shipped` The pycaps engine is the bundled default caption engine.
- **REQ-VID-032** `shipped` A profile, or a run through `--subtitle-engine`, selects the caption engine (`pycaps` or `ffmpeg`).
- **REQ-VID-033** `shipped` If pycaps is not installed and the fallback policy is `fallback_ffmpeg` (the bundled policy), the render uses the FFmpeg engine without manual intervention, detected before assembly.
- **REQ-VID-034** `shipped` The producer records the caption engine a run resolved in the run state.
- **REQ-VID-035** `shipped` The pycaps engine burns word-by-word karaoke captions after assembly, with per-word CSS animation and template-driven styling.
- **REQ-VID-036** `shipped` Where AI word tagging is on, the pycaps engine highlights the words a language model picks per segment.
- **REQ-VID-150** `shipped` Where AI word tagging is on and `pycaps.ai_tag_prompt_override` is set (the bundled value), the tagger is asked for about 15% of the words, chosen from prices, numbers, product nouns, outcome verbs and factual superlatives, and never articles, prepositions, auxiliaries, absolute praise words or a whole sentence.
- **REQ-VID-134** `shipped` If AI word tagging fails for a segment, `pycaps.ai_tagging_on_error: skip` (the bundled value) burns that segment without highlights, and `raise` hands the failure to the caption fallback policy.
- **REQ-VID-135** `shipped` If AI word tagging is on and the Gemini key is missing, the producer logs a warning and burns the captions without AI highlights.
- **REQ-VID-037** `shipped` The pycaps engine renders one caption track of up to two lines and has no two-part mode.
- **REQ-VID-038** `shipped` The FFmpeg engine burns SRT or ASS captions during assembly and supports two-part mode, karaoke and every positioning anchor.
- **REQ-VID-039** `shipped` If a pycaps burn fails under the `raise` fallback policy, the render aborts.
- **REQ-VID-040** `shipped` If a pycaps burn fails under the `warn_and_skip` policy, the render keeps the video without captions; no other policy produces a caption-less video.
- **REQ-VID-041** `shipped` If a pycaps burn fails under `fallback_ffmpeg` with a transcript and a video on disk, the render burns the captions with the FFmpeg engine; a missing transcript or video still aborts.
- **REQ-VID-042** `shipped` The pycaps engine renders with the CSS renderer; the pictex renderer is for previews only.
  - Why: pictex drops the gaps between words.
- **REQ-VID-043** `shipped` The pycaps engine picks one template per product, the same on every run, from a configurable pool (bundled: `explosive`, `word-focus`).
- **REQ-VID-044** `shipped` Where `force_sentence_case` is on (the bundled default), pycaps captions keep the transcript's casing whatever the template sets.
- **REQ-VID-045** `shipped` Where `mute_template_sound_effects` is on (the bundled default), caption templates add no sound to the render.

## Caption positioning

- **REQ-VID-046** `shipped` Captions anchor to the top, centre or bottom of the frame, or above or below the content.
- **REQ-VID-047** `shipped` Captions sit a configurable margin from their anchor edge (default 4%).
- **REQ-VID-048** `shipped` Captions align left, centre (default) or right.
- **REQ-VID-049** `shipped` The render positions captions relative to the actual bounds of the media on screen.
- **REQ-VID-050** `shipped` Caption boundaries avoid the TikTok, YouTube Shorts and Instagram Reels interface overlays (see [platform safe zones](../explanation/platform-safe-zones.md)); the zone is set globally and per profile, where a profile sets only the boundaries that differ.
- **REQ-VID-051** `shipped` The FFmpeg engine clamps caption position to the safe zone, keeping a caption's lowest pixel above the bottom boundary, and honours per-profile safe-zone overrides.
- **REQ-VID-052** `shipped` The pycaps engine limits caption width so centred text stays clear of the right-side boundary; it doesn't enforce the vertical boundaries.
- **REQ-VID-053** `shipped` The pycaps engine places the caption block as a lower third by default, its bottom at 75% of the frame height, below the safe-zone bottom of 65%.
  - Why: the centred source video on the product-video profiles ends near 66% of the frame.
- **REQ-VID-054** `shipped` The pycaps engine keeps the template's own alignment unless the profile overrides it.

## Caption text

- **REQ-VID-055** `shipped` Captions use a bold sans-serif font of weight 700 or more.
- **REQ-VID-056** `partial` The caption font size is `font_size_percent` of the frame height (bundled 7.5%, about 144 px on 1920) multiplied by `font_size_scale` (0.5 to 2.0, default 1.0, `--font-size-scale`).
  - Gap: only SRT captions read `font_size_percent`; ASS captions size from a fixed 4% of the frame height times `font_size_scale`, capped at 100 px, and pycaps captions take the template's size (#591).
- **REQ-VID-057** `partial` Captions have a white fill and an opaque black outline (2-4 px by style preset), with no background box.
  - Gap: FFmpeg karaoke draws black text with a white outline that fills yellow, the pycaps `explosive` template has a yellow base fill with an orange glow and no outline, and `word-focus` has a white fill with a 2 px black shadow and an orange box behind the active word (#591).
- **REQ-VID-058** `shipped` A caption holds at most 2 lines, each at most 80% of the frame width; on the FFmpeg engine a line also holds at most `max_words_per_line` words (bundled 3), while the pycaps engine splits lines by the template's character count.
- **REQ-VID-136** `shipped` On the FFmpeg engine, a caption line also holds at most `max_line_length` characters (bundled 30, `--max-line-length`).
- **REQ-VID-059** `shipped` A caption segment lasts between 0.6 s and 2.5 s.
- **REQ-VID-060** `shipped` Each narration word appears slightly before its audio onset, by a configurable lead.
- **REQ-VID-061** `shipped` The first few words of the opening hook get an extra lead on top of the base lead; the lead and the number of words are configurable per render.
- **REQ-VID-062** `shipped` Captions never split or alter a number: thousands separators, decimals and hyphenated tokens stay whole and keep their punctuation.
- **REQ-VID-063** `shipped` Where timing smoothing is on (the bundled default), caption word timings get a minimum time on screen per word (default 0.12 s), gaps shorter than a threshold merge into the preceding word (default 0.08 s), and the last word of a segment is held (default 0.2 s).
- **REQ-VID-147** `shipped` Captions carry no emoji: they come from the transcription of a voiceover whose script has its emojis removed, and no bundled caption template or style preset adds one.
- **REQ-VID-156** `planned #591` A caption segment's reading rate stays at or below a configured characters-per-second cap, and a segment over the cap merges into its neighbour.
- **REQ-VID-157** `planned #591` Caption lines break at phrase boundaries and never split a noun phrase or a product name.

## Two-part captions

- **REQ-VID-064** `shipped` Where two-part mode is on (FFmpeg engine only), the upper line shows a static link above the content.
- **REQ-VID-065** `shipped` Where two-part mode is on, the lower line shows the voiceover-synced transcription below the content.
- **REQ-VID-066** `shipped` Where two-part mode is on, both lines reposition for each visual segment.
- **REQ-VID-067** `shipped` Where two-part mode is on, the upper line shows only during detected call-to-action windows, or for the whole video when `use_full_duration` is set.

## Link line

- **REQ-VID-068** `shipped` Where `video_settings.upper_line.enabled` is set, the render draws a static line (the affiliate link, the link-in-bio address, or custom text) above the visual for the whole video, under either caption engine.
- **REQ-VID-069** `shipped` If the link line resolves to nothing or is estimated wider than the frame allows, the render skips it and logs why.

## Style presets

- **REQ-VID-070** `shipped` The FFmpeg engine offers the style presets minimal, modern (default), bold, animated and random.
- **REQ-VID-071** `shipped` The caption effects are karaoke (word-by-word highlight), fade and typewriter.
- **REQ-VID-072** `partial` A render uses one caption effect, chosen deterministically from the product id.
  - Gap: the choice for the same product differs from one run to the next.
- **REQ-VID-073** `shipped` Where randomisation is on, the caption font and colour pair are drawn from curated pools.
- **REQ-VID-148** `shipped` No bundled caption template or style preset flashes, and none scales a word beyond 1.1x: the `explosive` template's word zoom peaks at 1.10x, and `word-focus` has no animation.
- **REQ-VID-149** `shipped` A render's captions use at most three highlight colours, since one caption template or one colour pair styles the whole render.
  - Check: `explosive` uses three fills (a word before, during and after its narration), `word-focus` one box colour, and FFmpeg karaoke one sweep colour.
- **REQ-VID-155** `planned #591` A caption entrance animation lasts at most 250 ms, and a segment leaves with a hard cut or a fade of at most 80 ms.

## Cold open

- **REQ-VID-074** `shipped` Where pre-motion is on, the first image starts at a slight zoom and settles to 1.0 over its segment, so frame 0 is in motion.
- **REQ-VID-075** `shipped` Pre-motion is off by default and on in the short profile; a profile can turn it on or off.
- **REQ-VID-076** `shipped` The pre-motion peak zoom is configurable globally and per profile (default 1.10).
- **REQ-VID-077** `shipped` Where the hook overlay is on (the bundled default), the render shows a short headline as static centre-upper text for the first 1.5 s (configurable), sized relative to the captions, with no per-word reveal.
- **REQ-VID-154** `planned #591` The hook headline renders larger than the narration captions on every caption engine.
- **REQ-VID-078** `shipped` The on-frame disclosure is drawn above the hook overlay.
- **REQ-VID-079** `shipped` FFmpeg captions sit below the hook overlay; pycaps captions are burned after assembly and sit above both overlays.
- **REQ-VID-080** `shipped` When the hook headline is too long, the overlay wraps it to a configurable number of lines, each within a configurable share of the frame width, and shrinks the font when wrapping alone doesn't fit.
- **REQ-VID-081** `shipped` If the hook headline still doesn't fit at the minimum legible size, the overlay truncates it with an ellipsis and logs the truncation.
- **REQ-VID-082** `shipped` When no hook text is available, the render draws no overlay and doesn't fail.
- **REQ-VID-083** `shipped` The producer writes the hook headline for the screen, separately from the spoken script, so it doesn't repeat the first caption.
- **REQ-VID-084** `shipped` A product render's hook headline names the product category and never a model or SKU designation.
- **REQ-VID-085** `shipped` A topic render's hook headline names the symptom or the fix and nothing the script doesn't cover.
- **REQ-VID-086** `shipped` The hook headline is capped at a configurable word count.
- **REQ-VID-087** `shipped` If the generated headline reads as a conversational preamble or a refusal, the producer rejects it.
- **REQ-VID-088** `shipped` When no headline is available, the overlay shows the first sentence of the spoken script.
- **REQ-VID-089** `shipped` A re-render generates the headline when it is missing, skips generation when the overlay is off, and records the headline in the run state.
- **REQ-VID-090** `partial` Each render picks one named cold-open variant from a configurable pool, the same for a product on every run, and records the variant name in the run state.
  - Gap: every variant renders the same way; only the name differs.

## Profiles

- **REQ-VID-091** `shipped` Every visual, subtitle and video setting is configurable per profile.
- **REQ-VID-092** `shipped` A profile overrides caption settings in one nested `subtitle_settings` block, setting only the fields that differ; the nested `pycaps`, `two_part_subtitles` and `safe_zone` blocks merge field by field.
- **REQ-VID-093** `shipped` If the subtitle config or a profile override carries an unknown key, the config fails to load with an error naming it.
- **REQ-VID-094** `shipped` If a profile uses a legacy flat caption key (`subtitle_anchor`, `pycaps_template`, `two_part_subtitles` and the like), the config load is refused with an error naming the nested field to move it to.
- **REQ-VID-095** `partial` The short profile renders 15-30 s videos with a script of about 50-60 words.
  - Gap: the script word budget is global, so nothing sizes the short profile's script or holds its length to 15-30 s.

## Run state and resume

- **REQ-VID-137** `shipped` When the producer renders a product again, it skips each step recorded as done in `pipeline_state.json` whose artifacts still exist and belong to the same profile's run.
- **REQ-VID-138** `shipped` If a recorded artifact is missing or belongs to another profile's run, the producer re-runs from that step and deletes the stale outputs that would make the later steps reuse an old result.
- **REQ-VID-139** `shipped` When a step runs again, the producer forgets the completed steps that read its output, so the next full run redoes them.
- **REQ-VID-140** `shipped` When `--step <name>` is passed, the producer runs only that step, after loading the artifacts of the steps it depends on.
- **REQ-VID-141** `shipped` If a step that `--step` depends on is not complete, the producer refuses the run and names that step.
- **REQ-VID-142** `shipped` When `--clean` is passed, the producer deletes the files it generated for the product before rendering, including every profile's video and the run state, and keeps the scraped inputs.

## Product input

- **REQ-VID-143** `shipped` When `--product-index N` is passed, the producer renders only the product at 0-based position N in the input file.
- **REQ-VID-144** `shipped` If `--product-index` is combined with `--batch`, the producer refuses the run.
- **REQ-VID-145** `shipped` If `--product-index` is outside the input file's range, the producer exits non-zero naming the index and the product count.

## Stock visual media

- **REQ-VID-096** `shipped` The producer fetches stock footage from a configured provider and merges it into the same visual pool as the scraped product media, not only as a fallback.
- **REQ-VID-097** `shipped` Stock search terms come from `media_settings.stock_media_keywords`, overridable per profile; a profile that declares no terms inherits the global list.
- **REQ-VID-098** `shipped` A profile that declares an empty term list searches on the product title alone.
- **REQ-VID-099** `shipped` A profile controls whether scraped product imagery is used at all (`use_scraped_images`); the bundled `slideshow_stock` profile renders entirely from stock.
- **REQ-VID-100** `shipped` A topic's own search terms replace the profile and global lists rather than joining them.
  - Why: the provider joins every term into one query, so a mixed list searches for neither.
- **REQ-VID-101** `shipped` A profile that draws no visual from the scraped product gathers its footage after the script exists, searching on phrases derived from the narration.
- **REQ-VID-102** `shipped` The producer searches the narration phrases one at a time and pools the results.
- **REQ-VID-103** `shipped` The producer drops duplicate stock results across searches, so one item never appears twice in a render.
- **REQ-VID-104** `shipped` If deriving the narration phrases fails (no key, a provider failure, or an unusable answer), the render keeps the existing search terms.
- **REQ-VID-105** `shipped` If the stock provider fails, the render continues with a smaller visual pool.
- **REQ-VID-106** `shipped` If a stock provider key is missing, the producer stops at startup naming the variable and the profiles that need it, for profiles whose whole visual layer is stock.
- **REQ-VID-107** `shipped` A profile that also draws scraped media still renders without the stock provider key.
- **REQ-VID-108** `shipped` Where the stock relevance judge is on (the bundled default), the producer scores stock candidates against the script from their thumbnails and uses the best; candidates below `min_score` are used only to fill a shortfall.
- **REQ-VID-109** `shipped` If the relevance judge returns no scores, the producer uses a random sample of the candidates.
- **REQ-VID-110** `planned #555` Where the reuse guard is on, a stock clip or image used in a recent render is excluded while alternatives exist, and the ids each render uses are recorded so the rule survives cleanup.
  - On when: `stock_reuse_guard.enabled` is set after the reach-test readout (#540).

## Topic input

- **REQ-VID-111** `shipped` The producer renders a video from a topic (a title, a description and optional search terms) with no scraper run and no product directory.
- **REQ-VID-112** `shipped` A topic produces the same record the producer reads for a scraped product, and listing-only fields are empty rather than filled with plausible values.
- **REQ-VID-113** `shipped` The producer renders every topic in a topics file in turn.
- **REQ-VID-114** `shipped` If a topics file has a malformed entry, the run fails rather than skipping it.
- **REQ-VID-115** `shipped` The batch accepts topics as input and builds their records instead of scraping; a run with both products and topics handles both in one reported phase.
- **REQ-VID-116** `shipped` When a batch run has no input flags, it takes a configured number of topics in rotation alongside the configured keywords.
- **REQ-VID-117** `shipped` Topics or keywords given on the command line replace the configured set entirely.
- **REQ-VID-118** `shipped` The topic rotation advances with the date, so a daily run works through the list.
- **REQ-VID-119** `shipped` A topic renders only with profiles whose visuals come entirely from stock; a profile that draws product imagery is refused before the run starts.
- **REQ-VID-120** `shipped` On a run with both products and topics, each draws from its own profile pool, and a fixed profile that draws no stock media is refused.
- **REQ-VID-121** `planned #559` Where step lists are on, a topic script is written from a sourced step list (action, exact UI path, expected result and source per step), its length set by the step count, and a topic that forks by device becomes a series.
  - On when: `topic_scripts.step_list.enabled` is set after the reach-test readout (#540).
- **REQ-VID-122** `planned #559` Where step lists are on, a step with no source is refused, and a topic that can't be sourced is dropped.
  - On when: `topic_scripts.step_list.enabled` is set, on the same condition as REQ-VID-124.
- **REQ-VID-123** `planned #560` Where step visuals are on, each tutorial step is shown as it is spoken by a screen capture, a UI mockup with the exact labels or a diagram, stock footage covers only the opening symptom, and the video ends on the result and a path recap.
  - On when: `video_settings.step_visuals.enabled` is set after the reach-test readout (#540).
- **REQ-VID-124** `planned #561` Where explanatory graphics are on, a tutorial carries templated graphics (menu-path breadcrumb, step card, callout on a real capture, spec card, before/after, checklist) from a validated spec, one at a time, each tied to a script fact, and a failed graphic is skipped rather than failing the render.
  - On when: `video_settings.graphics.enabled` is set after the reach-test readout (#540).
- **REQ-VID-151** `planned #559` Where step lists are on, a topic enters the pool only when it is specific (one device family or app and one outcome), searchable, demonstrable and non-default, and a topic that asks for health, financial or legal advice is excluded.
  - On when: `topic_scripts.step_list.enabled` is set after the reach-test readout (#540).
- **REQ-VID-152** `planned #560` Where step visuals are on, a step marked error-prone stays on screen longer than an obvious one.
  - On when: `video_settings.step_visuals.enabled` is set after the reach-test readout (#540).
- **REQ-VID-153** `planned #561` Where explanatory graphics are on, each graphic type has two or three layout variants, and each video draws its variant reproducibly.
  - On when: `video_settings.graphics.enabled` is set after the reach-test readout (#540).

## Media validation

- **REQ-VID-125** `shipped` The scraper checks each product's media against the producer profile's requirements.
- **REQ-VID-126** `shipped` If a product has too little media, it is skipped, not failed.
