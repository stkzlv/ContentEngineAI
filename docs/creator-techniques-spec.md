# Creator Techniques: Technical Specs

The technical design for each requirement drawn from [creator-research.md](creator-research.md) and [ai-slop-research.md](ai-slop-research.md). The requirements themselves are in [requirements.md](requirements.md), marked `(planned, #N)`. One section per issue.

## Rules that apply to every spec

- **Off by default when it changes output.** The format-vs-format reach test needs the script, voice and sound held constant until its readout (#540). Anything that changes what a render looks or sounds like ships behind a switch that defaults to today's behaviour, and `tests/test_reach_test_holdout.py` gains a check for it. Measurement-only work (#547, #551) can land at any time.
- **Byte-identical when off.** Each spec's off state must leave the FFmpeg command, the prompt or the payload exactly as it is today, pinned by a test.
- **Seeded variation.** A choice drawn per render uses a salted MD5 of the product id (`<product_id>:<purpose>`), the pattern the CTAs and pauses already use (fonts and voices hash the bare product id with different slices). A product renders the same way every time, and a batch varies.
- **Record what was chosen.** Every drawn choice goes into `pipeline_state.json` beside `script_template` and `cta`, and is mirrored into the step entry so a truncating resume keeps it. #547 turns these records into a variety report, and #551 segments metrics by them.
- **Measure before enabling.** Each production spec lists the check that decides whether to enable it after the readout. Most of the evidence is creator opinion or ad research, so the pipeline's own analytics decide.

## #542 Motion on every still

**Today.** `_build_ken_burns_filter` in `src/video/assembler/visual_builder.py` applies a settle-zoom to the first image only. Later stills are static.

**Design.**
- A per-profile `still_motion` block in `video_production.yaml`: `enabled` (default false), `moves` (the pool: `push_in`, `pull_out`, `pan_left`, `pan_right`, `pan_up`), `max_zoom` (default 1.15) and `min_zoom` (1.0).
- Per still, draw a move from the pool with the seed `<product_id>:motion:<index>`, so consecutive stills differ and two products differ.
- Implement with an upscale then an animated `crop` (sub-pixel) and a final `scale` to the frame size, not `zoompan`, which rounds positions to whole pixels and shudders on slow moves. Keep the first image's settle-zoom as it is; this covers the rest.
- Respect the image band and the caption safe zone the assembler already computes, so motion never pushes the product under the captions.

**Tests.** The filter string for a three-still render carries a motion clause per still, with at least two distinct moves; the off state produces today's filter graph byte for byte; a rendered test clip shows frame-to-frame change on every still with no single-pixel oscillation in the motion path.

**Enable when.** Swipe-away and completion (#551) on a batch with motion are no worse than without.

## #543 End on the peak, with an optional seamless loop

**Today.** `outro_duration_sec: 1.0` (`config/core.yaml`) is added after the voiceover, partly to avoid AAC truncating the last word, and the music fades out over `music_fade_out_duration` (3.0 s).

**Design.**
- A per-profile `ending` setting: `outro` (default, today's behaviour), `peak` or `loop`.
- `peak`: the video ends `peak_margin_sec` after the last spoken word (default 0.25 s, enough for the AAC frame padding the outro was protecting; verify on a render). The music is cut at the same point with a fade no longer than the margin, not faded to silence over three seconds.
- `loop`: `peak`, plus the last visual segment ends on the frame 0 composition (the first image at its starting scale, without the hook overlay), so a replay reads as continuous. Implement by reusing the first image as the last segment's source and matching its crop.
- The CTA is spoken, so it stays the last sentence. `peak` removes only the silence after it.
- Relates to roadmap 1.8.

**Tests.** A `peak` render's duration equals the voiceover duration plus the margin within one frame; the last spoken word is intact in a Whisper transcript of the output; a `loop` render's last and first frames differ by less than a set mean pixel difference; `outro` produces today's command.

**Enable when.** Average percentage viewed rises and the last-word transcript check still passes.

## #544 Sparse event sound effects

**Today.** The mix is voiceover, music and (optionally) the signature sting, built by `AudioFilterBuilder.build_mix`.

**Design.**
- `audio_settings.sound_effects`: `enabled` (default false), `level_db` relative to the voice (default -15), `max_per_10_sec` (default 2), and one pool per event type: `transition`, `reveal`, `cta`. Each pool is a list of local files, at least five per type to avoid a template sound.
- Events come from data the pipeline already has: transition times from the visual chain, the reveal as the start of the sentence after the hook (from the Whisper word timings), the CTA as the start of the last sentence.
- Per event, draw a file with the seed `<product_id>:sfx:<event>`. Add each as an input with `volume` and `adelay` into the same `amix` as the music and sting, so `loudnorm` masters it with the rest.
- Enforce the per-10-second cap by dropping the lowest-priority events (transitions first). Never place an effect under a spoken word's first 100 ms.
- Ship without bundled files; the config points at a local directory, and a missing file warns and is skipped (the sting's rule).

**Tests.** With three events configured, the filter graph carries three delayed inputs at the configured level; the cap drops transition effects first; two products draw different variants; off adds nothing to the command.

**Enable when.** An A/B over a batch shows completion no worse with effects; stop at the first sign of lower engagement (the inverted-U finding).

## #545 Optional voice processing chain

**Today.** The voiceover enters the mix with only a volume adjustment.

**Design.**
- `audio_settings.voice_chain`: `enabled` (default false) and the parameters: `highpass_hz` (80), `harsh_cut_hz` (3000), `harsh_cut_db` (-2), `compressor` (threshold -18 dB, ratio 3, attack 5 ms, release 80 ms), `deess` (on), `air_shelf_db` (+1.5 above 9 kHz), `limiter` (on).
- Applied in `build_mix` to the voice chain before the `volume` stage: `highpass,equalizer,acompressor,deesser,highshelf,alimiter`. Captions are transcribed from the TTS output file, so the processed audio never reaches Whisper; the transcript is unchanged by construction.
- Record `voice_chain` (on or off) and the voice name per render, so #551 can compare voices and chains.

**Tests.** The filter clause matches the configured parameters; integrated loudness of a mixed test clip with the chain on stays within 0.5 LU of the same clip with the chain off (the mix already lands about 1 LU under the target, so compare the two, not either with the target); off produces today's command.

**Enable when.** A voice-by-chain comparison over at least 20 posts per cell shows no loss; the research suggests also trying a lower-pitched voice.

## #546 Snap visual cuts to music beats

**Today.** Segment boundaries come from the assembly strategy and the voiceover length; the music is chosen independently in the `download_music` step.

**Design.**
- `video_settings.beat_snap`: `enabled` (default false), `window_ms` (150), `min_segment_sec` (the profile's existing minimum).
- Detect beats once per track after the music is downloaded, with `librosa.beat.beat_track`, and cache them beside the track (`<track>.beats.json`). librosa is an optional dependency; without it the option warns and does nothing.
- Before building the `concat` or `xfade` offsets, move each boundary to the nearest beat within the window, skipping any move that would break the minimum segment length or push a boundary past the voiceover end.
- Boundaries move, the voiceover does not, so caption timing is untouched.

**Tests.** With a synthetic click track at a known tempo, most moved boundaries land within 30 ms of a beat; no segment drops below the minimum; a missing librosa leaves boundaries unchanged with a warning; off produces today's offsets.

**Enable when.** A blind listening comparison prefers it, or completion improves; the evidence is a lab result, so treat it as low priority.

## #547 Record render choices and report output variety

**Today.** `pipeline_state.json` records `script_template`, `cta`, the voice profile, `signoff` and some subtitle choices; the registry records `content_format`.

**Design.**
- Record every choice that shapes a render in the state: hook archetype, caption template, music track id, motion moves, transitions, effect variants, voice chain. Carry them into the published-products registry at publish time, as `content_format` is.
- A `variety` report (a new report type beside the existing analytics reports): for the last N published renders (default 14), the distribution per dimension, and an alert when one value exceeds a share threshold (default 60%) where the pool has more than one option.
- A script similarity check: character 5-gram Jaccard similarity between the new script and each of the last N scripts, with a warning above a threshold (default 0.5) logged at generation time and counted in the report. Warn, do not block.

**Tests.** A render records every listed choice; the report flags a dimension dominated by one value in a fixture; two near-identical scripts cross the similarity threshold and two unrelated ones do not.

**Measurement only.** This can land before the readout.

## #548 Script lint and search-phrase placement

**Today.** `validate_script_completeness` checks truncation, length floors and the CTA; the narrator profiles carry banned-phrase prose, which the model may ignore.

**Design.**
- `script_validation.lint`: `enabled` (default false), `banned_phrases` (a list: "it's not X, it's Y" shapes as regexes, "game-changer", "say goodbye to", "elevate", "seamless", "delve", "whether you're"), `max_sentence_words` (default 16), `max_words_per_sec` (default 2.8, applied to the profile's target duration).
- Run inside the existing `_validate` closure, so a failing script re-enters the retry loop like a missing CTA, with its own reason string. The last-resort fallback today rescues only a script whose sole defect is the CTA; extend it so a script that fails only the lint also ships (with a warning) rather than losing the render.
- Prompt rules for the hook's concreteness and the "but/therefore" chain render into `{CTA_RULE}` after the existing rules, like naturalism, behind `script_templates.hook_rules.enabled` (default false).
- Search-phrase placement: a prompt rule for the hook headline and each platform caption prompt to lead with the search phrase (the product keyword or the topic keyword), behind the same switch as the hook rules. The first spoken sentence already carries it through the existing audio-keyword rule.
- Search-phrase report: the phrase checked against the first spoken sentence, the hook headline and the first 60 characters of each platform caption, counted per render in the #547 report. The report is measurement only and can land before the readout; the placement rules wait for it.

**Tests.** Each banned shape rejects a fixture script; the length caps reject an over-long script; off leaves `_validate` unchanged; the report counts a fixture where the phrase is missing from the caption.

**Enable when.** Rejection rates on a batch stay low (the retry loop is paid per call) and the scripts read better on review.

## #549 Remove engagement-bait lines from the CTA pools

**Today.** Both pools carry a share request: "Share with someone who needs this." in `cta_options` and "Share it with whoever needs it." in `cta_options_topic`. Meta lists share requests as engagement bait.

**Design.**
- A bait-pattern list in a test (share, tag, vote, "comment <word>", emoji requests, follow-for-reward), checked against `cta_options`, `cta_options_topic` and the closing-line examples in every script template.
- Replace the flagged lines with genuine opinion or save prompts (for example "Save this for your next setup."). The first-comment extractor's `_CTA_MARKERS` has to gain the new openers, and its existing test asserts every configured CTA starts with one.
- Changing the pool changes the closing lines of both reach-test arms, so the pool edit waits for the readout. The test lands first with the current offenders listed as known exceptions, which the pool edit then removes.

**Tests.** The bait test fails on a pool containing a share request; the marker test passes with the new pool.

## #550 YouTube titles for products; Instagram hashtag range

**Today.** Product videos go to YouTube with the store listing title, cut to fit (the metadata validation warns on `data.json` titles over 100 characters). Three places set the Instagram hashtag range: `platform_metadata.instagram` and `platform_metadata_config` in `config/ai_services.yaml` (both 15-30), and `PLATFORM_LIMITS[Platform.INSTAGRAM]` in `src/publisher/models.py` (5-30), which `validate_limits` checks at publish. Posts actually carry 4 or 5.

**Design.**
- Find the path that sets the YouTube title for a product render (the metadata loader reads `title` from the product record) and have it use the generated YouTube title from the platform metadata step, bounded by `title_length_max`, keyword first. Fall back to a shortened listing title only when generation failed.
- Set all three Instagram ranges to 3-5 and correct the comments, so a 3- or 4-hashtag post no longer fails `validate_limits`. Trace which step currently caps Instagram at 5, and make one source decide, with the others derived from it or asserted equal by a test.

**Tests.** A product render's YouTube payload title is the generated one and within the maximum; the Instagram generator never returns more than 5 hashtags, and `validate_limits` accepts 3, 4 and 5.

**Reach test.** Titles and hashtags are held constant across both arms by the protocol. The title fix applies only to the product arm, so it waits for the readout; the hashtag config alignment changes nothing that ships and can land.

## #551 First-seconds metrics in the analytics sweep

**Today.** The sweep reads view timelines through the scheduling provider and stores day-2 and day-7 views and a durability ratio.

**Design.**
- Inventory what the provider's analytics endpoint returns per platform (likes, comments, shares, saves, impressions, reach, watch time). Store every available field per post in the metrics store, beside the view figures.
- YouTube's "viewed vs swiped away" and engaged views are exposed by the YouTube Analytics API (`engagedViews`), not necessarily by the provider. If the provider lacks them, add an optional YouTube Analytics reader behind its own credentials, off by default.
- Store an unavailable metric as unknown, never zero, the rule the day-N figures already follow.
- Extend the reports to segment each metric by `content_format` and by the #547 render choices.

**Tests.** The store round-trips the new fields and keeps unknowns unknown; a report segments a fixture by format and choice.

**Measurement only.** This can land before the readout, and it is what the other specs' "enable when" checks read.

## #552 Cover frames, including YouTube Shorts thumbnails

**Today.** No cover is produced. Since July 2026 YouTube accepts custom Shorts thumbnails on desktop for Partner Program channels; roadmap 4.7 records this.

**Design.**
- After assembly, render `cover.jpg` at 1080x1920 from the frame 0 composition: the hero image and the hook headline, with the headline inside the centred 3:4 area Instagram's grid crops to.
- Pass the cover in the publish payload for every platform the provider accepts one for. Check the provider's API for a Shorts thumbnail field; if it has none, record the gap and keep frame 0 as the YouTube lever.

**Tests.** Every render writes a cover of the right size with the headline inside the 3:4 area; the payload carries it where supported.

**Reach test.** The cover does not change the feed video, but it changes the profile grid and search presentation for both arms equally, so it can land before the readout if the payload change is verified on one post first.

## Specs from the AI-slop research

The sections below come from [ai-slop-research.md](ai-slop-research.md), which found that looking fully automated is itself the penalty. The same rules apply: off by default when output changes, byte-identical when off, recorded per render.

## #554 Prefer clean product images over seller infographics

**Today.** The producer uses the downloaded listing images in listing order (`step_gather_visuals` reads `downloaded_images`). Most are marketing composites with dense text, and captions and the hook headline are drawn over them.

**Design.**
- `video_settings.image_curation`: `enabled` (default false), `max_text_share` (default 0.15), `min_clean_images` (default 3).
- Score each image once after download with the multimodal judge the stock-relevance step already uses (`src/video/stock_relevance.py`): one call per image asking for the share of the frame covered by overlaid text and whether it is a composite. Cache the score beside the image, so a re-render pays nothing. A failed judgement is unknown and sorts after known scores, the stock judge's rule.
- Order images clean first. Drop images above `max_text_share` while at least `min_clean_images` remain; otherwise keep the least text-heavy ones.
- When the listing has a product video and the profile accepts video, prefer it over stills.
- Record the per-image scores and the chosen order in the state.

**Tests.** A fixture set with known scores is reordered clean first and trimmed only above the minimum; a failed judgement never removes an image; off leaves today's order.

**Enable when.** A side-by-side review of renders with and without it prefers the curated set, and swipe-away (#551) is no worse.

## #555 Do not reuse stock clips across recent renders

**Today.** Stock candidates come from the provider search and the relevance judge with no memory of earlier renders.

**Design.**
- A small append-only store under `outputs/state/` of `(stock_id, product_id, used_at)`, written when a render finishes, outside the product directory so cleanup does not remove it.
- Before judging, drop candidates used within `stock_reuse_window` renders (default 30). If fewer than the needed count remain, fill from the dropped set, least recently used first, and log it.
- `stock_reuse_guard.enabled` (default false). It changes which clips the topic arm shows, and the protocol holds each arm's visuals constant, so it ships off until the readout. The id store can record from day one, so the window is full when the guard is switched on.

**Tests.** A candidate used within the window is excluded; the fallback fills from the least recently used; the store survives a product directory's deletion; off records ids but leaves the candidate pool as today.

## #556 Normalise numbers, units and model names before TTS

**Today.** The sanitised script goes to the voice unchanged.

**Design.**
- A probe (a sibling of `tools/tts_tag_probe.py`) voices a fixed list of strings (`5000mAh`, `65W`, `2.4 GHz`, `1.83-inch`, `USB-C`, `IP68`, a SKU) and records the transcript, so misreadings are measured before anything is rewritten.
- A `tts_normalisation` table in config: unit spellings applied only after a number (`mAh` to "milliamp hours", `W` to "watts", `GHz` to "gigahertz"), decimal and range handling, and a small lexicon for brand and model terms. Applied in `TTSManager.generate_speech` to the text sent to the provider only; the script file, captions and state keep the written form, and captions come from Whisper on the audio, so spoken forms appear there as heard.
- Only entries the probe shows are misread go in the table.

**Tests.** Each table entry rewrites its fixture and leaves unit letters inside ordinary words alone ("Watch" stays "Watch"); the script file is unchanged; off sends today's text.

## #557 Evaluate a distinctive or owned narrator voice

An evaluation, not a feature. Compare the available voices, including lower-pitched ones, with the #439 pauses and the #545 chain, in a blind listening test on three scripts. Separately, record whether a clone of the operator's own voice is possible through the TTS providers in use, its cost, and each platform's rule for it (YouTube exempts an owned-voice clone from disclosure). The output is a recorded decision; a voice change waits for the readout.

## #558 Revisit the TikTok AI label

**Today.** `tiktok_settings.video_made_with_ai` is on for every post, and `docs/compliance.md` gave AI voiceover as the reason. TikTok's 2026-H2 guidelines exempt generic TTS narration.

**Design.**
- Correct the compliance row (done in this PR as a pending-correction note) and record the policy decision: keep the label on voluntarily, or turn it off, with the reason.
- Optionally add a bounded AI-role statement (for example "Voiced with AI. Researched and edited by a person.") to the profile bio or the caption template, behind a config key, since research found such a statement removes the penalty a bare label creates.
- Inspect a rendered file with `exiftool` or a C2PA reader for SynthID or C2PA metadata carried through from the TTS audio, and record whether it survives the mux.

**Reach test.** The label changes reach for both arms, so the config change waits for the readout.

