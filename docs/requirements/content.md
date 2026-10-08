# Content requirements

Ids use the prefix `REQ-CNT`. The format and the statuses are described in [the requirements index](README.md).

## AI services

- **REQ-CNT-001** `shipped` The producer generates text with Gemini as the primary provider and falls back to OpenRouter when every Gemini model fails.
- **REQ-CNT-002** `shipped` When the producer falls back to OpenRouter, it discovers the free models and excludes blocklisted models, models below a minimum context length, models that do not output text and models that reason in their output by default.
- **REQ-CNT-003** `shipped` The config sets the LLM retry policy, the validation thresholds and the model blocklist.
- **REQ-CNT-004** `shipped` The producer gives each configured model a bounded number of attempts before it moves to the next model.
- **REQ-CNT-005** `shipped` If LLM calls fail repeatedly, a circuit breaker stops further calls for a configured cool-down.
- **REQ-CNT-006** `shipped` If a generated video description is longer than `description_validation.max_chars` (default 900), the producer rejects it as model reasoning and tries the next model.
- **REQ-CNT-128** `shipped` `llm_settings.thinking_budget` sets the Gemini reasoning budget for every text call except the script fact check, and the bundled value 0 turns reasoning off.
  - Why: the fact check is the one reasoning-shaped call, so aligning it with the global budget weakens it.
- **REQ-CNT-129** `shipped` Where `random_model_selection` is on, the producer tries the discovered free OpenRouter models in random order; off (the bundled value), it tries the configured models first, in order, then the other discovered ones.
- **REQ-CNT-162** `planned #703` Script generation uses its own configured model, thinking budget and output limit, apart from the model the other text calls use, and the run state records which model wrote the script.

## Script templates

- **REQ-CNT-007** `shipped` The producer ships several script templates with distinct styles (curiosity hook, problem-solution, storytelling, comparison and others).
- **REQ-CNT-008** `shipped` Templates instruct calm, conversational delivery, with no high-energy, hype or clickbait phrasing.
- **REQ-CNT-142** `shipped` The product and topic narrator profiles instruct the LLM to target 30-40 seconds of speech at a normal pace, about 75-100 words.
- **REQ-CNT-163** `shipped` The narrator profiles take their spoken-length target (a seconds range and a words range) from `script_templates.target_length` per content type, overridable per video profile; the defaults render the profile text byte-identical to the shipped prompts.
- **REQ-CNT-009** `shipped` The producer selects a template per product deterministically, so a product gets the same template on every run.
- **REQ-CNT-010** `shipped` Where `script_templates.template_pool` lists templates, the producer selects only from them; an empty pool means every template.
- **REQ-CNT-011** `shipped` When `--script-template <name>` is passed, the producer uses that template for the run.
- **REQ-CNT-012** `shipped` The producer records the chosen template in `pipeline_state.json`.
- **REQ-CNT-013** `shipped` A topic render draws its template only from `script_templates.topic_templates`, and a product render never draws a topic template.
- **REQ-CNT-161** `planned #701` A topic whose title is a task ("How to ...") is written with a task-first template; the symptom-first and mistake-first templates are drawn only for a topic that names a symptom or a mistake.
- **REQ-CNT-014** `shipped` A topic render uses the topic narrator profile (`script_templates.narrator_profile_topic`) and the topic call-to-action list.
- **REQ-CNT-015** `shipped` The producer does not shorten a topic title with the product-alias heuristic.

## Product script rules

- **REQ-CNT-016** `shipped` Every product template instructs the LLM to open with a conversational hook that carries the long-tail search keyword (product category, price band, audience cue or pain point) within the first five seconds of speech.
- **REQ-CNT-017** `shipped` The product hook rule lists six hook patterns (price-first reveal, regret or contrarian, POV, outcome-first, numbered teardown, comparison) and names the literal search-query shape as an anti-pattern.
- **REQ-CNT-018** `shipped` Every product template instructs the LLM to state a concrete fact, result or observation about the product in the first line, and names setup framings ("Today I'll show you", "In this video") as anti-patterns.
- **REQ-CNT-019** `shipped` Every product template instructs the LLM to end the script with one short closing question or claim right before the call to action.
- **REQ-CNT-020** `shipped` Personal and storytelling templates close on a two-option opinion question; analytical and comparison templates close on a debatable but defensible spec claim.
- **REQ-CNT-021** `shipped` Where the product description states a measurement that can be quoted verbatim with its unit as a whole word, analytical templates close on a claim about that measurement.
  - Why: the spec branch gives no worked closing line, because a worked example gets copied onto products it does not fit.
- **REQ-CNT-022** `shipped` Where the description states no such measurement, analytical templates close on a material, shape or use claim that carries no numbers.
- **REQ-CNT-023** `shipped` Every product template instructs the LLM to include one trade-off or limitation of the product, one sentence at most.
- **REQ-CNT-024** `shipped` Scripts name the product the way a person says it aloud and never speak model or SKU designations.
- **REQ-CNT-025** `shipped` The template offers the short product alias as a suggestion, and the script uses the plain category noun when the alias does not read as a spoken name.
- **REQ-CNT-157** `planned #700` A product script never claims that the narrator owned, bought, received, used or tested the product, nor that other people talk about it; the first person speaks only for research and opinion.
- **REQ-CNT-158** `planned #700` A product script speaks no price, and no template example or hook pattern models one.
- **REQ-CNT-159** `planned #700` No script template or narrator profile quotes a whole example sentence a script could reuse; an example shows its shape with placeholders.
- **REQ-CNT-160** `planned #700` A product script says in one sentence who the product suits or who should skip it, beside its one trade-off.

## Topic script rules

- **REQ-CNT-026** `shipped` Every topic template instructs the LLM to state the fix within the first three seconds.
- **REQ-CNT-027** `shipped` Every topic template instructs the LLM to speak the search phrase within the first five seconds.
- **REQ-CNT-028** `shipped` Every topic template asks for one instruction per sentence.
- **REQ-CNT-029** `shipped` Every topic template forbids inventing a product to recommend.
- **REQ-CNT-030** `shipped` A topic script gives a menu path or a URL only when it can state it exactly (the labels in order, or the address in full) for a platform it names; otherwise it gives an observable on the device or says that it differs by device.
- **REQ-CNT-031** `shipped` A topic script closes on the result rather than on a debatable spec claim.
- **REQ-CNT-032** `shipped` A topic script carries one honest limit, placed among the steps rather than after them.
  - Why: a limit read last becomes the final spoken line before the call to action and leaves the viewer unsure the fix worked.
- **REQ-CNT-146** `held` Where step lists are on, a topic script names the app or settings screen where the steps start, before the first step.
  - On when: `llm_settings.topic_scripts.step_list.enabled` is set, on the same condition as REQ-VID-121.
- **REQ-CNT-147** `partial` A topic script names the most common mistake at the step where it happens.
  - Gap: only with step lists on (held) is the mistake named at its step; the free-form topic templates leave it out, or name it in the opening line as `topic_mistake_fix` does. Turning step lists on closes it.

## Call to action and closing line

- **REQ-CNT-033** `shipped` Every script, product or topic, ends on exactly one configured call to action, verbatim, as its final sentence (`script_templates.cta_options` for products, `cta_options_topic` for topics).
- **REQ-CNT-034** `shipped` The producer chooses one call to action per record, reproducibly, and renders only that line into the template rules.
- **REQ-CNT-035** `shipped` If a script's last sentence is not a configured call to action, validation rejects the script and the producer retries.
- **REQ-CNT-036** `shipped` If every attempt misses the call to action, the producer ships the first otherwise complete script with the chosen line appended, or substituted for a last sentence that reads as a paraphrased call to action, and logs a warning.
- **REQ-CNT-037** `shipped` When `--cta <line>` (or `script_templates.fixed_cta`) names a configured option, every record closes on that line; a line that is not configured is ignored.
- **REQ-CNT-038** `shipped` The producer records the chosen call to action and any spoken sign-off in `pipeline_state.json`.
- **REQ-CNT-039** `shipped` The per-platform caption generator places the script's closing line in the caption body, before the hashtag block.
- **REQ-CNT-040** `shipped` When no script is available, the caption falls back to the platform's standard search-optimised content with no closing line.
- **REQ-CNT-041** `planned #549` No configured call to action or closing-line example asks viewers to share, tag, vote, reply with a specific word or emoji, or follow for a promised payoff; closing questions ask for a choice or an experience.
  - On when: the call-to-action pools are edited after the reach-test readout (#540).
- **REQ-CNT-148** `planned #549` Every configured call to action uses an imperative verb and names an outcome or a destination.

## Script naturalism and signature

- **REQ-CNT-042** `shipped` When `script_templates.naturalism.intensity` is 0, the script prompt is unchanged.
- **REQ-CNT-043** `held` Where `script_templates.naturalism.intensity` is 1 or 2, scripts use contractions, spoken fillers and one emotional beat, and at 2 also one self-correction of wording.
  - On when: `script_templates.naturalism.intensity` is raised once naturalism is re-measured (#541) and after the reach-test readout, in stages (#540).
- **REQ-CNT-044** `held` Where naturalism is on, filler never lands in the first sentence, inside a number, name or claim, or in or after the call to action, and uses words the captions carry rather than sounds speech recognition drops.
  - On when: `script_templates.naturalism.intensity` is raised once naturalism is re-measured (#541) and after the reach-test readout, in stages (#540).
- **REQ-CNT-045** `held` Where the author signature is enabled, a script carries an opener of at most five words that starts the first sentence and runs into it with no comma, one transition where the script turns, and a sign-off right before the call to action; a tutorial written from a step list signs off after the recap of the path.
  - On when: `script_templates.signature.enabled` is set after the reach-test readout, topic arm first (#540, [design 0022](../design/0022-signature-lines.md)).
- **REQ-CNT-046** `held` Each signature element is drawn per render, reproducibly, from the render's own pools (`signature.topic` for a topic, `signature.product` for a product) below that arm's `use_rate`, so the signature recurs without appearing in every render.
  - On when: `script_templates.signature.enabled` is set after the reach-test readout (#540).
- **REQ-CNT-047** `held` Where a sign-off is spoken, the first comment and the platform captions quote the closing line, not the sign-off.
  - On when: `script_templates.signature.enabled` is set after the reach-test readout (#540).
- **REQ-CNT-153** `held` Where a signature pool lists `opener_templates`, an opener is drawn only for a script written from one of those templates; the bundled topic pool lists the templates that open on the task (`topic_from_steps`, `topic_answer_first`), so a symptom-first or mistake-first script keeps its opening.
  - On when: `script_templates.signature.enabled` is set after the reach-test readout (#540).
- **REQ-CNT-152** `shipped` While `script_templates.signature.enabled` is false, no signature line is drawn and the script prompt is unchanged, whatever the pools hold.

## Script checks

- **REQ-CNT-048** `shipped` If a generated script does not end with terminal punctuation, the producer treats it as truncated and retries.
- **REQ-CNT-049** `shipped` When a script is generated, the producer checks its falsifiable claims: a topic script with one grounded web search, a product script against the scraped title and description, prices excluded.
- **REQ-CNT-050** `shipped` When the check flags claims, the producer revises at most `script_fact_check.max_flags_to_revise` sentences (default 3) and refuses a revision whose length differs from the original by more than `script_fact_check.max_length_drift` (default 25%).
- **REQ-CNT-051** `shipped` If the fact check or the revision fails for any reason, the producer ships the original script.
- **REQ-CNT-151** `shipped` When every flagged claim's fix asks for its removal, the producer deletes those sentences without a rewrite, provided they are exactly the claims and no following sentence leans on them; otherwise the reviser repairs them, and a rewrite that repeats a removed claim's subject words is refused.
- **REQ-CNT-154** `shipped` The fact check refuses a revision that adds a sentence carrying fix wording a flagged claim's fix used ("instead of", "is located under", "is located in", "can be found under", "is found under") or that says a sentence twice where the original said it once; a fix that answers a claimed limit (a claim saying "limit" or "limited", or a number after "up to", "at most", "more than", "over", "fewer than", "less than" or "maximum of") with a universal ("regardless of", "no matter how", "no matter what", "any number of", "unlimited", "no limit") is carried out as a removal of the claim, and a revision that brings such a universal back, other than one another flag's fix uses, is refused.
- **REQ-CNT-155** `held` Where `llm_settings.script_validation.reject_copied_examples` is on, a generated script is retried when a sentence that is not a question, and not a configured call to action or one of its sentences, shares at least three, and at least three fifths, of the content words of a quoted example in its own prompt and some of those words appear nowhere in the product's title, description or keyword; when every attempt does so, the script ships without that sentence, and without a question just before it, if the rest passes validation.
  - On when: the reach-test readout (#540); measure first how often a quoted prompt example is still copied; the narrator profile's anecdote, the most copied one, was removed in 0.184.1 (REQ-CNT-156).
- **REQ-CNT-156** `partial` The narrator profile asks for one concrete detail taken from the product description and forbids inventing a personal moment, a day, a trip or a test with the product, and no script template models such a moment as an example.
  - Gap: the product narrator profile's voice example still models one ("So I picked this up last month ... Took it on a hike and never lost signal") (#700).
- **REQ-CNT-052** `shipped` Before TTS, the producer removes speaker labels, parenthetical stage directions, markdown (code-span backticks and link targets included), emojis and hashtags from the script, and unwraps square brackets, keeping the words inside; a generated script carrying bracketed text that neither the listing nor a step's UI path contains is retried, and ships only when every attempt carries some.
- **REQ-CNT-053** `held` Where script lint is enabled, the producer rejects a script that uses common machine-writing phrases, exceeds a sentence-length cap or exceeds a word count derived from the target duration, and retries; the script prompt states the sentence cap and, outside a tutorial, the word count.
  - On when: `script_validation.lint.enabled` is set after the reach-test readout, once rejection rates on a batch stay low and the scripts read better on review.
- **REQ-CNT-054** `held` Where the hook rules are enabled, the hook headline and every platform caption lead with the search phrase.
  - On when: `script_templates.hook_rules.enabled` is set after the reach-test readout (#540).
- **REQ-CNT-055** `shipped` A report shows, per render, whether the search phrase appears in the first spoken sentence, the hook headline and the start of each platform caption.

## Voice profiles

- **REQ-CNT-056** `shipped` The producer synthesises speech with Gemini TTS or Google Cloud TTS.
- **REQ-CNT-057** `shipped` The config defines named voice profiles, each with style direction, voice preferences and text markup rules.
- **REQ-CNT-058** `shipped` A voice profile can direct tone, energy and pacing.
- **REQ-CNT-059** `shipped` The default voice profile delivers calm, confident speech rather than high energy.
- **REQ-CNT-060** `shipped` A voice profile can set speaking rate and pitch.
- **REQ-CNT-061** `shipped` Where a profile has markup rules, the producer inserts pause tags at sentence boundaries (periods, exclamation marks and question marks).
- **REQ-CNT-143** `shipped` Where `silence_removal_enabled` is on (the bundled default), the producer trims only the leading and trailing silence below `silence_threshold_db` (bundled -50 dB) from the voiceover before transcription, and `silence_min_duration_sec` defaults to 0.1 s.
  - Why: the trim discards the audio inside that window, so a longer value cuts off a short final word.
- **REQ-CNT-062** `held` Where the selected voice profile has a `pause_plan`, the producer places no pause after the opening hook, longer pauses at paragraph breaks and before the closing line, and a reproducible per-product variation elsewhere.
  - On when: `default_voice_profile` or `voice_profile_pool` selects a profile with a `pause_plan` (such as `charon_varied`) after the reach-test readout, in stages (#540).
- **REQ-CNT-063** `shipped` The config rejects a pause plan that uses a tag not measured as silent, so no tag text reaches the captions.
- **REQ-CNT-130** `shipped` A probe tool measures which inline TTS tags a voice honours silently: one TTS call and one WAV file per tag, and a printed table of the gap each tag adds and any tag text the transcript carries.
- **REQ-CNT-064** `shipped` The producer selects a voice profile per product deterministically, so a product gets the same voice on every run.
- **REQ-CNT-065** `shipped` The producer resolves the voice profile in this order: `--voice-profile`, a draw from `voice_profile_pool`, the pinned `default_voice_profile`, a draw from all profiles.
- **REQ-CNT-066** `shipped` Where `default_voice_profile` is set and `voice_profile_pool` is empty, every render uses the pinned profile.
- **REQ-CNT-067** `shipped` Where `voice_profile_pool` lists profiles, selection is restricted to them.
- **REQ-CNT-068** `shipped` When `--voice-profile <name>` names a configured profile, the render uses it; an unknown name logs a warning and selection continues down the order.
- **REQ-CNT-069** `shipped` The producer records the voice profile name and the selected voice per render.
- **REQ-CNT-070** `shipped` The producer tries the voice profile's own provider first, then each provider in `provider_order`.
- **REQ-CNT-071** `shipped` When a Gemini voice fails, the producer strips the inline markup tags before it sends the text to a fallback provider, so no tag is spoken.
- **REQ-CNT-072** `shipped` Google Cloud voice choice falls back through ranked voice families (by default Chirp3, then Chirp, then Neural2, then any en-US voice).
- **REQ-CNT-131** `shipped` Where the Coqui TTS package is installed and `coqui` is listed in `tts_config.provider_order`, the producer can synthesise speech locally with Coqui TTS; the bundled order leaves it out.
- **REQ-CNT-073** `held` Where `audio_settings.voice_chain.enabled` is true, the producer treats the voiceover with filtering, gentle compression, de-essing and limiting before the mix, without changing its loudness target or its transcript.
  - On when: `audio_settings.voice_chain.enabled` is set after the reach-test readout (#540), once a voice-by-chain comparison over at least 20 posts per cell shows no loss.
- **REQ-CNT-074** `shipped` The producer records per render whether the voice chain was on, beside the voice name.
- **REQ-CNT-075** `held` Where TTS normalisation is enabled, numbers, units and model names the voice misreads are rewritten to speakable words in the text sent to TTS only.
  - On when: `tts_config.tts_normalisation.enabled` is set, with table entries, once `tools/tts_normalisation_probe.py` shows the voice misreading a string; on the pinned voice it found none.
- **REQ-CNT-076** `held` Where TTS normalisation is enabled, the script file and state keep the written form, and the captions show what the voice said.
  - On when: with REQ-CNT-075.

## Background music

- **REQ-CNT-077** `shipped` The music chain tries each provider configured in `audio_providers` in order and falls back to local stock files.
- **REQ-CNT-078** `shipped` The default chain is Jamendo, then Freesound, then local stock files.
- **REQ-CNT-079** `shipped` The producer requests tracks whose duration matches the voiceover length.
- **REQ-CNT-080** `shipped` Each music provider has its own circuit breaker.
- **REQ-CNT-081** `shipped` The producer records the chosen track's attribution: source, author, license URL and track id.
- **REQ-CNT-082** `shipped` A candidate track is downloaded once within a bounded timeout, and a failed download moves to the next candidate rather than retrying the same URL.
- **REQ-CNT-083** `shipped` The music step has a total time budget below the pipeline's per-step warning threshold; when the budget is spent, the chain falls back to local stock files.
- **REQ-CNT-084** `shipped` Where a provider ranks by popularity or rating alone, the producer accepts a candidate only when the query's terms match the track's tags or title.
- **REQ-CNT-085** `shipped` A query term matches any word it begins ("chill" matches "chillout"), and a provider that drew its own query is judged against that query.
- **REQ-CNT-086** `shipped` If no candidate from a provider matches the mood, the chain moves to the next provider rather than accept a mismatch.
- **REQ-CNT-087** `shipped` The audio summary names the terms the chosen track matched.

## Jamendo

- **REQ-CNT-088** `shipped` The Jamendo provider authenticates with a client id only, without OAuth2.
- **REQ-CNT-089** `shipped` The Jamendo provider searches in `fuzzytags` mode by default (any tag matches), and `search_mode` can switch it to `tags` (all tags match) or `search` (free text).
- **REQ-CNT-090** `shipped` Jamendo searches request instrumental tracks within the duration window, ordered by the month's popularity.
- **REQ-CNT-091** `shipped` If Jamendo returns no tracks, the provider retries a bounded number of times with a fresh query each time, and reports no tracks only after a run of empty answers.
- **REQ-CNT-092** `shipped` Where `search_queries` lists queries, the Jamendo provider draws one at random for each search attempt.
- **REQ-CNT-093** `shipped` The Jamendo provider prefers the track's download URL and falls back to the stream URL when download is not allowed.

## Freesound

- **REQ-CNT-094** `shipped` The Freesound provider uses OAuth2 for full-quality downloads and falls back to API-key previews.
- **REQ-CNT-095** `shipped` The Freesound provider refreshes the access token `freesound_token_refresh_buffer_sec` (default 60 seconds) before it expires.
- **REQ-CNT-096** `shipped` When the Freesound provider refreshes its token, it writes the rotated refresh token to `.env` in place of the old one.
  - Why: the service invalidates the old refresh token on use.
- **REQ-CNT-097** `shipped` If a token refresh fails, the provider attempts it once per run, logs one warning with the fitting remedy (the OAuth2 setup tool for a rejected token, none for an unreachable endpoint) and uses previews for the rest of the run.
- **REQ-CNT-098** `shipped` A setup tool guides the manual Freesound authorize step and writes the refresh token to `.env`.
- **REQ-CNT-099** `shipped` The Freesound provider searches with a duration filter and falls back to a general search.

## Final mix

- **REQ-CNT-100** `partial` The producer masters the final mix to a loudness target (default -14 LUFS, true peak -1 dBFS).
  - Gap: delivered files measure about -15 LUFS and peak at about -0.8 dBFS, because the true-peak ceiling forces dynamic normalisation and the AAC encode adds about 0.2 dB after it (#658).
- **REQ-CNT-101** `shipped` Where `music_ducking_enabled` is true, the music level drops while narration plays and recovers in the gaps; it is off by default.
- **REQ-CNT-132** `shipped` The mix plays the voiceover at `voiceover_volume_db` and the music at `music_volume_db` (bundled +3 dB and -24 dB).
- **REQ-CNT-133** `shipped` The music fades in over `music_fade_in_duration` (bundled 2 s) and out over the last `music_fade_out_duration` (bundled 3 s) of the video, or within `peak_margin_sec` where the ending is `peak` or `loop`.
- **REQ-CNT-134** `shipped` Where the ending is `outro` (the default), the video runs `outro_duration_sec` (bundled 1 s) past the end of the voiceover, so the music fade ends after the last spoken word.

## Captions and metadata

- **REQ-CNT-135** `shipped` The producer generates each render's social title, description and hashtags in one of two modes set by `description_settings.metadata_mode` or `--metadata-mode`: `unified` (the bundled value), one set for every platform, or `optimized`, one set per platform.
- **REQ-CNT-136** `shipped` In unified mode, the producer writes `metadata.json` with the listing title (for a product with `short_product_titles` on, the short title of REQ-PUB-008), an AI description with any hashtags removed, and hashtags derived from the title.
- **REQ-CNT-137** `shipped` In optimized mode, the producer writes `metadata_<platform>.json` for each platform enabled in `description_settings.platform_metadata` (`<platform>.enabled`), with an AI title, caption and hashtags within that platform's length and hashtag limits.
- **REQ-CNT-138** `shipped` If optimized mode produces metadata for no platform, or `platform_metadata.enabled` is false, the producer falls back to unified mode.
- **REQ-CNT-139** `shipped` In optimized mode, the producer also writes `UPLOAD_INSTRUCTIONS.txt` with each platform's metadata for manual upload; a failure to write it doesn't fail the step.
- **REQ-CNT-140** `shipped` When metadata from a previous run exists for the product, the producer reuses it instead of generating it again.
- **REQ-CNT-141** `shipped` Where `description_settings.enabled` is false, the producer skips metadata generation.
- **REQ-CNT-144** `shipped` In optimized metadata mode, a topic render's prompts ask for a YouTube title that front-loads the symptom, in the words a viewer would search, within its first 5-7 words, and for TikTok and Instagram captions that contain the search phrase.
- **REQ-CNT-145** `shipped` A topic render's description prompts ask for a description that leads with the symptom, and in optimized metadata mode the YouTube prompt asks for the symptom in the first sentence.
- **REQ-CNT-149** `held` Where `description_settings.short_product_titles` is on, a product video's title is built from its hook headline, so the two make the same promise.
  - On when: with REQ-PUB-008, at the reach-test readout (#540).
- **REQ-CNT-150** `planned #590` A product video's YouTube description carries no destination URL, and its call to action points at the profile link.

## Content pillars

- **REQ-CNT-102** `shipped` Pillars are named themes that group keywords and script templates; a keyword or a template can sit under more than one pillar.
- **REQ-CNT-103** `shipped` Pillars are the keys of `batch.keywords` in `config/scraper.yaml` and of the `script_templates` pillar maps (`pillars`, `pillar_preambles`, `pillar_audiences` and their `_topic` variants) in `config/ai_services.yaml`; the bundled config ships `value`, `novelty` and `utility`.
- **REQ-CNT-104** `shipped` Users can rename, add or remove pillars in config without code changes.
- **REQ-CNT-105** `shipped` Where `batch.keywords` is a map keyed by pillar, each scraped product carries its source keyword's pillar to the producer.
- **REQ-CNT-106** `shipped` `batch.keywords` also accepts a flat list, which attaches no pillar.
- **REQ-CNT-107** `shipped` If a keyword is listed under more than one pillar, config loading fails with an error naming the keyword and both pillars ([decision 0012](../decisions/0012-a-keyword-belongs-to-one-pillar.md)).
- **REQ-CNT-108** `shipped` A keyword passed on the command line carries its configured pillar, and a keyword not in the config carries none.
- **REQ-CNT-109** `shipped` A template can be listed under several pillars in `script_templates.pillars`.
- **REQ-CNT-110** `shipped` A render's pillar is `--pillar` when passed, else the pillar an earlier run recorded, else the product's keyword pillar.
- **REQ-CNT-111** `shipped` Where a product render has a pillar, the producer selects deterministically from that pillar's templates instead of the full pool.
- **REQ-CNT-112** `shipped` `--pillar <name>` narrows the product template pool, sets the prompt preamble and sets the audience hint for the run.
  - Why: a pillar shapes the script only; it does not filter which products a run includes, and nothing balances products across pillars.
- **REQ-CNT-113** `shipped` The producer and the batch both accept `--pillar`.
- **REQ-CNT-114** `shipped` The script prompt stacks, in order: the narrator profile (`script_templates.narrator_profile`), the pillar preamble when a pillar is set (`pillar_preambles`, or `pillar_preambles_topic` on a topic render), then the template with the product data.
- **REQ-CNT-115** `shipped` The platform caption generators (YouTube, TikTok, Instagram) receive the same narrator profile and pillar preamble as the script.
- **REQ-CNT-116** `shipped` When a pillar is set, the `{AUDIENCE}` placeholder takes `script_templates.pillar_audiences[pillar]` (or `pillar_audiences_topic[pillar]` on a topic render) instead of `target_audience`.
- **REQ-CNT-117** `shipped` If the pillar's audience entry is missing or empty, `{AUDIENCE}` falls back to `target_audience`.
- **REQ-CNT-118** `shipped` If `--pillar` names a pillar configured in none of the pillar maps, the run logs an info-level hint listing the configured pillars and continues with no template filter, preamble or audience override.
- **REQ-CNT-119** `shipped` Whenever the producer generates a script, it writes the full prompt to `outputs/<id>/temp/script_prompt.txt`.
- **REQ-CNT-120** `shipped` The producer records the render's pillar in `pipeline_state.json`, and a resumed run keeps it.
- **REQ-CNT-121** `shipped` Subtitle styling and the TTS voice are the same for every pillar.
- **REQ-CNT-122** `shipped` The bundled pillar preambles frame a `value` video around the deal, a `novelty` video around discovery and a `utility` video around the problem and its solution.
- **REQ-CNT-123** `shipped` The bundled audience hints target budget-conscious shoppers for `value`, curious early discoverers for `novelty` and practical problem-solvers for `utility`.

## Prompt hygiene

- **REQ-CNT-124** `shipped` The producer Unicode-normalises product titles and descriptions before they enter a prompt, folding mathematical-alphabet bold characters to plain ASCII.
- **REQ-CNT-125** `shipped` The producer replaces em dashes in the description with commas and en dashes with hyphens before prompting.
- **REQ-CNT-126** `shipped` Templates receive both the full product title (`{FULL_PRODUCT_NAME}`) and a short alias of a few words taken from the listing title (`{SHORT_PRODUCT_NAME}`).
- **REQ-CNT-127** `shipped` The narrator profile instructs the LLM to paraphrase a feature in its own words when the description carries a banned phrase or marketing fluff, rather than quote it.
