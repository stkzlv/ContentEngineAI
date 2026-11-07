# 0031. Narrator identity

- **Status:** Implemented
- **Issue:** #783
- **Requirements:** REQ-CNT-168, REQ-CNT-169, REQ-CNT-170

## Context

A recurring narrator is recognised by the same voice, sign-off, sound and caption look in every video. The pipeline already pins the voice (REQ-CNT-066) and can play a sting (`audio_settings.signature_sting`). The rest is drawn per render on purpose:

- signature lines recur below a use rate, so they don't appear in every render (REQ-CNT-046);
- music comes from a tag search, and caption templates from a pool;
- the variety report alerts when one value dominates a dimension (REQ-PUB-083, design 0006), so a pinned element would raise an alert on every report.

None of these settings knows the content kind, and the voice, the sting and the music are global.

## Goals

- One identity per content kind (`product`, `topic`, `commentary`) that pins the voice, a sign-off spoken on every render, the sting, a music mood or file list, and one caption template.
- Two renders of a kind on different subjects share every pinned element.
- The variety report lists a pinned dimension as pinned and doesn't alert on it, while it still alerts on the rest.
- The bundled config leaves every kind empty; an installation sets its values in its overlay (decision 0017).

## Non-goals

- Choosing an installation's identity. Those values are taste, and they live in the overlay.
- A per-kind video profile. The render profile still comes from the profile pools.

## Design

- **The block.** `narrator_identity` in `config/video_production.yaml` (`NarratorIdentities`, `src/video/config/identity_models.py`) has one `NarratorIdentity` per kind. Every field is optional. `content_kind(product)` reads the record: `commentary` for a topic with `topic_kind: commentary`, `topic` for any other topic, `product` otherwise.
- **Where it applies.** `config_for_record`, the per-record config copy that commentary topics already use (design 0030), writes the kind's pins into the copy. The shared config stays untouched for the next record. A record whose kind pins nothing, and isn't commentary, gets the config itself. Each pin writes the setting its consumer already reads:
  - `voice_profile` sets `tts_config.default_voice_profile` and empties `voice_profile_pool`. `--voice-profile` still wins, because the TTS manager checks the override first.
  - `signoff` sets the arm's new `pinned_signoff`.
  - `sting` replaces `audio_settings.signature_sting`.
  - `music_files` replaces `background_music_paths` and switches every entry in `audio_providers` off. The list isn't emptied, because an empty list makes the music step build a legacy Freesound provider. Only the beds are then drawn from.
  - `music_queries` replaces each provider's `search_queries` and the Freesound query.
  - `caption_template` sets the pycaps `template_name` and empties `template_pool`. A video profile's own pycaps template still merges over it.
- **The sign-off on every render.** The sign-off must be one sentence with words, because the script step places it as one sentence. `select_signature` returns the pinned sign-off whatever `enabled` and `use_rate` say. It still makes every draw, so pinning moves no other element. The prompt already asks for the drawn sign-off word for word. `place_signoff` gained `insert`: when the sign-off is pinned and the model left it out, it is added right before the closing CTA. A sign-off folded into another sentence counts as said and is not added again. An unpinned sign-off is never inserted, which keeps REQ-CNT-046's "not every render".
- **The report.** Each render-choices row records `content_kind` and `pinned`, the dimensions the identity fixes to one value: the voice, the caption template, and the music when there is exactly one file. The report writes those down when they are recorded, so it needs no config and older rows stay truthful. A dimension's dominance alert leaves out the rows that pin it. A pinned kind beside unpinned ones then alerts only on what the unpinned ones drew. The distribution line notes `(pinned in N)`.

## Alternatives considered

- **Per-kind settings threaded into each chooser** (TTS, music, captions, signature). Five consumers would each need the kind passed down. The per-record copy reuses the settings they already read.
- **The report reading the current config.** That would make old rows claim pins they never had, and the report runs without a config today.

## Rollout

No switch: an empty identity changes nothing, and that is the bundled default. An installation fills in its overlay to turn it on.

## As built

A commentary render ran with a probe identity (voice, sign-off, caption template, music query) in the overlay. The script ended on the sign-off and then the CTA. The TTS used the pinned voice. pycaps used the pinned template. The music search used the pinned query. The choices row recorded `commentary` with the voice and caption template as pinned.

## Open questions

None recorded.
