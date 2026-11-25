# 0040. Performed voice delivery aligned with the script

- **Status:** Accepted
- **Issue:** #855
- **Requirements:** REQ-CNT-179, REQ-CNT-180, REQ-CNT-181, REQ-CNT-182, REQ-CNT-183, REQ-CNT-184, REQ-CNT-185

## Context

The voice reads every line the same way. A profile sets one style for the whole script, and the pause plan (REQ-CNT-062) varies only the gaps between sentences. Lines that call for a dry aside, a pause before a reveal, or a short laugh at an absurd number sound the same as the rest.

Three findings shape the design:

- **The profile's style prompt never reaches the voice.** `src/video/tts.py` built `SynthesisInput(text=...)` and nothing else, because an earlier attempt had the prompt read aloud. Probes on `gemini-2.5-flash-tts` show it depends on the prompt's form: a descriptive prompt in the request's `prompt` field was read aloud in four of four takes, an instruction ending in a colon in none of three (C: our probes, small samples). REQ-CNT-185 sends only the instruction form (grade A for the field itself, from Google's Cloud TTS documentation).
- **Gemini 3.8 Flash TTS takes a style per request and a wider tag set.** It is served on the Gemini API's Interactions endpoint, not on Cloud TTS. A request carries the text with a `speech_metadata` annotation holding a natural-language style, and the text may contain inline non-verbal tags in angle brackets: `<sigh>`, `<laugh>`, `<breath>`, `<gasp>`, `<cough>`, `<throat-clearing>`, `<short pause>`, `<long pause>`. The model reads text strictly as a transcript, so any direction left in the text can be spoken (grade A, Google's Gemini API speech documentation). It bills output audio tokens, a few tenths of a cent per take at the listed rate.
- **A non-verbal sound can reach the captions as a word.** In a three-way listening probe (today's voice, 2.5 with tags and the prompt field, 3.8 with tags and a style per sentence group), Whisper transcribed a `<sigh>` on 3.8 as "ugh". The 2.5 take's tags were not transcribed. Both scripts otherwise matched their transcripts (one take each, grade C).

Tags and styles help only when they fit what the line says. A generic rule ("a breath every third sentence") puts a laugh on a serious line. So the delivery is planned from the script text, line by line, and checked against it.

## Goals

- A delivery plan per script that sets a style for each group of sentences and places a few vocal events, each justified by the words it sits on.
- A plan that can't drift from the script: its text equals the script word for word, and every event sits inside its segment.
- A provider for Gemini 3.8 Flash TTS that voices the plan segment by segment.
- Captions that show only the words, never a vocal event heard as a word.
- The style prompt on the existing Gemini voice actually reaching it.

## Non-goals

- Changing the script, its wording or its approval. The plan only adds delivery.
- Events inside a word, or sound effects mixed under the voice (the effects layer is separate).
- Replacing the pause plan on profiles that don't use the delivery plan.
- A second vendor. ElevenLabs v3 is an alternative, below.

## Design

### Delivery plan

A JSON file beside the script, `delivery_plan.json`:

- `segments`: an ordered list. Each segment has:
  - `text`: one or more whole sentences of the script, verbatim.
  - `style`: a short direction written from those lines ("dry, amused, slower on the number"). It describes how to read them, never what to say.
  - `events`: zero or more `{ "tag": "<sigh>", "before_word": 3 }` entries. `before_word` is the index of a word in `text`; the tag goes before it. `"after_last": true` places it at the segment's end instead.
  - `gap_after_ms`: the silence before the next segment.
- `version`: the schema version.

### Alignment check

Validation rejects a plan before TTS when:

- the segments' `text`, joined with single spaces, differs from the script after whitespace normalisation;
- a segment starts or ends inside a sentence;
- an event's index falls outside its segment, or its tag isn't in the measured allowlist;
- the plan has more events than `max_events` (bundled 4 per take), or more than one per segment;
- a `style` contains a quoted phrase from the script, which signals the direction is restating the text.

### Where the plan comes from

The script step writes the plan with one extra model call after the script passes its checks. The prompt gives the script and the event allowlist, and asks for groups that change delivery only where the meaning changes, a style per group from its lines, and events only where a line earns one (an absurd number, a turn, an aside). Most segments carry none. A plan that fails validation is retried once and otherwise dropped: the render falls back to the single-style path.

When a run stops after the script for review (roadmap H.4, REQ-VID-165), the plan is written beside the script and reviewed with it. A run resumed over an edited script or plan validates the plan again, and an edited script with a stale plan fails the alignment check rather than voicing the old text. A hand-written or edited plan that fails validation stops the render before TTS, naming the failed check, since it was approved as written; only a generated plan is retried and then dropped.

### Provider

A `gemini_api` provider calls the Interactions endpoint with `GEMINI_API_KEY`, which `collect_producer_secrets` adds:

- one request per segment: the segment text with its events inserted as inline tags, a `speech_metadata` annotation carrying the style, and the profile's voice;
- the WAV responses joined with `gap_after_ms` of silence, then the existing voice chain (REQ-CNT-073) and silence trim.

A profile selects it with `provider: gemini_api` and the existing `gemini_model_name: gemini-3.8-flash-tts`. A failed segment fails the whole voice, and the provider order then falls back with a single style. The fallback receives the script text without the plan's tags: REQ-CNT-071's stripper only matches square-bracket tags, so the angle-bracket tags are never written into text a fallback could receive.

### Tag allowlist

`tools/tts_tag_probe.py` (REQ-CNT-130) gains the Gemini API provider and the angle-bracket tags. Its results set `PERFORMED_TTS_TAGS`, beside `SILENT_TTS_TAGS` in `src/video/config/audio_models.py`: tags that change the delivery (a sound or a gap) and are never transcribed as their own text. A tag Whisper hears as a word may stay on the list, since the caption filter handles it, and the probe records the word it produced.

### Captions

After transcription, a word whose time falls within an event's window and that matches the probe's recorded transcriptions ("ugh", "ha", "hm") is dropped from the caption words. The window comes from the segment offsets. A word in the script is never dropped.

### Script cleaning

The REQ-CNT-052 cleaner runs on the script, before the plan exists. Tags are inserted only into the text sent to TTS, so the cleaner never sees them. The check is a test that a cleaned script with a plan reaches the provider with its tags intact.

### Style prompt fix

The Cloud TTS path sends a profile's `style_prompt` in `SynthesisInput.prompt` on models that accept it (`gemini-2.5-flash-tts`, `gemini-2.5-pro-tts`) and leaves the text unchanged, but only a prompt ending in a colon; any other is skipped with a warning. It sits behind its own switch, `tts_config.send_style_prompt`, which ships off. Every bundled profile's prompt is descriptive, so turning the switch on changes nothing until a profile gets an instruction-form prompt.

### Config

- `tts_config.performed_delivery.enabled` (false), `.max_events` (4), `.retry_on_invalid` (1).
- `tts_config.send_style_prompt` (false).
- A voice profile with `provider: gemini_api`.

### Tests

- The plan validator rejects each failure case above, one test per case.
- A plan joined back reproduces the script.
- The provider builds one request per segment with the annotation and inline tags (mocked HTTP), and joins the audio with the gaps.
- The caption filter drops an "ugh" at an event and keeps one in the script.
- The Cloud TTS request carries `prompt` when the switch is on, and only then.

## Alternatives considered

- **Tags and the prompt field on 2.5 alone.** No new endpoint, and the probe showed the tags silent. But 2.5 has a smaller tag set (`[sigh]`, `[laughing]`, `[uhm]`, pauses), one style per request, and no per-segment style without splitting calls anyway. It is kept as the fallback path, with the style fix.
- **One 3.8 request with all tags and one style.** Simpler, but the style can't follow the script's turns. The probe's per-group styles were the part that changed delivery most.
- **Rule-based tags** (a breath at every paragraph, a laugh after every number). Cheap and deterministic, but it ignores meaning, which is the failure this design exists to avoid.
- **ElevenLabs v3 audio tags.** A comparable tag and emotion set, at a new vendor and a higher price per take. It stays the fallback if 3.8 doesn't hold up in the A/B.

## Rollout

Ships off: `performed_delivery.enabled` false, `send_style_prompt` false, and no bundled profile on `gemini_api`. An installation turns on performed delivery in its private overlay after a listening A/B of one script, voiced with the plan and with today's profile, prefers the plan, and its transcript diff shows no spoken direction.

Remove the switch when: performed delivery has been the default for a month of renders with no take re-voiced for a spoken direction or a misplaced event. The single-style path stays as the fallback provider's mode.

## Open questions

- The event cap: 4 per take is a guess from one probe. The A/B should test 2 and 4.
- Whether the plan model call should use the script model or a cheaper one.
- Whether the 3.8 per-segment joins are audible at the bundled gaps, or need a short crossfade.
