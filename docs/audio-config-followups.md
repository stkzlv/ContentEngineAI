# Audio Configuration Follow-Ups

Deferred hygiene items on the audio-processing config surface
(`src/video/config/audio_models.py`, `config/*.yaml` audio sections).
Nothing here is a user-visible bug — the pipeline produces correct
output today. These are drift/documentation gaps worth closing next
time someone is in this area.

Related docs: [configuration.md](configuration.md) (general 3-level
config system), [subtitle-config-cleanup.md](subtitle-config-cleanup.md)
(neighbouring audit, subtitle side).

---

## 1. Silence-removal `start_duration` eats short trailing words

**Priority**: ~~low~~ **HIGH (fixed)** — was chopping the final word off
voiceovers on ~2% of runs (short final sentences with low-amplitude
final words like "tips", "tech").

**Status**: **FIXED**. YAML override lowered from `0.3s` to `0.1s`;
Pydantic field description rewritten to explain what the parameter
actually does.

### What was happening

`AudioProcessingSettings` in `src/video/config/audio_models.py:188-199`
declared these defaults:

- `silence_threshold_db: int = Field(-40, ...)`
- `silence_min_duration_sec: float = Field(0.1, ...)`

But `config/ai_services.yaml` overrode them to `-50 / 0.3`. The
justifying comment in the YAML read "0.3s prevents trimming the TTS
last_word_buffer (0.5s)" — which inverts what the parameter actually
does.

### Root cause (from isolated grid search)

The `silence_min_duration_sec` knob maps to ffmpeg's `silenceremove`
`start_duration` parameter. This is **not** "minimum silence duration
to trim". Per ffmpeg docs:

> Specify the amount of time that non-silence must be detected before
> it stops trimming audio. [...] a higher value of start_duration with
> a higher start_threshold will actually trim audio more aggressively.

Audio during this non-silence confirmation window is **discarded**,
not kept. So when the filter (running in reverse via `areverse`) hit
the trailing silence, then hit the last word, it started counting
toward 0.3s of sustained non-silence. Short words under ~0.4s (like
"tips", or the decaying tail of "tech") were entirely consumed by the
confirmation window and dropped from the output.

Isolated reproduction (see `~/tmp/gemini_repro.py` and
`~/tmp/silenceremove_grid.py`) confirmed:

- Gemini 2.5 Flash TTS produces complete audio in 30/30 runs across
  5 tail strategies × 2 scripts — no truncation at the synthesis layer
- Grid search over threshold × min_duration on the raw Gemini output:
  at `silence_min_duration_sec = 0.1` every threshold from -20 to -60
  dB preserves the last word; at 0.3s, thresholds -45 dB and lower
  drop the last word on the problematic script

### Fix applied

- `config/ai_services.yaml:586`: `silence_min_duration_sec: 0.3` → `0.1`
- `config/ai_services.yaml:583-586`: replaced misleading comment with
  an accurate description of the ffmpeg semantics
- `src/video/config/audio_models.py:194-199`: Pydantic `description=`
  rewritten to match the actual ffmpeg semantics instead of repeating
  the same misleading "minimum silence duration" framing

### Acceptance criteria

- [x] Find the YAML file that sets `silence_threshold_db: -50` /
  `silence_min_duration_sec: 0.3` → `config/ai_services.yaml:581,586`
- [x] Decide which value is correct → 0.1s (the code default); the
  0.3s override was based on a misread of the parameter semantics
- [x] Update both sides (YAML and Pydantic description) with the
  corrected understanding
- [x] Added comment in the YAML explaining the ffmpeg semantics so
  this doesn't get "fixed" back to 0.3s by someone trying to
  "preserve quiet speech endings"

---
