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

## 1. Silence-removal defaults drift between code and runtime

**Priority**: low — documentation/consistency issue, not a bug.

**Status**: not started.

### Context

`AudioProcessingSettings` in `src/video/config/audio_models.py:188-199`
declares these defaults:

- `silence_threshold_db: int = Field(-40, ...)`
- `silence_min_duration_sec: float = Field(0.1, ...)`

But production runs log the silenceremove filter executing at different
values. Example from `outputs/logs/producer.log` during a
`slideshow_images4` run on `B0CT2NQ7WG` (2026-04-19):

```
Trimmed silence from voiceover (threshold=-50dB, min_duration=0.3s)
```

So something in the YAML stack is overriding the code defaults by
`-10dB` and `+0.2s`. The triple-precedence system is working correctly
(YAML < env < CLI), but the code-declared defaults are stale relative
to the values actually shipping.

### Why it matters

- New contributors reading the Pydantic model get a misleading picture
  of what the pipeline actually uses.
- Field `description=` strings in `audio_models.py` explain the
  reasoning for `-40dB`/`0.1s`, but the shipped config no longer
  matches that reasoning.
- Any future "why is audio clipping?" debugging starts by trusting the
  code defaults, which costs time.

### Acceptance criteria

- [ ] Find the YAML file that sets `silence_threshold_db: -50` /
  `silence_min_duration_sec: 0.3` (likely `config/core.yaml` or the
  video-production profile YAML — grep `silence_threshold_db` and
  `silence_min_duration_sec` across `config/`).
- [ ] Decide which value is correct: the code default or the YAML
  override. Whichever is correct becomes the single source of truth.
- [ ] Update the other side to match, including the `description=`
  strings on the Pydantic fields if the numeric rationale changes
  (currently: "-40dB catches most background silence while preserving
  speech", "0.1s = 100ms is optimal for natural speech cadence").
- [ ] If the YAML value is deliberately more conservative for the
  current voice pool, add a one-line comment in the YAML explaining
  why the override exists.

### Investigation pointers

```bash
grep -rn "silence_threshold_db\|silence_min_duration_sec" config/
grep -rn "silence_threshold_db\|silence_min_duration_sec" src/
```

Trace the value through (Method 1 from the audit-config skill):

- Layer 1 (YAML): `config/<file>.yaml` sets the value
- Layer 2 (merged): `AudioProcessingSettings(**dict)` validates
- Layer 3 (runtime): `src/video/producer/steps.py:719-720` reads
  `audio_proc.silence_threshold_db` / `audio_proc.silence_min_duration_sec`

If the YAML layer sets the override, the other two will follow. If it
doesn't, an env var (`CONTENT_ENGINE_*` prefix) or CLI flag is
injecting the change.

### Effort

15-30 minutes. Pure reconciliation.

---
