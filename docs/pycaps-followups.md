# Pycaps Subtitle Engine — Follow-Ups

Tracker for work that was deliberately deferred out of the initial pycaps
integration (v0.36.0). Each item includes context, acceptance criteria, a
concrete implementation sketch, and rough effort so anyone can pick one up
without re-reading the investigation history.

Related docs: [pycaps-subtitles.md](pycaps-subtitles.md) (user-facing reference),
[CLAUDE.md](../CLAUDE.md) Pycaps Subtitle Engine Notes section.

---

## 1. AI word tagging via the existing Gemini key

**Priority**: high — the single feature that moves pycaps from "nice captions"
to "Submagic parity". Everything else here is cleanup or hardening.

**Status**: not started.

### Context

Pycaps ships a word-tagging system (`docs/TAGS.md` in the pycaps repo). Tags
become CSS classes on the rendered word spans so templates can style specific
words — highlight action verbs, color keywords gold, inject emoji after
product names, pop the current CTA. Three tagger rule types: `wordlist`,
`regex`, and `ai`.

The `ai` tagger calls an LLM with the transcript segment plus a prompt like
*"tag words that describe a concrete product benefit"* and gets back word
indices to mark. Upstream only ships a single `Gpt` provider
(`/home/user/github.com/pycaps/src/pycaps/ai/gpt.py`) hardcoded to OpenAI's
Responses API via `PYCAPS_OPENAI_API_KEY`. There's a clean abstract base
(`Llm` with `send_message` + `is_enabled`) and an `LlmProvider` singleton
swap point (`LlmProvider.set(my_llm)`), so adding a Gemini adapter is a
strictly-additive change.

ContentEngineAI already has `google-genai` as a main dependency
(`pyproject.toml:31`), the primary Gemini key wired through
`config.llm_settings.api_key_env_var`, and the secrets dict in both
`src/video/producer/cli.py` and `src/pipeline/global_batch.py`. No new deps,
no new env var, no new secrets plumbing — reuse what's there.

### Acceptance criteria

- [ ] A `GeminiLlm` adapter subclasses pycaps' `Llm` base and uses
  `google.genai.Client` under the hood.
- [ ] Defaults to `gemini-2.5-flash` (configurable via `PycapsSettings`).
- [ ] `step_burn_pycaps_subtitles` wires the adapter via
  `LlmProvider.set()` before calling `PycapsRenderer.render()` — only when
  pycaps is actually enabled, so default runs don't touch pycaps internals.
- [ ] Disabled cleanly when the Gemini key is missing: `is_enabled()` returns
  `False` and pycaps silently skips `ai` tagger rules (existing behaviour).
- [ ] One project-local pycaps template demonstrates the feature (e.g.
  highlight benefit words in gold, drop an emoji at CTA).
- [ ] Unit test monkey-patches `google.genai.Client` and asserts
  `send_message` round-trips.
- [ ] Docs updated: `pycaps-subtitles.md` gains an AI tagging section,
  `CLAUDE.md` gets a one-liner pointer.
- [ ] CLAUDE.md Module/Batch Alignment Rule: confirm no new env var is
  needed (Gemini key is already in both secrets dicts). Grep both files
  and note the outcome in the PR description.

### Implementation sketch

New file `src/video/pycaps_engine/gemini_llm.py`:

```python
"""Gemini adapter for pycaps AI word tagging."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pycaps.ai import Llm  # deferred import

logger = logging.getLogger(__name__)


class GeminiLlm:
    """pycaps Llm implementation backed by google-genai.

    Constructed lazily so nothing imports google.genai until pycaps
    actually needs AI tagging.
    """

    def __init__(self, api_key: str | None, model: str = "gemini-2.5-flash"):
        self._api_key = api_key
        self._model = model
        self._client = None

    def is_enabled(self) -> bool:
        return bool(self._api_key)

    def send_message(self, prompt: str, model: str | None = None) -> str:
        if not self._api_key:
            raise RuntimeError("GeminiLlm called without an API key")
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        resp = self._client.models.generate_content(
            model=model or self._model, contents=prompt
        )
        return resp.text or ""
```

Wiring in `step_burn_pycaps_subtitles` (before building the pipeline):

```python
gemini_key = ctx.secrets.get(ctx.config.llm_settings.api_key_env_var)
if gemini_key:
    from pycaps.ai import LlmProvider  # deferred — only when pycaps runs

    from src.video.pycaps_engine.gemini_llm import GeminiLlm
    LlmProvider.set(GeminiLlm(gemini_key, model=pycaps_settings.llm_model))
```

Extend `PycapsSettings` with:

```python
llm_model: str = Field(
    "gemini-2.5-flash",
    description="Gemini model used for pycaps AI word tagging",
)
enable_ai_tagging: bool = Field(
    False,
    description=(
        "Opt into pycaps AI word tagging via the Gemini key. "
        "Requires template rules with type=ai to have any effect."
    ),
)
```

Gate the `LlmProvider.set()` call on `pycaps_settings.enable_ai_tagging` so
installing the feature is a separate flip from enabling it. Default off —
templates with `type: ai` rules silently no-op until the user opts in.

### Risks and gotchas

- **Pycaps imports `pycaps.ai` lazily inside its template engine.** Confirm
  the `LlmProvider.set()` call actually gets picked up — the singleton is
  process-wide, but if pycaps imports its own `Gpt` before we override,
  we'd lose. Test path: set the provider before calling `pipeline.run()`
  and assert via a mock that `send_message` is hit instead of the real
  OpenAI client.
- **Rate limits and cost.** Gemini Flash is cheap but not free. Log the
  number of AI calls per render and surface it in `pycaps_metadata.json`.
  Long batch runs could generate hundreds of calls per day.
- **Gemini response format.** Pycaps' `Gpt` provider just returns
  `output_text`. Gemini responses may contain safety blocks, tool calls,
  or multi-part content. Strip to plain text and log anomalies.
- **Module/Batch Alignment Rule.** Both CLI entry points already include
  the Gemini key in their secrets dicts (grep `api_key_env_var` in
  `cli.py` and `global_batch.py` — it's there). Confirm before merging,
  no new wiring needed.

### Effort

1-2 days. Half a day for the adapter + wiring, half for tests, half for
docs + one demonstration template.

---

## 2. Unblock the mypy pin

**Priority**: medium — pure hygiene, doesn't affect users.

**Status**: **DONE** (PR #66, branch `chore/unpin-mypy`).

### Context

While installing the pycaps optional group during the main integration PR,
Poetry re-resolved and bumped mypy from 1.15 → 1.20. The newer mypy tightens
type checking around `dict[Literal[...], ...].get(str_key, ...)`, which
surfaces three pre-existing issues in `src/publisher/schedule.py` lines 949,
1006, 1009. None are bugs at runtime — the `str` values that flow through
those dicts are always one of the `Platform` literals, mypy just can't prove
it through the `.get()` overload set.

I pinned mypy below 1.20 in the pycaps PR to keep that PR's diff focused
and CI green. That pin now blocks future mypy upgrades until schedule.py
is cleaned up.

### Acceptance criteria

- [x] `src/publisher/schedule.py` type-checks cleanly under mypy 1.20+
- [x] Mypy pin in `pyproject.toml` reverted to `^1.10.1` (resolves to 1.20.1)
- [x] `poetry lock` regenerated and `poetry run mypy .` passes
- [x] No functional changes to schedule.py — only a type annotation on
  `platform_contents: dict[str, dict[str, Any]]`

### Implementation sketch

Read `src/publisher/schedule.py` around lines 940-1015. The pattern is
`platform_contents.get(p_name, {})` where `p_name` is a loop variable
typed `str` but semantically always a `Platform` literal. Fixes:

- Cast `p_name` via `cast(PlatformLiteral, p_name)` at loop entry, or
- Type `p_name` explicitly via an annotated loop variable, or
- Change the `platform_contents` dict type to use `str` keys (weaker
  typing but matches reality), or
- Add a `TypeGuard` function that narrows `p_name` to the literal union

Simplest: explicit `cast` at the loop variable. Three-line change.

### Effort

30 minutes. Small, isolated, and the fix is obvious once you read the
surrounding code.

---

## 3. Two-part subtitles + pycaps hybrid

**Priority**: low-medium — requested by the original two-part design but
nobody has asked for it yet.

**Status**: not started. v0.36.0 disables two-part entirely when pycaps is
selected (with a loud warning).

### Context

The existing `TwoPartSubtitleHandler` emits an upper static URL subtitle
(generated from `shortened_affiliate_link`) and a lower voiceover-synced
subtitle. Both are ASS/SRT files consumed by the FFmpeg assembler's
`build_dual_subtitle_graph`. Pycaps, by contrast, burns a single animated
caption track via a post-assembly render pass — there's no concept of a
second caption layer in the pycaps API.

The user-visible consequence: switching a profile from `subtitle_engine:
ffmpeg` to `subtitle_engine: pycaps` silently drops the affiliate URL from
the video. For now we log a warning in `step_generate_subtitles`, but the
right answer is a hybrid rendering path.

### Acceptance criteria

- [ ] A profile with both `two_part_subtitles_enabled: true` and
  `subtitle_engine: pycaps` produces a video with:
  - Upper static URL subtitle rendered via the existing FFmpeg ASS path
  - Lower animated pycaps captions rendered via the burn step
- [ ] No visible regression on single-line pycaps or ffmpeg-only profiles
- [ ] Fallback policy still works: if pycaps fails, the video still has
  the upper URL (from the FFmpeg layer) and the lower is whatever the
  ffmpeg subtitle path would have produced — not blank

### Implementation sketch

Two viable approaches:

**Approach A — two-pass composition (simpler)**:

1. In pycaps mode, `step_generate_subtitles` still generates the upper
   ASS file via `create_static_upper_subtitle` (unchanged).
2. `step_assemble_video` uses the single-subtitle path with
   `subtitle_path=upper_ass_file` — the assembler burns only the upper
   layer.
3. `step_burn_pycaps_subtitles` takes the upper-burned video and runs
   pycaps over it. Pycaps draws below the upper layer (which is already
   pixel-baked into the frame), so no clash.

Pros: zero pycaps API changes, minimal wiring.
Cons: extra FFmpeg pass over the full video.

**Approach B — pycaps CSS `::before` injection (more elegant, riskier)**:

Generate the upper text as a pycaps "document-level" element by injecting
a fixed `<div>` into the pycaps template via the CSS override hook. The
CSS would be position-absolute at the top, no animation. This requires
understanding how pycaps' `add_css_content` interacts with its word-level
rendering — may or may not be possible without template surgery.

**Recommendation**: start with Approach A. Ship the behaviour, revisit
Approach B if render time becomes a problem.

### Risks and gotchas

- **Positioning collision**: the upper ASS subtitle and the pycaps
  captions could overlap if visual bounds aren't computed correctly. Test
  on a top-heavy profile (`slideshow_images1`, centred image) and a
  full-frame profile.
- **`TwoPartSubtitleHandler.calculate_visual_bounds`** is currently called
  from `step_burn_pycaps_subtitles` directly. Need to confirm it returns
  sensible bounds when the upper subtitle is already burned in (it
  should — bounds come from the image geometry, not subtitle presence).
- **Assembler's `build_subtitle_graph`** already handles the "only upper,
  no lower" case when called with a single subtitle path. Verify by
  reading `subtitle_builder.py:build_subtitle_graph` — specifically the
  path that dispatches on `.ass` suffix.

### Effort

2-3 days. The hard part is test coverage across the profile matrix, not
the wiring.

---

## 4. CSS/Chromium integration test in CI

**Priority**: low — current CI coverage is good enough via the Pictex path.

**Status**: the existing integration test
(`tests/video/test_pycaps_integration.py`) uses `PictexSubtitleRenderer`
so it doesn't need Chromium. The CSS path — which is the production
default — is only verified manually.

### Context

The two pycaps renderers produce different output. Pictex uses Skia to
draw glyph-level primitives; the CSS renderer uses a headless Chromium
running real CSS including gradients, drop-shadows, blurs, and animations.
When upstream pycaps ships a change that affects the CSS codepath (custom
properties, keyframe handling, font loading), we won't notice until a
production render fails.

The blocker today is that CI doesn't install Chromium and the `pycaps`
Poetry group. Adding either is ~2 minutes of wall time and ~500 MB of
disk pressure on the runner.

### Acceptance criteria

- [ ] A CI job runs
  `tests/video/test_pycaps_integration.py::test_css_renderer_burns_captions_on_fixture`
  (new test) against the same 30s fixture with `renderer="css"`
- [ ] The job installs `--with pycaps` and runs `playwright install chromium`
- [ ] Total job runtime under 5 minutes (pin a timeout)
- [ ] Results surface via the existing GitHub status checks — no separate
  workflow unless we need different scheduling
- [ ] Skipped cleanly on PRs that don't touch `src/video/pycaps_engine/**`
  or `tests/video/test_pycaps_*` (path filter on workflow trigger)

### Implementation sketch

Option A: extend `.github/workflows/ci.yml` with a new job
`test-pycaps-integration`. Path filter via
`paths: ['src/video/pycaps_engine/**', 'tests/video/test_pycaps_*']`.
Reuse the existing caching strategy, add `poetry install --with pycaps`
and `poetry run playwright install chromium` steps. Run only
`tests/video/test_pycaps_integration.py -m integration`.

Option B: separate `.github/workflows/pycaps.yml` with its own schedule.
Worth it only if we add more pycaps-specific jobs later.

### Risks and gotchas

- **Chromium download flakiness**: playwright's downloader occasionally
  401s. Add a retry or cache the browser install.
- **Fixture drift**: the fixture video is a fake 30s color clip with a
  synthetic transcript — it won't catch real-world issues with product
  imagery and voiceovers. Accept this; the unit tests cover the logic,
  the integration test covers "can we invoke the library end-to-end".
- **Font rendering differences**: headless Chromium on Ubuntu and on a
  developer's macOS may pick different fallback fonts. The test asserts
  on codec/dimensions/duration, not visual output, so this doesn't
  matter.

### Effort

Half a day. The trick is keeping the CI job fast enough that nobody
complains.

---

---

## 5. Whisper timing post-processing

**Priority**: high — affects BOTH engines, biggest single readability win.

**Status**: **DONE** (PR #64, branch `feature/whisper-timing-smoothing`).

### Context

Vanilla OpenAI Whisper rounds word timestamps to whole seconds by default
(see [openai/whisper#435](https://github.com/openai/whisper/discussions/435)),
which produces flicker and uneven segment durations in karaoke-style
captions. Best-practice research
([docs/subtitle-best-practices.md](subtitle-best-practices.md) section 5)
specifies four post-processing rules that should be applied to every STT
result before handing it to either engine:

1. Clamp minimum word duration to **120 ms** — shorter words flash imperceptibly
2. Merge inter-word gaps under **80 ms** into the preceding word
3. Hold the last word of each segment **+200 ms** after audio end
4. Lead audio by **40 ms** so the word appears just before it's spoken

This benefits the existing FFmpeg/ASS karaoke path, the new pycaps path,
and any future rendering backends — it's a pure transform on the word
timings list.

### Acceptance criteria

- [x] New module `src/video/subtitle_timing_smoother.py` with two public
  functions: `smooth_word_timings` (flat list for FFmpeg) and
  `smooth_whisper_result_dict` (raw Whisper dict for pycaps)
- [x] Called from `stt_functions.py::generate_subtitles_with_whisper`
  after `_extract_word_timings`, before `save_whisper_transcript` —
  single call site serves both engines
- [x] Config in `config/subtitles.yaml` under `timing_smoothing` section
  (nested dict, not flat Pydantic fields). Passed through to smoother
  via `create_unified_subtitles` → `generate_subtitles_with_whisper`
- [x] 18 unit tests: empty input, single word, short gaps, segment
  boundary hold, gap merge, lead clamp, combined rules, custom params,
  input immutability, Whisper dict path
- [ ] Before/after fixture: same voiceover, same transcript, verify
  timing deltas are within the expected ranges

### Implementation notes (what was actually built)

Two public functions instead of one — the flat list and raw Whisper dict
have different key names (`start_time`/`end_time` vs `start`/`end`), so
smoothing both from a single call site in `generate_subtitles_with_whisper`
was cleaner than trying to unify the shapes.

Config flows as a nested `timing_smoothing` dict from YAML through to the
smoother kwargs. No flat Pydantic fields on `MergedSubtitleSettings` — the
config audit caught that as dead code and they were removed.

The before/after fixture test is still open — the unit tests cover the
logic exhaustively, but a visual comparison on a real voiceover would
confirm the perceptual improvement.

---

## 6. WhisperX (or whisper-timestamped) upgrade

**Priority**: medium — quality bump, follows #5.

**Status**: not started. Current `stt_functions.py` uses vanilla
`openai-whisper`.

### Context

Even with timing post-processing (#5), vanilla Whisper's word-level
timestamps are imprecise because the model wasn't trained for
word-boundary accuracy. Two viable upgrades:

- **WhisperX** ([m-bain/whisperX](https://github.com/m-bain/whisperX)) —
  adds wav2vec2 forced alignment on top of Whisper output. Most accurate,
  drop-in replacement for `whisper.transcribe()`. Adds a second model
  download (~360 MB for the wav2vec2 alignment model).
- **whisper-timestamped** ([linto-ai/whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)) —
  uses DTW on cross-attention weights. Works with existing Whisper
  models, slightly less accurate than WhisperX but no extra download.

### Acceptance criteria

- [ ] New config field `whisper_settings.timestamp_method` with values
  `vanilla | whisperx | timestamped` (default: `vanilla` for backward compat)
- [ ] Conditional import in `stt_functions.py`, fall through to vanilla
  if the optional dependency isn't installed
- [ ] Optional Poetry group `whisperx` (similar pattern to pycaps)
- [ ] Docs in `docs/subtitle-best-practices.md` section 5 updated with
  install instructions

### Risks and gotchas

- WhisperX has a heavier dependency chain (torch, pyannote, faster-whisper)
- Both alternatives emit slightly different output shapes — the raw dict
  structure for pycaps' `whisper_json` format needs verification with
  the actual library output
- **Don't break the existing flat-list extractor** (`_extract_word_timings`
  in `stt_functions.py`) — the smoother from #5 depends on it

### Effort

2 days. Mostly plumbing and testing across both engine paths.

---

## 7. Fix serif + low-contrast entries in font/color managers

**Priority**: low — cosmetic but violates the best-practice rules.

**Status**: not started. `src/video/font_color_manager.py` ships fonts
and color pairs that fail the research.

### Context

The curated font pool (`FontFamily` enum) includes `DMSerifDisplay-Regular` —
a serif, non-bold font. Best practice is strict on bold sans-serif.

The color pairs (`ColorPair` enum) include three amateur palettes:
- `VIBRANT` (light blue + dark red) — low contrast, doesn't meet WCAG AA
- `WARM` (orange + dark green) — amateur, uncommon in production captions
- `MODERN` (pink + purple) — amateur, low contrast

The surviving good pairs are `CLASSIC` (white + black, 21:1) and
`HIGH_CONTRAST` (yellow + dark blue — yellow is fine, outline should be
black).

### Acceptance criteria

- [ ] Remove `DM_SERIF` from `FontFamily`, or rename the enum entry and
  replace with another bold sans-serif (Inter Black? Anton?)
- [ ] Remove or replace `VIBRANT`, `WARM`, `MODERN` color pairs. Suggested
  replacements: `NEON_GREEN` (`#00FF4C` on black), `BRAND_YELLOW`
  (`#FFEB00` on black)
- [ ] Fix `HIGH_CONTRAST` outline from dark blue to black
- [ ] Update tests in `tests/video/` that reference the removed entries
- [ ] Deprecation path: if anyone's YAML still references the removed
  pair names, fall back to `CLASSIC` with a warning

### Effort

Half a day.

---

## 8. Drop `movement` effect from ASS presets

**Priority**: low — anti-pattern per research but harmless.

**Status**: partially done. YAML preset `animated` was switched from
`movement` to `karaoke` in the best-practices alignment commit. The
effect itself still exists in the code at
`src/video/unified_subtitle_generator.py` and can be re-enabled by
user YAML. Proper fix is to remove the effect implementation.

### Acceptance criteria

- [ ] Remove `movement` from the effect dispatch table
- [ ] Remove `movement_distance_pixels` field from `SubtitleEffectsSettings`
- [ ] Deprecation: if YAML still sets `effects: ["movement"]`, log a
  warning and fall back to `fade`
- [ ] Update unit tests that reference `movement`

### Effort

1-2 hours.

---

## 9. Custom `contentengine_benefit` pycaps template

**Priority**: medium — ships the research-backed starter recipe as a
first-class project template.

**Status**: not started.

### Context

The research in [docs/subtitle-best-practices.md](subtitle-best-practices.md)
section 9 specifies an exact starter recipe: Montserrat Black 72px,
white + black stroke, yellow active-word highlight, scale pop at 1.10,
fade+slide entrance over 160ms, ease-out-quint, no exit, 3-5 words per
line, single emoji per 8-word segment, derived positioning.

This is close to pycaps' `word-focus` preset but not identical. Shipping
our own template lets us bake in ContentEngineAI's brand voice and the
content-aware positioning that the existing integration computes at
runtime.

### Acceptance criteria

- [ ] New template dir `pycaps-templates/contentengine_benefit/` with
  `pycaps.template.json`, `style.css`, and `resources/` containing
  Montserrat Black TTF
- [ ] Loadable via `pycaps_template: "contentengine_benefit"` in
  profile config
- [ ] Font ships in `resources/` so headless Chromium renders correctly
  without system font dependencies
- [ ] Smoke test renders the 30s fixture with this template and compares
  visual output via a screenshot assertion
- [ ] Added to the default `template_pool` in
  `config/subtitles.yaml::subtitle_settings.pycaps.template_pool`

### Risks and gotchas

- Pycaps template loader resolves paths relative to the `pycaps.template.json`
  file. Make sure font path is `./resources/Montserrat-Black.ttf` not
  an absolute path.
- Verify TikTok/Reels safe zone via the derived offset at render time.
- Emoji injection requires the AI tagging follow-up (#1) — without it,
  the template can still hand-tag via regex for common benefit words
  ("free", "save", "deal", "new") or skip emoji entirely in v1.

### Effort

1-2 days. Font embedding + CSS tuning is the time sink.

---

## Notes on prioritisation

If you're picking one of these up and wondering where to start:

- **Want to ship user-visible value**: do #1 (AI word tagging). It's the
  feature that sells the engine.
- ~~**Want the biggest readability win for the smallest code change**:
  do #5 (timing smoothing). Affects BOTH engines.~~ **DONE** (PR #64).
- ~~**Want a fast win**: do #2 (mypy pin).~~ **DONE** (PR #66).
  Or #8 (drop `movement` effect), 1-2 hours.
- **Want to reduce open risk**: do #3 (two-part hybrid). Fixes the one
  thing the initial integration explicitly regresses.
- **Want to ship the starter recipe as a template**: do #9 (custom
  pycaps template). Requires #1 for full emoji support but can ship
  without it.
- **Want the quality ceiling**: do #6 (WhisperX upgrade). Follows #5.
- **Want clean-up**: do #7 (fix font/color managers) after research
  alignment is shipped.
- **Want to keep CI honest**: do #4 (Chromium integration test). Safety
  net, not a feature.

Suggested order: **#1 → #9 → #6 → #7 → #3 → #8 → #4**.

- ~~#5 is foundational~~ — **DONE** (PR #64). Clean timings are now in
  place for all downstream work
- #1 enables #9 (emoji injection in the template)
- #9 is the single most visible user-facing output
- #3, #4, #7, #8 are hygiene/safety and can slot in at any point
- #6 is the quality ceiling that makes everything else look better

Nothing here is blocked on anything else, so items can be picked up in
parallel. Items #6 and #7 benefit both engines — don't treat them as
pycaps-specific just because they're tracked in this doc.
