# ContentEngineAI Project Memory

## Project Overview

ContentEngineAI is an AI-powered video production pipeline for e-commerce platforms.

## Session Start

At the very beginning of every session, run this check:
```bash
pyenv version && python3 --version && which python3
```

Expected output:
- pyenv version: `ContentEngineAI` (set by `.python-version`)
- Python: `3.12.x`
- Path: pyenv shim (`~/.pyenv/shims/python3`)

The project uses a **pyenv virtualenv**. The `.python-version` file auto-activates it via pyenv shims. No manual activation, `PYENV_VIRTUAL_ENV` prefixes, or `PATH` overrides needed. Just use `python3`, `pytest`, `ruff` directly.

If pyenv version shows something else, fix with:
```bash
pyenv activate ContentEngineAI
```

## Private Overlay Files

The repo supports private, contributor-specific overlays that stay out of git. Any file matching `*.private.md` or living under `.business/` is gitignored (see `.gitignore`). Use this for business motivation, account-specific context, personal planning, or decisions you don't want to publish.

**Naming convention**: a private overlay sits next to its public counterpart with the same basename plus `.private.md`. Example: `docs/roadmap.md` (public) and `docs/roadmap.private.md` (private).

**At session start**: list any `*.private.md` files relevant to the current task and read them. They usually explain *why* a public item exists when the public doc only describes *what*. Typical check:
```bash
find . -name '*.private.md' -not -path './.venv/*' -not -path './outputs/*'
```

**Keeping public and private in sync** when a pair exists:
- Items and structure stay aligned. When an item is added, removed, reordered, or changes horizon in one file, mirror it in the other.
- Public describes the capability. Private adds motivation, constraints, and decisions.
- `Done when` criteria may differ: public stays generic and testable; private can reference signals (metrics, sample sizes, thresholds specific to the contributor's use case).
- The same rule applies to any other paired docs (e.g., `docs/strategy.md` + `docs/strategy.private.md`).

**Never leaks into the public tree**:
- Content or direct quotes from any `*.private.md` file.
- Account handles, follower counts, financial numbers, real persona names, or any contributor-specific identifiers.
- References to the private file's existence or path in commit messages, PR descriptions, issues, or committed code.
- Config values keyed to the contributor's accounts. Public YAML ships generic defaults; real values live in `.env` or in a gitignored override file.

This pattern is generic. Any contributor can create `docs/<public-doc>.private.md` (or a `.business/` subtree) for their own overlay without project-side configuration changes.

## Logs

Pipeline logs are in `outputs/logs/`:
- `global_pipeline.log` — batch pipeline (scrape + produce + publish)
- `scraper.log` — standalone scraper runs
- `producer.log` — standalone video production
- `publisher.log` — standalone publishing

## Resource discipline (read before running anything below)

The scraper and the producer are the heavy commands. The producer pipeline peaks around 2-2.5 GB RSS per render (Whisper STT, FFmpeg encoding, pycaps Chromium) and runs for 3-6 minutes on a single 30-45s output. The scraper drives Botasaurus + Chromium and holds RAM for the duration of a search. Running either bare while the user is working on the same machine causes systemd-oomd to kill unrelated session apps (Chrome, VSCode) — see the 0.44.0 changelog for why we now ship `MemorySwapMax=0` in the lowpri cgroup.

**Rule: full scrape and full produce ALWAYS go through `make scrape-lowpri` / `make produce-lowpri` (or `make batch-lowpri` for the global pipeline).** These targets wrap the command in a `systemd-run --user --scope` cgroup with `MEM_LIMIT` (default 8G) + `nice` (default 10) + `MemorySwapMax=0`. Tune via `MEM_LIMIT=6G NICE_LEVEL=15` when thrashing. The bare `poetry run python -m src.scraper.amazon.scraper` and `poetry run python -m src.video.producer` forms are reserved for:

- Unit tests / pytest runs (no full render).
- Single-step debug runs that pass `--step <name>` and skip the heavy steps.
- Dry runs (`--dry-run`).
- One-second invocations that just print help or load config.

If the command will scrape products, render audio/video, or run the full pipeline end-to-end, use lowpri. No exceptions for "I just need one quick test render" — quick test renders are exactly when the user is also using the machine.

When invoking lowpri, pass the args via `ARGS="..."`:

```bash
make produce-lowpri ARGS="outputs/<ASIN>/data.json slideshow_images1 --clean --debug"
make scrape-lowpri  ARGS="--product-ids B0XXXXXXXX --debug"
make batch-lowpri   ARGS="--product-ids B0XXXXXXXX --profile slideshow_images1 --debug"
```

For batch operations, `make batch-lowpri` is documented below as the default for global pipeline runs. Apply the same rule to single-product runs by reaching for `make produce-lowpri` first.

## Essential Commands

```bash
# Core workflow
poetry run python -m src.scraper.amazon.scraper --keywords <ASIN> --debug --clean
poetry run python -m src.video.producer outputs/<ASIN>/data.json slideshow_images1 --debug

# Batch scraping (product IDs)
poetry run python -m src.scraper.amazon.scraper --product-ids B0ASIN1 B0ASIN2 B0ASIN3 --debug

# Batch scraping (keywords with filters)
poetry run python -m src.scraper.amazon.scraper --keywords "wireless earbuds" "headphones" --min-price 20 --max-price 100 --min-rating 4.0 --debug

# Batch scraping (mixed mode with fail-fast)
poetry run python -m src.scraper.amazon.scraper --product-ids B0ASIN1 --keywords "product" --fail-fast --debug

# Scraping from URLs (shortened or full Amazon URLs)
poetry run python -m src.scraper.amazon.scraper --product-ids "https://tr.ee/mUk1eH" --output-dir tmp --debug

# Batch scraping from file with chunked processing
poetry run python -m src.scraper.amazon.scraper --input-file products.txt --output-dir tmp --batch-size 10 --debug

# Batch video production (fixed profile)
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug

# Batch video production (random profile per product - deterministic)
poetry run python -m src.video.producer --batch --random-profile --debug

# Batch video production (random from specific pool)
poetry run python -m src.video.producer --batch --random-profile --profile-pool slideshow_images1 video_sequential --debug

# Batch video production (specific products only)
poetry run python -m src.video.producer --batch --random-profile --product-ids B0ASIN1 B0ASIN2 --debug
make produce-lowpri ARGS="--batch --random-profile --product-ids B0ASIN1 B0ASIN2 --debug" MEM_LIMIT=6G

# Global batch pipeline (always use make batch-lowpri for batch runs)
make batch-lowpri ARGS="--product-ids B0ASIN1 B0ASIN2 --profile slideshow_images1 --debug"

# Global batch pipeline (random profiles with filters)
make batch-lowpri ARGS="--keywords 'wireless earbuds' --max-products 10 --min-price 20 --min-rating 4.0 --random-profile --debug"

# Global batch pipeline (skip publishing)
make batch-lowpri ARGS="--keywords 'smart watch' --skip-publish --debug"

# Global batch pipeline (clean stale outputs before run)
make batch-lowpri ARGS="--product-ids B0ASIN1 --clean --debug"

# Scraping only (low priority)
make scrape-lowpri ARGS="--keywords 'wireless earbuds' --debug"

# Video production only (low priority)
make produce-lowpri ARGS="--batch --batch-profile slideshow_images1 --debug"

# Tune resource limits if needed (defaults: MEM_LIMIT=8G, NICE_LEVEL=10)
make batch-lowpri ARGS="--product-ids B0ASIN1 --debug" MEM_LIMIT=6G NICE_LEVEL=15

# Publish single product (auto-schedules to next slot)
poetry run python -m src.publisher.late single B0ASIN1 --debug
make publish ARGS="single B0ASIN1 --debug"

# Schedule all unpublished products
make publish ARGS="schedule --debug"
make publish-lowpri ARGS="schedule --debug" MEM_LIMIT=6G NICE_LEVEL=15

# Publish to specific platforms
poetry run python -m src.publisher.late single B0ASIN1 --platform youtube --platform tiktok --debug

# Published products registry
poetry run python -m src.publisher.late registry --rebuild --outputs-dir outputs
poetry run python -m src.publisher.late registry --rebuild --scan-dir tmp --outputs-dir outputs

# Performance monitoring
poetry run python tools/performance_report.py --report-type summary
poetry run python tools/performance_report.py --report-type trends --days 30
poetry run python tools/performance_report.py --report-type detailed --limit 10
poetry run python tools/performance_report.py --report-type comparison
poetry run python tools/performance_report.py --report-type regressions --window 10
poetry run python tools/performance_report.py --report-type detailed --format csv --limit 5
```

## Code Standards

- **Naming**: snake_case functions, PascalCase classes, UPPER_CASE constants
- **Type Annotations**: Use modern Python typing (`dict[str, Any]`, `| None`)
- **Error Handling**: Specific exceptions (never bare `except Exception`), structured logging
- **Logging**: Use lazy format (`logger.debug("msg: %s", val)`) not f-strings. No emojis in log messages (existing emoji-laden lines are pre-existing tech debt to clean up over time; new code emits plain text).
- **Configuration**: Centralized in `src/video/config/` (Pydantic models)
- **Secrets wiring**: When adding new env vars (API keys, credentials) to any module, verify they're included in the secrets dict in BOTH `src/pipeline/global_batch.py` AND `src/video/producer/cli.py`. These are the two entry points that pass secrets to the pipeline. Missing a key means the feature works in tests but silently falls back in production runs. The audio provider `audio_providers[].settings` env vars are read dynamically; other modules use hardcoded lists.

### Audio Module Notes

- **Provider platform**: `src/audio/` uses `BaseAudioProvider` ABC + `AudioProviderRegistry` + `AudioManager` chain pattern (same as publisher module)
- **Adding a provider**: create `src/audio/new_provider.py` with `@register_audio_provider` decorator, add enum value to `AudioProvider`, import in `__init__.py`, enable in YAML
- **Provider chain**: configured via `audio_providers` list in `video_production.yaml`. Tried in order, first successful download wins, local files are last resort
- **Jamendo**: uses `client_id` auth only (no OAuth2), `fuzzytags` search for genre/mood matching, random query selection from configured pool
- **Freesound**: `FreesoundProvider` wraps existing `FreesoundClient` (don't modify the 728-line client directly). OAuth2 for full quality, API key for previews
- **Config patterns**: Jamendo uses `audio_providers[].settings` dict, Freesound uses legacy `freesound_*` fields on `AudioSettings`. Both work, new providers should use the `settings` dict pattern
- **CircuitBreaker**: use public `record_success()`/`record_failure()` methods, not private `_on_success()`/`_on_failure()`
- **`silence_min_duration_sec` is a trim-MORE knob, not a trim-LESS knob**: the field in `AudioProcessingSettings` maps to ffmpeg `silenceremove` `start_duration`, which is the continuous non-silence window the filter must detect before it stops trimming. Audio during that window is **discarded**, not kept. So larger values trim MORE aggressively, and short trailing words (under ~0.4s, e.g. "tips", "tech") get eaten if `start_duration` exceeds the word length. Keep at 0.1s or below. Lives in `config/ai_services.yaml::audio_processing`.

### Video Module Notes

- **Profile settings**: `get_profile_merged_settings()` returns typed `MergedProfileSettings` (not dicts). Access via `.video_settings.field` and `.subtitle_settings.field`. Use `.model_dump()` when downstream functions need dicts.
- **Disclosure overlay is the LAST video filter**: `apply_disclosure_overlay` in `src/video/assembler/overlay_builder.py` rewrites the subtitle builder's terminal `copy[v_out]` no-op into a drawtext that produces `[v_out]` directly. If a future change makes the subtitle builder emit a different terminal filter shape (anything other than `<input>copy[v_out]`), the overlay step logs a warning and skips silently — there's no overlay corruption, but `#ad` won't appear. Grep for the `copy[v_out]` literal when refactoring subtitle_builder. Pycaps overlays its captions on the assembler output, so the overlay survives the pycaps step automatically.
- **Disclosure config defaults are conservative on size, aggressive on margin**: 0.45 size_factor (slightly under FTC's 50-60% band) and 0.12 vertical margin (clears YouTube Shorts top header at ~10% and TikTok top username at 0-8%). Pydantic floors `size_factor` at 0.2; the runtime also floors the rendered fontsize at 8px so even a tiny subtitle base font doesn't produce illegible disclosures.
- **Profile-level field gating** (silent-drop class of bugs): a profile-level YAML override only reaches the assembler when ALL THREE conditions hold: (1) field declared on `VideoSettings` in `src/video/config/visual_models.py`, (2) field declared as `<type> | None = Field(None, ...)` on `VideoProfile` in the same file, (3) field listed in the `_collect_overrides` field map in `src/video/config/core_models.py::get_profile_merged_settings`. Miss any one and Pydantic's default `extra="ignore"` swallows the override silently — no warning, no test failure. When adding a new profile-overridable field, grep both files for similar fields (e.g., `image_top_position_percent`) and follow the pattern. Audit follow-up tracked as a GitHub issue with the `follow-up` label.
- **Subtitle config model**: `SubtitleSettings` (in `src/video/config/subtitle_models.py`) is the single source of truth — both the config-layer merge output and runtime generator input. Strict (`extra="forbid"`); YAML typos throw at load. `from_legacy_dict()` translates renames (`max_subtitle_duration` → `max_duration`, `min_subtitle_duration` → `min_duration`) and drops underscore-prefixed runtime side-channel keys.
- **Subtitle profile overrides**: `VideoProfile.subtitle_settings: PartialSubtitleSettings | None` is a single nested block; `partial.merge_into(base)` deep-merges (nested sub-models like `pycaps`, `two_part_subtitles`, `safe_zone` merge per-field). The `@model_validator(mode="before")` shim on `VideoProfile` migrates legacy flat keys (`subtitle_anchor`, `pycaps_template`, `two_part_subtitles`, ...) to the nested block at load with a `DeprecationWarning` for one release.
- **Video centering**: Use `video_vertical_align: center` in profile config. Setting `video_top_position_percent: 0.0` puts video at top, not center. The `center` value triggers FFmpeg's `(oh-ih)/2` pad expression.
- **Subtitle positioning**: For centered images with mixed aspect ratios (landscape + portrait), use AVERAGE `video_top` across all images to balance positioning
- **Visual bounds**: Calculated from actual image dimensions, not frame dimensions
- **Two-part subtitles**: Upper (static URL) + Lower (voiceover-synced) handled in `two_part_subtitles.py`. Upper subtitle is dynamically repositioned per visual segment in `subtitle_builder._create_content_aware_upper_ass_file()` using actual geometry from the assembler
- **Upper subtitle positioning gotcha**: The pre-assembly `calculate_visual_bounds()` can produce wrong bounds (e.g., `ctx.scraped_videos` empty). The real fix is in the assembler where per-segment `VisualGeometry` data is available
- **Scraper-producer alignment**: Pass `profile_uses_videos` to scraper so media validation counts only what the profile actually uses
- **TTS voice profiles**: Configured in `config/subtitles.yaml` under `tts_config.voice_profiles`. Profiles specify provider (`google_cloud` or `gemini`), style prompt, voice criteria, and markup rules
- **Voice selection precedence** (highest first): `--voice-profile` CLI override, non-empty `voice_profile_pool` (random across pool, testing path), `default_voice_profile` (pinned voice for unattended runs), random across all profiles (back-compat). Bundled config ships `default_voice_profile: charon` and an empty pool, so no-flag runs always pick Charon. To opt back into random for A/B, set `voice_profile_pool` non-empty.
- **Named voice profiles**: `puck`, `charon`, `fenrir`, `orus` exist as A/B candidates. Each pins one Gemini TTS voice via `voice_criteria.name_contains`. Same pattern works for any Gemini voice name from the catalog (see `docs/tts-voice-profiles.md`).
- **Gemini TTS**: Uses same `google.cloud.texttospeech` SDK but with `SynthesisInput(text=..., prompt=...)`. Requires `Vertex AI User` IAM role on the service account. Falls back to Google Cloud TTS on failure
- **Gemini ignores numeric `speaking_rate`**: empirically the Gemini TTS API does not honor the numeric `speaking_rate` parameter for Gemini-model voices (1.05 vs 1.00 A/B produces near-identical duration). Pacing direction for Gemini realistically flows through the `style_prompt` ("at a relaxed conversational pace", "never rushed") rather than `AudioConfig.speaking_rate`. The rate field IS honored on Chirp 3 HD voices via the same Cloud TTS client.
- **TTS hash slice**: Voice profiles use md5 hex `[16:24]` (fonts use `[0:8]`, colors `[8:16]`, voice within profile `[24:32]`)
- **TTS metadata**: Profile name and voice name saved in `pipeline_state.json` under `create_voiceover.tts_metadata`
- **Script templates**: 15 prompt templates in `src/ai/prompts/scripts/` with different styles (curiosity hook, problem-solution, storytelling, etc.). Configured in `config/ai_services.yaml` under `llm_settings.script_templates`. Selection is deterministic per product using salted md5 hash (`md5(product_id + ":script_template")`). Override with `--script-template NAME` CLI arg. Template name saved in `pipeline_state.json` under `generate_script.script_template`
- **LLM provider fallback**: Gemini is primary, OpenRouter is automatic fallback. Configured via `llm_settings.fallback_provider` in `ai_services.yaml` (self-referencing `LLMSettings`). Both `global_batch.py` and `cli.py` must include fallback provider's API key env var in the secrets dict. Shared dispatch in `src/ai/llm_client.py`
- **LLM config fields**: `model_blocklist`, `min_context_length`, `retry_attempts`/`retry_min_wait_sec`/`retry_max_wait_sec`, and `script_validation` (min_chars, min_words) all live on `LLMSettings`. Don't hardcode these in generator code
- **`.env` safety**: `update_env_file()` in `freesound_client.py` only updates existing keys, never adds new lines
- **Timing smoother**: `src/video/subtitle_timing_smoother.py` post-processes raw Whisper word timestamps before either engine. Four rules: min duration 120ms, gap merge 80ms, segment-end hold +200ms, audio lead 40ms. Wired in `generate_subtitles_with_whisper` (single call site for both engines). Config: `subtitle_settings.timing_smoothing` nested dict in YAML, flows through `extra="allow"` on `MergedSubtitleSettings`. No flat Pydantic fields — the nested dict is passed directly to smoother kwargs.
- **Run paths registration is in TWO places**: adding a new path to `src/video/config/core_models.py::get_run_paths()` is not enough. The runtime `ctx.run_paths` is built by `src/video/producer/state.py::get_video_run_paths()`, which constructs a separate `legacy_paths` dict from the core_models output. New keys must be registered in BOTH places or `ctx.run_paths['<key>']` raises `KeyError` at runtime. The `legacy_paths.update({...})` block at the bottom of `get_video_run_paths` is where additional keys go.
- **Content pillar pipeline**: pillar-aware script generation is wired through `script_templates.{pillars, pillar_preambles, pillar_audiences, narrator_profile}` in `config/ai_services.yaml` and consumed in `src/ai/script_generator.py`. The `--pillar` flag on producer + global_batch sets `ctx.state["pillar"]`; `step_generate_script` passes it to `generate_ai_script`, which filters templates, prepends the per-pillar preamble (after the channel-wide narrator profile), and substitutes `{AUDIENCE}` from `pillar_audiences`. Unknown pillars log an info-level hint and gracefully no-op all three. The fully-rendered prompt always lands at `outputs/<asin>/temp/script_prompt.txt`.
- **Prompt rule placement**: per-script imperative rules go in each template's `## Rules` block in `src/ai/prompts/scripts/*.md`. Channel-wide voice direction goes in `narrator_profile`. The narrator profile sits ~40 lines from the active task in the rendered prompt, so per-script rules buried there bind less reliably (the trade-off honesty rule was the canary case). When adding a rule, ask whether it must apply per-script (template `## Rules`) or describes persistent voice (narrator profile). Inspect `outputs/<asin>/temp/script_prompt.txt` after a run to see what the LLM actually got.

### Pycaps Subtitle Engine Notes

- **Bundled default is pycaps; Pydantic default is ffmpeg.** `config/subtitles.yaml` sets `subtitle_settings.subtitle_engine: "pycaps"` so users running the bundled config get animated captions out of the box. The `SubtitleSettings` Pydantic field still defaults to `"ffmpeg"` for programmatic construction without YAML. Default install does NOT pull pycaps — run `poetry install --with pycaps` (+ `poetry run playwright install chromium` for the CSS renderer). Forks without the optional group degrade silently to FFmpeg via `pycaps.fallback_policy: "fallback_ffmpeg"` in the bundled YAML.
- **Whisper transcript is a literal drop-in**: the raw `result_w` dict from `stt_functions.py:118` is exactly pycaps' `whisper_json` format. `generate_subtitles_with_whisper(..., transcript_out_path=...)` serialises it unconditionally when set (NOT gated by `debug_mode` — pycaps production runs need the file).
- **Post-assembly burn step**: `step_burn_pycaps_subtitles` runs after `assemble_video` and short-circuits when `engine != "pycaps"`, so it's safe to always include in the pipeline graph. Uses `asyncio.to_thread` because pycaps is sync.
- **Content-aware offset formula**: `offset = (bounds.y + bounds.height + 0.02) - 0.95`, clamped to `[-0.9, 0]`. Matches pycaps' `LayoutUtils.get_vertical_alignment_position` bottom-anchor formula so captions land in the whitespace below the product image. See `src/video/pycaps_engine/renderer.py::layout_from_visual_bounds`
- **Template selection**: deterministic md5 hash `md5(product_id + ":pycaps_template")[0:8] % len(pool)`. Empty pool falls back to `template_name`. Same pattern as font/colour/script-template selection
- **Two-part incompatibility**: when engine=pycaps, `step_generate_subtitles` warns and disables `two_part_subtitles_enabled` (now `two_part_subtitles.enabled` after nesting) for that run. Single-line only in v1. The upper URL is NOT rendered.
- **Benchmark numbers** (30s 1080x1920 portrait, 40-word transcript, CSS renderer): word-focus 0.70x realtime, hype 0.79x, minimalist 0.67x. Peak RSS ~420 MB per render. CSS is faster than pictex on production-length clips.
- **Font randomization**: the project's `subtitle_randomize_fonts` doesn't affect pycaps — templates ship their own `@font-face` in `resources/`. Custom fonts go into a template directory, not into the global font pool.
- **Font/color pools**: live in `config/subtitles.yaml` under top-level `font_pool` / `color_pool` keys, validated by `FontPoolEntry` / `ColorPoolEntry` on `VideoConfig`. To add or remove an entry, edit YAML — no Python changes needed. `FontManager` and `ColorManager` load from these pools; selection methods return string names (not enums). Old pair names (`vibrant`, `warm`, `modern`) fall back to `classic` with a warning for backwards compatibility.
- **Fallback policy**: `raise` (Pydantic default) aborts the pipeline if pycaps is unavailable or fails. `fallback_ffmpeg` (bundled YAML default in `config/subtitles.yaml`) switches to the FFmpeg subtitle engine for that run; this is what forks without `--with pycaps` actually hit. `warn_and_skip` keeps the video without subtitles (not recommended). The availability check runs early in `step_generate_subtitles`, before committing to the pycaps-only path.
- **Module/Batch Alignment Rule applied**: the four CLI flags (`--subtitle-engine`, `--pycaps-template`, `--pycaps-template-pool`, `--pycaps-renderer`) live on BOTH `src/video/producer/cli.py` AND `src/pipeline/global_batch.py`. Same names, same choices, same dotted override keys (`subtitle_settings.subtitle_engine`, `subtitle_settings.pycaps.*`). Grep both files when touching either.
- **3-level merge supports nested dotted keys**: `VideoConfig.get_profile_merged_settings()` understands `subtitle_settings.pycaps.<field>` and folds them into the nested `PycapsSettings` model. The same path works in `cli_overrides` dicts.
- **Pycaps upstream is alpha (0.2.1)**: pinned in `pyproject.toml` to a specific git SHA. Upgrade deliberately. If upstream stalls, fork to `ContentEngineAI/pycaps`.
- **AI word tagging via Gemini**: `subtitle_settings.pycaps.enable_ai_tagging: true` wires `GeminiLlm` (in `src/video/pycaps_engine/gemini_llm.py`) into pycaps' `LlmProvider` singleton during `step_burn_pycaps_subtitles`. Pycaps' base `Llm.send_message` declares `model` as required but real call sites pass only `prompt`, so the adapter (and the upstream `Gpt`) defaults `model="gemini-2.5-flash"`. Adapter is lazy — `google.genai` not imported until first call. Per-segment Gemini errors are governed by `ai_tagging_on_error` (default `skip` = swallow + drop tag), distinct from `fallback_policy` (render-fatal). Built-in templates `neo-minimal` and `explosive` ship `type: ai` rules out of the box and demo the feature with no project-local template needed.
- **Follow-up work**: tracked as GitHub Issues with the `pycaps` label.

### CI/CD Gotchas (mypy version drift)

- **mypy error codes differ between versions.** CI installs mypy from `poetry.lock`; local dev may have a different version from `pip install` overrides. The `google` namespace package triggers `attr-defined` on mypy 1.15 but `import-untyped` on 1.19. Never use inline `# type: ignore[specific-code]` for cross-version issues. Use a module-level override in `pyproject.toml` with BOTH codes: `disable_error_code = ["import-untyped", "attr-defined"]`.
- **Always run `poetry run mypy .` (whole project), not `mypy src/`.** CI runs `mypy .` which scans 267 files including tests. `mypy src/` only checks 144. The difference can hide errors.
- **Before pushing, run the exact CI commands** from `.github/workflows/ci.yml`: `poetry run ruff check .`, `poetry run ruff format --check .`, `poetry run mypy .`. Don't substitute with `make lint` or `mypy src/` — they can diverge.
- **`warn_unused_ignores = true`** is enabled. Bare `# type: ignore` works but module-level overrides in `pyproject.toml` are cleaner because they survive mypy version changes without unused-ignore noise.
- **Poetry explicit source pins don't cascade to transitive deps.** `torch` is pinned to `pytorch-cpu` (`source = "pytorch-cpu"`), but `torchaudio` pulled transitively by `coqui-tts` still resolves from the default PyPI index (CUDA wheel), failing at import on CPU boxes with `libcudart.so.13: cannot open shared object file`. Fix: pin `torchaudio` explicitly to the same source. Any PyTorch-ecosystem package used at runtime needs its own source entry — don't assume `torch`'s pin propagates.
- **`pytest --cov` + transitive torch.** Coverage instrumentation re-imports modules, which makes torch's `overrides.py` raise `RuntimeError("function '_has_torch_function' already has a docstring")` the second time it loads. If a module imports a torch-using package at module load (e.g. `from TTS.api import TTS`), pytest-cov sessions crash before any test runs. Fix: defer the import to first use via `importlib.util.find_spec("X")` for the availability check + a lazy `from X import Y` inside a function. Pattern is in `src/video/tts.py::_load_coqui_tts_class`. Same pattern works for any torch-transitive dependency that doesn't need to load until a feature path actually executes.

## Session Continuity

After every context compaction (session continuation), run `/github-workflow` to check CI status and catch any issues from the previous session. This is non-optional.

## Development Guidelines

- Use Poetry for dependency management
- Use imperative commit messages (e.g., "Add subtitle generation")
- **NEVER mention Claude Code, AI tools, or assistants in commits/PRs — no `Co-Authored-By`, no AI references anywhere**
- Document project status in relevant documentation files
- Create implementation plans for features/fixes before coding

**Important Documentation**:
- **CONTRIBUTING.md**: GitHub Flow workflow, branch naming, code style, testing requirements
- **docs/development.md**: Architecture, performance optimization, component development, debugging
- **docs/versioning.md**: Semantic versioning rules, release process, version support policy

*These files are automatically read by the github-workflow skill during iteration start and releases.*

### Git Commit & PR Guidelines

**Commit Messages**:
- Use imperative mood (e.g., "Add feature", not "Added feature" or "Adds feature")
- Keep first line under 50 characters
- **CRITICAL: NEVER include `Co-Authored-By` trailers, author attributions, or any mention of Claude Code / AI tools / assistants**
- Keep messages short and simple
- Explain what and why, not how
- Track follow-up work as GitHub Issues with the `follow-up` label, not as `docs/*-followups.md` files. Issues survive renames, link cleanly from PRs, and don't bit-rot when section numbers shift.

**Pull Request Descriptions**:
- **CRITICAL: NEVER mention authors, Claude Code, AI tools, or assistants in PR titles or descriptions**
- Keep descriptions short and simple
- Use PR template if available in `.github/`
- Focus on what changed, why it changed, and how to test
- Don't reference internal follow-up/todo tracker docs in PR descriptions. Describe the change on its own terms.

## Development Workflow (GitHub Flow)

ContentEngineAI follows **GitHub Flow** - a branch-based workflow for features and bug fixes.

### Branch Management

1. **Create Branch from Main**:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Branch Naming Conventions**:
   - `feature/` - New features
   - `bugfix/` - Bug fixes
   - `hotfix/` - Critical fixes
   - `docs/` - Documentation updates

### Quality Gates (Required Before PR)

```bash
make lint          # Ruff, MyPy, Bandit, Vulture, Safety
make test          # Pytest with coverage
make security      # Security scans
```

**Code Standards Enforced**:
- 88-character line limit
- Modern Python typing (`dict[str, Any]`, `| None`)
- Unit tests >90% coverage, Integration >80% coverage
- Security scanning with Bandit and Safety

### Development Commands

```bash
make dev-setup     # Complete development environment setup
make quick-check   # Essential checks (ruff + type-check)
make full-check    # All checks (lint + security + test-cov)
make lint-fix      # Run linting with automatic fixes
make test-cov      # Run tests with coverage report
```

### Pull Request Process

1. **Push and Create PR**:
   ```bash
   git add .
   git commit -m "Add subtitle generation"  # Imperative messages
   git push origin feature/your-feature-name
   ```

2. **PR Requirements**:
   - Target `main` branch
   - Use conventional commit format (`feat:`, `fix:`, `docs:`)
   - Complete PR template checklist
   - All CI checks must pass
   - Include tests for new functionality

3. **CI/CD Pipeline**:
   - **CI Workflow**: Runs on push/PR to main (lint, test, coverage)
   - **Security Workflow**: Weekly scans + PR checks
   - **Release Workflow**: Triggered by version tags

### Merge Process

- Squash merge for clean history
- Address all review feedback
- Ensure all CI checks pass

### Release Process

**Version Bumping**:
- Follow semantic versioning: `MAJOR.MINOR.PATCH`
- Determine version bump based on changes:
  - **Major** (e.g., 1.0.0 → 2.0.0): Breaking API changes
  - **Minor** (e.g., 0.17.0 → 0.18.0): New features (backward compatible)
  - **Patch** (e.g., 0.17.0 → 0.17.1): Bug fixes only
- Update version in `pyproject.toml`

**Releases are automated via CI/CD**:
1. Review changes and determine version bump
2. Update version in `pyproject.toml` and code files
3. Update `CHANGELOG.md` with release notes following [Keep a Changelog](https://keepachangelog.com/) format
4. Commit version bump: `git commit -m "Bump version to 0.18.0"`
5. Merge PR and switch to main branch
6. Create and push version tag: `git tag -a v0.18.0 -m "Release v0.18.0"`
7. CI workflow automatically creates GitHub release with:
   - Release notes extracted from CHANGELOG.md
   - Build artifacts (wheel and source distribution)
   - Tests and linting verification

**Note**: Do not manually create GitHub releases - CI handles this when tags are pushed

### Dependency Updates (Dependabot)

Dependabot PRs are batched into patch releases per `docs/versioning.md`:

1. Dependabot PRs stay open until the next patch release cycle
2. At release time: `gh pr checkout <PR>`, rebase onto main, install deps, run full test suite
3. Bump version in `pyproject.toml`, add a "Dependencies" section in CHANGELOG
4. Commit version bump on the Dependabot branch, force-push (rebase changed history), squash-merge
5. Tag and push from main as usual

Security-critical updates can trigger an immediate patch release without waiting.

## Publisher Module Notes

- **Zernio (formerly Late)**: The platform rebranded from Late to Zernio. API is identical, old `getlate.dev` endpoints redirect. Our codebase still uses the old `late-sdk` package, `LATE_API_KEY` env var, and `src/publisher/late/` directory structure. Planned migration: switch to `zernio-sdk`, rename env var to `ZERNIO_API_KEY`, update imports. Both old and new SDK packages work during a 6-month grace period. No rush, but should be done eventually.
- **TikTok content disclosure**: Posts **must** include `commercial_content_type: "brand_organic"` and `is_brand_organic_post: true` in `tiktokSettings`. Without these, TikTok rejects with "Commercial content disclosure is enabled but no option selected". The fix is in `src/publisher/late/client.py` — settings are sent both per-platform (`platformSpecificData.tiktokSettings`) and at top-level (`tiktok_settings`).
- **YouTube AI-content disclosure**: every YouTube payload from `_build_sdk_platforms` carries `containsSyntheticMedia: true`. Required by YouTube's policy for AI-generated content; this pipeline's output qualifies on every render. The flag is set even when no per-platform content is provided.
- **Caption-leading disclosure**: `PublishMetadata.format_content()` prepends `disclosure` (default `#ad`) on a line of its own, ahead of the description and hashtag block, satisfying FTC's "clear and conspicuous" placement. Case-insensitive dedup drops `#ad`/`AD`/`Ad` from the hashtag list so the disclosure never appears twice. Override the field per-render to inject language-matched variants (`#publi` for Spanish, etc.) without code changes.
- **Zernio SDK gaps**: YouTube `paid_promotion` and Instagram paid-partnership tagging are NOT exposed by the SDK (verified via SDK source + Zernio docs full-text search + per-platform reference pages). Manual workaround documented in `docs/compliance.md` (per-video YouTube Studio checkbox + Instagram post-edit "Add Paid Partnership Label"). Don't bypass Zernio with direct platform APIs unless volume crosses ~5/day or analytics show real ranking impact.
- **Registry pillar source is `pipeline_state.json`, not `data.json`**: `_read_pillar_from_state` in `product_registry.py` reads from the producer's state file because that's what the producer actually used. Reading from `data.json` requires the unshipped #82 (scraper-side pillar attachment) and would tag rows with the keyword's pillar rather than the rendered video's pillar.
- **Fixing failed TikTok posts**: Use Zernio SDK `posts.aupdate()` to set correct `tiktokSettings` per-platform, then the platform status resets from `failed` → `pending` and auto-publishes. No need to call `retry()` — the update triggers re-publish automatically. Calling `retry()` after that gives 409 "Post is currently publishing".
- **Zernio SDK post methods**: `create`, `get`, `update`, `delete`, `retry`, `list` (+ async variants `acreate`, etc.)
- **SDK returns Pydantic enums, not strings**: `accounts.list()` returns `SocialAccount` objects with `platform` as a `Platform5` enum. Downstream code (`cli.py`, `batch.py`, `global_batch.py`, `schedule.py`) treats `acc["platform"]` as a lowercase string. Normalize in `get_accounts()` by unwrapping to `.value` before returning the dict, otherwise `.lower()` or direct string comparison fails with `AttributeError`.
- **Publishing modes**: Unified (default) = 1 post to all platforms; platform-specific (`--platform-specific` or `use_platform_specific_content: true`) = 1 post per platform with optimized metadata. Shared helper: `src/publisher/publish_modes.py`.
- **Config loading gotcha**: `publisher.yaml` may contain keys not in `PublisherConfig` dataclass (e.g. deprecated `backoff_multiplier`). The config loader in `src/publisher/config.py` strips unknown keys before constructing `PublisherConfig(**config_dict)`.
- **`.env` file**: Auto-loaded by the CLI via `load_dotenv()`. No manual sourcing needed.
- **CLI format**: `poetry run python -m src.publisher.late single <PRODUCT_ID> --debug`. Takes product ID (not file paths). Video is auto-discovered from `outputs/<product_id>/`. Don't pass `--immediate` or `--platform-specific` by default.
- **`publish_history.json` `published_at` is queue time, not actual publish time**: The field is written when `client.posts.create()` succeeds, i.e. when Zernio accepts the post into its scheduler. The video is still queued and won't go live until `scheduledFor`. To answer "is this video actually live yet", query Zernio directly via `client.posts.get(post_id)`. Sorting `publish_history.json` by `published_at` to find "the latest live videos" is wrong: it gives you the latest-queued, not the latest-published.
- **Reading Zernio post status**: `client.posts.get(pid)` returns a `PostGetResponse` whose payload is nested under `.post` (Pydantic alias). Dump with `resp.model_dump(by_alias=True, mode="json")["post"]`. Without `by_alias=True`, every aliased field comes back as `None` and you get a misleading "no status / no scheduled_for" reading. Per-platform `status` values: `pending`, `published`, `failed`, `partial`. Real publish time is `platforms[*].publishedAt` (camelCase, only set once the platform actually publishes).

## Link-in-Bio Module Notes

- **CLI flags**: `--link-in-bio` and `--no-link-in-bio` override `link_in_bio.enabled` config for single publish
- **Affiliate URL fallback**: Uses `affiliate_link` field first, falls back to `url` if unavailable
- **Image fallback**: Uses `images[0]` URL first, falls back to `downloaded_images[0]` local file upload
- **Lnk.Bio auth**: Requires HTTP Basic Auth (not form-encoded), plus `User-Agent: ContentEngineAI/1.0` header to bypass Cloudflare
- **Lnk.Bio API endpoints**: Auth: `POST /oauth/token`, Add: `POST /oauth/v1/lnk/add`, List: `GET /oauth/v1/lnk/list`, Edit: `POST /oauth/v1/lnk/edit` (undocumented; in-place title edit, same id, same position), Delete: `POST /oauth/v1/lnk/delete`. `LnkBioProvider` exposes `add_link` / `list_links` / `delete_link` but no `update_link` — reach the edit endpoint directly until that gap closes. Full protocol notes in `docs/lnkbio-api.md`.
- **Lnk.Bio OAuth scope is hard-capped at `basic`**: nineteen alternative scope strings (`full`, `read_write`, `all`, `links.write`, ...) all return `unsupported scope`. There is no premium scope. Don't try to widen access via OAuth.
- **`/lnk/list` 50 ceiling is an API page size, not a bio cap**: pagination is not exposed (page/offset/cursor/etc. all return the same first 50). The bio itself has no link quota on the free plan. To enumerate beyond 50, scrape the public bio page HTML; don't try to paginate the API.
- **Non-blocking**: Failures never block video publishing; logged as warnings
- **`created_at` is link-add time, not platform publish time**: The bio link is added right after `posts.create` (queue time), but YouTube/TikTok/Instagram only go live when Zernio's scheduler fires `scheduledFor`. The bio link can be clickable for days before the corresponding video is up. Don't use lnk.bio `created_at` to verify a video is actually published; use Zernio's `platforms[*].status` instead.
- **Free plan has unlimited links**: lnk.bio's free tier has no link quota (their headline differentiator vs Linktree). Paid tiers unlock customization (themes, custom domain, analytics), not link count. Safe to keep adding without worrying about a cap.

## Scraper Module Notes

- **URL support**: Scraper accepts full URLs (including shortened URLs like tr.ee) via `--product-ids` or `--input-file`. URLs are detected by `startswith("http")`, navigated directly in the browser, and ASIN is extracted from the redirected URL via regex `/dp/([A-Z0-9]{10})`.
- **CLI args**: `--input-file FILE` (one URL/ASIN per line), `--batch-size N` (process in chunks), `--output-dir DIR` (override output directory).
- **Botasaurus output dir override**: Botasaurus framework callbacks don't accept custom parameters. Use module-level `set_output_dir()` in `botasaurus_output.py` before running the scraper. The `_effective_dir()` helper resolves: explicit param > module override > None (config default).
- **Variable initialization in browser_functions.py**: Variables used after `if is_url / elif is_asin / else` branching (like `count_products_with_media`, `products_with_media_count`, `max_products`) must be initialized **before** the branch, not inside one branch.
- **Two scraping code paths**: The standalone scraper CLI uses `scrape_products_unified()` which has a cycling loop (`_scrape_until_validated_count_reached`). The global batch uses `scrape_batch_browser()` + `process_raw_products()` with its own page retry loop in `global_batch.py`. Changes to validation logic must be tested through both paths.
- **Page retry**: Global batch retries with additional search pages when products fail media validation (`max_retry_pages` in scraper YAML). Only applies to keyword searches, not ASINs/URLs.
- **Standalone CLI must load `.env` at startup**: `src/scraper/amazon/scraper.py::main()` calls `load_dotenv()` before parsing args. Without it, env-only secrets like `AMAZON_ASSOCIATE_TAG` are invisible to `build_affiliate_url`, which silently falls back to returning the input URL unchanged. The global batch entry point in `src/pipeline/global_batch.py::main()` does the same. Any new CLI entry point that reads env vars needs the same call.
- **`build_affiliate_url` logs WARNING on missing associate tag**: in `src/scraper/amazon/utils.py`. The fallback (return input URL unchanged) is preserved for backward compatibility, but the warning makes the silent-revenue-loss class of bug grep-able in `outputs/logs/scraper.log`. Future work to harden this is tracked as a GitHub issue with the `follow-up` label.
- **URL shortener default is `bare` (no-op)**: bundled `config/url_shortener.yaml` ships `provider: bare`. Picsee is opt-in via `provider: picsee` + `PICSEE_API_KEY`. The `_shorten_affiliate_links` function bypasses the API-key gate for the bare provider. Architectural note: the consumer still reads the YAML inline rather than via the Pydantic `URLShortenerSettings` model. Tracked as a config-hygiene follow-up.

## Module/Batch Alignment Rule

**CRITICAL**: Standalone module CLIs (publisher, scraper, producer) and `global_batch.py` often have parallel implementations of the same logic (scheduling, validation, retry, cleanup). When fixing or adding behavior in one path, **proactively check the other path** for the same issue or missing feature. Don't wait for it to break separately. The batch pipeline re-implements logic from standalone modules rather than calling them, so drift is common and silent.

## Available MCP Servers

The project has access to these MCP servers for enhanced development capabilities:

### Context7 Server
- **Purpose**: Library documentation and code examples
- **Usage**: Get up-to-date documentation for any library
- **Tools**: `resolve-library-id`, `query-docs`
- **Example**: Get Next.js documentation, React hooks examples, Python library docs

### GitHub Server
- **Purpose**: GitHub repository management and automation
- **Capabilities**:
  - Repository operations (create, fork, search)
  - Issue management (create, update, comment, sub-issues)
  - Pull request workflow (create, review, merge, status)
  - Workflow automation (run, cancel, retry)
  - Code search and file operations
- **Integration**: Use for automating PR creation, issue tracking, code reviews
