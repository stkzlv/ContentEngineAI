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

**A foreign `VIRTUAL_ENV` breaks more than it looks like.** If another project's virtualenv is active in the shell (`echo $VIRTUAL_ENV` points outside this repo), it sets `VIRTUAL_ENV`, leads `PATH`, and is what Poetry reports, because `poetry.toml` sets `virtualenvs.create=false`. `poetry run <anything>` then fails with `Please change python executable via the "env use" command`, so the CI gates can't be run through `poetry run` at all. Deactivate it, or run the gates through the venv directly (`~/.pyenv/versions/ContentEngineAI/bin/ruff`, `.../mypy`, `.../pytest`). The same hijack is why the `*-lowpri` targets consult `.python-version` before any ambient source.

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

**Rule: full scrape and full produce ALWAYS go through `make scrape-lowpri` / `make produce-lowpri` (or `make batch-lowpri` for the global pipeline).** These targets wrap the command in a `systemd-run --user --scope` cgroup with `MEM_LIMIT` (default 6G) + `nice` (default 15) + `MemorySwapMax=0`. Tune via `MEM_LIMIT=4G NICE_LEVEL=19` when thrashing. `nice`/`ionice` are CPU/IO priority only and do nothing for OOM; `MemoryMax` + `MemorySwapMax=0` are what contain a memory blow-up to the pipeline cgroup instead of letting it (or systemd-oomd) kill unrelated session apps. The bare `poetry run python -m src.scraper.amazon.scraper` and `poetry run python -m src.video.producer` forms are reserved for:

- Unit tests / pytest runs (no full render).
- Single-step debug runs that pass `--step <name>` and skip the heavy steps.
- Dry runs (`--dry-run`).
- One-second invocations that just print help or load config.

If the command will scrape products, render audio/video, or run the full pipeline end-to-end, use lowpri. No exceptions for "I just need one quick test render" — quick test renders are exactly when the user is also using the machine.

**The `*-lowpri` recipes deliberately do NOT use `poetry run` — don't "simplify" them back to it.** `systemd-run --user --scope` starts the process through the user service manager, which doesn't carry the caller's virtualenv, so `poetry run python` inside the scope resolves an interpreter without the project's dependencies and the run dies on import (which module varies by entry point). The recipes resolve an interpreter that can actually import a project dependency and exec it directly with `PATH` forwarded. `poetry run python` is also unusable as the probe: with `virtualenvs.create=false` in `poetry.toml` it reports the base interpreter, not the project venv. Only the lowpri targets need this; the plain targets run outside the scope and `poetry run` is correct there.

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

# Tune resource limits if needed (defaults: MEM_LIMIT=6G, NICE_LEVEL=15)
make batch-lowpri ARGS="--product-ids B0ASIN1 --debug" MEM_LIMIT=4G NICE_LEVEL=19

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

## End-to-End Pipeline Test Cases

After any change that alters runtime behavior, run the real path it touches and inspect the real artifact (file, video, log line, published post), not just a green test run. Match the check to the change: scraper change -> a scrape; producer/subtitle/audio change -> a produce + `ffprobe`/frame check; publisher change -> the publish-option runbook; config-model change -> the one path that consumes the field; cross-phase change -> a full batch. When the changed logic exists in both a standalone module CLI and `global_batch` (Module/Batch Alignment Rule), verify BOTH paths. Trust the artifact over the exit code (the global batch now exits non-zero when nothing completes end-to-end, but partial failures still exit 0 — grep the phase-summary log lines to confirm). Full change-type -> check table and worked runbooks in `docs/testing.md` ("Verifying a change end-to-end").

The full-pipeline cases below are one instance. Manual full-pipeline checks (scrape -> produce -> publish) for one random config product, random profile. The scrape, produce, and publish paths each have a standalone module CLI and a `global_batch` variant that re-implement the same logic, so all three cases below exercise different code (see Module/Batch Alignment Rule).

**On this Wayland box, wrap any producer run in `xvfb-run -a`** — the CSS pycaps renderer (bundled default) needs an X display or its per-word screenshots hang. `pictex` doesn't, but **`pictex` is preview-only and must not be used for published output**: it renders words with no gaps between them (`Likemyphonewentfrom`), silently and without error. Use `xvfb-run -a` with the CSS renderer instead. See the pycaps notes below and issue #174.

Pick a random keyword from the config pool:
```bash
KW=$(sed -n '/^  keywords:/,/^  [a-z]/p' config/scraper.yaml | grep -oE '^\s+- "[^"]+"' | sed -E 's/^\s+- "([^"]+)"/\1/' | shuf -n1)
```

**Case 1 — batch pipeline (one command, `global_batch`):**
```bash
xvfb-run -a make batch-lowpri ARGS="--keywords '$KW' --max-products 1 --products-per-keyword 1 --random-profile --debug"
```

**Case 2 — separate modules via make (lowpri cgroup):**
```bash
make scrape-lowpri ARGS="--keywords '$KW' --max-products 1 --debug"          # note the ASIN
xvfb-run -a make produce-lowpri ARGS="--batch --random-profile --product-ids <ASIN> --debug"
make publish ARGS="single <ASIN> --debug"                                    # add --force to republish an already-published product
```

**Case 3 — separate modules, no makefile (bare, normal mode):** bypasses the lowpri memory cap (see Resource discipline), so only when the machine is otherwise idle.
```bash
poetry run python -m src.scraper.amazon.scraper --keywords "$KW" --max-products 1
xvfb-run -a poetry run python -m src.video.producer --batch --random-profile --product-ids <ASIN>
poetry run python -m src.publisher.late single <ASIN>
```

**Verify each stage:**
- **Scrape**: `outputs/<ASIN>/data.json` + `images/` exist; log says `complete: N validated products collected`.
- **Produce**: `outputs/<ASIN>/video_<ASIN>_<profile>.mp4` exists; log says `Pipeline execution completed: 8 completed, 0 skipped, 0 failed`; the random profile is in the log line `with profile '<name>'`.
- **Publish**: log says `Post created successfully: <id> (status: scheduled)`; the product dir is auto-cleaned after a successful publish (media lives on Zernio's CDN).
- **Registry**: `grep <ASIN> outputs/published_products.json`.
- **Authoritative publish check** (the post actually reached the scheduler): `client.posts.get(<id>).model_dump(by_alias=True, mode="json")["post"]` -> top `status` plus `platforms[*].status` / `scheduledFor`. Don't trust `publish_history.json` `published_at` (that's queue time, not live time).

**Caveats baked in from real runs:**
- **Normal-mode scrape is reliable** now that the browser window size is clamped to desktop widths in `_BROWSER_CONFIG` (`src/scraper/amazon/config.py`). It was `WindowSize.RANDOM`, which could draw a narrow/mobile width that triggers Amazon's responsive layout the desktop card selectors miss, silently yielding 0 products. `--debug` is no longer needed just for scrape reliability (it still pins a fixed window and adds verbosity). A genuine 0-product run on Wayland is a different cause (no X display) — see `docs/troubleshooting.md`.
- **`--force` republishes** a product already published (default off). Fresh scrapes don't need it.
- **Coqui TTS `No espeak backend` ERROR is non-fatal** and appears only when `espeak-ng` isn't installed (it's the unused fallback provider failing to load; Gemini TTS is primary). Install `espeak-ng` (`sudo apt install -y espeak-ng`) to silence it.
- **Random profile is per-run, not pinned** — the same product can draw a different profile across runs.

### Publish-option verification (every publishing path)

Full runbook in `docs/testing.md` -> "End-to-end publish-option verification". Use it to exercise all publishing options before a publisher refactor/release. Render each product with `make batch-lowpri ARGS="... --skip-publish"`, then publish one option combo per product and verify on Zernio. Real posts are created (immediate goes live; scheduled ships at the next slot — `python -m src.publisher.late delete <POST_ID>` to drop one). Cover both publish paths (`single` and `schedule auto`) and both modes (unified, `--platform-specific`) — they re-implement the same logic.

Operational gotchas the runbook captures (must-know inline):
- **Verify surfaces**: `outputs/publish_history.json` / `published_products.json` / `schedule.json` live at the outputs root and survive product-dir cleanup — diff their counts. Authoritative live status is `client.posts.get(<id>)` `.post` (`by_alias=True`), not `publish_history.json` (queue time). Disclosures to confirm: YouTube `containsSyntheticMedia: true`, TikTok `platformSpecificData.tiktokSettings.commercial_content_type: brand_organic` + `is_brand_organic_post: true` (both unified and platform-specific).
- **link-in-bio** runs on the `single` path and reads `outputs/<asin>/data.json`. Config defaults it ON — pass `--no-link-in-bio` to skip. It runs in BOTH branches: after a successful publish, and on the already-published early return, where `cli.py` deliberately calls `update_link_in_bio_safe(...)` before returning ("a fully-published product needs no video or upload: keep its bio link fresh and exit"). So a no-op `single` run on an already-published product still touches the bio — it does not return before the link-in-bio step. That branch also forces `enabled=True` on the passed config, so only `--no-link-in-bio` (which clears the `link_in_bio_enabled` gate) actually skips it. After cleanup the dir is gone; to set the link later, reconstruct a minimal `data.json` (`title` + `affiliate_link`) from `published_products.json` in a temp dir and call `LinkInBioManager.update(<asin>, <tmp_outputs>)`.
- **`schedule auto` needs `--auto-resolve`** to take an alternative when the preferred slot conflicts (2h `min_post_spacing`); without it the product is counted failed (and the schedule path exits non-zero, unlike the batch pipeline which exits 0 on partial failure — grep phase summaries, not `$?`).
- **Cleanup** skips while a leg is still `publishing` (immediate runs keep the dir) and runs once `scheduled` (dir removed after tracking/registry writes — the #175 path).
- **The #177 SDK crash blocks verification**: a published TikTok leg with `platformPostUrl: ""` makes `posts.list`/`get` raise, so `verify-comments` hard-fails and slot-occupancy degrades. Read status via the raw REST API and first-comment delivery via `get_post_comments(<platformPostId>, <accountId>)`, both of which bypass the failing model.

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
- **Jamendo CDN requires HTTP/2**: `prod-1.storage.jamendo.com` (behind Cloudflare/nginx) serves the actual MP3 only over HTTP/2; HTTP/1.1 requests get a blocking text/html page. aiohttp 3.x only supports HTTP/1.1, so `JamendoProvider.download()` uses `asyncio.create_subprocess_exec("curl", ...)` instead, which negotiates HTTP/2 by default. The search API (`api.jamendo.com`) works fine with HTTP/1.1 and still uses aiohttp. See `src/audio/jamendo_provider.py:176-198`.
- **Jamendo CDN is slow without proxy**: even with HTTP/2 via curl, the Jamendo CDN can be very slow in some network environments (~90 B/s, 3.3 MB tracks time out in 180s). Through a HTTP/SOCKS proxy, downloads typically complete in under 120s. The provider chain falls through to Freesound if Jamendo times out. If a run is missing background music, a proxy/VPN may be needed.
- **Freesound**: `FreesoundProvider` wraps existing `FreesoundClient` (don't modify the 728-line client directly). OAuth2 for full quality, API key for previews
- **Config patterns**: Jamendo uses `audio_providers[].settings` dict, Freesound uses legacy `freesound_*` fields on `AudioSettings`. Both work, new providers should use the `settings` dict pattern
- **CircuitBreaker**: use public `record_success()`/`record_failure()` methods, not private `_on_success()`/`_on_failure()`
- **`silence_min_duration_sec` is a trim-MORE knob, not a trim-LESS knob**: the field in `AudioProcessingSettings` maps to ffmpeg `silenceremove` `start_duration`, which is the continuous non-silence window the filter must detect before it stops trimming. Audio during that window is **discarded**, not kept. So larger values trim MORE aggressively, and short trailing words (under ~0.4s, e.g. "tips", "tech") get eaten if `start_duration` exceeds the word length. Keep at 0.1s or below. Lives in `config/ai_services.yaml::audio_processing`.

### Video Module Notes

- **Profile settings**: `get_profile_merged_settings()` returns typed `MergedProfileSettings` (not dicts). Access via `.video_settings.field` and `.subtitle_settings.field`. Use `.model_dump()` when downstream functions need dicts.
- **Disclosure overlay is the LAST video filter**: `apply_disclosure_overlay` in `src/video/assembler/overlay_builder.py` rewrites the subtitle builder's terminal `copy[v_out]` no-op into a drawtext that produces `[v_out]` directly. Chains that end some other way are normalized first by `_ensure_copy_terminal`, which re-points the terminal filter at `[v_pre_overlay]` and appends the missing no-op — that is what lets both overlays apply on the content-aware ASS path, which ends with `ass='...'[v_out]` and used to drop them silently. The remaining unrecoverable shape is a terminal that doesn't produce `[v_out]` at all; that still logs a warning and skips, so `#ad` won't appear. Any new terminal filter shape in subtitle_builder should be checked against that helper. Pycaps overlays its captions on the assembler output, so the overlay survives the pycaps step automatically.
- **FFmpeg filtergraph commas inside expressions need backslash-escaping, not quoting.** Phase 1.2c's hook overlay first shipped with `enable='between(t,0,X)'` per the FFmpeg docs' "quote to protect commas" rule. The filter failed at render with `Missing ')' or too many args in 'between(t'` because FFmpeg's parser still split at the inner commas when the filter sits inside a comma-separated chain (multiple drawtexts + the disclosure rewrite). The reliable form is `enable=between(t\,0\,X)` — backslash-escape the commas inside the expression. Same trick applies to any FFmpeg expression with embedded commas (`if`, `lt`, `gte`, `min`, `max`). The "wrap in single quotes" form is documented but unreliable in practice for filters embedded in larger chains. Verified with a synthetic `ffmpeg -vf "drawtext=...:enable=between(t\,0\,1.5)"` call.
- **Inline `text=` with an apostrophe corrupts inside a multi-filter chain; use `textfile=`.** The exit/reenter escape `'\''` (close-quote, backslash-quote, open-quote) survives a standalone `-vf` drawtext but NOT the assembler's multi-filter `-filter_complex` chain: when another filter follows, FFmpeg swallows the drawtext's own trailing args (`fontsize=`, `x=`, `enable=`) as literal text, rendering them as tiny garbage at the frame's top-left and dropping the intended text. Hit on the hook overlay for any hook whose first sentence had a contraction (`you're`, `it's`, `don't`); apostrophe-free hooks (the USB-C test case) hid it. Fix: write each line to a temp file and reference `textfile='<path>'` (colon-escape the path) — the same approach `subtitle_builder` already uses. `build_hook_drawtext` does this; `_escape_drawtext_text` is now only safe for the disclosure's config-controlled, apostrophe-free text (`#ad`/`#publi`). Alternative escapes tested: `'\''` mangles, `\\\'` drops the apostrophe, `textfile=` renders it verbatim. Verified by rendering the real `build_hook_drawtext` output through ffmpeg and viewing the frame. The disclosure overlay was later moved to `textfile=` too, since `disclosure_overlay.text` is user-configurable per language and a value like `Pub d'affiliation` would hit exactly this bug.
- **`textfile=` removes the quoting layer but NOT drawtext's text expansion — escape `%` and `\` in the file contents.** Both failures are silent and FFmpeg still exits 0: a raw `%` is read as the start of a `%{...}` function (logs `Stray %`, draws NOTHING for that line — the frame comes out byte-identical to one with no drawtext at all), and a raw `\` is swallowed (`a \ b` renders as `a  b`). Escape `\` -> `\\` first, then `%` -> `\%`, so the marker added for `%` survives. This bit the hook overlay immediately after the apostrophe fix above: switching from `text=` to `textfile=` dropped the `%` escaping that `_escape_drawtext_text` had been applying, and the hook headline prompt explicitly asks for price and percentage lines, so every percent-bearing hook silently vanished from the render. `_escape_drawtext_textfile` in `overlay_builder.py` is the single escaper for textfile contents; the older `_escape_drawtext_text` is for inline `text=` only. Lesson: when swapping how text reaches FFmpeg, re-check every metacharacter the old path handled, not just the one you set out to fix.
- **Disclosure config defaults are conservative on size, aggressive on margin**: 0.45 size_factor (slightly under FTC's 50-60% band) and 0.12 vertical margin (clears YouTube Shorts top header at ~10% and TikTok top username at 0-8%). Pydantic floors `size_factor` at 0.2; the runtime also floors the rendered fontsize at 8px so even a tiny subtitle base font doesn't produce illegible disclosures.
- **Profile-level field gating** (silent-drop class of bugs): a profile-level YAML override only reaches the assembler when ALL THREE conditions hold: (1) field declared on `VideoSettings` in `src/video/config/visual_models.py`, (2) field declared as `<type> | None = Field(None, ...)` on `VideoProfile` in the same file, (3) field listed in the `_collect_overrides` field map in `src/video/config/core_models.py::get_profile_merged_settings`. Miss any one and Pydantic's default `extra="ignore"` swallows the override silently — no warning, no test failure. When adding a new profile-overridable field, grep both files for similar fields (e.g., `image_top_position_percent`) and follow the pattern. Audit follow-up tracked as a GitHub issue with the `follow-up` label.
- **Subtitle config model**: `SubtitleSettings` (in `src/video/config/subtitle_models.py`) is the single source of truth — both the config-layer merge output and runtime generator input. Strict (`extra="forbid"`); YAML typos throw at load. `from_legacy_dict()` translates renames (`max_subtitle_duration` → `max_duration`, `min_subtitle_duration` → `min_duration`) and drops underscore-prefixed runtime side-channel keys.
- **Subtitle profile overrides**: `VideoProfile.subtitle_settings: PartialSubtitleSettings | None` is a single nested block; `partial.merge_into(base)` deep-merges (nested sub-models like `pycaps`, `two_part_subtitles`, `safe_zone` merge per-field). The `@model_validator(mode="before")` shim on `VideoProfile` migrates legacy flat keys (`subtitle_anchor`, `pycaps_template`, `two_part_subtitles`, ...) to the nested block at load with a `DeprecationWarning` for one release.
- **Video centering**: Use `video_vertical_align: center` in profile config. Setting `video_top_position_percent: 0.0` puts video at top, not center. The `center` value triggers FFmpeg's `(oh-ih)/2` pad expression.
- **Subtitle positioning**: For centered images with mixed aspect ratios (landscape + portrait), use AVERAGE `video_top` across all images to balance positioning
- **Visual bounds**: Calculated from actual image dimensions, not frame dimensions
- **Safe zone has two model homes**: `TextRenderingSettings.safe_zone` (global, `core_models.py`) and `SubtitleSettings.safe_zone` (per-profile mergeable). They reconcile one-way in `core_models.py::_build_subtitle_base` (`base["safe_zone"] = text_rendering.safe_zone`). pycaps reads the profile-merged `subtitle_settings.safe_zone`; `subtitle_builder._get_safe_zone()` must prefer the same merged value, else a profile-level safe-zone override applies to pycaps but is silently dropped on the FFmpeg burn. Defaults live in `constants.py`; `config/subtitles.yaml::text_rendering.safe_zone` restates them, so edit both or they diverge.
- **ASS captions are center-anchored (`\an5`)**: `\pos` y is the text CENTER, not its bottom. Any safe-zone clamp must subtract half the line height (`clamp_to_safe_zone(..., text_half_height_px=int(font_size/2))`), or a clamped caption's bottom spills past `max_y` into the platform UI zone.
- **Two-part subtitles**: Upper (static URL) + Lower (voiceover-synced) handled in `two_part_subtitles.py`. Upper subtitle is dynamically repositioned per visual segment in `subtitle_builder._create_content_aware_upper_ass_file()` using actual geometry from the assembler
- **Upper subtitle positioning gotcha**: The pre-assembly `calculate_visual_bounds()` can produce wrong bounds (e.g., `ctx.scraped_videos` empty). The real fix is in the assembler where per-segment `VisualGeometry` data is available
- **Scraper-producer alignment**: Pass `profile_uses_videos` to scraper so media validation counts only what the profile actually uses
- **TTS voice profiles**: Configured in `config/subtitles.yaml` under `tts_config.voice_profiles`. Profiles specify provider (`google_cloud` or `gemini`), style prompt, voice criteria, and markup rules
- **Voice selection precedence** (highest first): `--voice-profile` CLI override, non-empty `voice_profile_pool` (random across pool, testing path), `default_voice_profile` (pinned voice for unattended runs), random across all profiles (back-compat). Bundled config ships `default_voice_profile: charon` and an empty pool, so no-flag runs always pick Charon. To opt back into random for A/B, set `voice_profile_pool` non-empty.
- **Named voice profiles**: `puck`, `charon`, `fenrir`, `orus` exist as A/B candidates. Each pins one Gemini TTS voice via `voice_criteria.name_contains`. Same pattern works for any Gemini voice name from the catalog (see `docs/tts-voice-profiles.md`).
- **Gemini TTS**: Uses same `google.cloud.texttospeech` SDK but with `SynthesisInput(text=...)`. Requires `Vertex AI User` IAM role on the service account. Falls back to Google Cloud TTS on failure.
- **Gemini TTS `prompt` parameter reads aloud instead of styling**: the `SynthesisInput(prompt=...)` field is documented as a style directive but the model treats it as spoken content prepended to the script. Every voice profile's `style_prompt` was audibly read before the script started ("warm but grounded, like a trusted friend recommending something over coffee"). Fix: don't pass `prompt` at all. Voice character is controlled by voice name selection (Charon, Puck, etc.). The `style_prompt` field still exists in config for documentation but isn't wired to the API.
- **Gemini ignores numeric `speaking_rate`**: empirically the Gemini TTS API does not honor the numeric `speaking_rate` parameter for Gemini-model voices (1.05 vs 1.00 A/B produces near-identical duration). Pacing for Gemini is controlled by the voice character itself, not by API parameters. The rate field IS honored on Chirp 3 HD voices via the same Cloud TTS client.
- **TTS hash slice**: Voice profiles use md5 hex `[16:24]` (fonts use `[0:8]`, colors `[8:16]`, voice within profile `[24:32]`)
- **TTS metadata**: Profile name and voice name saved in `pipeline_state.json` under `create_voiceover.tts_metadata`
- **Script templates**: 15 prompt templates in `src/ai/prompts/scripts/` with different styles (curiosity hook, problem-solution, storytelling, etc.). Configured in `config/ai_services.yaml` under `llm_settings.script_templates`. Selection is deterministic per product using salted md5 hash (`md5(product_id + ":script_template")`). Override with `--script-template NAME` CLI arg. Template name saved in `pipeline_state.json` under `generate_script.script_template`
- **LLM provider fallback**: Gemini is primary, OpenRouter is automatic fallback. Configured via `llm_settings.fallback_provider` in `ai_services.yaml` (self-referencing `LLMSettings`). Both `global_batch.py` and `cli.py` must include fallback provider's API key env var in the secrets dict. Shared dispatch in `src/ai/llm_client.py`
- **LLM config fields**: `model_blocklist`, `min_context_length`, `retry_attempts`/`retry_min_wait_sec`/`retry_max_wait_sec`, and `script_validation` (min_chars, min_words) all live on `LLMSettings`. Don't hardcode these in generator code
- **`.env` safety**: `update_env_file()` in `freesound_client.py` only updates existing keys, never adds new lines
- **Timing smoother**: `src/video/subtitle_timing_smoother.py` post-processes raw Whisper word timestamps before either engine. Four rules: min duration 120ms, gap merge 80ms, segment-end hold +200ms, audio lead 40ms. Wired in `generate_subtitles_with_whisper` (single call site for both engines). Config: `subtitle_settings.timing_smoothing` nested dict in YAML, flows through `extra="allow"` on `MergedSubtitleSettings`. No flat Pydantic fields — the nested dict is passed directly to smoother kwargs.
- **Run paths registration is in TWO places**: adding a new path to `src/video/config/core_models.py::get_run_paths()` is not enough. The runtime `ctx.run_paths` is built by `src/video/producer/state.py::get_video_run_paths()`, which constructs a separate `legacy_paths` dict from the core_models output. New keys must be registered in BOTH places or `ctx.run_paths['<key>']` raises `KeyError` at runtime. The `legacy_paths.update({...})` block at the bottom of `get_video_run_paths` is where additional keys go.
- **Content pillar pipeline**: pillar-aware script generation is wired through `script_templates.{pillars, pillar_preambles, pillar_audiences, narrator_profile}` in `config/ai_services.yaml` and consumed in `src/ai/script_generator.py`. The `--pillar` flag on producer + global_batch sets `ctx.state["pillar"]`; `step_generate_script` passes it to `generate_ai_script`, which filters templates, prepends the per-pillar preamble (after the channel-wide narrator profile), and substitutes `{AUDIENCE}` from `pillar_audiences`. Unknown pillars log an info-level hint and gracefully no-op all three. The fully-rendered prompt always lands at `outputs/<asin>/temp/script_prompt.txt`. Platform caption generators (YouTube, TikTok, Instagram) also receive the narrator profile and pillar preamble via `generate_with_llm()` so captions match the video's voice.
- **Keyword-to-pillar attachment**: `config/scraper.yaml` `batch.keywords` is a dict keyed by pillar name (`{value: [...], novelty: [...], utility: [...]}`). The config loader builds a `keyword_pillar_map` on `BatchConfig`, and the batch controller sets `product.pillar` on each keyword-sourced product. The producer and global_batch fall back to `product.pillar` when `--pillar` is not set. Flat-list keyword shape is accepted for backward compatibility (no pillar attached). **Build the map from config regardless of where the keyword list came from.** Both loaders (`src/pipeline/config.py`, `src/scraper/amazon/config.py`) apply CLI-over-YAML precedence to the keyword *list*, but the map is a property of the config and must be populated either way; building it inside the YAML-only branch means every `--keywords` run writes an empty pillar, silently, because a missing pillar is indistinguishable from an unconfigured keyword.
- **Prompt rule placement**: per-script imperative rules go in each template's `## Rules` block in `src/ai/prompts/scripts/*.md`. Channel-wide voice direction goes in `narrator_profile`. The narrator profile sits ~40 lines from the active task in the rendered prompt, so per-script rules buried there bind less reliably (the trade-off honesty rule was the canary case). When adding a rule, ask whether it must apply per-script (template `## Rules`) or describes persistent voice (narrator profile). Inspect `outputs/<asin>/temp/script_prompt.txt` after a run to see what the LLM actually got.
- **Keep each rule compact; long rules with nested sub-bullets crowd attention from neighbouring rules**. A Phase 1.5 closing-line rule that was reliably honoured got crowded out when the sibling Phase 1.1 rule grew into a six-bullet pattern list. Collapsing the list to a prose paragraph and referencing the doc instead restored Phase 1.5 enforcement. Same class as the placement gotcha above: prompt attention is finite. If a rule needs more than a couple of inline shapes, point at an external doc rather than inlining every example. Smoke-test the LLM output after any rule expansion.
- **Body framing and Rules-block instructions must describe the same beat, not compete for it**. A template's body (lines before `## Rules`) reads as structural directive to the LLM — "Open with an anecdote / scene / arrival moment". If the Rules block then adds an audio-keyword hook requirement, the LLM follows the body's "anecdote first" framing and lands the keyword too late. Symptom: first 5 seconds of TTS carry no product, category, or price band; the Rule's anti-pattern check ("read it out loud") never fires because the LLM thinks it complied. Fix is to merge the angle and the keyword into one body instruction with keyword-embedding example openers, not to copy the Rule into the body or strip the Rule. The Rule stays as the canonical spec; the body framing matches it. Audit pattern: for each template, ask whether the opening body sentence and the first Rules-block bullet describe the same first sentence of script output, or compete for it.
- **Examples teach the LLM by demonstration; when examples contradict a rule, examples win**. Sibling to the body-vs-Rules gotcha above, hit on the platform caption prompts in `src/ai/prompts/{tiktok,instagram,youtube}_*.md`. Each prompt has a rule asking for one behaviour (e.g., end the caption body with the script's closing engagement-bait line) AND an `Examples` block showing 3-4 sample outputs that demonstrate the OPPOSITE shape (end with `#ad`, end with the URL, etc.). The LLM follows the example shape, not the rule, because examples sit closer to the output position and look like ground truth. The TikTok prompt compounded this with a body framing of "NO creative hooks" that the LLM read as forbidding the engagement-bait closing line; the fix carved out the closing line as a named exception in both the body framing AND the example block. Audit pattern: for each platform prompt, check that every example output already demonstrates the rule. If three examples end one way and the rule asks for another, the rule loses.
- **A required rule that doesn't fit the product makes the LLM fabricate**. Phase 1.5 required analytical templates to close with a "debatable spec claim." On a passive product (phone holder, hook, organizer — no contestable numeric spec), the LLM had no spec to argue about, so it invented one and walked it back in the same sentence ("eight hours of battery life is the sweet spot for phone holders, even though this doesn't have a battery"). The class of bug is: any rule that assumes a property the product may not have will produce fabrication on the products that lack it. Fix: branch the rule on a mechanical keyword self-check ("does the description contain W, mAh, Hz, GHz, ports, hours of battery, ..."). Provide both a spec-shaped close and a fallback shape (material-or-use claim) and have the LLM pick exactly one. Anchor the rule against the actual canary case in the negative-example slot ("don't claim battery life for a phone holder"). Audit pattern: for each "always do X" rule, ask whether X assumes a property the product may not have. If yes, the rule needs a conditional branch.
- **A worked example inside a conditional branch teaches its SUBJECT, not just its shape — delete the example, don't add another warning.** The branch above still fabricated on its second outing: it produced "They say four ports is the minimum you need, but honestly, three is usually enough" for a Bluetooth tracker tag, despite already carrying the explicit "never invent a spec the product doesn't have" prohibition. The spec branch demonstrated itself with `"Most people only need two ports, but three is usually better."` and the output is that example with the numbers shuffled — the model copied the subject (ports) onto a product with none. Compounding it, the branch condition said to scan the description for tokens including "ports", and the product text had no standalone "port", only `supports` / `Portable` / `Important`; a substring scan matches all three. Fix that worked: strip the worked example from the branch that fabricates (the model must then derive the subject from the value it was told to quote), keep examples only on branches where a wrong subject is impossible (the non-numeric material-or-use branch), and make the self-check demand a verbatim quote with the unit attached as a whole word, naming the substring trap inline. Note the existing prompt test asserted the OLD rule's strings and blocked the fix — a prompt test must assert the current contract, including that the removed example is ABSENT.

### Pycaps Subtitle Engine Notes

- **Bundled default is pycaps; Pydantic default is ffmpeg.** `config/subtitles.yaml` sets `subtitle_settings.subtitle_engine: "pycaps"` so users running the bundled config get animated captions out of the box. The `SubtitleSettings` Pydantic field still defaults to `"ffmpeg"` for programmatic construction without YAML. Default install does NOT pull pycaps — run `poetry install --with pycaps` (+ `poetry run playwright install chromium` for the CSS renderer). Forks without the optional group degrade silently to FFmpeg via `pycaps.fallback_policy: "fallback_ffmpeg"` in the bundled YAML.
- **Whisper transcript is a literal drop-in**: the raw `result_w` dict from `stt_functions.py:118` is exactly pycaps' `whisper_json` format. `generate_subtitles_with_whisper(..., transcript_out_path=...)` serialises it unconditionally when set (NOT gated by `debug_mode` — pycaps production runs need the file).
- **Post-assembly burn step**: `step_burn_pycaps_subtitles` runs after `assemble_video` and short-circuits when `engine != "pycaps"`, so it's safe to always include in the pipeline graph. Uses `asyncio.to_thread` because pycaps is sync.
- **Content-aware offset formula**: `offset = (bounds.y + bounds.height + 0.02) - 0.95`, clamped to `[-0.9, 0]`. Matches pycaps' `LayoutUtils.get_vertical_alignment_position` bottom-anchor formula so captions land in the whitespace below the product image. See `src/video/pycaps_engine/renderer.py::layout_from_visual_bounds`
- **Template selection**: deterministic md5 hash `md5(product_id + ":pycaps_template")[0:8] % len(pool)`. Empty pool falls back to `template_name`. Same pattern as font/colour/script-template selection
- **Two-part incompatibility**: when engine=pycaps, `step_generate_subtitles` warns and disables `two_part_subtitles_enabled` (now `two_part_subtitles.enabled` after nesting) for that run. Single-line only in v1. The upper URL is NOT rendered.
- **Benchmark numbers** (30s 1080x1920 portrait, 40-word transcript, CSS renderer): word-focus 0.70x realtime, hype 0.79x, minimalist 0.67x. Peak RSS ~420 MB per render. CSS is faster than pictex on production-length clips.
- **CSS renderer on Ubuntu 26.04 needs a Playwright override**: `playwright install chromium` fails with `does not support chromium on ubuntu26.04-x64` (Playwright's platform detection emits `ubuntu26.04-x64`, but the registry only ships up to `ubuntu24.04`; affects ≤1.60.0, fixed in unreleased 1.61). Force the binary-compatible 24.04 build with `PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64`. At runtime it's set automatically for the CSS renderer on Ubuntu-like distros at version 26+ by `_ensure_playwright_chromium_platform()` in `src/video/pycaps_engine/renderer.py` — the single source of truth, covering standalone producer, batch, `make`, and tests (it runs before any Playwright launch, so the Makefile needs no env wiring). It matches Playwright's own distro set (`ubuntu`, `pop`, `neon`, `tuxedo`) and uses setdefault semantics, so an explicit env wins. Only the one-time `playwright install chromium` still needs a manual prefix. Full writeup in `docs/troubleshooting.md`.
- **CSS renderer needs a virtual display on Wayland/headless boxes**: after the install override above, the `css` renderer launches Chromium but every `page.screenshot` hangs (`Timeout 30000ms exceeded`) because headless Chrome can't rasterize a frame without a usable X display (same reason the scraper runs under Xvfb). Confirmed independent of Chrome version (bundled 145 and system 149 both hang). Wrap the producer in `xvfb-run -a` (apt `xvfb`). The `pictex` renderer is browserless and sidesteps both the install override and the Xvfb requirement, but it is **preview-only** and cannot be used as the workaround: it drops the gaps between words. Use `xvfb-run -a` with the CSS renderer. See `docs/troubleshooting.md`.
- **Font randomization**: the project's `subtitle_randomize_fonts` doesn't affect pycaps — templates ship their own `@font-face` in `resources/`. Custom fonts go into a template directory, not into the global font pool.
- **Font/color pools**: live in `config/subtitles.yaml` under top-level `font_pool` / `color_pool` keys, validated by `FontPoolEntry` / `ColorPoolEntry` on `VideoConfig`. To add or remove an entry, edit YAML — no Python changes needed. `FontManager` and `ColorManager` load from these pools; selection methods return string names (not enums). Old pair names (`vibrant`, `warm`, `modern`) fall back to `classic` with a warning for backwards compatibility.
- **Fallback policy**: `raise` (Pydantic default) aborts the pipeline if pycaps is unavailable or fails. `fallback_ffmpeg` (bundled YAML default in `config/subtitles.yaml`) switches to the FFmpeg subtitle engine **only for the pycaps-unavailable case**, caught early in `step_generate_subtitles` (before assembly, so the assembler burns ffmpeg captions); this is what forks without `--with pycaps` hit. `warn_and_skip` keeps the video without subtitles (not recommended). The two cases are distinct: any **burn-step** failure (pycaps installed but the burn can't produce captions — missing transcript, missing assembled video, or a render failure like the CSS renderer with no display) happens after assembly, so there's no ffmpeg burn to fall back to. All three branches route through `_handle_pycaps_burn_failure` in `steps.py` via `return _handle_pycaps_burn_failure(...)` (folded return so a skip can't fall through to the video swap): `warn_and_skip` keeps the caption-less video, but `raise` and `fallback_ffmpeg` both abort rather than silently ship a caption-less video reported as success.
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
- **`openai-whisper==20240930` won't build in a fresh PEP517 env.** Its legacy `setup.py` imports `pkg_resources`, which setuptools removed in `>=81`. Poetry's isolated build env always grabs the latest setuptools, so `poetry install` fails on a clean venv with `ModuleNotFoundError: No module named 'pkg_resources'`. Fix: `pip install 'setuptools>=78.1.0,<79' wheel` into the active venv, then `VIRTUALENV_SYSTEM_SITE_PACKAGES=true poetry install` (the build env then sees the venv's 78.x setuptools). `PIP_CONSTRAINT` does NOT propagate into Poetry's build subprocess. Never `pip install openai-whisper` directly to work around it — pip pulls `torch` (CUDA wheel) from PyPI over the `pytorch-cpu` source pin, Poetry then thinks the venv is satisfied and skips the CPU build; recover by recreating the venv. Full writeup in `docs/troubleshooting.md`.
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
- **Affiliate literal phrase is opt-in and defaults to OFF**: configured via `config/publisher.yaml::affiliate_disclosure`. When enabled, the phrase is injected into every post's caption between the `#ad` line and the description. Supports non-Amazon programs via the `program` key. The closing-frame overlay is deferred. The default is off because the phrase asserts membership of the named program, so emitting it without an active account misstates a material connection; the config loader also falls back to the dataclass defaults when the YAML section is missing **or empty**, which is why the default itself has to be off rather than only the bundled YAML. Distinct from the `#ad` disclosure, which is unconditional.
- **Zernio SDK gaps**: YouTube `paid_promotion` and Instagram paid-partnership tagging are NOT exposed by the SDK (verified via SDK source + Zernio docs full-text search + per-platform reference pages). Manual workaround documented in `docs/compliance.md` (per-video YouTube Studio checkbox + Instagram post-edit "Add Paid Partnership Label"). Don't bypass Zernio with direct platform APIs unless volume crosses ~5/day or analytics show real ranking impact.
- **Registry pillar source is `pipeline_state.json`, not `data.json`**: `_read_pillar_from_state` in `product_registry.py` reads from the producer's state file because that's what the producer actually used. `data.json` now carries `pillar` from the scraper-side attachment, but the registry intentionally reads the producer's state to tag rows with the pillar the video was actually rendered under (CLI override wins over product-level pillar).
- **Fixing failed TikTok posts**: Use Zernio SDK `posts.aupdate()` to set correct `tiktokSettings` per-platform, then the platform status resets from `failed` → `pending` and auto-publishes. No need to call `retry()` — the update triggers re-publish automatically. Calling `retry()` after that gives 409 "Post is currently publishing".
- **Fixing failed Instagram legs (`instagram container error: ERROR`)**: distinct from the TikTok fix above. This is a transient IG Graph container failure (`errorCategory: platform_rejected`, no `platformPostId`), not a settings problem — when YouTube/TikTok got the same media fine, the file is almost always OK. Use `posts.retry(post_id)` (NOT `update()`, which is for payload changes like the TikTok disclosure case): it flips the failed leg `failed` → `processing` and auto-republishes, leaving already-published platforms untouched (no duplicates). No re-render needed — Zernio keeps the upload on its own CDN (`media.zernio.com`), independent of Vercel Blob retention, so the media survives even after the product dir + blob are cleaned up. Resolves to `published` in ~1 min; poll `posts.get`. The `verify-delivery` command sweeps recent posts and WARNs on any `partial`/`failed` one with its failing platform and error, so a silently-dropped leg stops being invisible (`src/publisher/partial_post_sweep.py::sweep_partial_posts` is the reusable check).
- **Zernio SDK post methods**: `create`, `get`, `update`, `delete`, `retry`, `list` (+ async variants `acreate`, etc.)
- **SDK returns Pydantic enums, not strings**: `accounts.list()` returns `SocialAccount` objects with `platform` as a `Platform5` enum. Downstream code (`cli.py`, `batch.py`, `global_batch.py`, `schedule.py`) treats `acc["platform"]` as a lowercase string. Normalize in `get_accounts()` by unwrapping to `.value` before returning the dict, otherwise `.lower()` or direct string comparison fails with `AttributeError`.
- **Publishing modes**: Unified (default) = 1 post to all platforms; platform-specific (`--platform-specific` or `use_platform_specific_content: true`) = 1 post per platform with optimized metadata. Shared helper: `src/publisher/publish_modes.py`.
- **Config loading gotcha**: `publisher.yaml` may contain keys not in `PublisherConfig` dataclass (e.g. deprecated `backoff_multiplier`). The config loader in `src/publisher/config.py` strips unknown keys before constructing `PublisherConfig(**config_dict)`.
- **`.env` file**: Auto-loaded by the CLI via `load_dotenv()`. No manual sourcing needed.
- **CLI format**: `poetry run python -m src.publisher.late single <PRODUCT_ID> --debug`. Takes product ID (not file paths). Video is auto-discovered from `outputs/<product_id>/`. Don't pass `--immediate` or `--platform-specific` by default.
- **Shared link-in-bio post-publish hook**: `update_link_in_bio_safe()` in `src/publisher/link_in_bio/manager.py` is the single post-publish hook for all four publish paths (single, schedule, batch, global batch). Never raises — failures only WARN. Defaults to an enabled config. Pass `LinkInBioConfig(enabled=False)` to skip. Every new publish path should call this hook.
- **Registry writes are protected by `.bak` files**: `save_registry()` renames each existing `published_products.json` / `published_products.csv` to `<name>.bak` before writing the new copy. `rebuild_registry()` merges scanned entries into the existing registry instead of replacing it, so rows whose product directories were cleaned up after publishing stay in the registry. Restore from `.bak` if a write needs to be reverted.
- **`record_publish` writes are isolated per platform**: the publish-record loop in `cli.py::_record_publish_results` wraps each `record_publish` call in `try/except OSError` and logs the failure, so a tracking write that fails for one platform doesn't drop the others or surface as a publish error after Zernio already accepted the post. `outputs_dir` is resolved to an absolute path at `cmd_single` entry so anything that changes cwd downstream can't redirect tracking writes.
- **Title and description are clamped before publish**: `PublishMetadata.clamp_to_limits()` trims either field on a word boundary with an ellipsis when it exceeds the platform's hard cap (YouTube 100/5000, TikTok/Instagram 2200). Called from both `_publish_unified` and `_publish_platform_specific`. **The call must stay ahead of `_build_platform_contents_with_comments`**, which copies `metadata.title` into the per-platform payload: clamping afterwards would leave the copy unclamped and YouTube rejects an over-cap title. Scraped Amazon titles routinely run past 100 chars (190+ is common), so this is the normal path, not an edge case.
- **`platform_contents` entries are the authoritative per-platform payload, not a comment side-channel.** `_build_platform_contents_with_comments` builds them and `LatePublisher._build_sdk_platforms` reads `content` and `title` back out of the same dict. A partial entry is silently destructive: a missing `content` blanked `customContent`, and a missing `title` meant no YouTube title was sent at all, so the platform derived one from the caption's first line, which the `#ad` disclosure leads. Add every field the consumer reads when adding a key to either side. Hashtag-count violations stay WARN — auto-fixing them would invent or drop tags.
- **YouTube rejects descriptions with literal `<` / `>` characters**: YouTube returns `Video description is invalid` when the caption contains unescaped angle brackets (e.g., a product claim like `<10ms UPS`). The unified caption is used as the YouTube description, so any `<` or `>` from the script/description must be sanitized before publish (e.g., rewrite `<10ms` as `sub-10ms`). To fix an already-failed post, `posts.update(post_id, content=sanitized)` the unified content and then `posts.retry(post_id)`.
- **The two tracking files answer different questions; only `publish_history.json` backs the duplicate guard.** `publish_history.json` is `{"posts": {"<ASIN>:<platform>": {...}}}` — one entry per product/platform pair, keyed by the **compound** `product_id:platform` string. `is_already_published` (`src/publisher/tracking.py`) looks up exactly that key, so it is the file that decides whether `single` / `schedule` skip a product. `published_products.json` is a flat list of product rows (`product_id`, `title`, `url`, `affiliate_url`, `pillar`) with **no publish fields at all** — it records what was produced, not what went live, and the production phase writes a row even on a `--skip-publish` batch. Two consequences: a registry row alone never blocks a publish, and any script that probes history with a bare ASIN (`posts.get("B0...")`) always misses, because the key needs the `:platform` suffix. Verifying "was this published" from the bare ASIN is a silent false negative — use the compound key, or Zernio.
- **Force-republish overwrites the tracking row, losing the previous post_id.** `record_publish` writes `posts["<ASIN>:<platform>"]` unconditionally, so a `single --force` on an already-published product replaces the old `post_id` with the new one while BOTH posts stay live on Zernio. Local history then shows only the newest. To find the older post, scan Zernio (`GET /api/v1/posts?page=N&limit=50`) for the ASIN in `content` rather than trusting local state.
- **Local tracking can drift from Zernio in both directions; reconcile against Zernio, not against the other local file.** Observed on a real audit: two `publish_history` entries pointed at post_ids that return HTTP 404 `post_not_found` (posts deleted on Zernio, local row kept, so those products stay blocked from republishing), and one product live on all three platforms was absent from both local files (so `single` would have cheerfully created a duplicate). Neither direction is detectable by diffing the two local files against each other — only a Zernio sweep finds them.
- **`publish_history.json` `published_at` is queue time, not actual publish time**: The field is written when `client.posts.create()` succeeds, i.e. when Zernio accepts the post into its scheduler. The video is still queued and won't go live until `scheduledFor`. To answer "is this video actually live yet", query Zernio directly via `client.posts.get(post_id)`. Sorting `publish_history.json` by `published_at` to find "the latest live videos" is wrong: it gives you the latest-queued, not the latest-published.
- **Reading Zernio post status**: `client.posts.get(pid)` returns a `PostGetResponse` whose payload is nested under `.post` (Pydantic alias). Dump with `resp.model_dump(by_alias=True, mode="json")["post"]`. Without `by_alias=True`, every aliased field comes back as `None` and you get a misleading "no status / no scheduled_for" reading. Per-platform `status` values: `pending`, `published`, `failed`, `partial`. Real publish time is `platforms[*].publishedAt` (camelCase, only set once the platform actually publishes).
- **Empty-string `platformPostUrl` breaks the SDK's strict URL model**: a published TikTok leg often returns `platformPostUrl: ""` (TikTok gives no URL), and `None` validates but `""` doesn't, so a raw `client.posts.list`/`get` raises (`Input should be a valid URL, input is empty`) on any page containing such a post — persistently, since a published leg keeps `""` (a `processing` leg after `posts.retry` also shows `""` but clears in ~1 min). `LatePublisher.list_posts` / `get_status` (and everything routed through them: `verify-comments`, slot-occupancy detection, blob-retention) tolerate this via `_posts_list_safe` / `_posts_get_safe`: they try the normal SDK call, and on the validation error refetch the raw response, coerce `""` -> `None` (`_coerce_empty_platform_urls`), and validate the sanitized payload. A **direct** `client.posts.list`/`get` call, or a one-off script, still hits the crash — read via the raw REST API (`GET https://zernio.com/api/v1/posts/<id>` or `?page=N&limit=50` with `Authorization: Bearer $LATE_API_KEY`) to bypass it, and check first-comment delivery via the comments inbox (`pub.get_post_comments(platformPostId, accountId)`).
- **Retrying failed/partial posts (no CLI command for it)**: there's no built-in "retry failed" subcommand. List posts via the raw paginated SDK (`pub.client.posts.list(page, limit=50)`, `model_dump(by_alias=True)`), filter to the target `scheduledFor` date and top status in `{failed, partial}`, then `pub.client.posts.retry(post_id)` per post. `retry` re-publishes only the failed leg(s) and leaves already-`published` platforms untouched (no duplicates). This is the right call for a transient `TikTok upload failed or timed out` leg too (the disclosure-settings `update()` path is only for the "no option selected" rejection). `load_dotenv()` with no args resolves the `.env` relative to the calling script's own directory, so a one-off script living outside the repo root won't find the project `.env` — pass the repo's `.env` path explicitly (`load_dotenv(REPO_ROOT / ".env")`).
- **First comments fail silently; verify via the inbox, not `posts.get`**: Zernio accepts `platformSpecificData.firstComment` (YouTube + Instagram; TikTok returns None) and reports the post `published` with no error, but the comment can fail to post on-platform. The `posts.get` response only echoes the `firstComment` input. The platform inbox is the only signal: `client.comments.get_inbox_post_comments(post_id=<platformPostId>, account_id=<accountId._id>)` returns the delivered comments, and our first comment is the only one with `from.isOwner == true` (comment text is in `message`, not `text`). The `verify-comments` CLI subcommand sweeps recent published posts and WARNs on misses; `src/publisher/comment_verify.py::verify_post_first_comments` is the reusable check. Delivery is per-account flaky (seen one platform stop posting comments for a week while still publishing videos) — re-auth the failing account in Zernio. The comment also lags the video publish, so an inline check right after publish races it; the sweep on already-live posts is the reliable path. `firstComment` works on Instagram Reels despite Zernio docs claiming feed-only.
- **`PLATFORM_LIMITS` rows in `src/publisher/models.py` must reflect each platform's hard cap, not prompt-side soft targets.** YouTube row is 5000 (description hard cap), Instagram is 2200 (caption hard cap), TikTok is 2200 (caption hard cap). If a row matches the prompt's "optimal" target instead (e.g. 150 for TikTok), every caption that exceeds the soft target trips a false-positive validation WARN at publish time. The publisher loader logs and proceeds today, but #109's open policy decision could turn validation failures into blockers — at which point a too-tight row would brick every publish. Soft targets belong in the prompt config (`caption_length_optimal`, `caption_length_seo`), where they shape LLM output but don't gate the publisher.
- **Vercel Blob is a staging area, not permanent storage**: videos >4 MB upload to the user's Blob store (`LATE_VERCEL_TOKEN`) via the SDK's `upload_large`; Zernio fetches the blob URL when the scheduled post goes live, after which the blob is dead weight. `blob_retention` in `config/publisher.yaml` (age + total-size policy) trims the store once after each publish run on all three publish paths; `src/publisher/blob_retention.py` holds the logic. Blobs referenced by posts that aren't fully `published` are never deleted (`LatePublisher.get_unpublished_media_urls()` — the normalized `list_posts` drops mediaItems, so it reads the raw paginated SDK response). Retention failures only WARN; without it the 1 GB free tier fills in months and Vercel pauses store access, breaking every >4 MB upload.

## Link-in-Bio Module Notes

- **CLI flags**: `--link-in-bio` and `--no-link-in-bio` override `link_in_bio.enabled` config for single publish
- **Affiliate URL fallback**: Uses `affiliate_link` field first, falls back to `url` if unavailable
- **Image fallback**: Uses `images[0]` URL first, falls back to `downloaded_images[0]` local file upload
- **Lnk.Bio auth**: Requires HTTP Basic Auth (not form-encoded), plus `User-Agent: ContentEngineAI/1.0` header to bypass Cloudflare
- **Lnk.Bio API endpoints**: Auth: `POST /oauth/token`, Add: `POST /oauth/v1/lnk/add`, List: `GET /oauth/v1/lnk/list`, Edit: `POST /oauth/v1/lnk/edit` (undocumented; in-place edit of title AND destination URL via an optional `link` param, same id, same position, same image and `created_at`; `title` is replaced by whatever you send, so echo the current one back when only rewriting the URL), Delete: `POST /oauth/v1/lnk/delete`. `LnkBioProvider` exposes `add_link` / `list_links` / `delete_link` but no `update_link` — reach the edit endpoint directly until that gap closes. Full protocol notes in `docs/lnkbio-api.md`.
- **Lnk.Bio OAuth scope is hard-capped at `basic`**: nineteen alternative scope strings (`full`, `read_write`, `all`, `links.write`, ...) all return `unsupported scope`. There is no premium scope. Don't try to widen access via OAuth.
- **`/lnk/list` 50 ceiling is an API page size, not a bio cap**: pagination is not exposed (page/offset/cursor/etc. all return the same first 50). The bio itself has no link quota on the free plan. Fetching the public page with curl is NOT the workaround: it returns only the newest ~48, the rest being rendered client-side, so both automated sources truncate newest-first on a bio that really held 300 links. Their agreement is a shared blind spot, not corroboration, and any "is the list complete" test built on the two of them gives a confident wrong answer. Enumerate by opening the bio in a browser, letting it load, and saving the page; each anchor carries `data-id`, `href`, and the full `title`. Those ids drive `/lnk/edit` and `/lnk/delete` directly, so the cap limits discovery, not modification.
- **Non-blocking**: Failures never block video publishing; logged as warnings
- **`created_at` is link-add time, not platform publish time**: The bio link is added right after `posts.create` (queue time), but YouTube/TikTok/Instagram only go live when Zernio's scheduler fires `scheduledFor`. The bio link can be clickable for days before the corresponding video is up. Don't use lnk.bio `created_at` to verify a video is actually published; use Zernio's `platforms[*].status` instead.
- **Free plan has unlimited links**: lnk.bio's free tier has no link quota (their headline differentiator vs Linktree). Paid tiers unlock customization (themes, custom domain, analytics), not link count. Safe to keep adding without worrying about a cap.

## Scraper Module Notes

- **URL support**: Scraper accepts full URLs (including shortened URLs like tr.ee) via `--product-ids` or `--input-file`. URLs are detected by `startswith("http")`, navigated directly in the browser, and ASIN is extracted from the redirected URL via regex `/dp/([A-Z0-9]{10})`.
- **CLI args**: `--input-file FILE` (one URL/ASIN per line), `--batch-size N` (process in chunks), `--output-dir DIR` (override output directory).
- **Botasaurus output dir override**: Botasaurus framework callbacks don't accept custom parameters. Use module-level `set_output_dir()` in `botasaurus_output.py` before running the scraper. The `_effective_dir()` helper resolves: explicit param > module override > None (config default).
- **Variable initialization in browser_functions.py**: Variables used after `if is_url / elif is_asin / else` branching (like `count_products_with_media`, `products_with_media_count`, `max_products`) must be initialized **before** the branch, not inside one branch. Same trap for imports: a local `import X` inside a conditional branch makes `X` function-local for the **whole** function, so any path that skips the branch hits an unbound local when it touches `X` (a redundant `import time` inside `if DEBUG_MODE:` did exactly this and crashed every non-debug keyword scrape). Keep imports at module level. The function runs behind a Botasaurus `@browser` wrapper so unit tests mock it out; drive it with a mock `Driver` (see `tests/scraper/test_browser_scrape_impl.py`) to actually execute the body.
- **Two scraping code paths**: The standalone scraper CLI uses `scrape_products_unified()` which has a cycling loop (`_scrape_until_validated_count_reached`). The global batch uses `scrape_batch_browser()` + `process_raw_products()` with its own page retry loop in `global_batch.py`. Changes to validation logic must be tested through both paths.
- **Page retry**: Global batch retries with additional search pages when products fail media validation (`max_retry_pages` in scraper YAML). Only applies to keyword searches, not ASINs/URLs.
- **Price/rating extraction in `product_extractor.py`**: `_normalize_price` infers the decimal separator from the value, so it handles both US (`$1,234.56`) and EU (`1.234,56`) formats; a lone separator followed by exactly 3 digits is treated as thousands grouping, not a decimal. Price selectors are scoped to the core price block and skip `.a-text-price` so the struck-through list price isn't read; when only `.a-price-whole`/`.a-price-fraction` are present, `_price_from_parts` reconstructs cents instead of truncating to whole dollars. Rating is never set from the detail page in the scrape path: `ProductData.__post_init__` always sources it from `serp_rating`, and `_product_to_dict` must emit `rating` (it was dropped before, so the fallback never reached `data.json`).
- **Standalone CLI must load `.env` at startup**: `src/scraper/amazon/scraper.py::main()` calls `load_dotenv()` before parsing args. Without it, env-only secrets like `AMAZON_ASSOCIATE_TAG` are invisible to `build_affiliate_url`, which silently falls back to returning the input URL unchanged. The global batch entry point in `src/pipeline/global_batch.py::main()` does the same. Any new CLI entry point that reads env vars needs the same call.
- **`build_affiliate_url` logs WARNING on missing associate tag**: in `src/scraper/amazon/utils.py`. The fallback (return input URL unchanged) is preserved for backward compatibility, but the warning makes the silent-revenue-loss class of bug grep-able in `outputs/logs/scraper.log`. Future work to harden this is tracked as a GitHub issue with the `follow-up` label.
- **`scrapers.amazon.affiliate_links.enabled: false` means "no program", not "misconfigured"**: the missing-tag WARNING exists to catch a mistake, so an install with no affiliate account got a revenue-loss warning per product plus URLs carrying whatever tracking params the SERP happened to attach. With the flag off, `build_affiliate_url` strips to a bare `https://www.amazon.com/dp/<ASIN>` and logs at DEBUG. The flag governs the missing-tag path only: an explicit `associate_tag` or `AMAZON_ASSOCIATE_TAG` still wins, so it can't silently discard a working program. Read via `_affiliate_links_enabled()`, which checks `AMAZON_AFFILIATE_LINKS_ENABLED` first (a blank value falls through, so `FOO=` doesn't read as false) and defaults to enabled (a missing/unreadable config must not silence the warning).
- **URL shortener default is `bare` (no-op)**: bundled `config/url_shortener.yaml` ships `provider: bare`. Picsee is opt-in via `provider: picsee` + `PICSEE_API_KEY`. The `_shorten_affiliate_links` function bypasses the API-key gate for the bare provider. Architectural note: the consumer still reads the YAML inline rather than via the Pydantic `URLShortenerSettings` model. Tracked as a config-hygiene follow-up.
- **Headful Chrome on Wayland needs an X display, and a *visible* one freezes CDP**: the scraper forces `--ozone-platform=x11` (in `_build_browser_config`) so Chrome draws on an X server, not a real Wayland window. Normal runs use Botasaurus's Xvfb (`enable_xvfb_virtual_display=True`). Unsetting `WAYLAND_DISPLAY` alone is not enough — libwayland defaults to the `wayland-0` socket, so the explicit ozone flag is required. A real on-screen window on a live Wayland session makes Chromium's CDP endpoint go unresponsive (`Failed to connect to Chrome URL`, then per-navigation `Response not received` ~400s hangs), so debug runs on Wayland route to Xvfb too (guarded on `WAYLAND_DISPLAY` in the `force_real_browser` branch) and are invisible. To watch a debug scrape, `make scrape-watch` runs it on a dedicated Xvfb + `x11vnc` (VNC at `localhost:5900`). `x11vnc` refuses to start on a detected Wayland session, so the target launches it with `WAYLAND_DISPLAY`/`XDG_SESSION_TYPE` truly unset (`env -u`) — an empty value isn't enough for `getenv`. The browser launch profile and navigation timing are logged at DEBUG to triage display/CDP issues from `outputs/logs/scraper.log` alone.
- **NO_PROXY CIDR caveat for Chrome CDP**: Botasaurus connects to Chrome's DevTools endpoint at `127.0.0.1` via Python's `urllib`. Python's proxy bypass check does not parse CIDR notation (`127.0.0.0/8`), so if a system proxy is configured, `127.0.0.1` requests get routed through the proxy and fail with `502 Bad Gateway` or timeout. The fix (in `browser_functions.py` at module import) adds the exact `127.0.0.1` to `NO_PROXY`. The same issue affects the Zernio SDK (httpx), which also reads proxy env vars — if `ALL_PROXY` is set to a `socks://` URL, httpx raises `ValueError: Unknown scheme for proxy URL`. Workaround: unset `ALL_PROXY` / `all_proxy` before running the publisher.
- **Vercel security checkpoint through proxies**: Requests to Zernio API through an HTTP proxy can hit a Vercel WAF challenge (HTTP 403 with `X-Vercel-Mitigated: challenge`). Retrying after a brief pause usually succeeds, as the challenge is per-request, not per-IP. The no-proxy path avoids the challenge but is significantly slower on some networks.
- **TikTok account restrictions are not retryable**: `This TikTok account has been restricted from posting. Please check your TikTok account status.` is a TikTok-side account restriction. Retrying via Zernio `posts.retry()` produces the same error. The account must be resolved in TikTok's app or support.

## Module/Batch Alignment Rule

**CRITICAL**: Standalone module CLIs (publisher, scraper, producer) and `global_batch.py` often have parallel implementations of the same logic (scheduling, validation, retry, cleanup). When fixing or adding behavior in one path, **proactively check the other path** for the same issue or missing feature. Don't wait for it to break separately. The batch pipeline re-implements logic from standalone modules rather than calling them, so drift is common and silent.

- **The random profile-pool "all profiles" fallback exists in FOUR places**: `src/video/producer/utils.py::load_profile_pool`, two spots in `src/pipeline/global_batch.py`, and `src/pipeline/config.py::validate_global_batch_config`. The last one is the one that actually matters for the batch: it populates an empty `config.profile_pool` with the profile list, and that populated pool is what `select_profile_for_product` selects from. The two `global_batch.py` spots build local pools used elsewhere. So a change to "which profiles are random-selectable" (e.g. excluding `base`) has to land in `config.py` too, or the batch keeps using the old set. Grep all four for `video_profiles` when touching random selection.
- **Two `SearchParameters` classes hold the same concept**: `src/scraper/config_models.py` (Pydantic) and `src/scraper/amazon/models.py` (dataclass extending `BaseSearchParameters`). The global batch's `scraper_filters` uses the dataclass one; its defaults can drift from the Pydantic one (the `sort_order` default did: dataclass inherited the base's `"relevance"`, Pydantic had `"relevanceblender"`). When changing a search-param default, change both or confirm which path consumes it.
- **The batch publish phase does NOT apply the duplicate-publish guard**: `single` / `schedule` skip a product's already-published platforms via `is_already_published` (override with `--force`), but `global_batch.py::_execute_publishing_phase` publishes every produced product unconditionally. So a batch that re-scrapes an already-published ASIN (same keyword returning the same top result) schedules a NEW post for it, on top of the existing one. This is accepted behavior: the batch re-publishes. Note the asymmetry within the batch itself: the registry write dedups (`add_to_registry` refreshes the row, logs "already in registry") and link-in-bio dedups (`Link-in-bio skipped (duplicate)`), but the post creation doesn't, so tracking keys get overwritten with the new post_id while a second Zernio post exists. If you want per-product dedup in the batch, filter `produced_videos` against `publish_history.json` before the publish phase.

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
