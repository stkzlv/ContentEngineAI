# End-to-end checks: the worked cases

<!--
Moved out of CLAUDE.md (#454). The rule -- run the real path a
change touches and judge the artifact, not the exit code -- stays there;
these are the three worked invocations and the caveats they are built
from.
-->

The full-pipeline cases below are one instance. Manual full-pipeline checks (scrape -> produce -> publish) for one random config product, random profile. The scrape, produce, and publish paths each have a standalone module CLI and a `global_batch` variant that re-implement the same logic, so all three cases below exercise different code (see Module/Batch Alignment Rule).

**`xvfb-run -a` is optional for producer runs, kept in the examples as the fallback.** The CSS pycaps renderer (bundled default) used to hang on its per-word screenshots without an X display; re-measured on 2026-09-02 with Playwright 1.58.0 and the bundled Chromium, the burn succeeds with no display reachable at all (`DISPLAY`, `WAYLAND_DISPLAY` and `XDG_SESSION_TYPE` unset, empty `XDG_RUNTIME_DIR`), and no faster under Xvfb: 33s at 466 MB peak against 37s at the same peak. If `Page.screenshot: Timeout 30000ms exceeded` appears, the wrapper is the fix. `poetry.lock` has pinned Playwright 1.58.0 since before the hang was recorded, and 1.58.0 pins Chromium revision 1208 (145), the same build the June hang was recorded on, so neither the Playwright version nor the Chromium revision moved. The browser binaries under `~/.cache/ms-playwright` were reinstalled on 2026-07-30, after the entry was written, as that same revision 1208; the reinstall is the only dated change and is unconfirmed as the cause. `pictex` never needed a display, but **`pictex` is preview-only and must not be used for published output**: it renders words with no gaps between them (`Likemyphonewentfrom`), silently and without error. See [subtitles.md](subtitles.md) and issues #174 and #349.

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
- **Registry**: `grep <ASIN> outputs/state/published_products.json`.
- **Authoritative publish check** (the post actually reached the scheduler): `client.posts.get(<id>).model_dump(by_alias=True, mode="json")["post"]` -> top `status` plus `platforms[*].status` / `scheduledFor`. Don't trust `publish_history.json` `published_at` (that's queue time, not live time).

**Caveats baked in from real runs:**
- **Normal-mode scrape is reliable** now that the browser window size is clamped to desktop widths in `_BROWSER_CONFIG` (`src/scraper/amazon/config.py`). It was `WindowSize.RANDOM`, which could draw a narrow/mobile width that triggers Amazon's responsive layout the desktop card selectors miss, silently yielding 0 products. `--debug` is no longer needed just for scrape reliability (it still pins a fixed window and adds verbosity). A genuine 0-product run on Wayland is a different cause (no X display) — see `docs/troubleshooting.md`.
- **`--force` republishes** a product already published (default off). Fresh scrapes don't need it.
- **Coqui TTS is no longer a dependency**, so a default install logs `Coqui TTS library not available; this provider will be disabled.` once and moves on. The provider code and its config block are kept, and `provider_order` omits `coqui`. Re-enabling is NOT just `pip install coqui-tts` plus a YAML edit. 0.27.5 also needs `transformers >=4.57,<5` (it imports `isin_mps_friendly`, removed in `transformers` 5) and `torchcodec` (required on torch 2.9+), and **`torchcodec` has to come from the PyTorch CPU index** — the default PyPI wheel is CUDA-flavoured and dies on `libnvrtc.so.13`, which is the same source-pins-don't-cascade trap recorded for `torchaudio` in [ci-and-dependencies.md](ci-and-dependencies.md). With all three in place the provider loads fine on the pinned torch 2.13 (verified). Get any of them wrong and it fails quietly: `find_spec` doesn't execute the module, so `COQUI_AVAILABLE` stays `True`, config validation keeps `coqui`, and the break is one WARN at first synthesis while every render falls through to the next provider. The `No espeak backend` ERROR is a separate, later failure that only appears once Coqui imports at all.
- **Random profile is per-run, not pinned** — the same product can draw a different profile across runs.

### Publish-option verification (every publishing path)

Full runbook in `docs/testing.md` -> "End-to-end publish-option verification". Use it to exercise all publishing options before a publisher refactor/release. Render each product with `make batch-lowpri ARGS="... --skip-publish"`, then publish one option combo per product and verify on Zernio. Real posts are created (immediate goes live; scheduled ships at the next slot — `python -m src.publisher.late delete <POST_ID>` to drop one). Cover both publish paths (`single` and `schedule auto`) and both modes (unified, `--platform-specific`) — they re-implement the same logic.

Operational gotchas the runbook captures (must-know inline):
- **Verify surfaces**: `publish_history.json` / `published_products.json` / `schedule.json` live under `outputs/state/` (durable-state directory; legacy root copies migrate on first touch) and survive product-dir cleanup — diff their counts. Authoritative live status is `client.posts.get(<id>)` `.post` (`by_alias=True`), not `publish_history.json` (queue time). Disclosures to confirm: YouTube `containsSyntheticMedia` matches `publisher.yaml::synthetic_media_disclosure` (off by default, so `false` on a stock config); TikTok `platformSpecificData.tiktokSettings.commercial_content_type` is `brand_organic` + `is_brand_organic_post: true` on an affiliate render and `none` + `false` on a topic render (both unified and platform-specific).
- **link-in-bio** runs on the `single` path and reads `outputs/<asin>/data.json`. Config defaults it ON — pass `--no-link-in-bio` to skip. It runs in BOTH branches: after a successful publish, and on the already-published early return, where `cli.py` deliberately calls `update_link_in_bio_safe(...)` before returning ("a fully-published product needs no video or upload: keep its bio link fresh and exit"). So a no-op `single` run on an already-published product still touches the bio — it does not return before the link-in-bio step. That branch also forces `enabled=True` on the passed config, so only `--no-link-in-bio` (which clears the `link_in_bio_enabled` gate) actually skips it. After cleanup the dir is gone; to set the link later, reconstruct a minimal `data.json` (`title` + `affiliate_link`) from `published_products.json` in a temp dir and call `LinkInBioManager.update(<asin>, <tmp_outputs>)`.
- **`schedule auto` needs `--auto-resolve`** to take an alternative when the preferred slot conflicts (2h `min_post_spacing`); without it the product is counted failed (and the schedule path exits non-zero, unlike the batch pipeline which exits 0 on partial failure unless `--strict` is passed — grep phase summaries, not `$?`).
- **Cleanup** skips while a leg is still `publishing` (immediate runs keep the dir) and runs once `scheduled` (dir removed after tracking/registry writes — the #175 path).
- **The #177 SDK crash blocks verification**: a published TikTok leg with `platformPostUrl: ""` makes `posts.list`/`get` raise, so `verify-comments` hard-fails and slot-occupancy degrades. Read status via the raw REST API and first-comment delivery via `get_post_comments(<platformPostId>, <accountId>)`, both of which bypass the failing model.
