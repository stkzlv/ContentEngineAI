# Content Pillars: Follow-Ups

Tracks work left over from the initial content-pillars iteration. Pick up when ready.

## State as of this doc

What's already on `main`:

- Keyword pool grouped by pillar via comment markers in `config/scraper.yaml::batch.keywords` (still a flat list at the schema level).
- Script template -> pillar mapping in `config/ai_services.yaml::script_templates.pillars`. 15 templates assigned across `value` (6), `novelty` (6), `utility` (8).
- Channel-wide narrator profile in `config/ai_services.yaml::script_templates.narrator_profile`. Templates carry only their hook-specific rules; universal rules and anti-AI-tells live in the profile.
- Per-pillar preambles in `config/ai_services.yaml::script_templates.pillar_preambles`. Stacked above the template at runtime.
- `--pillar <name>` flag on both `src/video/producer/cli.py` and `src/pipeline/global_batch.py`. CLI override wins over saved state and over any pillar the product carries.
- `apply_prompt_preambles` stacks narrator profile + pillar preamble + template content with blank-line separators.
- `select_script_template(... pillar=...)` filters the active pool to the pillar's template list, with a fallback to the unfiltered pool when the intersection is empty.

What's missing:

The pipeline only honors a pillar when the user explicitly passes `--pillar`. Without the flag, products have no pillar context and the generator runs against the full template pool. The work below makes unattended batches balance across pillars naturally by attaching each product's source-keyword pillar to its data.

## 3b. Source-keyword pillar attached to product data

### Schema change

Replace `batch.keywords: list[str]` with `batch.keyword_pillars: dict[str, list[str]]` in `config/scraper.yaml`. Update the Pydantic model in `src/scraper/config_models.py` and the loader in `src/scraper/amazon/config.py`.

Add accessors:

- `config.all_keywords() -> list[str]` flattens the dict for callers that want a flat list.
- `config.pillar_for(keyword: str) -> str | None` returns the pillar a keyword belongs to, or `None` if the keyword isn't mapped.

The CLI `--keywords` arg keeps passing a flat list; only the YAML config changes shape. CLI keywords don't have an attached pillar by default (they could be defaulted from `--pillar` if set, otherwise None).

### Call-site updates

About fifteen places read `config.keywords` today. They need to switch to `config.all_keywords()`:

- `src/scraper/amazon/batch_controller.py`: six references including the search loop and progress logging.
- `src/scraper/amazon/config.py`: loader currently reads `yaml_batch.get("keywords", [])`; switch to `yaml_batch.get("keyword_pillars", {})` and build the model accordingly.
- `src/pipeline/config.py:125`: `"keywords": config.keywords` in the producer-overrides dict.
- `src/pipeline/global_batch.py`: keyword summary printout around the configuration banner.
- `src/scraper/amazon/scraper.py`: a few references via `args.keywords`. CLI list stays flat, so most of these stay as-is. Verify which ones touch the YAML-derived list.

Tests under `tests/scraper/`, `tests/test_global_batch_config.py`, and `tests/test_pipeline_graph.py` reference `keywords` and need updating in lockstep.

### Product data + producer wiring

- Add `pillar: str | None = None` to whichever product data class hits `outputs/<asin>/data.json`. Likely `BaseProductData` or the Amazon-specific subclass; check `src/scraper/base/models.py` and the JSON serialization path.
- In the scraper batch loop, when a product is scraped via a keyword search, set `product.pillar = config.pillar_for(keyword)`. Direct ASIN / URL inputs leave it unset (no source keyword).
- In `step_generate_script` (`src/video/producer/steps.py`), when `ctx.state.get("pillar")` is None, fall back to the product's pillar field before passing to `generate_ai_script`. CLI override still wins because orchestration sets `ctx.state["pillar"]` before this step runs when `--pillar` is provided.

### Tests

- Loader tests: `keyword_pillars` parses correctly, `all_keywords()` flattens in config order, `pillar_for()` resolves keywords across pillars and returns None for unknowns.
- Scraper integration test: a keyword search attaches the correct pillar to the resulting product. A direct-ASIN scrape leaves `product.pillar` as None.
- Producer step test: when `ctx.state["pillar"]` is set, it wins; when only `product.pillar` is set, the product value flows through; when neither is set, the generator runs against the full pool.

## 4. Registry persistence

### pipeline_state.json

The full `ctx.state` dict already serializes, so once `ctx.state["pillar"]` is set the value lands in the state file automatically. After 3b, run a small sanity check that the resume path picks the pillar back up correctly across a partial run.

### published_products.{json,csv}

- Add a `pillar` field to the registry record in `src/publisher/product_registry.py`.
- Update the CSV writer's column list and the JSON schema. Backfill is fine: missing entries stay empty.
- `registry --rebuild` should pick up pillar from each product's `data.json` (added by 3b) so existing outputs end up tagged after a rebuild.

## 5. CHANGELOG entries for follow-up work

CHANGELOG entries land per iteration under `[Unreleased]`. The version bump itself happens at release time, after 3b and 4 are in.

When 3b lands, append to `[Unreleased]`:

- `Added`: source-keyword pillar attached to each product so unattended batches balance across pillars without `--pillar`.
- `**Breaking**` (if `keyword_pillars` replaces `keywords` in `config/scraper.yaml`): note the YAML schema change in `Changed`.

When 4 lands, append:

- `Changed`: published-products registry rows now carry a `pillar` column.

At release time, the accumulated `[Unreleased]` section becomes the new version's notes (currently 0.42.x, so a minor bump to 0.43.0 fits since the pillar feature is additive). Move the entries to `## [0.43.0] - <date>` and start a fresh `[Unreleased]` block.

## 6. Description preprocessing for prompt quality

A review of the rendered prompt for B0FXB188B8 showed Amazon product descriptions still leak marketing copy patterns even after Unicode normalization and em/en dash replacement. Examples from one description:

- "Engineered for ultimate portability" — banned word in a positive context.
- "always prepared for road trips, emergencies, and beyond" — generic SEO closer.
- "keeping all your essential tech connected wherever you go" — Amazon catch-all.
- "What's in the Box: 1* ..." — packaging boilerplate that adds no script value.

These read as AI rhythm, and the LLM may borrow the phrasing in output. Two paths:

### Heuristic strip in `format_prompt`
A regex pass that drops "What's in the Box" lines, replaces banned phrases with neutral alternatives, and trims obvious marketing closers. Risk: false positives are easy on natural-language input. The description is the only source of product detail; aggressive stripping can lose useful information.

### LLM-based pre-cleaner
A first-pass LLM call that rewrites the description into clean factual prose before the script-writing call. Better quality, doubles the per-script LLM cost, adds a failure mode (cleaner returns garbage).

### Recommendation
Don't ship either yet. The narrator-profile rule "If the description contains any banned phrase or marketing fluff, paraphrase the underlying feature in your own words" already covers the worst case (LLM quoting banned words). The softer issue is rhythm mimicry. Revisit after watching real script outputs for whether the LLM actually leaks Amazon copy or paraphrases it cleanly. If clean, skip permanently. If leaky, the heuristic strip is the cheaper first step.

## Order of operations

1. 3b first: it's the largest piece and unlocks unattended balanced batches.
2. 4 right after 3b: trivial once the product carries a pillar.
3. Release after both land. The current state is functional only when the user passes `--pillar`; a release that ships pillars but only honors them with an explicit flag would be a half-feature.
4. 6 is unblocked but waiting on real-script-output data to know whether it's worth the work.
