# Content research reference

`python -m src.research <stage>` measures what the configuration should change: search demand for the scraper keywords and the topic pool, and (in later releases) text-only script samples and their verification. It reads configuration and writes a report; it never edits configuration. Design: [0023](../design/0023-content-research.md). How to run it: [the research guide](../guides/research.md).

## Command

| Argument | Meaning |
|---|---|
| `demand` | Measure Google Trends interest, Google autocomplete, Wikipedia pageviews and recent Stack Exchange questions, write `demand.json`, then the report |
| `sample` | Generate text-only scripts through the producer's script step for the pool topics under each variant and the most recent scraped products, write `samples.json`, then check them and render the report. A rerun into the same run directory clears that variant's earlier samples first, because the script step would otherwise reuse a script already on disk |
| `check` | Re-run the measured checks over `samples.json` (no model call), write `checks.json`, render the report |
| `verify` | Check every topic sample's steps and claims with one grounded Gemini call each (the LLM key from `.env`), write `verification.json`, render the report |
| `report` | Re-render `report.md` from the records already in the run directory, with no request |
| `--config PATH` | Research settings (default `config/research.yaml`) |
| `--out DIR` | Run directory (default `<outputs>/reports/research-<date>`) |

Exit codes: 0 on success; 2 when pytrends is not installed (`poetry install --with research`), when `verify` finds no `samples.json`, or when the LLM API key is not set. It loads `.env` for `PIPELINE_TOPICS_FILE` and the outputs root, as the batch does.

## Inputs

- **Scraper keywords:** `batch.keywords` in `config/scraper.yaml`, every pillar.
- **Topic pool:** the pool the batch reads (`PIPELINE_TOPICS_FILE`, then `global_batch.topics_file`, then `global_batch.topics` in `config/pipeline.yaml`). A topic is measured by its optional `search` field, where its title is too long for Google Trends, and otherwise by its title without "How to".

## `config/research.yaml`

| Key | Default | Meaning |
|---|---|---|
| `countries` | `[US]` | Countries to measure, as Google Trends codes |
| `request_pause_sec` | `10` | Seconds before each Trends request; it answers 429 to a faster client |
| `max_retries` | `2` | Retries for a failed request, each after a longer pause |
| `products.anchor` | required | The broad term every keyword's interest is a share of; keep it the same between runs |
| `products.drop_below` | `0.25` | A keyword under this share of the keywords' median in every country, and not rising (last quarter at most 1.1 times the same quarter a year earlier), is a drop candidate |
| `products.related_from` | `5` | Rising searches next to this many of the strongest keywords become add candidates: the 20 fastest-rising are measured (twelve months only) and kept when at or above the keywords' median share in some country |
| `products.wikipedia` | `[]` | Entries `{keyword, article}`: the English Wikipedia article a scraper keyword's views are read from; a keyword no longer scraped is skipped. Redirects are followed and the report names the article reached. Map a keyword only to an article about that product |
| `topics.anchor` | required | The term every topic's interest is a share of |
| `topics.suggest_stems` | `[]` | Autocomplete stems; suggestions no pool topic covers are listed |
| `topics.stack_exchange.sources` | `[]` | Stack Exchange sites to read, each `{site, tagged}`; `site` is the API name (`apple`, `android`, `superuser`), `tagged` one tag or empty for the whole site |
| `topics.stack_exchange.days` | `90` | Questions asked in this many days |
| `topics.stack_exchange.pages` | `3` | Pages of 100 questions per source (1-10), newest first; a source with more questions in `days` than this holds is read only for its newest. Without a key the API allows 300 requests a day |
| `topics.stack_exchange.top` | `10` | Questions listed per source, by views per day since asked |
| `sample.topics` | `16` | Pool topics to sample, in pool order |
| `sample.products` | `8` | The most recently scraped products to sample (under `shipped` only) |
| `sample.profile` | `slideshow_stock` | The profile whose name the run paths use; the script step reads nothing else from it |
| `sample.variants` | all three | `shipped` (required, the baseline), `step_lists` (step lists on), `task_answer_first` (a "How to" topic on `topic_answer_first` only) |
| `sample.band` | `[75, 100]` | The word band a script is checked against; a step-list script uses its step count's band |
| `verify.model` | `gemini-3.7-flash` | The model the verification call uses, with Google Search grounding |
| `verify.timeout_seconds` | `90` | Seconds before a verification call is recorded as failed |

Every block refuses an unknown key. Variants are applied in memory; no YAML is changed.

## Outputs

In the run directory:

- `demand.json`: per keyword and topic, per country, `share` of the anchor (median week over median week), `trend` (the last quarter's median over the same quarter a year earlier, from five years; above 1.1 is a rise, below 1/1.1 a fall, in the report and the drop rule alike) and `peak_month` (the month that peaked in at least two complete years, by at least 1.2 times that year's median month and at least 1 in Trends' units; a year whose median month is zero does not vote; "-" when none recurs); a keyword with no interest a year ago and none now reads a flat trend of 1.0; the drop and add candidates, each add candidate with its measured shares; the suggestions, the uncovered ones alternating between stems; `products.wikipedia`, per mapped keyword, the `article` reached, its median monthly `views` over the last 12 complete months and its `trend` relative to all of English Wikipedia (null when the article or the request was missing); `topics.questions`, per Stack Exchange source, the ranked questions with title, link, views, views per day, tags and whether one is answered (null when the request failed); and `missing`, every request that failed, the five-year request included (a missing five-year reading shows its peak month as "-").
- `samples.json`: per sample, its variant and kind, the script, template, CTA, hook headline, sign-off, step count and fact-check record, or the error that failed or dropped it. The scripts themselves are under `samples/<variant>/`.
- `checks.json`: the measured checks per sample and the per-variant summary.
- `verification.json`: per topic sample, each claim's verdict (correct, wrong, outdated or unverified), source URL, quote and correction, and the tally; or the error of a failed call. A verdict without a source is unverified.
- `report.md`: the recommended changes first, then the demand tables and candidate lists, the variant comparison and the verification. A failed request shows as "no data", never as zero.

## Recommended changes

A variant is recommended when a smaller share of its verified scripts (those with at least one verdict that cites a source) has a wrong or outdated step than under `shipped`, with no smaller share in the word band and no larger share of template misfits; otherwise it is kept, or left undecided without verification. `step_lists` maps to `llm_settings.topic_scripts.step_list.enabled: true`; `task_answer_first` maps to `llm_settings.script_templates.topic_templates: [topic_answer_first]` when every sampled topic is a task, and otherwise needs a pipeline change. Keyword drops and adds and uncovered topic searches are listed for consideration. Nothing is applied.
