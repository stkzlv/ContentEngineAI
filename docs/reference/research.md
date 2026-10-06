# Content research reference

`python -m src.research <stage>` measures what the configuration should change: search demand for the scraper keywords and the topic pool, and (in later releases) text-only script samples and their verification. It reads configuration and writes a report; it never edits configuration. Design: [0023](../design/0023-content-research.md). How to run it: [the research guide](../guides/research.md).

## Command

| Argument | Meaning |
|---|---|
| `demand` | Measure Google Trends interest and Google autocomplete, write `demand.json`, then the report |
| `report` | Re-render `report.md` from the records already in the run directory, with no request |
| `--config PATH` | Research settings (default `config/research.yaml`) |
| `--out DIR` | Run directory (default `<outputs>/reports/research-<date>`) |

Exit codes: 0 on success; 2 when pytrends is not installed (`poetry install --with research`). It loads `.env` for `PIPELINE_TOPICS_FILE` and the outputs root, as the batch does.

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
| `products.drop_below` | `0.25` | A keyword under this share of the keywords' median in every country, not rising, is a drop candidate |
| `products.seeds` | `[]` | Broad searches whose rising related searches are add candidates |
| `topics.anchor` | required | The term every topic's interest is a share of |
| `topics.suggest_stems` | `[]` | Autocomplete stems; suggestions no pool topic covers are listed |

Every block refuses an unknown key.

## Outputs

In the run directory:

- `demand.json`: per keyword and topic, per country, `share` of the anchor, `recent_ratio` (last quarter's mean over the year's; above 1 is a rise) and `peak_month` (from five years); the drop and add candidates; the suggestions; and `missing`, every request that failed.
- `report.md`: the tables and candidate lists. A failed request shows as "no data", never as zero.
