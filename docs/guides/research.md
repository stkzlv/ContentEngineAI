# Researching what to change

The content research measures demand and writes a report that answers which scraper keywords to add or drop and which questions the topic pool misses. Reference: [research.md](../reference/research.md). Why it exists: [design 0023](../design/0023-content-research.md).

## Run it

```bash
poetry install --with research      # pytrends, once
make research ARGS="demand"         # about 15 minutes for 50 keywords, 16 topics, 2 countries
```

The report is `outputs/reports/research-<date>/report.md`. Google Trends is rate-limited: the requests are paced, and one that still fails is listed under "Missing data". Run again later rather than lowering `request_pause_sec`.

To re-render the report from a run's records without new requests:

```bash
make research ARGS="report --out outputs/reports/research-2026-10-06"
```

## Read it

- **Shares are relative.** A share of 2.0 means twice the anchor term's interest. Compare within one table; the product and topic tables have different anchors.
- **Drop candidates** are far below the keywords' median in every country and not rising. Check the peak month before dropping one: a gift item with a December peak reads low in October.
- **Add candidates** are rising related searches for the product seeds, by growth. A large growth figure on a small base is common; measure a candidate against the anchor (add it to the keywords and run again) before scraping it.
- **Uncovered searches** are autocomplete suggestions no pool topic covers. They show phrasing, not volume. A good new topic names one device or app and one outcome, and passes the topic filter: check it with `python -m tools.check_topic_pool`.
