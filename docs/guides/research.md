# Researching what to change

The content research measures demand and writes a report that answers which scraper keywords to add or drop and which questions the topic pool misses. Reference: [research.md](../reference/research.md). Why it exists: [design 0023](../design/0023-content-research.md).

## Run it

```bash
poetry install --with research      # pytrends, once
make research ARGS="demand"         # about 15 minutes for 50 keywords, 16 topics, 2 countries
```

The report is `outputs/reports/research-<date>/report.md`. Google Trends is rate-limited: the requests are paced, and one that still fails is listed under "Missing data". Run again later rather than lowering `request_pause_sec`.

To sample scripts under each variant (model calls, no scraping or rendering; about four calls per script):

```bash
make research ARGS="sample"
```

Then check the samples' steps against official documentation (one grounded call per topic script):

```bash
make research ARGS="verify"
```

To re-render the report from a run's records without new requests:

```bash
make research ARGS="report --out outputs/reports/research-2026-10-06"
```

## Read it

- **Shares are relative.** A share of 2.0 means twice the anchor term's interest. Compare within one table; the product and topic tables have different anchors.
- **Drop candidates** are far below the keywords' median in every country and not rising. Check the peak month before dropping one. The trend compares a quarter with the same quarter a year earlier, so a holiday item does not read as falling in autumn.
- **Add candidates** are rising searches next to the strongest keywords, kept only when they measure at or above the keywords' median share somewhere; the shares are listed beside each. They are still searches, not products: check each names something to scrape.
- **Recommended changes** lead the report. A variant is recommended only on fewer scripts with wrong or outdated steps, without costing length or template fit; read its evidence line, especially how many topics it dropped. A held switch (step lists) stays off until the reach-test readout whatever the report says.
- **Verification** lists each wrong or outdated step with the page that says so. The verifier is a model too: spot-check a few verdicts against their sources before acting on one.
- **Script samples** compare the variants on one set of topics. A variant is worth turning on when it fixes more than it breaks: fewer template misfits and more search phrases in the first sentence count for it; dropped topics, scripts out of band and more fact-check rewrites count against it. Read a few scripts in `samples/` before deciding; the checks measure form, not whether a step is right.
- **Uncovered searches** are autocomplete suggestions no pool topic covers. They show phrasing, not volume. A good new topic names one device or app and one outcome, and passes the topic filter: check it with `python -m tools.check_topic_pool`.
