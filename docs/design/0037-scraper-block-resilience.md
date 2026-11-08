# 0037. Scraper block resilience

- **Status:** Accepted
- **Issue:** #812, #813, #814, #815
- **Requirements:** REQ-SCR-081, REQ-SCR-082, REQ-SCR-083, REQ-SCR-084, REQ-SCR-085, REQ-SCR-087

## Context

Scheduled product runs lose most of their scrape time to the site's block, and the scraper doesn't see it as one. Two weeks of batch logs show:

- **The block is a page titled just `Amazon.com`, not the error page.** The throttle tracker (REQ-SCR-044) watches for "Sorry! Something went wrong!", which appeared once in that window. The bare-title page appeared after the fifth to tenth search of a run and then answered every later search. Each such search waited out the result-card selectors, about 66 s, and was counted as a keyword with no products; one run lost 9 inputs in 835 s.
- **Runs make far more requests than they need.** With `--max-products 1`, the batch's scrape phase browses every input, about ten searches and ten product pages, before validating any of them, so a run reaches the block before it stops.
- **Every run is a new, inconsistent visitor.** No profile or cookies, a random window size, and a random user agent from the driver library's list, which stops at Chrome 120 while the installed browser is far newer. The user-agent flag changes only the string; the client hints and `navigator.userAgentData` still report the real version.

Other scrapers report a "continue shopping" interstitial on search pages [B] ([SerpApi roadmap #4501](https://github.com/serpapi/public-roadmap/issues/4501), October 2026). One report, from another regional store, found that its button leads to `/errors_page/validateCaptcha` and back to the same wall, and that the wall was gone about an hour later [C] ([toolshed #93](https://github.com/ocampor/toolshed/issues/93), October 2026). A scraper classifier published with its markers lists the captcha page, the interstitial (`bm-verify`), the not-found page and the small error page, and backs off 60, 120 and 240 s on an interstitial while never retrying a captcha [B] ([amz-cli](https://github.com/tamnd/amz-cli)). The driver's own `is_bot_detected()` knows only Cloudflare and PerimeterX pages, and the scraper's `bypass_cloudflare=True` does nothing on this site.

## Goals

- A block is recognised on any page the scraper opens, within seconds, and named in the log and the run summary.
- A run that meets a captcha or bare-title page stops browsing instead of spending a minute per remaining input; an error page or interstitial gets the throttle tracker's retry or dead-query handling.
- A batch run makes only the searches it needs to reach `max_products`.
- With the identity switch on, consecutive runs present as the same browser.

## Non-goals

- Solving captchas, or clicking through the interstitial. Its button was reported to lead back to the same wall, and a solved captcha is still a flagged visitor.
- Proxies or residential IP rotation. They cost money and are the owner's call.
- A request-only (no browser) scraper. One public report finds plain requests with honest headers served reliably [B], but it is a single source; it can be tried after this ships.

## Design

- **Page classifier.** One function returns the page kind from the URL, title and a bounded read of the body: `results`, `product`, `error_page`, `captcha`, `interstitial`, `bare_title`, `not_found` or `unknown`. Markers: title exactly `Amazon.com` with no results container; `validateCaptcha` in the URL or a form action, or the captcha prompt text; the "continue shopping" prompt or `bm-verify`; the not-found page's marker; the existing error-page title. It runs after the search navigation, after each product page, and after the return to the results page. The never-called captcha helper in `base/browser_utils.py` is folded in or removed.
- **Blocks by kind.** The four blocking kinds (`error_page`, `captcha`, `interstitial`, `bare_title`) raise one exception carrying the kind. `captcha` and `bare_title` end the run's browsing and are not retried in the same run, since every later search in the logs met the same page. `error_page` and `interstitial` go to the throttle tracker as the error page does today: the growing-wait retry while nothing in the run has succeeded (REQ-SCR-044, REQ-SCR-045), a dead query when other inputs succeed (REQ-SCR-046). `not_found` is not a block: it fails that input only. The run summary lists blocks by kind beside dead queries (REQ-SCR-048).
- **Bounded waits.** Card lookup runs only on a `results` page. On a page the classifier has seen loaded, each selector wait uses a short configured bound (a few seconds) instead of the driver default; `global_settings.browser_config` already declares `element_selection_timeout`, `search_result_wait_reduced` and `search_result_timeout`, which no code reads, so #815 wires one of them or removes them rather than adding a third, and the body text is read once per page.
- **One input at a time in the batch.** The batch scrape phase runs search, product pages, download and validation per input, and stops as soon as the validated count reaches `max_products`, the way the standalone scraper does (REQ-SCR-038). The page-retry setting is unified across both paths in #817.
- **Consistent identity, behind a switch.** `scrapers.amazon.consistent_identity` (default `false` in the bundled config): no `--user-agent` override, so the browser sends its own; a persistent Botasaurus profile (`tiny_profile`, cookies only) under the outputs state directory; a window size drawn once per profile and stored with it, still desktop-width; product pages opened by clicking the result card instead of a fresh navigation. With the switch off, REQ-SCR-039 and REQ-SCR-041 hold as today.
- **Logging.** A search or product page that yields nothing logs its kind at INFO, and for a results page the card counts found and skipped (REQ-SCR-087), so a normal run says why it found nothing.
- **Tests.** Saved pages of each kind against the classifier; a blocked search raising within the bound; a batch with `--max-products 1` whose first input validates making one search; the identity switch on and off passing the expected browser options. Saved pages go in `tests/fixtures` with made-up product ids.

## Alternatives considered

- **Space runs far apart as the fix.** A long gap between runs delays the block to around the fifth to tenth search but each run still makes about ten searches, and a scheduled week takes most of a day.
- **Pin a newer user-agent string.** It drifts again with the next browser update and still disagrees with the client hints unless they are overridden too; the browser's own user agent can't drift.
- **Click through the interstitial.** In the one report of it, its button led to the captcha and back into the same page [C].

## Rollout

The classifier, the block handling, the bounded waits, the logging and the batch change ship on: they change how the scraper fails and how many requests it makes, not what it produces. The identity switch ships off. Turn it on for a week of scheduled runs and compare block pages per run with the week before, read from the batch log.

Remove the switch when: a week with it on has fewer block pages per run than the week before and no new failure kind.

## Open questions

- Whether the bare-title page clears with time, as the interstitial was reported to. If it does, it should get the growing-wait retry too; the logs so far show it lasting the rest of the run.
