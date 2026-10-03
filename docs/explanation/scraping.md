# Scraping: why the scraper behaves as it does

This page explains the scraper's browser mode, how it meets Amazon's anti-bot defences, how it tells a rate limit from a query that never works, and why its media thresholds and keyword rotation exist. The flags and keys it mentions are listed in [the scraper reference](../reference/scraper.md), the steps of a scrape are in [the scraping guide](../guides/scraping.md), and the requirements are in [the scraper requirements](../requirements/scraper.md). The defects behind the code, and what catches each one, are in [the scraper module notes](../notes/scraper.md).

## Browser mode and anti-bot detection

The scraper drives a real Chrome through Botasaurus (a fork of `nodriver`, itself descended from `undetected-chromedriver`). Two facts drive the design: Botasaurus is unreliable in headless mode, and Amazon runs a serious anti-bot stack. The scraper answers both with a real browser that runs headful, and with human-like navigation (`REQ-SCR-039` to `REQ-SCR-041`).

### Why the scraper never runs headless

Every browser config path hardcodes `headless: False` (`src/scraper/amazon/config.py` and `browser_functions.py::_build_browser_config`). Headless is avoided for two separate reasons, both observed in this project and confirmed upstream:

1. **Detection.** In headless mode Botasaurus and nodriver don't fully patch the browser fingerprint. The user agent still reports `HeadlessChrome/<version>`, and other headless signals leak: missing plugins, GPU and renderer mismatches, `navigator.webdriver` traces. Anti-bot services flag this at once. Botasaurus's own docs warn that headless mode is identified by services such as Cloudflare and DataDome, and recommend it only for sites with no bot protection. Amazon is not such a site.
2. **Stability.** nodriver's headless startup path has crash bugs. This project hit a `StopIteration` during the headless connection setup (hence the `# Disabled - causes StopIteration in headless mode` comments), and upstream tracks a related `TypeError: cannot unpack non-iterable NoneType` in the headless connection code. Headful avoids both.

The cost of headful is that Chrome needs a display. On X11 (Ubuntu 22 and earlier) the session exports `DISPLAY`, so the need is invisible. On Wayland (Ubuntu 26) `DISPLAY` is empty, so headful Chrome has nowhere to draw and `google_get` hangs until the 60 s document-ready timeout. That failure looks like an anti-bot block in the logs but is purely a missing display; [troubleshooting](../guides/troubleshooting.md#scraper-times-out--0-products-on-wayland) has the fix.

### Virtual display

Botasaurus supports a headful but invisible browser through a virtual framebuffer. With `headless=False` and `enable_xvfb_virtual_display=True`, it starts a `pyvirtualdisplay` Xvfb session itself (`botasaurus_driver/core/config.py`). This is the mode for unattended scraping: a real, non-headless browser with no visible window, so the fingerprint stays clean and nothing appears on screen.

Two details matter:

- Botasaurus starts Xvfb on its own only when `is_vmish` is true (Docker, a VM with `VM=true`, Gitpod or Kubernetes). A normal Linux desktop is not `is_vmish`, so the scraper requests the virtual display explicitly with `enable_xvfb_virtual_display=True`.
- The Xvfb binary comes from the `xvfb` apt package. The `pyvirtualdisplay` Python package is installed with the project but only wraps the binary. If the binary is missing, Botasaurus prints a one-line notice and falls back to `--headless=new`, which is the detectable and unstable mode again. Grep `outputs/logs/scraper-<date>.log` for `install Xvfb` to catch this.

Debug mode (`--debug`) depends on the session:

- On an X11 desktop it uses the live session display (`enable_xvfb_virtual_display=False`), so the browser window is visible. If no display is found, it falls back to a virtual display.
- On a Wayland session it runs on a virtual Xvfb display with no visible window, because a headful window on Wayland freezes Chromium's DevTools protocol: DevTools doesn't connect, then each navigation hangs on `Response not received`. `make scrape-watch` starts a dedicated Xvfb plus `x11vnc` so the browser can be watched at `localhost:5900`.

Running the module directly with `--debug` behaves the same way; only `make scrape-watch` adds the VNC view.

### Amazon's anti-bot stack

Amazon doesn't use Cloudflare. Amazon.com is fronted by CloudFront and protected by AWS WAF plus Amazon's own "Robot Check" page. The layers a scraper meets:

| Layer | What it does |
|---|---|
| AWS WAF silent challenge | Background JavaScript checks and a lightweight proof of work that issue a token before content loads. No user interaction: a real browser passes, and a thin HTTP client or a leaky headless browser fails. |
| Robot Check CAPTCHA | Amazon's image-text CAPTCHA page, shown when the silent challenge is not satisfied or the risk is high. |
| Fingerprinting | TLS (JA3), HTTP/2 frame order, header consistency, and JavaScript checks of the browser environment (the headless signals above). |
| IP reputation and rate limits | Blocks on datacenter IPs, per-IP request-rate thresholds, detection of repeated patterns. |
| Behavioural analysis | AWS WAF targeted protections score traffic statistics (timing, navigation patterns, the prior URL) for signs of coordinated bots. |

This is why the scraper navigates with `driver.google_get(url, bypass_cloudflare=True)` and human-like cursor motion rather than fetching the URL directly: an organic referrer and real browser behaviour are what satisfy the silent AWS WAF challenge. A direct `requests`-style fetch gets the Robot Check at once.

### Cloudflare Turnstile and the bypass flag

The `bypass_cloudflare=True` argument on `google_get` is generic Botasaurus machinery, not Amazon-specific. Cloudflare Turnstile runs non-interactive JavaScript challenges (proof of work and browser-environment signals), and most visitors never see a widget. When a visible Turnstile checkbox does appear, Botasaurus's `solve_cloudflare_captcha.py` walks the iframe and shadow DOM and clicks it with a human-like cursor. Since Amazon doesn't run Cloudflare, this path is mostly dormant for Amazon. It still matters because `google_get` routes through the same flow, and the `wait_till_document_is_ready` step that times out on a missing display lives in that module. Treat a 60 s document-ready timeout as a display or navigation failure first, not as a Cloudflare block.

Sources: [Botasaurus](https://github.com/omkarcloud/botasaurus), [nodriver headless bot detection (undetected-chromedriver #2003)](https://github.com/ultrafunkamsterdam/undetected-chromedriver/issues/2003), [nodriver headless exception (#2120)](https://github.com/ultrafunkamsterdam/undetected-chromedriver/issues/2120), [AWS WAF Bot Control](https://docs.aws.amazon.com/waf/latest/developerguide/aws-managed-rule-groups-bot.html), [Amazon CAPTCHA and AWS WAF overview](https://2captcha.com/p/amazon-captcha-bypass), [Cloudflare Turnstile](https://developers.cloudflare.com/turnstile/).

## Throttling and dead queries

Amazon answers a rate limit and a query that never works with the same page, `Sorry! Something went wrong!`, so the run separates them by what its other inputs did. A rate limit blocks the connection, so nothing else gets through either; a dead query is specific to itself, and its neighbours keep working.

The run acts on that (`REQ-SCR-044` to `REQ-SCR-048`):

- An input whose neighbours are also failing is treated as throttled. It waits, doubling from `throttle_backoff_base_sec` up to `throttle_backoff_max_sec`, and retries the same input.
- An input that has failed `dead_query_after` times in a run where something else got through, and has never itself returned products, is named a dead query and skipped, because no wait fixes it. A keyword that delivered earlier in the run is never called dead, whatever its later pages do.
- The run summary reports the two separately, so a keyword that needs replacing isn't confused with one that needed a longer gap.

`dead_query_after` is 3 rather than 1 because the two backoffs it implies are the wait that separates the cases: a rate limit affecting only this input has had several minutes to clear before the verdict.

Waiting is capped for the run as a whole by `throttle_max_total_wait_sec`, not only per input. Per-input budgets don't compose: five blocked inputs at fifteen minutes each is over an hour of an unattended run asleep, and by the time the second one exhausts its budget with nothing having succeeded, the answer is already known. The cap is compared against what the next retry would cost, so it is a ceiling rather than a tripwire. A success doesn't reset it: resetting reads fairer but stops the cap from capping anything, since a run broken up by occasional successes would get a fresh allowance after each one.

Consecutive inputs are also paced by `inter_input_delay_sec` on the happy path. Back-to-back searches are the pattern that draws the block, and the pause costs seconds against a scrape measured in tens of them.

The backoff ceiling is 10 minutes because the block was measured clearing when runs were spaced roughly 8 minutes apart; a schedule topping out in seconds retries inside the block every time and loses the input. Raising the ceiling alone does nothing unless `throttle_max_attempts` is high enough to reach it.

### A rate limit that starts mid-run

One case the classification gets wrong, deliberately: a rate limit that begins partway through a run, after something has already succeeded. Every input after it is waited on for two backoffs and then named a dead query, not just the first. The verdict is wrong for all of them, so the summary lists most of the run as dead queries rather than one bad keyword. It is misattributed, but not mistaken about which inputs were lost or how many. Reading it correctly would need a probe request the scrape doesn't otherwise make.

## Why media thresholds exist

A scraped product is only useful if the producer can render it, so the scraper keeps a product only when its validated media meet the producer's minimums: `min_total_media`, `min_images_if_no_video` and `min_images_with_video`, which it reads from `video_settings` in `config/video_production.yaml` (`REQ-SCR-021`, `REQ-VID-125`). A product that fails is removed with its output directory (`REQ-SCR-023`), instead of reaching the producer and failing there after the scrape has been counted as a success.

With `count_products_with_media: true`, only products that pass count toward `products_per_keyword` and `max_products`. Many search results fail, so taking exactly the target from each page would leave most keyword searches short. The scraper fetches a multiple of the target (`prefetch_multiplier`, capped by `max_batch_size`) and scans further result pages up to `max_pages` (`REQ-SCR-033`). A URL or an ASIN names one product, so for those inputs a further page would only resolve the same listing again, and they get one pass.

Where the profile uses no video, videos are neither extracted nor counted (`REQ-SCR-017`, `REQ-SCR-022`), so an image-only profile isn't held to a rule that counts videos it never uses.

## Keyword pool rotation

A configured keyword pool isn't searched in config order. The keyword loop stops once `max_products` is collected, so it only ever reaches the first few entries, and taking them from the top returned the same products every run. The pool is rotated by date instead: every keyword stays in the list, so a barren search still moves on to the next one, but the starting point advances each day by what one run consumes (`max_products` over `products_per_keyword`), so consecutive days reach different keywords (`REQ-SCR-006`).

It is a reorder rather than the slice the global batch takes, because the standalone loop relies on the rest of the pool as a fallback. Keywords passed with `--keywords` are used exactly as given and aren't rotated (`REQ-SCR-007`): a date-dependent order is surprising in a tool used to reproduce a problem.

## URL shortening

The canonical Amazon URL is about 50 characters and fits every supported platform's caption budget after the disclosure, description and hashtags. The characters a shortener saves don't justify the vendor dependency for most setups, so `bare` is the recommended default (`REQ-SCR-063`).

The `picsee` provider remains for setups with existing `stte.psee.io` short codes, for example in captions already published. One caveat: Picsee captures the input URL verbatim rather than canonicalising it again, so a short code minted while the affiliate URL was bad (no tag, or a stale tag) keeps redirecting to that bad URL until its target is updated through Picsee's API or dashboard. The `bare` provider avoids this class of problem by design.

Amazon's own `amzn.to` shortener isn't available programmatically: SiteStripe mints its short codes in the browser, and there's no public API. If you want short URLs in captions and have SiteStripe access, mint them by hand and bypass this layer.

## The affiliate program flag

A missing associate tag is usually a mistake, so by default it logs a warning on every scrape. An install with no affiliate program says so with `affiliate_links.enabled: false` rather than leaving a dead account's tag in the config, since a tag belonging to a terminated account is still sent to Amazon on every link. The environment variable overrides the YAML so that an install can declare "no program" from `.env` without editing a tracked config file, as the tag itself prefers the environment.
