# Publishing: how and why

This page explains how the publisher schedules, retries, cleans up and measures posts, and why each mechanism works the way it does. The settings are in `config/publisher.yaml` and the commands in `src/publisher/late/cli.py`; both are listed in [the publisher reference](../reference/publisher.md), and the tasks are in [the publishing guide](../guides/publishing.md). The requirements are in [the publisher requirements](../requirements/publisher.md) and [the compliance requirements](../requirements/compliance.md). The defects behind this code, and what catches each one, are in [the publisher notes](../notes/publisher.md) and [the link-in-bio notes](../notes/link-in-bio.md); read them before you change the module.

For the disclosure stack the publisher produces (FTC, Amazon Associates, platform policy) and the manual steps per video, see [Compliance](compliance.md).

## The provider

The publisher posts through [Zernio](https://zernio.com), a multi-platform scheduling service, to YouTube, TikTok and Instagram (`REQ-PUB-001`, `REQ-PUB-002`); the platform choices also accept Facebook, Twitter and LinkedIn. Zernio was formerly named Late, and the integration still uses the legacy `late-sdk` package and the `LATE_API_KEY` variable. Old `getlate.dev` and `late.dev` URLs redirect to `zernio.com`.

The publisher uploads each video, creates a post that targets several platforms, and reads each post's status back (`REQ-PUB-003`). Videos over 4 MB are staged in a Vercel Blob store first, and Zernio fetches them from there when the post goes live.

## Unified and platform-specific posts

In unified mode, the default, the publisher creates one post per product for all target platforms, with one metadata set (`REQ-PUB-099`). Each platform still gets its own `platformSpecificData` block (the YouTube title, the TikTok disclosure settings, the first comment), so per-platform behaviour works in both modes.

In platform-specific mode (`--platform-specific` or `use_platform_specific_content: true`), the publisher creates a separate post per platform, each with that platform's title, description and hashtags from `metadata_<platform>.json` (`REQ-PUB-100`). It costs one post id per platform instead of one shared id, in exchange for metadata tuned to each platform's audience and limits.

Titles and descriptions over a platform's hard cap are trimmed on a word boundary with an ellipsis before they reach the provider (`REQ-PUB-102`). The cap applied is the smallest among the platforms a caption reaches, measured on the composed caption, disclosure and hashtags included (`REQ-PUB-103`, `REQ-PUB-104`). Hashtag counts outside a platform's range are only logged (`REQ-PUB-106`), because fixing them would mean inventing or dropping tags.

## Per-platform profile routing

`profiles` maps each platform to a video profile, so YouTube Shorts can take a short cut while TikTok and Instagram keep a longer one (`REQ-PUB-015`). The producer must render the routed profile before the publisher runs; without it, the publisher falls back to the first render in the product directory (`REQ-PUB-016`). A post uploads one file for all its platforms, the render routed to the first platform in its target list (`REQ-PUB-017`), so different files on different platforms need platform-specific posts.

## Scheduling into recurring slots

Without `--immediate` or `--schedule`, the publisher schedules each post into the next free slot from `recurring_schedule.slots`, a weekly list of days and times in a timezone (`REQ-PUB-019`, `REQ-PUB-020`). The bundled schedule has one slot a day at 10:00 Europe/Berlin.

`schedule` in scheduled mode works through the backlog:

1. It loads the recurring schedule and scans the outputs directory for products not already published to every target platform, one render per product (`REQ-PUB-021`, `REQ-PUB-023`).
2. It reads the occupied slots from the provider's whole post list, with no date horizon, and from the local schedule; a slot is taken when either holds a post at that time (`REQ-PUB-025`). If the provider's list can't be read, it schedules around the local schedule alone (`REQ-PUB-026`).
3. It assigns each product the first free slot and creates one unified post, or one post per platform in platform-specific mode, all at the same time.
4. A product that finds no slot within the search limit (100 slots) counts as failed. The command does not fall back to publishing at once.
5. It reports the scheduled times and slot assignments.

### Schedule validation and conflicts

Before a post is created, the validator applies `schedule_validation`: no duplicate product, platform and time; a minimum spacing between posts on the same platform; no times in the past; and a daily cap counted across platforms.

When a slot is occupied or fails validation, the scheduler searches for alternatives starting from the preferred time and sorts them by proximity. It suggests 5. With `--auto-resolve` it takes the first one (`REQ-PUB-027`); without it, it logs the suggestions, such as `Suggested alternatives: 2026-01-20T14:00:00, 2026-01-22T10:00:00...`, and skips the product (`REQ-PUB-028`). Each resolution records the original time, the reason it failed, the alternatives, whether it was resolved automatically, and the time used.

### The local schedule

Every scheduling path (`single`, `schedule` and the global batch) writes its posts to `outputs/state/schedule.json` (`REQ-PUB-029`), and `calendar` lists that file and nothing else. Because local state can fall behind the provider, `calendar` first reads the provider's count of upcoming posts and warns when the two differ (`REQ-PUB-031`). Slot selection never depended on the local file alone: it always unions the provider's posts with it.

## The duplicate guard

The publisher records each post it creates, per product and platform, in `publish_history.json` (`REQ-PUB-045`), and skips a product already published to a target platform (`REQ-PUB-046`). Reruns are therefore safe by default. `--force` bypasses the guard to republish deliberately, for example after re-rendering a video (`REQ-PUB-047`).

On `single`, a product already published to every requested platform still gets its link-in-bio entry refreshed before the command exits (`REQ-PUB-067`), so a rerun touches the bio. Pass `--no-link-in-bio` for a rerun that changes nothing.

`schedule --immediate` does not consult the guard and writes no publish history, so it republishes and later checks can't see its posts (gaps in `REQ-PUB-046` and `REQ-PUB-057`).

## Immediate batches and the retry queue

`schedule --immediate` publishes the backlog at once instead of scheduling it (`REQ-PUB-022`):

1. It scans the outputs directory for product directories.
2. It picks one render per product (`video_<asin>_*.mp4`, honouring `profiles`), so a product rendered under two profiles is published once.
3. It loads each product's platform metadata.
4. It uploads one video at a time, waiting a random whole number of seconds between `stagger_delay_min` and `stagger_delay_max` after each (`REQ-PUB-034`).
5. It continues past a failure, unless `--fail-fast` is passed.
6. It prints a summary.

The stagger keeps a burst of uploads under the provider's rate limit. At the defaults, ten products take about 7 to 15 minutes: 10 to 30 seconds of upload each plus 30 to 60 seconds between them. Lowering `stagger_delay_min` speeds a batch up at the risk of hitting the limit.

A product that fails or is skipped goes into the retry queue in `publish_history.json`, with its platforms, error, original scheduled time and a retry count (`REQ-PUB-032`). `--retry-failed` publishes only the queued products (`REQ-PUB-033`): it keeps their original scheduled times, removes each one that succeeds, increments the count of each that fails again, and reports `Retry queue is empty - no failed items to retry` when there is nothing to do. Successful products are never reprocessed.

Immediate mode posts no first comment and no affiliate phrase (gaps in `REQ-PUB-043` and `REQ-PUB-068`), and `--dry-run` does not stop it from publishing (gap in `REQ-PUB-024`).

## Cleanup and its grace period

Rendered product directories are large, so the publisher removes a product's directory once its post is confirmed (`REQ-PUB-048`). The removal is irreversible unless an archive is written, so every step leans towards keeping data:

- **Verification.** With `verify_before_delete` on, cleanup asks the provider for each post's status first; a published or scheduled leg counts as successful (`REQ-PUB-052`).
- **Settling.** A platform reporting `publishing` has neither finished nor failed, so cleanup re-checks it on a doubling delay until every platform is final or `settle_timeout_sec` runs out (`REQ-PUB-054`). This matters on `--immediate` runs, where the provider's scheduler takes roughly 30 to 90 seconds and a single check right after the post is created always reads `publishing`. Waiting stops early once the verdict can't change: one failed leg already sinks a `require_all_platforms` run, and one published leg already carries a run that doesn't require all. A post read as `scheduled` is final on the first read, and a dry run never waits.
- **All platforms.** With `require_all_platforms` on, a product is removed only when every target platform succeeded (`REQ-PUB-053`). A product that failed on one platform keeps its files for a retry.
- **Grace period.** `keep_published_days` keeps a product for that many days after publication (`REQ-PUB-055`).
- **Archive.** `archive_before_delete` writes a ZIP to `archive_dir` first (`REQ-PUB-056`).
- **Audit.** Every removal is logged with the product id, platforms and post URLs, and appended to `outputs/cleanup_audit.json`.

A post scheduled for next week counts as successful, so cleanup can remove a product before its video is live. Its video survives on the provider's own CDN, independent of the local files and of the Blob store.

## Link-in-bio

After a publish, the publisher adds the product's affiliate link to a link-in-bio page (`REQ-PUB-062`), the destination the Instagram first comment points at. Lnk.Bio is the supported provider (`REQ-PUB-063`).

The manager reads `outputs/<product_id>/data.json`: `title` becomes the link title, truncated to `max_title_length`; `affiliate_link` is the destination, or `url` without one; `images[0]` is the thumbnail, or the first file in `downloaded_images`. It skips a product whose id already appears in a link on the page. When `max_links` is above 0 and the page is full, it removes the oldest link first (`REQ-PUB-064`). A failure is logged as a warning and never blocks publishing (`REQ-PUB-065`).

The duplicate check sees only the first page of links the provider's list call returns, so a link older than that window can be added again; the protocol details, including that 50-link page size, are in [the Lnk.Bio API notes](../reference/lnkbio-api.md).

## The affiliate phrase

The Amazon Associates Operating Agreement requires a literal identification phrase wherever program content appears. With `affiliate_disclosure.enabled` on, the publisher places it in the caption of every post that carries a material connection, between the `#ad` line and the description (`REQ-PUB-068`, `REQ-PUB-070`).

It is off by default, and stays off when the section is absent or empty (`REQ-PUB-069`), because the phrase asserts membership of the named program. Enable it when you join, and disable it again if the account closes or you leave, because claiming it otherwise misstates a material connection. `phrase` and `program` default to the Amazon values, so joining that program needs only `enabled: true`; other programs override the phrase (`REQ-PUB-071`).

The phrase is separate from `#ad`, which leads the caption on any render with a material connection, whatever the program. Both follow the same recorded decision, so a caption and a frame can't disagree about whether a render is promotional (`REQ-CMP-007`).

The phrase flows through the code like this:

1. `load_publisher_config()` parses `affiliate_disclosure` into `AffiliateDisclosureConfig`.
2. `cmd_single()` and the global batch pass `disclosure_phrase` to `publish_product()` only when it is enabled.
3. `publish_product()` sets `PublishMetadata.affiliate_disclosure`, and `format_content()` places it between the disclosure line and the description.
4. The phrase is included in `PublishMetadata.to_dict()` and `PublisherConfig.to_dict()`.

## The first comment

The publisher can post a first comment on each YouTube and Instagram post instead of putting links in the caption (`REQ-PUB-035`, `REQ-PUB-038`). Meta's ranking deprioritises posts with outbound links in the description, so a comment keeps the caption clean. TikTok is always skipped, because the provider's API doesn't support `firstComment` there.

The bundled templates spend the YouTube comment on the script's closing line, because YouTube renders URLs in Shorts comments as plain text, and point Instagram at the bio link (`REQ-PUB-037`).

1. After the metadata loads, `build_first_comment()` renders the platform's template with data from `outputs/<product_id>/data.json`.
2. Each platform gets its own comment from its template in `first_comment.platforms`.
3. The comments travel in `platform_contents` to `publisher.publish()`, alongside each platform's caption and title. That dict is the authoritative per-platform payload, not a comment side channel: the client reads `content` and `title` from the same entry, so an entry carrying only a comment blanks the caption and sends no title.
4. The provider receives each one as `firstComment` in that platform's `platformSpecificData`.
5. If the data a template needs is missing, that comment is skipped with a warning and the post is still published (`REQ-PUB-042`).

This works in both publishing modes: in unified mode each platform entry in the one post carries its own `firstComment`. The comment is additive; descriptions stay as they are.

The provider reports a post published without confirming that its comment posted, which is why `verify-comments` exists (`REQ-PUB-044`).

## Blob retention

The Vercel Blob store is a staging area. Zernio fetches a video from its blob URL when a scheduled post goes live, after which the blob is dead weight; without retention the store fills the free tier (1 GB) and Vercel pauses access, which breaks every upload over 4 MB.

Retention runs once after each publish run (`single`, `schedule` and the global batch). It deletes blobs older than `max_age_days`, then deletes the oldest until the store is under `max_total_mb` (`REQ-PUB-058`). A blob referenced by a post that isn't fully published is always kept, whatever the policy (`REQ-PUB-059`). `max_age_days` must exceed the longest time a post can sit scheduled; the auto-scheduler looks ahead up to 8 weeks only when the calendar is full, and the protection covers scheduled posts anyway. A failure logs a warning and never affects publishing (`REQ-PUB-060`); the step skips silently when disabled or when no Blob token is set (`REQ-PUB-061`).

## The delivery sweep

Zernio reports a post accepted into its scheduler, not delivered. A platform leg that fails at publish time leaves the post `partial` with no alert, and the fix, `posts.retry(post_id)`, works only while Zernio still holds the upload on its CDN.

So the check behind `verify-delivery` also runs on every publish run: `single` once it has a publisher, `schedule` in both modes, and the global batch (`REQ-PUB-005`). It reads the live per-platform status of the `limit` most recent posts (`REQ-PUB-006`) and logs a warning per failing leg with the post id, platform and error category, plus the call that fixes it: `posts.retry`, or `posts.update` first when the payload was rejected, as with the TikTok disclosure error.

The post a run just created is still pending and isn't judged; the value is in earlier runs' posts, which have fired since. A sweep failure logs a warning and never affects a publish already accepted. A `single` run on a product already published everywhere returns before it builds a publisher, so it doesn't sweep; run `verify-delivery` by hand then, or for a wider window.

## The published-products registry

The registry records every published product in `outputs/state/published_products.json` and `published_products.csv` (`REQ-PUB-085`), as a durable list that outlives cleanup. Each row holds the product id, title, canonical URL, affiliate URL and `content_format`, the arm `registry --summary` counts by (`REQ-PUB-086`, `REQ-PUB-093`).

1. After a successful publish, `add_to_registry()` reads the product's `data.json`.
2. It extracts the title, the URL normalised to `https://www.amazon.com/dp/<ASIN>`, and the affiliate URL.
3. It appends a new product, or refreshes the row of a republished one (`REQ-PUB-087`, `REQ-PUB-088`). A refresh with identical data writes nothing (`REQ-PUB-089`).
4. It writes both files, renaming each existing file to `<name>.bak` first (`REQ-PUB-098`), so a write that drops or corrupts entries can be recovered.
5. A failure logs a warning and never blocks publishing.

`registry --rebuild` merges the rows it finds into the existing registry, so rows of products whose directories were cleaned up stay (`REQ-PUB-096`, `REQ-PUB-097`).

## Retries and rate limiting

Each API call makes up to `max_retries` attempts with an exponential delay of 1, 2, then 4 seconds; the delay is not configurable, and the old `backoff_multiplier` key is ignored. Network errors, timeouts, server errors and unexpected client errors are retried. Authentication failures (401, 403) and validation errors (400, 422) are not, because a retry returns the same answer.

A rate-limited call (429) waits the `Retry-After` header's seconds, or 60 seconds without one, then retries. Zernio's standard tier allows 100 requests an hour and the Pro tier 1,000 ([pricing](https://zernio.com/pricing)). To stay under the limit, keep the default 30 to 60 second stagger on immediate batches, watch the retry warnings in debug logs, and move to a higher tier if publishing volume needs it.

## Post analytics

`analytics` stores, for each published post, its cumulative views at day 2 and day 7 and a durability ratio: the views after the first 30 days divided by the views within them (`REQ-PUB-072`, `REQ-PUB-073`). Ranking by durability answers a different question from ranking by total views: a post that spiked and stopped can outrank one still earning months later on totals, and at day 7 the two look the same.

### Why the sweep runs on a schedule

The provider's per-post timeline has a retention horizon of roughly five weeks. Past it, a post's rows start at a recent date rather than at publication, so day-2 and day-7 are no longer reachable and the durability ratio can't be computed. Nothing widens the window: `from_date` makes no difference at any post age.

Figures already captured are safe. Each post's row is merged field by field, so a later, shorter reading never replaces a measured value with an absent one (`REQ-PUB-078`); what it can't do is recover a figure that was never taken in time. The one exception is a day-N figure a later sweep finds was measured before every platform had started reporting, described next, where the point is that the figure was never true rather than merely stale.

### A day-N figure counts every platform or none

Platforms start reporting on their own lag, one commonly by days, and a leg's first row carries its whole lifetime total to that date rather than that day's increment. A figure taken before a leg started would count only part of the post, while `views_total` counts all of it, so the two would describe different things.

A format comparison that ranks arms on median day-7 views would rank a post understated that way below an identical one, for a reason that is reporting lag rather than reach. So `views_day_2`, `views_day_7` and `durability_ratio` are unknown when a platform that appears later in the series hadn't reported by the cutoff (`REQ-PUB-076`), the same rule that applies to a window the timeline hasn't reached: unknown, not a small number (`REQ-PUB-074`).

The cost is coverage, and it isn't small. Measured against the live API on 2026-08-25 over the 60 most recent published posts, 57 of them multi-platform:

| | Counting a lagging leg's absence as zero | Reporting it unknown |
|---|---|---|
| day-2 available | 56 | 37 |
| day-7 available | 51 | 33 |

So 18 of 51 day-7 figures, a third, were understated by a leg that hadn't started reporting. That is the size of the bias the old rule carried into a comparison, and roughly a third of posts is the coverage the rule gives up to remove it.

The trade is deliberate. A missing figure is visible and can be excluded; a quietly understated one is neither. It does mean a comparison needs more posts than its target sample to end up with that many usable ones.

### Reading a blank figure

Three causes produce a blank, and `timeline_end` alone doesn't separate them: a post that aged past the retention horizon before its first sweep looks the same as one with a silent leg. Read `lagged_cutoff_days` in `outputs/state/post_metrics.json` first; it names the cutoffs a leg was silent for. Failing that, `timeline_end` earlier than the cutoff means the window hadn't closed when the sweep ran, and at or past it with no marker means the retained rows begin after the cutoff.

A sweep still withholds its own figure wherever the retained window covers the cutoff and the legs disagree; it just doesn't mark the post, because marking withdraws figures other sweeps took. A figure is withdrawn only on the evidence of a sweep whose record still reaches back to publication (`REQ-PUB-077`). Past the retention horizon every leg's rows begin at the window edge, so a leg absent from that first date looks identical to one that started late, and a ratio measured while the record was whole must not be discarded on that reading, because no later sweep can recompute it. For the same reason the merge keeps a durability ratio taken from a full record over one computed from a truncated window, which divides by a partial figure and reads higher.

The sweep that stores a figure is usually not the one that can tell it was biased. A daily run reaches a young post while the slow platform has no rows at all, so one leg looks like the whole post. The marker is what a later sweep uses to withdraw a stored number, and it is never unset: a sweep whose rows all begin past the lag sees no disagreement between legs, which says nothing about whether the figure was biased when it was taken. Figures captured before this rule existed are corrected the same way, on the next sweep that observes the lag.

### Why daily

Most of a short-form post's views arrive in the first day or two, and one platform's analytics rows take 48 to 72 hours to finalise, so a same-day reading of a fresh post is still settling; the merge corrects it on the next run. The durability ratio is the binding constraint: it needs a post past day 30 while its rows still reach back to publication, and retention is about five weeks, so the window is roughly five days wide. A weekly sweep can step over a post's only window and never produce a ratio for it.

Repeat runs are safe, because readings merge per field and a later, better figure replaces an earlier partial one. They aren't free: a sweep costs one timeline call per measured post plus the paging to list them, so the bundled size is roughly 52 requests, about half of one hour on the standard tier's 100. A daily sweep fits but shouldn't share its hour with a publish run; several times an hour doesn't fit. A rate-limited timeline call isn't retried, and the sweep walks newest-first, so what a 429 costs is the oldest posts measured, the ones whose ratio window is about to close.

`analytics.limit` must exceed the number of posts published inside the retention horizon (`REQ-PUB-081`). At one post a day the horizon holds about 35 posts, so the default 50 leaves headroom; two posts a day puts 70 inside it, and 50 would quietly lose the oldest 20. Raise it with the cadence.

### Why the timer is built the way it is

`outputs/` is local and gitignored, and the API key comes from `.env`, so the capture belongs on the machine that owns the data rather than in CI (`REQ-PUB-082`).

The sweep size lives in `config/publisher.yaml` and nowhere else. The unit passes no `--limit`, so editing the YAML takes effect on the next run with no reinstall, and the scheduled sweep can't drift from `make analytics`.

The unit names the interpreter instead of going through `poetry run` or `make`. A user service doesn't inherit your login shell's environment, so its `PATH` has no pyenv shims, and because `poetry.toml` sets `virtualenvs.create = false`, `poetry run python` then resolves the base interpreter rather than the project environment. The service would fail at the first import, daily, while the figures it exists to capture age out. This is the same trap the `*-lowpri` targets work around for `systemd-run`. The installer resolves the interpreter and refuses to install a unit whose interpreter can't import the project, because a file that merely exists proves nothing.

Two properties of the unit matter because both failures would be silent:

- `Persistent=true` runs a window the machine slept through on the next boot rather than skipping it. Retention is finite, so a missed sweep is a permanent hole in the record, not a late reading. Cron misses a machine that was asleep, which is why the timer is the better of the two.
- `TimeoutStartSec=` is set explicitly. systemd disables the start timeout by default for `Type=oneshot` units, and a hung request would then leave the unit activating forever; systemd refuses to start a second instance, so every later firing is dropped and the unit never reaches `failed`, and the failure handler never runs either. A stuck sweep would look exactly like a working one.

A sweep that measured posts and captured none of them exits non-zero, so the failure channels fire. Every timeline call failing is a broken sweep, not a quiet one, and exiting 0 there would keep the timer green while the figures expire. A single post failing stays a warning, because a partial reading is still worth storing, and an account with no published posts isn't an error at all.

A sweep whose timelines all come back empty is a different case and isn't treated as a failure: in that sweep it can't be told apart from an account whose posts are too young to have rows, and failing there would fail a young account daily. What is reported instead is every post with a stored view count returning none at once. Posts age out of the provider's timeline one at a time, so all of them going quiet in the same sweep points at the reader, not age. The one case where this fires without a reader fault is an account quiet long enough for every post in the measured window to age out; it reports once there rather than on every sweep, because a post already known to have gone quiet stops counting. It is recorded as a warning and in `outputs/logs/analytics-failures.log`, so `make analytics-timer-status` shows it; a warning alone would sit behind one line per measured post and never reach the last few journal lines that command prints. Stored figures are untouched.

The unit templates and the reasons they are rendered rather than parameterised are in [`deploy/README.md`](../../deploy/README.md).
