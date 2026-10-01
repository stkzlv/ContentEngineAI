# Publisher requirements

Ids use the prefix `REQ-PUB`. The format and the statuses are described in [the requirements index](README.md).

## Service integration

- **REQ-PUB-001** `shipped` The publisher publishes through the Zernio scheduling service (formerly Late).
- **REQ-PUB-002** `shipped` The publisher posts to YouTube, TikTok and Instagram.
- **REQ-PUB-003** `shipped` The publisher uploads each video through the provider's API and reads each post's status back from it.
- **REQ-PUB-004** `shipped` If a published platform leg reports no post URL, status checks, first-comment verification, slot-occupancy detection and upload-store retention still complete.
- **REQ-PUB-005** `shipped` Where `delivery_sweep.enabled` is on (default on), the publisher checks recent posts after each publish run and reports every post whose status is `partial` or that has a failed platform leg, naming the platform and its error.
- **REQ-PUB-006** `shipped` The delivery sweep inspects the `delivery_sweep.limit` most recent posts (default 25).
- **REQ-PUB-007** `shipped` The `verify-delivery` command runs the delivery check on demand over the `--limit` most recent posts (default 25).
- **REQ-PUB-008** `planned #550` Product videos publish to YouTube with a short written title within the configured maximum, not the store listing title.

## Accounts and configuration

- **REQ-PUB-009** `shipped` The publisher config can define several named provider accounts under `accounts`, with `default_account` naming the one used when none is chosen.
- **REQ-PUB-010** `shipped` The `--account NAME` option selects the provider account a run uses.
- **REQ-PUB-011** `shipped` The number of API retries (`max_retries`, default 3) and the request timeout (`timeout`, default 120 seconds) are set in the publisher config.
- **REQ-PUB-012** `shipped` If `config/publisher.yaml` exists and cannot be read or parsed, the publisher stops instead of applying defaults.
  - Why: the defaults publish immediately, so falling back on a parse error turns a scheduled run into a live one.
- **REQ-PUB-013** `shipped` If one recurring slot in the config is invalid, the publisher skips that slot with a warning and keeps the others.
- **REQ-PUB-014** `shipped` If the `cleanup` section is invalid, the publisher keeps that section's `enabled` and `archive_before_delete` values.

## Per-platform profile routing

- **REQ-PUB-015** `shipped` Where `profiles` maps a platform to a video profile, the publisher picks that profile's render for the post.
- **REQ-PUB-016** `shipped` If the mapping is unset or the routed profile has no render for the product, the publisher falls back to the first render in the product directory.
- **REQ-PUB-017** `shipped` Each post uploads one file to all its platforms: the render routed to the first platform in its target list.

## Scheduling

- **REQ-PUB-018** `shipped` The `single` command publishes a product immediately with `--immediate`, or at a given time with `--schedule "YYYY-MM-DD HH:MM:SS"`.
- **REQ-PUB-019** `shipped` Recurring slots are configured under `recurring_schedule.slots` as a day of the week, a time and a timezone, which defaults to `recurring_schedule.timezone`.
- **REQ-PUB-020** `shipped` When no publish time is given, the publisher schedules the post into the next free recurring slot.
- **REQ-PUB-021** `shipped` The `schedule` command schedules every unpublished product in the outputs directory into recurring slots.
- **REQ-PUB-022** `shipped` When `--immediate` is passed, the `schedule` command publishes those products at once instead of scheduling them.
- **REQ-PUB-023** `shipped` The `schedule` command posts each product once, even when the product has renders under several profiles.
- **REQ-PUB-024** `partial` When `--dry-run` is passed, the `schedule` command shows the slot each product would take and publishes nothing.
  - Gap: combined with `--immediate`, it publishes.
- **REQ-PUB-025** `shipped` A slot counts as taken when either the provider or the local schedule holds a post at that time.
- **REQ-PUB-026** `shipped` If the provider's post list cannot be read, the publisher schedules around the local schedule alone.
- **REQ-PUB-027** `shipped` When `--auto-resolve` is passed and the chosen slot fails schedule validation, the publisher moves the post to the first free alternative slot.
- **REQ-PUB-028** `shipped` If the chosen slot fails schedule validation without `--auto-resolve`, the publisher reports alternative slots and skips the product.
- **REQ-PUB-029** `shipped` Every post scheduled through `single`, `schedule` or the batch is recorded in the local schedule.
- **REQ-PUB-030** `shipped` The `calendar` command lists the local schedule, filtered by `--platform`, `--status`, `--date-from` and `--date-to`.
- **REQ-PUB-031** `shipped` If the provider holds a different number of upcoming posts than the local schedule, `calendar` warns with both counts.
- **REQ-PUB-032** `shipped` If `schedule --immediate` fails or skips a product, the publisher adds the product to a retry queue.
- **REQ-PUB-033** `shipped` When `--retry-failed` is passed, `schedule --immediate` publishes only the products in the retry queue and removes each one that succeeds.
- **REQ-PUB-034** `shipped` Between posts, `schedule --immediate` waits a random delay between `stagger_delay_min` and `stagger_delay_max` seconds (default 30 to 60).

## First comment

- **REQ-PUB-035** `shipped` Where `first_comment.enabled` is on, the publisher posts a first comment on each post, built from that platform's template.
- **REQ-PUB-036** `shipped` First comments are off unless the config enables them, and the bundled config enables them.
- **REQ-PUB-037** `shipped` The bundled config posts the script's closing line as the YouTube first comment, and the product title with a link-in-bio pointer as the Instagram first comment.
- **REQ-PUB-038** `shipped` The publisher posts first comments on YouTube and Instagram, and never on TikTok.
- **REQ-PUB-039** `shipped` A first-comment template can use the placeholders `{affiliate_link}`, `{product_title}`, `{hashtags}` and `{closing_line}`, and needs data only for the placeholders it uses.
- **REQ-PUB-040** `shipped` The `{affiliate_link}` placeholder takes the shortened affiliate link when one exists, and the full link otherwise.
- **REQ-PUB-041** `shipped` Where `first_comment.move_hashtags_to_comment` is on, the publisher moves Instagram hashtags from the caption into the first comment.
- **REQ-PUB-042** `shipped` If the data a template needs is missing, the publisher skips that first comment with a warning and still publishes the post.
- **REQ-PUB-043** `partial` The publisher posts first comments in both unified and platform-specific publishing modes.
  - Gap: `schedule --immediate` posts no first comment.
- **REQ-PUB-044** `shipped` The `verify-comments` command checks recent published posts on each platform and warns on every YouTube or Instagram post missing its first comment.
  - Why: the provider reports a post published without confirming that its comment posted.

## Duplicate publish protection

- **REQ-PUB-045** `shipped` The publisher records each post it publishes, per product and platform.
- **REQ-PUB-046** `partial` If a product is already published to a target platform, the publisher warns and skips it.
  - Gap: `schedule --immediate` publishes it again.
- **REQ-PUB-047** `shipped` When `--force` is passed, the publisher republishes an already-published product.

## Post-publication cleanup

- **REQ-PUB-048** `shipped` Where `cleanup.enabled` is on (default on), the publisher removes a product's directory after the product is published.
- **REQ-PUB-049** `shipped` When `--no-cleanup` is passed, the publisher skips cleanup for that run.
- **REQ-PUB-050** `shipped` The `cleanup --all` command refuses to run without `--confirm`, unless `--dry-run` is passed.
- **REQ-PUB-051** `shipped` The `cleanup --platform` option limits the platforms whose publication cleanup checks.
- **REQ-PUB-052** `shipped` Where `cleanup.verify_before_delete` is on (default on), cleanup confirms each post's status before deleting, and a post that is published or scheduled counts as successful.
- **REQ-PUB-053** `shipped` Where `cleanup.require_all_platforms` is on (default on), cleanup deletes a product only when every target platform succeeded.
- **REQ-PUB-054** `shipped` While a platform reports that it is still publishing, cleanup re-checks it until every platform is final or `cleanup.settle_timeout_sec` (default 300) runs out.
- **REQ-PUB-055** `shipped` Where `cleanup.keep_published_days` is above 0, cleanup waits that many days after publication before deleting.
- **REQ-PUB-056** `shipped` Where `cleanup.archive_before_delete` is on, cleanup writes a ZIP archive of the product directory to `cleanup.archive_dir` before deleting it.
- **REQ-PUB-057** `partial` Before a product directory is removed, the publisher writes the product's publish history and registry entry on every publish path.
  - Gap: `schedule --immediate` writes no publish history, so later duplicate checks cannot see its posts.

## Upload store retention

- **REQ-PUB-058** `shipped` After each publish run, the publisher deletes Vercel Blob uploads older than `blob_retention.max_age_days`, then deletes the oldest until the store is within `blob_retention.max_total_mb`.
- **REQ-PUB-059** `shipped` Upload retention keeps every upload referenced by a post that is not fully published.
- **REQ-PUB-060** `shipped` If upload retention fails, the publisher logs a warning and the publish result is unchanged.
- **REQ-PUB-061** `shipped` Upload retention is skipped when `blob_retention.enabled` is off or no Blob token is configured.

## Link-in-bio

- **REQ-PUB-062** `shipped` After publishing a product, the publisher adds the product's affiliate link to a link-in-bio page.
- **REQ-PUB-063** `shipped` The publisher supports Lnk.Bio as the link-in-bio provider.
- **REQ-PUB-064** `shipped` Where `link_in_bio.max_links` is above 0, the publisher removes the oldest link when the page reaches that count; 0 (the default) sets no limit.
- **REQ-PUB-065** `shipped` If the link-in-bio update fails, the publisher logs a warning and the publish result is unchanged.
- **REQ-PUB-066** `partial` Link-in-bio updates are on by default, and `--no-link-in-bio` skips them for one run.
  - Gap: the flag exists only on `single`, not on `schedule` or the batch.
- **REQ-PUB-067** `shipped` When `single` is run without `--force` on a product already published to every target platform, the publisher refreshes the product's link-in-bio entry and exits without publishing.

## Affiliate program phrase

- **REQ-PUB-068** `partial` Where `affiliate_disclosure.enabled` is on, the publisher places the configured phrase in the caption of every post that carries a material connection, in both unified and platform-specific modes, from the publisher CLI and the batch alike.
  - Gap: `schedule`, with or without `--immediate`, adds no phrase.
- **REQ-PUB-069** `shipped` The affiliate phrase is off by default, including when the `affiliate_disclosure` section is absent or empty.
  - Why: the phrase asserts membership of the named program, so an unconfigured install must not publish it.
- **REQ-PUB-070** `shipped` The affiliate phrase sits between the leading disclosure line and the description.
- **REQ-PUB-071** `shipped` The phrase and the program name are configurable, and the default phrase is the Amazon Associates identification phrase.

## Post analytics

- **REQ-PUB-072** `shipped` The `analytics` command stores, for each published post, its cumulative views at day 2 and day 7 and a durability ratio.
- **REQ-PUB-073** `shipped` The durability ratio is the views after the first 30 days divided by the views within them.
- **REQ-PUB-074** `shipped` If a post has not reached a day-N window, that figure is unknown, not the running total.
- **REQ-PUB-075** `shipped` If a post has no views in the first 30 days, its durability ratio is unknown, not 0.0.
- **REQ-PUB-076** `shipped` If any platform leg had not started reporting by a day-N cutoff, that day-N figure is unknown.
- **REQ-PUB-077** `shipped` When a later sweep finds that a stored day-N figure was taken before every leg reported, it withdraws that figure.
- **REQ-PUB-078** `shipped` A later measurement never replaces a stored figure with an unknown one, except a figure withdrawn under the leg-reporting rule.
- **REQ-PUB-079** `shipped` Analytics reports can rank posts by durability, and sort unknown figures last.
- **REQ-PUB-080** `shipped` When `--rank-only` is passed, the `analytics` command ranks stored figures without contacting the provider.
- **REQ-PUB-081** `shipped` Each analytics sweep measures the `analytics.limit` most recent published posts (default 50).
  - Check: the limit exceeds the number of posts published within the provider's retention window of about five weeks.
- **REQ-PUB-082** `shipped` The `make install-analytics-timer` target installs a daily analytics sweep that runs a missed sweep after downtime and can notify the operator on failure.
- **REQ-PUB-083** `planned #547` The choices that shape each render (template, hook archetype, voice, caption template, music, motion, transitions, effects) are recorded, and a report shows their distribution over recent renders, with an alert when one value dominates or two scripts are near-identical.
- **REQ-PUB-084** `planned #551` The analytics sweep stores each first-seconds and quality metric a platform exposes (engaged views and viewed-vs-swiped on YouTube, watch time or completion on TikTok, sends or shares per reach on Instagram), records an unavailable metric as unknown, and segments them by format arm and render choice.

## Published products registry

- **REQ-PUB-085** `shipped` The publisher keeps a registry of published products in the outputs directory, as `published_products.json` and `published_products.csv`.
- **REQ-PUB-086** `shipped` Each registry row holds the product id, product title, canonical URL, affiliate URL and content-format arm.
- **REQ-PUB-087** `shipped` After each successful publish, the publisher adds the product to the registry, with one row per product.
- **REQ-PUB-088** `shipped` When a product is republished, the publisher refreshes its registry row with the latest data.
- **REQ-PUB-089** `shipped` If a refresh carries identical data, the registry files are not rewritten.
- **REQ-PUB-090** `shipped` The registry loader gives a row missing a field that field's default, and ignores columns the registry no longer has.
- **REQ-PUB-091** `shipped` If one registry row cannot be read, the loader skips that row and keeps the rest.
- **REQ-PUB-092** `partial` If the registry file cannot be parsed, adding an entry keeps the existing rows.
  - Gap: after a parse failure, adding an entry rewrites the registry with that one row.
- **REQ-PUB-093** `shipped` The content-format arm records whether the video came from a topic or a scraped product, read from the product record.
- **REQ-PUB-094** `shipped` The `registry --summary` command counts published products per content-format arm, and counts rows written before the arm existed as `unlabelled`.
- **REQ-PUB-095** `shipped` The registry CSV has one column per registry field.
- **REQ-PUB-096** `shipped` The `registry --rebuild` command rebuilds the registry from scraped data directories (`--scan-dir`), merging the rows it finds into the existing registry.
- **REQ-PUB-097** `shipped` A rebuild keeps the rows of products whose directories were cleaned up after publishing.
- **REQ-PUB-098** `shipped` Before each registry write, the publisher keeps the previous JSON and CSV files as `<name>.bak`.

## Captions and platform limits

- **REQ-PUB-099** `shipped` In unified mode (the default), the publisher sends one post with one metadata set to every target platform.
- **REQ-PUB-100** `shipped` In platform-specific mode (`--platform-specific` or `use_platform_specific_content: true`), the publisher sends one post per platform, each with that platform's title, description and hashtags.
- **REQ-PUB-101** `shipped` The publisher checks each caption against the platform's character limits.
- **REQ-PUB-102** `shipped` If a title or description exceeds a platform's hard cap, the publisher trims it on a word boundary and adds an ellipsis.
- **REQ-PUB-103** `shipped` The cap applied to a caption is the smallest cap among the platforms the caption reaches.
- **REQ-PUB-104** `shipped` The cap applies to the composed caption: the disclosure line, the affiliate phrase, the description, the hashtag block and the blank lines between them.
- **REQ-PUB-105** `shipped` Every publish path (`single`, `schedule`, `schedule --immediate` and the batch) applies the caption cap rules.
- **REQ-PUB-106** `shipped` If a caption's hashtag count is outside the platform's range, the publisher logs a warning and keeps the tags as written.
- **REQ-PUB-107** `shipped` The publisher appends the product id as the last hashtag of each product caption.
- **REQ-PUB-108** `planned #567` Each platform's hashtag count stays within that platform's own limit.
- **REQ-PUB-109** `shipped` The publisher sends a distinct video title to every platform that accepts one.
- **REQ-PUB-110** `shipped` If a YouTube post has no title, the publisher refuses to publish it.
  - Why: without one the platform titles the video from the caption's first line, which is the disclosure.
- **REQ-PUB-111** `shipped` The title sent to each platform is the trimmed title.
