# Publisher reference

This page lists the publisher's commands, configuration keys, environment variables, file formats, webhook payloads and Python API. The publisher posts rendered videos through the Zernio scheduling service, formerly named Late; the code still uses the `late-sdk` package, the `LATE_API_KEY` variable and the `src/publisher/late/` package.

For step-by-step tasks, see [the publishing guide](../guides/publishing.md). For why the publisher behaves as it does, see [the publishing explanation](../explanation/publishing.md). The requirements are in [the publisher requirements](../requirements/publisher.md), and direct SDK and REST usage is in [the Zernio client reference](zernio-client.md).

## Command line

Every command runs as `python -m src.publisher.late <command> [options]`. The `make publish` and `make publish-lowpri` targets run the same entry point with `ARGS="<command> [options]"`.

### Global options

| Option | Description |
|---|---|
| `--account NAME` | Use the named account from the `accounts` section of `config/publisher.yaml`. Goes before the command name. An unknown name stops the run with the list of configured accounts. |

Every command takes `--debug`, which turns on debug logging. It goes after the command name. Logs are written to `outputs/logs/publisher-<date>.log`.

### `list-accounts`

Lists the social accounts connected to the provider account.

```bash
python -m src.publisher.late list-accounts [--debug]
```

### `single`

Publishes one product. The video is found in `outputs/<product_id>/`.

```bash
python -m src.publisher.late single <product_id> [options]
```

| Option | Default | Description |
|---|---|---|
| `product_id` | required | Product id, such as an ASIN. The product directory must exist under the repository's `outputs/`. |
| `--platform NAME` | `youtube`, `tiktok`, `instagram` | Target platform. Repeat for several. One of `youtube`, `tiktok`, `instagram`, `facebook`, `twitter`, `linkedin`. The `default_platforms` key does not change this default. |
| `--schedule DATETIME` | none | Publish at this time. Accepts `YYYY-MM-DD HH:MM:SS`, `YYYY-MM-DDTHH:MM:SS`, `YYYY-MM-DD HH:MM` and `YYYY-MM-DDTHH:MM`, read as UTC. Takes precedence over `--immediate`. |
| `--immediate` | off | Publish at once. |
| `--force` / `--no-force` | `--no-force` | Republish a product that is already published to every requested platform. Without it, the command skips platforms already published. |
| `--no-cleanup` | off | Skip post-publication cleanup for this run. |
| `--link-in-bio` | from config | Update the link-in-bio page even when `link_in_bio.enabled` is off. |
| `--no-link-in-bio` | from config | Skip the link-in-bio update. Wins over `--link-in-bio`. |
| `--platform-specific` | off | Create one post per platform with that platform's metadata. Same as `use_platform_specific_content: true`. |

With neither `--schedule` nor `--immediate`, the command schedules the post into the next free slot from `recurring_schedule.slots`, and fails if no slot is configured.

When every requested platform is already published and `--force` is not passed, the command refreshes the product's link-in-bio entry (unless link-in-bio is off) and exits without uploading.

After a publish, the command records the post in `publish_history.json` and, for a scheduled post, in `schedule.json`; adds the product to the registry; updates the link-in-bio page; runs cleanup; then runs blob retention and the delivery sweep.

### `schedule`

Publishes every product in the outputs directory that is not already published to every target platform, either into recurring slots or at once. `schedule` and `schedule auto` are the same command.

```bash
python -m src.publisher.late schedule [auto] [options]
```

| Option | Default | Applies to | Description |
|---|---|---|---|
| `--platform NAME` | `youtube`, `tiktok`, `instagram` | both modes | Target platform. Repeat for several. Same choices as `single`. |
| `--outputs-dir PATH` | the repository's `outputs/` | both modes | Directory to scan for product directories. |
| `--immediate` | off | | Publish at once instead of scheduling into slots. |
| `--dry-run` | off | both modes | Publish nothing. Scheduled mode shows the slot each product would take; immediate mode lists the products it would publish and contacts no provider. |
| `--no-cleanup` | off | both modes | Skip post-publication cleanup for this run. |
| `--auto-resolve` | off | scheduled mode | When a slot fails schedule validation, use the first free alternative. |
| `--force` / `--no-force` | `--no-force` | both modes | Include products already published to every target platform. |
| `--link-in-bio`, `--no-link-in-bio` | `link_in_bio.enabled` | both modes | Update, or skip, the link-in-bio page after each publish. |
| `--fail-fast` | off | immediate mode | Stop at the first failed product. |
| `--retry-failed` | off | immediate mode | Publish only the products in the retry queue. |

Each product is posted once, even when it has renders under several profiles: the scanner picks one render per product, honouring `profiles`. Scheduled mode creates one unified post, or one per platform with `use_platform_specific_content`; immediate mode creates one post per platform, each with that platform's metadata, or another platform's where it has none. Both attach the first comment and, where `affiliate_disclosure` is on, the program phrase, and write each post to the publish history. `--dry-run` with `--retry-failed` lists the retry queue.

The command exits with status 1 when any product fails. `recurring_schedule.enabled: false` or an empty slot list stops scheduled mode with an error.

### `calendar`

Lists the posts in the local schedule (`outputs/state/schedule.json`). `calendar` and `calendar list` are the same command.

```bash
python -m src.publisher.late calendar [list] [options]
```

| Option | Description |
|---|---|
| `--platform NAME` | Show only posts for this platform. |
| `--status STATUS` | Show only posts with this status: `pending`, `scheduled` or `failed`, the statuses the local schedule records. Whether a post went live is on the provider; `verify-delivery` checks it. |
| `--date-from DATE` | Show only posts at or after this date. ISO 8601, read as UTC when no offset is given. |
| `--date-to DATE` | Show only posts at or before this date. Same format. |

Before listing, the command reads the provider's count of upcoming posts and warns when it differs from the local count, naming the number missing. Posts scheduled through `single` or the global batch before 0.121.5 were never written locally and appear only in that warning.

Each post prints its product id, scheduled time in UTC, platforms, status, and, when known, its post id and slot index.

### `cleanup`

Removes the directories of published products from the outputs directory.

```bash
python -m src.publisher.late cleanup (--product-id ID | --all) [options]
```

| Option | Description |
|---|---|
| `--product-id ID` | Clean up one product. Exactly one of `--product-id` and `--all` is required. |
| `--all` | Clean up every published product. Requires `--confirm` unless `--dry-run` is passed. |
| `--platform NAME` | Platform whose publication cleanup checks. Repeat for several. Defaults to `default_platforms`. |
| `--outputs-dir PATH` | Directory to scan. Defaults to the repository's `outputs/`. |
| `--dry-run` | Show what would be removed and remove nothing. |
| `--confirm` | Confirm an `--all` run. |

The summary line reads `Products: <n> cleaned, <n> skipped, <size> freed`.

### `delete`

Deletes a post from the provider.

```bash
python -m src.publisher.late delete <post_id> [--debug]
```

| Option | Description |
|---|---|
| `post_id` | The provider's post id. |

### `registry`

Maintains the published-products registry.

```bash
python -m src.publisher.late registry (--rebuild | --summary) [options]
```

| Option | Description |
|---|---|
| `--rebuild` | Rebuild the registry from every `<product_id>/data.json` under the scan directory, merging into the existing registry. Wins when both actions are given. |
| `--summary` | Count published products per content-format arm. Rows written before the arm existed count as `unlabelled`. |
| `--outputs-dir PATH` | Outputs root whose `state/` directory holds the registry files. Defaults to the repository's `outputs/`. |
| `--scan-dir PATH` | Directory to scan for product data. Defaults to `--outputs-dir`. |

One of `--rebuild` and `--summary` is required.

### `analytics`

Captures day-2 and day-7 views and a durability ratio for recent published posts, and ranks posts by durability.

```bash
python -m src.publisher.late analytics [options]
```

| Option | Default | Description |
|---|---|---|
| `--limit N` | `analytics.limit` (50) | How many recent published posts to measure. |
| `--rank-only` | off | Rank stored figures without contacting the provider. Publisher config still loads first, so an API key must be configured. |
| `--outputs-dir PATH` | the repository's `outputs/` | Outputs root; `post_metrics.json` lives under its `state/` directory. |

`make analytics` runs the same command with no `--limit`. A sweep that measured posts and captured none of them exits with status 1.

### `verify-comments`

Checks the most recent published posts and warns on every YouTube or Instagram post that is missing its first comment.

```bash
python -m src.publisher.late verify-comments [--limit N] [--outputs-dir PATH]
```

| Option | Default | Description |
|---|---|---|
| `--limit N` | 25 | Number of recent published posts to check. |
| `--outputs-dir PATH` | the repository's `outputs/` | Directory holding `publish_history.json`, used to name products in the output. |

### `verify-delivery`

Checks the most recent posts and warns on every post whose status is `partial` or `failed`, naming the failing platform and its error.

```bash
python -m src.publisher.late verify-delivery [--limit N] [--outputs-dir PATH]
```

| Option | Default | Description |
|---|---|---|
| `--limit N` | 25 | Number of recent posts to check. |
| `--outputs-dir PATH` | the repository's `outputs/` | Directory holding `publish_history.json`, used to name products in the output. |

## Environment variables

Settings resolve in this order, highest first: command-line options, environment variables, `config/publisher.yaml`. Ten settings have environment overrides; nothing under `cleanup`, `link_in_bio`, `first_comment`, `blob_retention`, `delivery_sweep`, `affiliate_disclosure`, `analytics`, `tiktok_settings`, `recurring_schedule` or `schedule_validation` does.

| Variable | Setting | Notes |
|---|---|---|
| `LATE_API_KEY`, then `PUBLISHER_API_KEY` | `api_key` | Required. At least 10 characters. Keys start with `sk_live_` or `sk_test_`. |
| `BLOB_READ_WRITE_TOKEN`, then `LATE_VERCEL_TOKEN`, then `PUBLISHER_VERCEL_TOKEN` | `vercel_token` | Vercel Blob token. Required to upload videos over 4 MB. |
| `PUBLISHER_PROVIDER` | `provider` | |
| `PUBLISHER_IMMEDIATE` | `immediate_publish` | `true`, `1` or `yes` is true; anything else is false. |
| `PUBLISHER_MAX_RETRIES` | `max_retries` | Integer. |
| `PUBLISHER_TIMEOUT` | `timeout` | Seconds, float. |
| `PUBLISHER_DEFAULT_PLATFORMS` | `default_platforms` | Comma-separated, such as `youtube,tiktok`. |
| `PUBLISHER_PRIVACY_YOUTUBE`, `PUBLISHER_PRIVACY_TIKTOK`, `PUBLISHER_PRIVACY_INSTAGRAM` | `privacy_settings` | Loaded, but no publish path reads `privacy_settings`. |

The link-in-bio provider reads `LNKBIO_CLIENT_ID` and `LNKBIO_CLIENT_SECRET`; both are required while `link_in_bio.enabled` is on.

`config/publisher.yaml` does no variable expansion, so `${LATE_API_KEY}` there is stored literally and is long enough to pass the key-length check. Set credentials in the environment or `.env`.

## Configuration file

The file is `config/publisher.yaml`. A missing file is allowed, and everything then comes from the environment and the defaults below; a file that exists and cannot be parsed stops the publisher (`REQ-PUB-012`).

In the tables, "Default" is the value used when the key is absent, and "Bundled" is the value in the shipped file where it differs.

### Top-level keys

| Key | Type | Default | Bundled | Description |
|---|---|---|---|---|
| `provider` | string | `late` | | Publishing provider. Only `late` is implemented. |
| `api_key` | string | none | | Single-account key. Prefer `LATE_API_KEY`. |
| `vercel_token` | string | none | | Single-account Blob token. Prefer `LATE_VERCEL_TOKEN`. |
| `immediate_publish` | bool | `true` | `false` | Publish at once instead of scheduling. Read by the global batch only; the publisher CLI publishes at once only with `--immediate`. |
| `default_platforms` | list | `youtube`, `tiktok`, `instagram` | | Platforms the global batch and `cleanup` use when none are given. `single` and `schedule` default to the same three platforms regardless. |
| `use_platform_specific_content` | bool | `false` | | One post per platform with that platform's metadata, instead of one post for all. |
| `profiles` | map | `{}` | commented out | Platform name to video profile. See [`profiles`](#profiles). |
| `schedule_time` | string | none | | Fixed ISO 8601 publish time the global batch uses when it is given none. |
| `max_retries` | int | `3` | | Attempts per API call, including the first. Must be 0 or more. |
| `timeout` | float | `120.0` | | Seconds per API request. Must be above 0. |
| `backoff_multiplier` | float | | `2.0` | Deprecated. Stripped by the loader; the retry delay is not configurable. |
| `stagger_delay_min` | int | `30` | | Minimum seconds between posts in an immediate batch. Must be 0 or more. |
| `stagger_delay_max` | int | `60` | | Maximum seconds between posts. Must be at least `stagger_delay_min`. |
| `privacy_settings` | map | `{}` | `youtube: public`, `tiktok: public`, `instagram: everyone` | Loaded, but no publish path reads it. TikTok privacy is set by `tiktok_settings.privacy_level`. |
| `synthetic_media_disclosure` | bool | `false` | | Sends YouTube's altered-or-synthetic-content flag (`containsSyntheticMedia`) on every YouTube post. See [Compliance](../explanation/compliance.md). |

### `accounts`

Named provider accounts (`REQ-PUB-009`). With an `accounts` section, the chosen account's `api_key` and `vercel_token` replace the top-level ones. Without `--account`, a set `LATE_API_KEY` still overrides the default account's key; `--account` overrides both.

| Key | Type | Description |
|---|---|---|
| `accounts.<name>.api_key` | string | Required. An account without it is skipped with a warning. |
| `accounts.<name>.vercel_token` | string | Optional Blob token. |
| `accounts.<name>.description` | string | Free text. |
| `accounts.<name>.default_platforms` | list | Loaded, but no command reads it. |
| `default_account` | string | Account used without `--account`. Defaults to the first account listed. |

```yaml
accounts:
  production:
    api_key: sk_live_prod_key_12345
    vercel_token: vercel_prod_token
    description: Production account
  staging:
    api_key: sk_live_staging_key_123
    description: Staging/test account
    default_platforms: [youtube]
default_account: production
```

### `recurring_schedule`

| Key | Type | Default | Bundled | Description |
|---|---|---|---|---|
| `enabled` | bool | `false` | `true` | `schedule` refuses to run in scheduled mode when off. `single` reads the slots either way. |
| `timezone` | string | `UTC` | `Europe/Berlin` | IANA timezone for the slots. |
| `slots` | list | `[]` | daily at `10:00:00` | Each slot has `day_of_week` (`monday` to `sunday`), `time` (`HH:MM:SS`, 24-hour) and an optional `timezone` that defaults to the section's. An invalid slot is skipped with a warning (`REQ-PUB-013`). |

The number of alternatives suggested on a conflict is fixed at 5 and is not read from the file.

### `schedule_validation`

| Key | Type | Default | Description |
|---|---|---|---|
| `min_post_spacing_hours` | int | `2` | Minimum hours between posts on the same platform. 0 turns the check off. |
| `prevent_duplicates` | bool | `true` | Refuse the same product, platform and time twice. |
| `allow_past_schedules` | bool | `false` | Accept a time in the past. |
| `max_posts_per_day` | int | `10` | Posts per calendar day, counted across platforms. 0 means no limit. |

### `cleanup`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Remove a product's directory after it is published. |
| `verify_before_delete` | bool | `true` | Check each post's status with the provider first. A published or scheduled post counts as successful. |
| `require_all_platforms` | bool | `true` | Remove a product only when every target platform succeeded. |
| `settle_timeout_sec` | float | `300` | Seconds to keep re-checking a platform that reports `publishing`. 0 or less checks once. |
| `settle_initial_delay_sec` | float | `30` | Delay before the second check. Each later delay doubles, and the last is trimmed so the delays sum to `settle_timeout_sec`. 0 or less checks once. |
| `keep_published_days` | int | `0` | Days after publication before a product is removed. 0 removes it at once. |
| `archive_before_delete` | bool | `false` | Write a ZIP of the product directory before removing it. |
| `archive_dir` | path | `outputs/archive` | Where archives go, as `<product_id>_<timestamp>.zip`. |
| `preserve_metadata` | bool | `false` | Loaded, but cleanup does not read it. |
| `preserve_logs` | bool | `true` | Loaded, but cleanup does not read it. |

If the section is invalid, the loader falls back to the defaults but keeps `enabled` and `archive_before_delete` (`REQ-PUB-014`).

### `link_in_bio`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Add the product's link to the bio page after each publish. |
| `provider` | string | `lnkbio` | Provider. Only `lnkbio` is implemented. |
| `max_links` | int | `0` | When above 0, remove the oldest link once the page holds this many. 0 means no limit. |
| `max_title_length` | int | `80` | Truncate link titles past this length, ending in `...`. At least 10. |

The Lnk.Bio protocol is described in [the Lnk.Bio API notes](lnkbio-api.md).

### `first_comment`

| Key | Type | Default | Bundled | Description |
|---|---|---|---|---|
| `enabled` | bool | `false` | `true` | Post a first comment on each YouTube and Instagram post. TikTok is always skipped. |
| `move_hashtags_to_comment` | bool | `false` | | Move Instagram hashtags from the caption into the comment, as `{hashtags}`. |
| `platforms` | map | `{}` | see below | Platform name to comment template. |

Bundled templates: `youtube: "{closing_line}"`, and for `instagram` the product title, a newline, then a link emoji followed by `Link in bio!`.

| Placeholder | Source |
|---|---|
| `{affiliate_link}` | `shortened_affiliate_link`, else `affiliate_link`, from `data.json` |
| `{product_title}` | `title` from `data.json` |
| `{hashtags}` | The metadata hashtags, only when `move_hashtags_to_comment` is on, Instagram only |
| `{closing_line}` | The script's closing line |

A template needs data only for the placeholders it uses (`REQ-PUB-039`).

### `affiliate_disclosure`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Put the phrase in the caption of every post that carries a material connection. Off when the section is absent or empty. |
| `phrase` | string | `As an Amazon Associate I earn from qualifying purchases` | The literal phrase. |
| `program` | string | `amazon` | Program name. Metadata only; it does not change how the phrase renders. |

### `blob_retention`

| Key | Type | Default | Bundled | Description |
|---|---|---|---|---|
| `enabled` | bool | `false` | `true` | Trim the Vercel Blob store after each publish run. |
| `max_age_days` | int | `30` | | Delete blobs older than this. |
| `max_total_mb` | int | `500` | | Then delete the oldest until the store is under this size. |

### `delivery_sweep`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Check recent posts for failed platform legs after each publish run. |
| `limit` | int | `25` | Number of recent posts to check. |

### `tiktok_settings`

Shipped commented out. A section with an unknown key falls back to all defaults with a warning.

| Key | Type | Default | Description |
|---|---|---|---|
| `privacy_level` | string | `PUBLIC_TO_EVERYONE` | TikTok privacy level. |
| `allow_comment` | bool | `true` | Allow comments. |
| `allow_duet` | bool | `false` | Allow duets. |
| `allow_stitch` | bool | `false` | Allow stitches. |
| `commercial_content_type` | string | `brand_organic` | `brand_organic`, `brand_content` or `none`. A render with no material connection always sends `none`. |
| `is_brand_organic_post` | bool | `true` | A render with no material connection always sends `false`. |
| `content_preview_confirmed` | bool | `true` | Sent on every post. |
| `express_consent_given` | bool | `true` | Sent on every post. |
| `video_made_with_ai` | bool | `true` | TikTok's AI-generated-content label, sent as `videoMadeWithAi` beside `tiktokSettings` rather than inside it. |

### `analytics`

| Key | Type | Default | Description |
|---|---|---|---|
| `limit` | int | `50` | Posts each `analytics` sweep measures. A whole number of at least 1. A value that is not a mapping, or is refused, falls back to 50 with a warning. |

### `profiles`

Maps a platform to a video profile from `config/video_production.yaml::video_profiles`. The publisher picks `video_<asin>_<profile>.mp4` for that platform when it exists, and the first `video_<asin>_*.mp4` otherwise. A post uploads one file for all its platforms: the render routed to the first platform in its target list.

```yaml
profiles:
  youtube: slideshow_short_20s
  tiktok: slideshow_images1
  instagram: slideshow_images1
```

## Platform limits

`PLATFORM_LIMITS` in `src/publisher/models.py` holds each platform's hard cap. A title or description over the cap is trimmed on a word boundary with `...` before publishing; a hashtag count outside the range is logged as a warning and kept.

| Platform | Title | Caption or description | Hashtags |
|---|---|---|---|
| YouTube | 100 | 5,000 | 3 to 15 |
| TikTok | none | 2,200, hashtags included | 3 to 5 |
| Instagram | none | 2,200 | 5 to 30 |

## Product directory files

### Metadata files

For each platform, the publisher reads the first of these that it can load from `outputs/<product_id>/`:

1. `metadata.json`, written by the producer in unified metadata mode (the default).
2. `metadata_<platform>.json`, written by the producer with `--metadata-mode optimized`.

The producer's `UPLOAD_INSTRUCTIONS.txt` is a guide for uploading by hand; the publisher does not read it.

In unified publishing mode the post uses the first metadata found among its platforms; in platform-specific mode a platform with no metadata of its own uses another platform's. A product with no metadata at all fails with `No metadata found for <product_id>`.

The JSON fields the publisher reads:

| Field | Type | Description |
|---|---|---|
| `title` | string | Video title. Required for YouTube. |
| `description` | string | Required. Trailing hashtags are stripped from it. |
| `hashtags` | list | Hashtags, with or without the leading `#`. |
| `keywords` | list | Keywords. |
| `carries_affiliate_content` | bool | Whether the render has a material connection to disclose. Absent means `true`. |

```json
{
  "platform": "youtube",
  "title": "Amazing Wireless Earbuds - Premium Sound Quality",
  "description": "Check out these incredible wireless earbuds! 30-hour battery life, active noise cancellation and premium sound.",
  "hashtags": ["WirelessEarbuds", "TechReview", "AudioGear"],
  "keywords": ["wireless earbuds", "tech review"],
  "product_id": "B0BTYCRJSS",
  "carries_affiliate_content": true
}
```

## State files

The publisher keeps its state in `outputs/state/`. A legacy copy at the outputs root is moved there the first time it is read.

| File | Contents |
|---|---|
| `publish_history.json` | One record per product and platform (`product_id`, `platform`, `post_id`, `published_at`), the retry queue, and webhook state. Backs the duplicate guard. `published_at` is the time the post was queued, not the time it went live. |
| `schedule.json` | The local schedule `calendar` lists. Written by `single`, `schedule` and the global batch. |
| `post_metrics.json` | Figures captured by `analytics`, merged per field. |
| `published_products.json`, `published_products.csv` | The published-products registry. Each write keeps the previous file as `<name>.bak`. |

Cleanup appends a record per removed product to `outputs/cleanup_audit.json`, with the archive path when one was written.

The retry queue sits under `retry_queue` in `publish_history.json`:

```json
{
  "retry_queue": {
    "B0ABC123": {
      "product_id": "B0ABC123",
      "platforms": ["youtube", "tiktok"],
      "error": "Rate limit exceeded",
      "scheduled_time": "2025-01-20T10:00:00Z",
      "failed_at": "2025-01-17T14:30:00Z",
      "retry_count": 1
    }
  }
}
```

Each registry row holds the product id, title, canonical URL (`https://www.amazon.com/dp/<ASIN>`), affiliate URL and `content_format`.

## API retry policy

Each API call makes up to `max_retries` attempts. The delay between attempts is `2 ** (attempt - 1)` seconds: 1 second, then 2, then 4.

| Response | Behaviour |
|---|---|
| 401, 403 | Fails at once with an authentication error naming the first four characters of the key. |
| 400, 422 | Fails at once with a validation error. |
| 429 | Waits the `Retry-After` header's seconds, or 60 without one, then retries. |
| 5xx, other 4xx | Retries with the delay above. |
| Connection error, timeout | Retries with the delay above. |

Videos over 4 MB upload through the Vercel Blob store, up to 500 MB.

## Webhooks

`WebhookHandler` in `src/publisher/webhooks.py` processes the provider's webhook calls and records the results in `publish_history.json`.

### Events

| Event | Description |
|---|---|
| `post.scheduled` | Post scheduled. |
| `post.published` | Post published. |
| `post.failed` | Post failed on all platforms. |
| `post.partial` | Post succeeded on some platforms. |
| `account.disconnected` | A social account's token expired. |

The event type is read from the payload's `event` or `type` field, and the event id from `eventId` or `id`. Without an id, the handler builds one from the event type, post id and timestamp.

### Signature

Payloads are signed with HMAC-SHA256 of the raw body, keyed by the webhook secret, hex-encoded and sent in the `X-Late-Signature` header. Without a secret the handler logs a warning and skips verification.

```python
import hmac, hashlib
signature = hmac.new(
    key=secret.encode("utf-8"),
    msg=payload_bytes,
    digestmod=hashlib.sha256
).hexdigest()
```

### Handler

```python
from pathlib import Path
from src.publisher import WebhookHandler

handler = WebhookHandler(
    secret="your-webhook-secret",
    outputs_dir=Path("outputs")
)
```

Flask:

```python
from flask import Flask, request, jsonify
from src.publisher import WebhookHandler, WebhookVerificationError

app = Flask(__name__)
handler = WebhookHandler(secret="your-webhook-secret")

@app.route("/webhooks/late", methods=["POST"])
def handle_late_webhook():
    try:
        event = handler.process_webhook(
            payload=request.data,
            signature=request.headers.get("X-Late-Signature")
        )
        return jsonify({
            "status": "ok",
            "event_id": event.event_id,
            "event_type": event.event_type.value
        })
    except WebhookVerificationError as e:
        return jsonify({"error": str(e)}), 401
```

FastAPI:

```python
from fastapi import FastAPI, Request, HTTPException
from src.publisher import WebhookHandler, WebhookVerificationError

app = FastAPI()
handler = WebhookHandler(secret="your-webhook-secret")

@app.post("/webhooks/late")
async def handle_late_webhook(request: Request):
    try:
        body = await request.body()
        event = handler.process_webhook(
            payload=body,
            signature=request.headers.get("X-Late-Signature")
        )
        return {
            "status": "ok",
            "event_id": event.event_id,
            "event_type": event.event_type.value
        }
    except WebhookVerificationError as e:
        raise HTTPException(status_code=401, detail=str(e))
```

### Idempotency

The handler records each processed `event_id` and skips an event it has seen. It keeps the last 1,000 events (`WEBHOOK_EVENT_HISTORY_LIMIT`).

### Webhook state

```python
from src.publisher.webhooks import (
    get_post_status,
    get_disconnected_accounts,
)

status = get_post_status("post_123", outputs_dir)
if status:
    print(f"Status: {status['status']}")
    print(f"URLs: {status['published_urls']}")

disconnected = get_disconnected_accounts(outputs_dir)
for acc in disconnected:
    print(f"Account {acc['account_id']} disconnected")
```

`clear_webhook_events(outputs_dir)` clears the processed-event history.

## Python API

### Publish programmatically

```python
import asyncio
from pathlib import Path
import aiohttp
from src.publisher import create_publisher, PublisherProvider

async def publish_video():
    async with aiohttp.ClientSession() as session:
        publisher = create_publisher(
            provider=PublisherProvider.LATE,
            api_key="sk_live_your_key",
            session=session,
            vercel_token="your_vercel_token",  # Optional
            timeout=30.0,
            max_retries=3,
        )

        if not await publisher.authenticate():
            print("Authentication failed")
            return

        accounts = await publisher.get_accounts()
        youtube_account = next(
            acc for acc in accounts if acc["platform"] == "youtube"
        )

        media_id = await publisher.upload_media(Path("outputs/B0ABC/video.mp4"))

        result = await publisher.publish(
            media_id=media_id,
            platforms=[{
                "platform": "youtube",
                "account_id": youtube_account["account_id"],
            }],
            content="Amazing product video! #viral",
            scheduled_time=None,  # Immediate
        )
        print(f"Published: {result['post_id']}")

asyncio.run(publish_video())
```

`create_publisher_from_config(config, session)` in `src/publisher/registry.py` builds a publisher carrying every setting of a loaded `PublisherConfig`.

### `BasePublisher`

`BasePublisher` in `src/publisher/base.py` is the interface every provider implements. A subclass missing any of these eight abstract members cannot be instantiated:

| Member | Signature |
|---|---|
| `provider` | property returning a `PublisherProvider` |
| `authenticate` | `async () -> bool` |
| `get_accounts` | `async () -> list[dict[str, str]]` |
| `upload_media` | `async (file_path, progress_callback=None) -> str` (media id) |
| `publish` | `async (media_id, platforms, content, scheduled_time=None, ...) -> dict` |
| `get_status` | `async (post_id) -> dict` |
| `list_posts` | `async (status=None) -> list[dict]` |
| `delete_post` | `async (post_id) -> bool` |

`LatePublisher.publish` also takes `platform_contents`, a per-platform payload of `content`, `title` and first comment, and `carries_affiliate_content`.

### `publish_product`

`src/publisher/publish_modes.py` holds the publish orchestration the CLI, the global batch and the scheduler share:

```python
from src.publisher.publish_modes import publish_product

results = await publish_product(
    publisher=publisher,
    media_id="media_123",
    product_id="B0ABC",
    platforms=[{"platform": "youtube", "account_id": "acc_123"}],
    outputs_dir="outputs",
    platform_specific=False,   # True for per-platform metadata
    schedule_time=None,        # datetime for scheduled posts
    disclosure_phrase=None,    # affiliate phrase, when enabled
)
```

### Tracking and the retry queue

`src/publisher/tracking.py` writes `publish_history.json` atomically (temporary file, then rename). Its functions include `record_publish()`, `is_already_published()`, `load_tracking()` and `save_tracking()`, and for the retry queue:

```python
from src.publisher.tracking import (
    get_retry_queue,
    get_retry_queue_count,
    clear_retry_queue,
)

items = get_retry_queue(outputs_dir)
print(f"Failed items: {len(items)}")

cleared = clear_retry_queue(outputs_dir)
print(f"Cleared {cleared} items")
```

### Constants

`src/publisher/constants.py`:

| Constant | Value | Use |
|---|---|---|
| `DEFAULT_OUTPUTS_DIR` | the repository's `outputs/` | Default outputs directory |
| `SDK_LIST_PAGE_SIZE` | `100` | Page size for SDK list calls |
| `MAX_CONCURRENT_CLEANUPS` | `3` | Concurrent cleanup operations |
| `LATE_DIRECT_UPLOAD_MAX_BYTES` | 4 MB | Largest direct upload; larger files go through Vercel Blob |
| `LATE_MAX_UPLOAD_SIZE_BYTES` | 500 MB | Largest upload |
| `LATE_DEFAULT_RETRY_AFTER_SEC` | `60` | Wait after a 429 with no `Retry-After` |
| `LATE_API_KEY_MIN_LENGTH` | `10` | Shortest accepted API key |
| `DEFAULT_EXPONENTIAL_BACKOFF_BASE` | `2` | Base of the retry delay |
| `WEBHOOK_EVENT_HISTORY_LIMIT` | `1000` | Webhook events kept for idempotency |
| `SCHEDULE_MAX_SLOT_SEARCH_ATTEMPTS` | `100` | Slots tried before a product counts as failed |

### Link-in-bio providers

A provider implements `BaseLinkInBioProvider` from `src/publisher/link_in_bio/base.py`:

```python
class BaseLinkInBioProvider(ABC):
    async def authenticate(self) -> bool: ...
    async def add_link(
        self,
        title: str,
        url: str,
        image: str | None = None,
        image_file: Path | None = None,
    ) -> dict[str, object]: ...
    async def list_links(self) -> list[dict[str, object]]: ...
    async def delete_link(self, link_id: str | int) -> bool: ...
```

Register a provider in `create_link_in_bio_manager()` in `src/publisher/link_in_bio/manager.py`.

## External resources

- [Zernio documentation](https://docs.zernio.com) and [API reference](https://docs.zernio.com/api)
- [Zernio dashboard](https://zernio.com/dashboard)
- [Zernio pricing](https://zernio.com/pricing)
- [Zernio status](https://zernio.com/status)
