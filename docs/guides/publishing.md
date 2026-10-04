# Publishing videos

This guide shows how to publish rendered videos to YouTube, TikTok and Instagram through the Zernio scheduling service, schedule a backlog, recover from failures, clean up afterwards and capture post analytics. Every command and key is listed in [the publisher reference](../reference/publisher.md); why each mechanism works the way it does is in [the publishing explanation](../explanation/publishing.md). Error messages are covered in [Troubleshooting](troubleshooting.md#publisher-issues).

## Quick start

```bash
# 1. Get an API key: sign up at https://zernio.com, then Dashboard -> Developers

# 2. Add credentials to .env
echo "LATE_API_KEY=sk_live_your_key_here" >> .env
echo "LATE_VERCEL_TOKEN=vercel_blob_rw_xxx" >> .env  # For videos over 4 MB

# 3. Connect your social accounts at https://zernio.com/dashboard/accounts

# 4. Check the connection
poetry run python -m src.publisher.late list-accounts --debug

# 5. Publish one product into the next free slot
poetry run python -m src.publisher.late single B0BTYCRJSS --debug

# Or publish it at once to two platforms
poetry run python -m src.publisher.late single B0BTYCRJSS \
  --platform youtube --platform tiktok --immediate --debug

# 6. Publish every product in outputs/ at once
poetry run python -m src.publisher.late schedule --immediate \
  --platform youtube --platform tiktok --platform instagram \
  --debug

# 7. Check that first comments landed on recent posts
poetry run python -m src.publisher.late verify-comments --limit 25 --debug
```

## Set up the publisher

1. Create a Zernio account at [zernio.com](https://zernio.com).
2. Create an API key under Dashboard -> Developers. It starts with `sk_live_` or `sk_test_`.
3. To publish videos over 4 MB, create a Vercel Blob token: Vercel Dashboard -> Storage -> Create Blob -> Settings -> Token.
4. Connect YouTube, TikTok, Instagram and any other platforms under Dashboard -> Accounts, and check that each account is active and authorized.
5. Add the credentials to `.env`:

   ```bash
   # Required
   LATE_API_KEY=sk_live_your_api_key_here

   # Required for videos over 4 MB
   LATE_VERCEL_TOKEN=vercel_blob_rw_your_token_here
   ```

   Retries, the request timeout and the default platforms are set in `config/publisher.yaml`.

   Never commit `.env`; use `.env.example` as the template.

6. To keep link-in-bio updates on (the bundled default), add your Lnk.Bio credentials too:

   ```bash
   LNKBIO_CLIENT_ID=your_client_id
   LNKBIO_CLIENT_SECRET=your_client_secret
   ```

7. Review `config/publisher.yaml`. The keys you're most likely to change:

   ```yaml
   provider: late
   immediate_publish: false
   default_platforms: [youtube, tiktok, instagram]
   timeout: 120.0              # TikTok needs longer processing time
   recurring_schedule:
     enabled: true             # Schedule into the next free slot
     timezone: "Europe/Berlin"
   cleanup:
     enabled: true             # Remove products once published
   ```

8. Check the setup by listing the connected accounts:

   ```bash
   poetry run python -m src.publisher.late list-accounts --debug
   ```

## Publish one product

`single` finds the product's video in `outputs/<product_id>/` and its metadata alongside it.

```bash
# Schedule into the next free recurring slot
poetry run python -m src.publisher.late single B0BTYCRJSS --debug

# Publish at once to YouTube
poetry run python -m src.publisher.late single B0BTYCRJSS \
  --platform youtube --immediate

# Publish at once to three platforms
poetry run python -m src.publisher.late single B0BTYCRJSS \
  --platform youtube --platform tiktok --platform instagram \
  --immediate --debug

# Schedule for a specific time (UTC)
poetry run python -m src.publisher.late single B0BTYCRJSS \
  --platform youtube \
  --schedule "2025-01-20 14:00:00" \
  --debug
```

The same command runs through make: `make publish ARGS="single B0BTYCRJSS --debug"`.

## Publish a product from scratch

1. Configure credentials as in [Set up the publisher](#set-up-the-publisher).
2. Check the connection:

   ```bash
   poetry run python -m src.publisher.late list-accounts --debug
   ```

3. Scrape the product and render a video. The producer writes `metadata.json`; add `--metadata-mode optimized` to write per-platform `metadata_<platform>.json` files instead.

   ```bash
   make scrape-lowpri ARGS="--keywords B0BTYCRJSS --debug"
   make produce-lowpri ARGS="outputs/B0BTYCRJSS/data.json slideshow_images1 --metadata-mode optimized --debug"
   ```

4. Publish at once to test:

   ```bash
   poetry run python -m src.publisher.late single B0BTYCRJSS \
     --platform youtube --immediate --debug
   ```

5. Check the post at [zernio.com/dashboard/posts](https://zernio.com/dashboard/posts).

## Schedule the backlog

`schedule` places every unpublished product into the next free recurring slots.

1. Set the slots in `config/publisher.yaml`:

   ```yaml
   recurring_schedule:
     enabled: true
     timezone: "Europe/Berlin"
     slots:
       - day_of_week: monday
         time: "10:00:00"
       - day_of_week: tuesday
         time: "10:00:00"
       # ... one slot a day at 10:00
   schedule_validation:
     min_post_spacing_hours: 2
     prevent_duplicates: true
     allow_past_schedules: false
     max_posts_per_day: 10
   ```

2. Preview the slots each product would take:

   ```bash
   poetry run python -m src.publisher.late schedule \
     --platform youtube --platform tiktok --platform instagram \
     --dry-run --debug
   ```

3. Schedule them:

   ```bash
   poetry run python -m src.publisher.late schedule \
     --platform youtube --platform tiktok --platform instagram \
     --debug
   ```

4. If a slot fails validation, the log suggests alternatives and the product is skipped. To take the first free alternative automatically, add `--auto-resolve`:

   ```bash
   poetry run python -m src.publisher.late schedule \
     --platform youtube --auto-resolve --debug
   ```

## Publish the backlog at once

`schedule --immediate` uploads every product in turn, waiting 30 to 60 seconds between uploads.

```bash
# Every product, to youtube, tiktok and instagram
poetry run python -m src.publisher.late schedule --immediate --debug

# Two platforms, stopping at the first failure
poetry run python -m src.publisher.late schedule --immediate \
  --platform youtube --platform tiktok --fail-fast --debug
```

`--dry-run` lists the products the run would publish and publishes nothing. A product already published to every target is skipped unless you pass `--force`.

## Retry failed products

A product that fails in `schedule --immediate` goes into the retry queue.

1. Run the batch:

   ```bash
   poetry run python -m src.publisher.late schedule --immediate \
     --platform youtube --platform tiktok --debug
   ```

2. Check what failed:

   ```bash
   poetry run python -m src.publisher.late calendar list --status failed
   ```

3. Retry only the failed products:

   ```bash
   poetry run python -m src.publisher.late schedule --immediate \
     --platform youtube --platform tiktok --retry-failed --debug
   ```

## Republish a product

By default the publisher skips a product already published to a platform, so reruns are safe. Pass `--force` to post it again, for example after re-rendering the video:

```bash
# Publish one product again
poetry run python -m src.publisher.late single B0ABC123 --force --debug

# Schedule the whole backlog, including products already published
poetry run python -m src.publisher.late schedule --force --debug
```

A rerun of `single` without `--force` on a fully published product still refreshes its link-in-bio entry. Add `--no-link-in-bio` for a rerun that changes nothing.

## Publish one post per platform

By default a product goes out as one post to every platform. To send each platform its own post with its own metadata, render with `--metadata-mode optimized` and pass `--platform-specific`, or set `use_platform_specific_content: true`:

```bash
# One post to all three platforms
poetry run python -m src.publisher.late single B0ABC \
  --platform youtube --platform tiktok --platform instagram \
  --immediate

# Three posts, each with its platform's metadata
poetry run python -m src.publisher.late single B0ABC \
  --platform youtube --platform tiktok --platform instagram \
  --platform-specific --immediate

# The same through the global batch
make batch-lowpri ARGS="--keywords earbuds --max-products 1 --random-profile --platform-specific --debug"
```

## Use several provider accounts

1. Define the accounts in `config/publisher.yaml`:

   ```yaml
   accounts:
     brand_a:
       api_key: sk_live_brand_a_key
     brand_b:
       api_key: sk_live_brand_b_key
   default_account: brand_a
   ```

2. Publish with the default account, or pick one with `--account` before the command name:

   ```bash
   poetry run python -m src.publisher.late single B0ABC123 --immediate
   poetry run python -m src.publisher.late --account brand_b single B0ABC123 --immediate
   ```

Without `--account`, a `LATE_API_KEY` in the environment replaces the default account's key.

## Check the schedule

```bash
# Every scheduled post
poetry run python -m src.publisher.late calendar list --debug

# One platform
poetry run python -m src.publisher.late calendar list \
  --platform youtube --debug

# A date range
poetry run python -m src.publisher.late calendar list \
  --date-from "2025-12-19" \
  --date-to "2025-12-25" \
  --debug

# One status
poetry run python -m src.publisher.late calendar list \
  --status scheduled \
  --debug
```

The list is the local schedule. When the provider holds a different number of upcoming posts, the command warns with both counts.

## Check delivery and first comments

The provider reports a post accepted, not delivered, and a first comment can fail without an error. After a batch goes live:

```bash
# Warn on every recent post with a failed platform leg
poetry run python -m src.publisher.late verify-delivery --limit 25

# Warn on every recent YouTube or Instagram post missing its first comment
poetry run python -m src.publisher.late verify-comments --limit 25
```

The delivery check also runs after every publish run. To fix a failed leg, see [Instagram container errors](troubleshooting.md#instagram-container-errors-and-partial-posts) for a transient failure and [Repair a failed TikTok post](#repair-a-failed-tiktok-post) for a rejected payload.

## Repair a failed TikTok post

If TikTok rejects a post with `Commercial content disclosure is enabled but no option selected`, update the post's TikTok settings through the SDK. The update republishes the failed platform on its own.

```python
import asyncio, late, os

async def fix_tiktok(post_id: str):
    client = late.Late(api_key=os.environ["LATE_API_KEY"])

    # Update platform-level TikTok settings with correct disclosure
    platforms = [
        {"platform": "youtube", "accountId": "<youtube_account_id>"},
        {
            "platform": "tiktok",
            "accountId": "<tiktok_account_id>",
            "platformSpecificData": {
                "tiktokSettings": {
                    "privacy_level": "PUBLIC_TO_EVERYONE",
                    "allow_comment": True,
                    "allow_duet": False,
                    "allow_stitch": False,
                    "commercial_content_type": "brand_organic",
                    "is_brand_organic_post": True,
                    "content_preview_confirmed": True,
                    "express_consent_given": True,
                },
                # Measured: an update REPLACES platformSpecificData rather than
                # merging it, so anything omitted here is stripped from the
                # post. Leaving this out republishes an AI-voiced video with
                # no AI-content label, silently, because the field is passed
                # through as a raw dict and nothing rejects its absence.
                "videoMadeWithAi": True,
            },
        },
        {"platform": "instagram", "accountId": "<instagram_account_id>"},
    ]

    # Update triggers automatic re-publish (no retry() needed)
    result = await client.posts.aupdate(post_id, platforms=platforms)
    print(f"TikTok status: {result.post.platforms[1].status}")
    # Status changes: failed -> pending -> processing -> published

asyncio.run(fix_tiktok("your_post_id"))
```

Don't call `retry()` after the update: it returns 409 `Post is currently publishing`. Wait about 30 seconds and check the status with `aget()`.

## Clean up published products

With `cleanup.enabled` on, every publish run removes the directories of products it confirmed. To clean up by hand:

1. Preview what would be removed:

   ```bash
   poetry run python -m src.publisher.late cleanup --all --dry-run --debug
   ```

2. Confirm the posts are live:

   ```bash
   poetry run python -m src.publisher.late verify-delivery --limit 25
   ```

3. Review the audit trail in `outputs/logs/publisher-<date>.log`.
4. Run the cleanup:

   ```bash
   # Every published product
   poetry run python -m src.publisher.late cleanup --all --confirm --debug

   # One product
   poetry run python -m src.publisher.late cleanup \
     --product-id B0BTYCRJSS \
     --debug
   ```

To skip cleanup for a single run, pass `--no-cleanup`:

```bash
poetry run python -m src.publisher.late single B0ABC \
  --platform youtube --immediate --no-cleanup

poetry run python -m src.publisher.late schedule --immediate \
  --platform youtube --platform tiktok \
  --no-cleanup --debug
```

### Clean up safely

A cautious setup keeps an archive and a grace period:

```yaml
cleanup:
  enabled: true
  verify_before_delete: true      # Always keep enabled
  require_all_platforms: true     # Only clean up if every platform succeeded
  archive_before_delete: true     # ZIP the directory first
  archive_dir: "outputs/archive"
  keep_published_days: 7          # Wait 7 days after publication
```

Checklist:

- [ ] Run `--dry-run` first on production data.
- [ ] Keep `require_all_platforms: true` when publishing to several platforms.
- [ ] Turn on `archive_before_delete` for valuable content.
- [ ] Check the Zernio dashboard to confirm posts are live.
- [ ] Review `outputs/cleanup_audit.json` afterwards.
- [ ] Keep products for at least 7 days (`keep_published_days: 7`).

Avoid these:

- Running `cleanup --all --confirm` without a `--dry-run` preview.
- Turning off `verify_before_delete` in production.
- Setting `keep_published_days: 0` without archiving.
- Cleaning up before posts are published, not just scheduled.

If a cleanup removed something by mistake:

1. Look in the archive directory, `outputs/archive/`.
2. Find the removed products in `outputs/cleanup_audit.json`.
3. Without an archive, scrape and render the products again.

## Capture post analytics on a schedule

Day-2 and day-7 figures can be captured only while a post is inside the provider's five-week retention window, so run the sweep daily rather than once at the end of a comparison. [The publishing explanation](../explanation/publishing.md#post-analytics) says why.

1. Run a sweep by hand to check it works:

   ```bash
   # Measure recent published posts and store the figures
   python -m src.publisher.late analytics

   # Re-rank stored figures without contacting the provider
   python -m src.publisher.late analytics --rank-only
   ```

2. Install the daily timer:

   ```bash
   make install-analytics-timer
   ```

   This renders a systemd user timer, installs and enables it, runs one sweep, and checks that `state/post_metrics.json` under the outputs root changed (`OUTPUTS_DIR` moves the root; `python -m src.utils.outputs_paths` prints it). It needs no root, and with lingering enabled it runs whether or not you're logged in.

3. Check on it later with `make analytics-timer-status`. Remove it with `make uninstall-analytics-timer`; the captured figures stay.

Settings live in two files, split by what reads them:

| Setting | Lives in | Why there |
|---|---|---|
| How many posts a sweep measures | `config/publisher.yaml::analytics.limit` | Behaviour, read by both the manual and the scheduled run |
| Schedule, timeouts, failure reporting, paths | `deploy/schedule.env` | Shapes the unit files, which systemd reads before any of this project's code runs |

To change the second set, copy `deploy/schedule.env.example` to `deploy/schedule.env` and edit what you need. The copy is gitignored and every key is optional, so the installer works before you write it. Re-run the installer after editing it, because systemd doesn't pick up a changed `OnCalendar` on its own. When you changed only the schedule, run `./deploy/install-timer.sh --no-run`: it re-renders and re-arms the timer without spending a sweep's worth of API calls.

A failed sweep is recorded three ways: in the journal, appended to `outputs/logs/analytics-failures.log`, and as a desktop notification when a session is there to receive it. The log file is the durable one, and `make analytics-timer-status` shows it. Set `NOTIFY_ON_FAILURE=0` to install no failure handler.

Cron works too, but it has no equivalent of the timer's `Persistent=true` and misses a sweep while the machine sleeps:

```cron
@daily cd /path/to/ContentEngineAI && ~/.pyenv/versions/ContentEngineAI/bin/python -m src.publisher.late analytics
```

## Plan a week of content

1. Scrape and render the week's products without publishing them:

   ```bash
   make batch-lowpri ARGS="--keywords 'wireless earbuds' --max-products 7 --products-per-keyword 7 --profile slideshow_images1 --skip-publish --debug"
   ```

2. Schedule them into the recurring slots, one a day with the bundled schedule:

   ```bash
   poetry run python -m src.publisher.late schedule \
     --platform youtube --platform tiktok --platform instagram \
     --debug
   ```

3. Review the schedule:

   ```bash
   poetry run python -m src.publisher.late calendar list --debug
   ```

4. If conflicts were skipped, schedule again with `--auto-resolve`:

   ```bash
   poetry run python -m src.publisher.late schedule \
     --platform youtube --auto-resolve --debug
   ```
