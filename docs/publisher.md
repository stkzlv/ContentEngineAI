# Publisher Module - Social Media Publishing

[![Late.dev Integration](https://img.shields.io/badge/Late.dev-Integrated-brightgreen)](https://late.dev)
[![Version](https://img.shields.io/badge/version-0.19.0-blue)](../CHANGELOG.md)

**Automatically publish your generated videos to social media platforms via Late.dev**

The Publisher module provides a complete solution for distributing your AI-generated product videos across YouTube, TikTok, Instagram, Facebook, Twitter, and LinkedIn. It integrates seamlessly with [Late.dev](https://late.dev) for multi-platform publishing with scheduling, metadata management, and batch processing.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Setup](#-setup)
- [CLI Usage](#-cli-usage)
- [Configuration](#-configuration)
- [Platform Metadata](#-platform-metadata)
- [Batch Publishing](#-batch-publishing)
- [Publishing Schedule & Calendar](#-publishing-schedule--calendar)
- [Post-Publication Cleanup](#-post-publication-cleanup)
- [Error Handling](#-error-handling)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)

---

## ✨ Features

- **🌐 Multi-Platform Publishing**: YouTube, TikTok, Instagram, Facebook, Twitter, LinkedIn
- **📅 Auto-Scheduling**: Automatically finds first available unoccupied slot in recurring schedule
- **📆 Calendar Management**: View and filter all scheduled posts by platform, date, and status
- **🔄 Batch Publishing**: Upload multiple videos with automatic rate limiting
- **🗑️ Auto-Cleanup**: Automatically remove published products from outputs directory
- **📝 Platform-Specific Metadata**: Auto-loads AI-generated titles, descriptions, hashtags
- **⚡ Smart Uploads**: Large files (>4MB) automatically routed through Vercel CDN
- **🔁 Retry Logic**: Exponential backoff for rate limits and network errors
- **✅ Progress Tracking**: Real-time upload progress with callbacks
- **🎯 CLI Interface**: Simple command-line interface for all operations

---

## 🚀 Quick Start

```bash
# 1. Get Late.dev API credentials
# Sign up at https://late.dev and get your API key from Dashboard → Developers

# 2. Configure credentials in .env
echo "LATE_API_KEY=sk_live_your_key_here" >> .env
echo "LATE_VERCEL_TOKEN=vercel_blob_rw_xxx" >> .env  # Optional, for large files >4MB

# 3. Connect your social media accounts
# Visit https://late.dev/dashboard/accounts and connect platforms

# 4. Verify setup - list connected accounts
poetry run python -m src.publisher.late list-accounts --debug

# 5. Publish a video immediately
poetry run python -m src.publisher.late single \
  --video outputs/B0BTYCRJSS/video_B0BTYCRJSS_sequential.mp4 \
  --platform youtube --platform tiktok \
  --immediate --debug

# 6. Batch publish all videos in outputs directory
poetry run python -m src.publisher.late batch \
  --platform youtube --platform tiktok --platform instagram \
  --immediate --debug
```

---

## 🔧 Setup

### 1. Late.dev Account Setup

1. **Create Account**: Sign up at [https://late.dev](https://late.dev)
2. **Get API Credentials**:
   - Navigate to Dashboard → Developers
   - Create a new API key (starts with `sk_live_` or `sk_test_`)
   - (Optional) Get Vercel token for large file uploads (>4MB)
3. **Connect Social Accounts**:
   - Go to Dashboard → Accounts
   - Connect YouTube, TikTok, Instagram, and other platforms
   - Verify accounts are active and authorized

### 2. Environment Configuration

Create or update your `.env` file:

```bash
# Required: Late.dev API Key
LATE_API_KEY=sk_live_your_api_key_here

# Optional: Vercel Blob Token for large file uploads (>4MB)
# Required if publishing videos larger than 4MB
# Get from: Vercel Dashboard → Storage → Create Blob → Settings → Token
LATE_VERCEL_TOKEN=vercel_blob_rw_your_token_here

# Optional: Override default settings
LATE_TIMEOUT=30.0
LATE_MAX_RETRIES=3
```

**Security Note**: Never commit `.env` to version control. Use `.env.example` as a template.

### 3. Publisher Configuration

The publisher uses a three-tier configuration system with the following precedence (highest to lowest):

1. **CLI Arguments** (highest priority)
2. **Environment Variables** (`.env`)
3. **Configuration File** (`config/publisher.yaml`, lowest priority)

**Key Configuration** (`config/publisher.yaml`):

```yaml
provider: late
immediate_publish: true
default_platforms: [youtube, tiktok, instagram]
timeout: 120.0              # TikTok needs longer processing time
recurring_schedule:
  enabled: true             # Auto-schedule to next available slot
  timezone: "Europe/Berlin"
cleanup:
  enabled: true             # Auto-cleanup after successful publish
```

See [Configuration](#-configuration) for full options.

### 4. Verify Setup

Test your configuration by listing connected accounts (see [CLI Usage](#-cli-usage) for details).

---

## 💻 CLI Usage

The publisher provides three main commands: `list-accounts`, `single`, and `batch`.

### Command: `list-accounts`

List all connected social media accounts.

```bash
poetry run python -m src.publisher.late list-accounts [--debug]
```

**Options:**
- `--debug`: Enable verbose debug logging

**Example:**

```bash
poetry run python -m src.publisher.late list-accounts --debug
```

---

### Command: `single`

Publish a single video to one or more platforms.

```bash
poetry run python -m src.publisher.late single \
  --video <path> \
  --platform <platform> \
  [--platform <platform2> ...] \
  [--immediate | --schedule <datetime>] \
  [--debug]
```

**Required Arguments:**
- `--video <path>`: Path to video file (must exist)
- `--platform <name>`: Target platform (can be specified multiple times)
  - Valid platforms: `youtube`, `tiktok`, `instagram`, `facebook`, `twitter`, `linkedin`

**Publishing Options:**
- `--immediate`: Publish immediately (default if neither flag specified)
- `--schedule <datetime>`: Schedule for future publishing
  - Format: `YYYY-MM-DD HH:MM:SS` or `YYYY-MM-DDTHH:MM:SS`
  - Example: `2025-01-20 14:00:00` or `2025-01-20T14:00:00`

**Other Options:**
- `--debug`: Enable verbose debug logging

**Examples:**

```bash
# Publish immediately to YouTube
poetry run python -m src.publisher.late single \
  --video outputs/B0BTYCRJSS/video_B0BTYCRJSS_sequential.mp4 \
  --platform youtube \
  --immediate

# Publish to multiple platforms
poetry run python -m src.publisher.late single \
  --video outputs/B0BTYCRJSS/video_B0BTYCRJSS_sequential.mp4 \
  --platform youtube --platform tiktok --platform instagram \
  --immediate --debug

# Schedule for future publishing
poetry run python -m src.publisher.late single \
  --video outputs/B0BTYCRJSS/video_B0BTYCRJSS_sequential.mp4 \
  --platform youtube \
  --schedule "2025-01-20 14:00:00" \
  --debug
```

**Metadata Loading:**

The `single` command automatically loads platform-specific metadata from:
1. `outputs/<PRODUCT_ID>/metadata_<platform>.json` (preferred)
2. `outputs/<PRODUCT_ID>/UPLOAD_INSTRUCTIONS.txt` (fallback)

If no metadata is found, it uses a basic content template.

---

### Command: `batch`

Publish all videos in the outputs directory to specified platforms.

```bash
poetry run python -m src.publisher.late batch \
  --platform <platform> \
  [--platform <platform2> ...] \
  --immediate \
  [--outputs-dir <path>] \
  [--fail-fast] \
  [--debug]
```

**Required Arguments:**
- `--platform <name>`: Target platform (can be specified multiple times)
- `--immediate`: Publish immediately (scheduled publishing not supported in batch mode)

**Optional Arguments:**
- `--outputs-dir <path>`: Directory to scan for videos (default: `outputs`)
- `--fail-fast`: Stop processing on first failure (default: continue processing)
- `--debug`: Enable verbose debug logging

**Examples:**

```bash
# Batch publish all videos to YouTube and TikTok
poetry run python -m src.publisher.late batch \
  --platform youtube --platform tiktok \
  --immediate --debug

# Publish to all platforms with fail-fast
poetry run python -m src.publisher.late batch \
  --platform youtube --platform tiktok --platform instagram \
  --immediate --fail-fast --debug

# Specify custom outputs directory
poetry run python -m src.publisher.late batch \
  --platform youtube \
  --outputs-dir /path/to/custom/outputs \
  --immediate
```

**Batch Behavior:**

- Scans `outputs` directory for product folders
- Finds videos matching pattern: `video_*_sequential.mp4` or `video_*_slideshow.mp4`
- Loads metadata for each product from `metadata_<platform>.json`
- **Creates separate posts per platform when platform-specific metadata exists**
  - Example: 10 products × 3 platforms = 30 posts total
  - Each platform receives optimized content from its metadata file
- Staggers uploads with random delays (30-60s by default) to avoid rate limits
- Skips products without videos or metadata (continues processing unless `--fail-fast`)
- Reports summary statistics at completion

**Batch Summary Output:**

```
================================================================================
BATCH PUBLISHING SUMMARY
================================================================================
Total products: 10
✅ Successful: 8
❌ Failed: 2
⏭️  Skipped: 0
⏱️  Duration: 8m 45s
================================================================================
```

---

## ⚙️ Configuration

<details>
<summary><strong>Configuration Precedence</strong></summary>

The publisher uses a **three-tier configuration system** with the following precedence:

```
CLI Arguments (highest) → Environment Variables → Configuration File (lowest)
```

**Priority Examples:**

```bash
# CLI argument overrides config file
poetry run python -m src.publisher.late single \
  --video video.mp4 \
  --platform youtube \
  --immediate
# → Uses YouTube (CLI) even if publisher.yaml specifies TikTok

# Environment variable overrides config file
export LATE_API_KEY=sk_live_new_key
# → Uses new key instead of publisher.yaml value

# Config file provides defaults
# If no CLI args or env vars, uses config/publisher.yaml defaults
```

</details>

<details>
<summary><strong>Configuration File Structure</strong></summary>

**File**: `config/publisher.yaml`

```yaml
# === Provider Settings ===
provider: late                      # Publisher provider (only "late" supported)
api_key: ${LATE_API_KEY}           # API key (use env var for security)
vercel_token: ${LATE_VERCEL_TOKEN} # Vercel token for large files (optional)

# === Publishing Defaults ===
immediate_publish: true             # Default to immediate vs scheduled
default_platforms:                  # Platforms to use if none specified
  - youtube
  - tiktok
  - instagram

# === Retry & Timeout ===
max_retries: 3                     # Maximum retry attempts on failure
timeout: 120.0                     # HTTP request timeout (TikTok needs longer)
backoff_multiplier: 2.0            # Exponential backoff multiplier (2^n)

# === Batch Settings ===
stagger_delay_min: 30              # Min delay between batch uploads (seconds)
stagger_delay_max: 60              # Max delay between batch uploads (seconds)

# === Privacy Settings ===
privacy_settings:
  youtube: public                  # public, unlisted, private
  tiktok: public                   # public, friends, private
  instagram: everyone              # everyone, followers, private
  facebook: public                 # public, friends
  twitter: public                  # public
  linkedin: public                 # public, connections
```

</details>

<details>
<summary><strong>Environment Variables</strong></summary>

All configuration values can be overridden via environment variables:

```bash
# Required
export LATE_API_KEY=sk_live_your_key

# Optional (for large files >4MB)
export LATE_VERCEL_TOKEN=vercel_blob_rw_xxx
export LATE_TIMEOUT=60.0
export LATE_MAX_RETRIES=5
export LATE_STAGGER_MIN=10
export LATE_STAGGER_MAX=30
```

</details>

<details>
<summary><strong>Platform-Specific Settings</strong></summary>

**YouTube:**
- Privacy: `public`, `unlisted`, `private`
- Supports scheduled publishing
- Max title length: 100 characters
- Max description length: 5000 characters

**TikTok:**
- Privacy: `public`, `friends`, `private`
- Supports scheduled publishing
- Max caption length: 2200 characters (includes hashtags)

**Instagram:**
- Privacy: `everyone`, `followers`, `private`
- Caption length: 2200 characters
- Supports scheduled publishing (Reels)

</details>

---

## 📝 Platform Metadata

### Overview

The publisher automatically loads platform-specific metadata generated by the video producer. This metadata includes AI-optimized titles, descriptions, hashtags, and formatting for each platform.

### Platform-Specific Content Architecture

**Important**: When platform-specific metadata is enabled, the publisher creates **separate posts for each platform** rather than a single multi-platform post. This is necessary because Late.dev's API stores custom content per platform but doesn't apply it during publishing.

**Single vs. Multiple Posts:**

```bash
# Without platform-specific metadata (default)
# → 1 post published to all 3 platforms (same content)
poetry run python -m src.publisher.late single \
  --video outputs/B0ABC/video.mp4 \
  --platform youtube --platform tiktok --platform instagram \
  --immediate

# With platform-specific metadata files present
# → 3 separate posts (each with platform-optimized content)
#   - YouTube post: metadata_youtube.json content + title
#   - TikTok post: metadata_tiktok.json content
#   - Instagram post: metadata_instagram.json content
```

**Scheduling Implications:**

When scheduling videos with platform-specific metadata:
- Each platform receives its own scheduled post
- All platforms for the same product use the same time slot
- Schedule tracking creates separate entries per platform
- Example: 3 products → 9 scheduled posts (3 per product)

### Metadata File Location

Metadata files are stored in the product directory root:

```
outputs/
└── B0BTYCRJSS/
    ├── video_B0BTYCRJSS_sequential.mp4
    ├── metadata_youtube.json
    ├── metadata_tiktok.json
    ├── metadata_instagram.json
    └── UPLOAD_INSTRUCTIONS.txt  # Fallback
```

### Metadata JSON Format

**YouTube** (`metadata_youtube.json`):

```json
{
  "platform": "youtube",
  "title": "Amazing Wireless Earbuds - Premium Sound Quality",
  "description": "Check out these incredible wireless earbuds!\n\n🔥 Key Features:\n✅ 30-hour battery life\n✅ Active noise cancellation\n✅ Premium sound quality\n\n#WirelessEarbuds #TechReview #AudioGear",
  "tags": ["wireless earbuds", "tech review", "audio"],
  "product_id": "B0BTYCRJSS"
}
```

**TikTok** (`metadata_tiktok.json`):

```json
{
  "platform": "tiktok",
  "title": "Amazing Earbuds 🔥",
  "description": "These earbuds are incredible! 30hr battery + ANC 🎧\n\n#WirelessEarbuds #TechTok #FYP",
  "tags": ["wirelessearbuds", "techtok", "fyp"],
  "product_id": "B0BTYCRJSS"
}
```

**Instagram** (`metadata_instagram.json`):

```json
{
  "platform": "instagram",
  "title": "Premium Wireless Earbuds",
  "description": "Game-changing wireless earbuds 🎧\n\n✨ 30-hour battery\n✨ Active noise cancellation\n✨ Premium sound\n\n#WirelessEarbuds #TechReview #AudioGear",
  "tags": ["wirelessearbuds", "techreview", "audiogear"],
  "product_id": "B0BTYCRJSS"
}
```

### Fallback to UPLOAD_INSTRUCTIONS.txt

If JSON metadata files are not found, the publisher falls back to parsing `UPLOAD_INSTRUCTIONS.txt`:

```text
Product ID: B0BTYCRJSS

YouTube:
Title: Amazing Wireless Earbuds - Premium Sound Quality
Description: Check out these incredible wireless earbuds with 30-hour battery life.

TikTok:
Title: Amazing Earbuds 🔥
Description: These earbuds are incredible! #WirelessEarbuds #TechTok

Instagram:
Title: Premium Wireless Earbuds
Description: Game-changing wireless earbuds 🎧 #WirelessEarbuds
```

### Generate Metadata

Platform metadata is automatically generated when producing videos with the `--target-platform` flag:

```bash
# Generate metadata for all platforms
poetry run python -m src.video.producer \
  outputs/B0BTYCRJSS/data.json \
  slideshow_images1 \
  --target-platform multi \
  --debug

# This creates:
# - outputs/B0BTYCRJSS/metadata_youtube.json
# - outputs/B0BTYCRJSS/metadata_tiktok.json
# - outputs/B0BTYCRJSS/metadata_instagram.json
# - outputs/B0BTYCRJSS/UPLOAD_INSTRUCTIONS.txt
```

---

## 🔄 Batch Publishing

<details>
<summary><strong>Overview & Workflow</strong></summary>

Batch publishing processes multiple videos from the `outputs` directory sequentially with automatic rate limiting.

**How Batch Publishing Works:**

1. **Discovery**: Scans `outputs` directory for product folders
2. **Video Detection**: Finds videos matching patterns:
   - `video_<PRODUCT_ID>_sequential.mp4`
   - `video_<PRODUCT_ID>_slideshow.mp4`
3. **Metadata Loading**: Loads platform-specific metadata for each product
4. **Sequential Upload**: Uploads videos one at a time with stagger delays
5. **Error Handling**: Continues on failure (unless `--fail-fast` specified)
6. **Summary Report**: Displays statistics after completion

**Stagger Delays:**

To avoid rate limiting, batch publishing adds random delays between uploads:

```yaml
# config/publisher.yaml
stagger_delay_min: 30  # Minimum 30 seconds between uploads
stagger_delay_max: 60  # Maximum 60 seconds between uploads
```

**Calculation**: Random delay between min and max (uniform distribution)

**Fail-Fast Mode:**

By default, batch publishing continues even if individual uploads fail. Use `--fail-fast` to stop on first failure:

```bash
# Stop on first error
poetry run python -m src.publisher.late batch \
  --platform youtube \
  --immediate \
  --fail-fast
```

**Batch Performance:**

**Expected timing** (for 10 products):
- Upload time per video: ~10-30s (depending on file size)
- Stagger delays: 30-60s between uploads
- Total time: ~7-15 minutes for 10 videos

**Optimization tips:**
- Reduce `stagger_delay_min` for faster processing (risk of rate limits)
- Use `--fail-fast` to catch issues early
- Filter products before batch publishing

</details>

---

## 📅 Publishing Schedule & Calendar

<details>
<summary><strong>Calendar View</strong></summary>

List and manage all scheduled posts with filtering capabilities:

```bash
# View all scheduled posts
poetry run python -m src.publisher.late calendar list --debug

# Filter by platform
poetry run python -m src.publisher.late calendar list \
  --platform youtube --debug

# Filter by date range
poetry run python -m src.publisher.late calendar list \
  --date-from "2025-12-19" \
  --date-to "2025-12-25" \
  --debug

# Filter by status (scheduled, published, failed)
poetry run python -m src.publisher.late calendar list \
  --status scheduled \
  --debug
```

**Calendar Response:**
```
================================================================================
SCHEDULED POSTS
================================================================================
[1/5] 2025-12-19 09:00:00 UTC (10:00 CET)
      Product: B09LYF2ST7
      Platforms: youtube, tiktok, instagram
      Post IDs: 6943cf76... (instagram), 6943cf78... (tiktok), 6943cf7a... (youtube)
      Status: scheduled
--------------------------------------------------------------------------------
[2/5] 2025-12-20 14:00:00 UTC (15:00 CET)
      Product: B0BTYCRJSS
      Platforms: youtube, instagram
      Post IDs: 6944a12b..., 6944a12d...
      Status: scheduled
--------------------------------------------------------------------------------
...
```

</details>

<details>
<summary><strong>Recurring Schedule Configuration</strong></summary>

Configure recurring publishing times for automated queue-based publishing:

**Configuration** (`config/publisher.yaml`):

```yaml
# === Recurring Schedule ===
recurring_schedule:
  enabled: true
  timezone: "Europe/Berlin"  # CET timezone
  slots:
    - day_of_week: monday
      time: "10:00:00"
    - day_of_week: tuesday
      time: "10:00:00"
    - day_of_week: wednesday
      time: "10:00:00"
    # ... daily slots at 10:00 AM CET
```

**CLI Usage:**

```bash
# Schedule videos to next available recurring slots
poetry run python -m src.publisher.late schedule auto \
  --outputs-dir outputs \
  --debug

# Preview schedule without publishing
poetry run python -m src.publisher.late schedule auto \
  --outputs-dir outputs \
  --dry-run \
  --debug

# Skip to specific slot number
poetry run python -m src.publisher.late schedule auto \
  --outputs-dir outputs \
  --start-slot 3 \
  --debug
```

**Auto-Scheduling Behavior:**
1. Loads recurring schedule from configuration
2. Scans outputs directory for unpublished videos
3. **Queries Late.co API** to find occupied slots (8-week lookahead)
4. Finds first available unoccupied slot by comparing scheduled times
5. **Creates separate posts per platform when metadata files exist**
   - Reads `metadata_youtube.json`, `metadata_tiktok.json`, `metadata_instagram.json`
   - Each platform gets its own post with platform-specific content
   - All platforms for same product scheduled to same time slot
6. Falls back to immediate publishing if all slots occupied
7. Reports scheduled times and slot assignments

### Schedule Validation

The publisher enforces schedule validation rules:

**Validation Rules:**
- No duplicate scheduling (same product + platform + time)
- Minimum spacing between posts on same platform (configurable: 1-24 hours)
- Timezone-aware datetime validation
- Platform-specific posting hour restrictions (if configured)

**Configuration** (`config/publisher.yaml`):

```yaml
# === Schedule Validation ===
schedule_validation:
  min_post_spacing_hours: 2      # Minimum hours between posts on same platform
  prevent_duplicates: true        # Block duplicate product+platform+time
  allow_past_schedules: false     # Block scheduling in the past
  max_posts_per_day: 10          # Platform rate limit (per platform)
```

</details>

---

## 🗑️ Post-Publication Cleanup

<details>
<summary><strong>Automatic Cleanup</strong></summary>

Automatically remove successfully published product directories from outputs after confirmed publication:

**Configuration** (`config/publisher.yaml`):

```yaml
# === Post-Publication Cleanup ===
cleanup:
  enabled: true                         # Enable automatic cleanup
  verify_before_delete: true            # Verify publication success before cleanup
  require_all_platforms: true           # Only cleanup if published to ALL configured platforms

  # Per-platform cleanup settings
  platforms:
    youtube:
      auto_cleanup: true
    tiktok:
      auto_cleanup: true
    instagram:
      auto_cleanup: true
    facebook:
      auto_cleanup: false               # Keep outputs for Facebook posts
```

**Cleanup Behavior:**
- **Verification**: Confirms publication success via API status check before deletion
- **Multi-Platform Validation**: Requires successful publication to ALL configured platforms (unless `require_all_platforms: false`)
- **Audit Logging**: Logs all deleted directories with product IDs, platforms, and post URLs
- **Selective Cleanup**: Respects per-platform `auto_cleanup` settings

**CLI Override:**

```bash
# Disable cleanup for single publish
poetry run python -m src.publisher.late single \
  --video outputs/B0ABC/video.mp4 \
  --platform youtube \
  --immediate \
  --no-cleanup

# Disable cleanup for batch publish
poetry run python -m src.publisher.late batch \
  --platform youtube --platform tiktok \
  --immediate \
  --no-cleanup \
  --debug
```

</details>

<details>
<summary><strong>Manual Cleanup</strong></summary>

Clean up specific products or all published products:

```bash
# Clean up specific product
poetry run python -m src.publisher.late cleanup \
  --product-id B0BTYCRJSS \
  --debug

# Clean up all published products
poetry run python -m src.publisher.late cleanup \
  --all \
  --debug

# Preview cleanup without deletion
poetry run python -m src.publisher.late cleanup \
  --all \
  --dry-run \
  --debug
```

**Safety Features:**
- **Dry Run Mode**: Preview what would be deleted without actually deleting
- **Verification**: Double-checks publication status before deletion
- **Logging**: Detailed audit trail of all cleanup operations
- **Backup Option**: Optional archive to ZIP before deletion (configurable)

**Advanced Configuration:**

```yaml
cleanup:
  enabled: true
  verify_before_delete: true
  require_all_platforms: true

  # Archive before cleanup
  archive_before_delete: false
  archive_dir: "outputs/archives"

  # Retention period
  keep_published_days: 0              # 0 = immediate, 7 = keep for 7 days

  # What to preserve
  preserve_metadata: false            # Keep metadata JSON files
  preserve_logs: true                 # Keep log files in outputs/logs/
```

**Cleanup Summary:**

```
================================================================================
CLEANUP SUMMARY
================================================================================
Total products evaluated: 10
✅ Cleaned up: 7 products
   - B0BTYCRJSS (youtube, tiktok, instagram)
   - B0CPSY5HJY (youtube, instagram)
   - B0CTTZJRL6 (youtube, tiktok)
   ...
⏭️  Skipped: 3 products
   - B0ABC123 (not published to all platforms)
   - B0DEF456 (publication failed)
   - B0GHI789 (cleanup disabled for platform)
💾 Disk space freed: 1.2 GB
================================================================================
```

</details>

---

## ⚠️ Error Handling

<details>
<summary><strong>Retry Logic & Rate Limiting</strong></summary>

The publisher implements automatic retry with exponential backoff for transient errors:

**Retry Strategy:**
- Initial retry: 2 seconds
- Second retry: 4 seconds (2^1 × 2s)
- Third retry: 8 seconds (2^2 × 2s)
- Max retries: 3 (configurable)

**Retryable Errors:**
- Network timeouts and connection errors
- Server errors (500, 502, 503, 504)
- Rate limit errors (429) with `Retry-After` header

**Non-Retryable Errors:**
- Authentication failures (401, 403)
- Validation errors (400, 422)
- Resource not found (404)

### Rate Limiting

**Late.dev Rate Limits** (as of v0.17.0):
- Standard tier: 100 requests/hour
- Pro tier: 1000 requests/hour

**Rate Limit Handling:**
1. Publisher detects 429 status code
2. Extracts `Retry-After` header (seconds to wait)
3. Waits specified time before retry
4. Falls back to exponential backoff if no header

**Best Practices:**
- Use batch mode with default stagger delays (30-60s)
- Monitor rate limit headers in debug logs
- Upgrade Late.dev tier for higher limits

</details>

<details>
<summary><strong>Common Error Messages</strong></summary>

Common error scenarios and their messages:

**Authentication Failed:**
```
ERROR: Authentication failed - check your API key
```
**Solution**: Verify `LATE_API_KEY` in `.env` is correct

**Video Not Found:**
```
ERROR: Video file not found: outputs/B0ABC/video.mp4
```
**Solution**: Check file path and verify video was generated

**No Connected Accounts:**
```
ERROR: No connected accounts found
```
**Solution**: Connect platforms at https://late.dev/dashboard/accounts

**Metadata Not Found:**
```
WARNING: No metadata found for youtube, using basic content
```
**Solution**: Generate metadata with `--target-platform multi` flag

</details>

---

## 🔧 Troubleshooting

<details>
<summary><strong>Authentication Issues</strong></summary>

**Problem**: `Authentication failed - check your API key`

**Solutions:**
1. Verify API key format (should start with `sk_live_` or `sk_test_`)
   ```bash
   grep LATE_API_KEY .env
   ```

2. Test authentication directly:
   ```bash
   poetry run python -m src.publisher.late list-accounts --debug
   ```

3. Regenerate API key:
   - Go to https://late.dev/dashboard/developers
   - Revoke old key
   - Create new key
   - Update `.env`

4. Check for trailing whitespace:
   ```bash
   # Remove whitespace
   export LATE_API_KEY=$(echo $LATE_API_KEY | xargs)
   ```

</details>

<details>
<summary><strong>Rate Limit Errors</strong></summary>

**Problem**: `Rate limit exceeded (429)`

**Solutions:**
1. Check current rate limit tier:
   ```bash
   poetry run python -m src.publisher.late list-accounts --debug
   # Check response headers for X-RateLimit-Remaining
   ```

2. Increase stagger delays in batch mode:
   ```yaml
   # config/publisher.yaml
   stagger_delay_min: 60  # Increase to 60 seconds
   stagger_delay_max: 120 # Increase to 2 minutes
   ```

3. Reduce concurrent uploads:
   - Process fewer products per batch
   - Split large batches into smaller chunks

4. Upgrade Late.dev tier:
   - Standard: 100 req/hour
   - Pro: 1000 req/hour
   - Visit https://late.dev/pricing

</details>

<details>
<summary><strong>Upload Failures</strong></summary>

**Problem**: `Upload failed: Network timeout`

**Solutions:**
1. Check internet connection:
   ```bash
   ping late.dev
   ```

2. Increase timeout:
   ```bash
   export LATE_TIMEOUT=60.0  # Increase to 60 seconds
   ```

3. Verify file size and format:
   ```bash
   ffprobe outputs/B0ABC/video.mp4
   # Check codec (H.264), size (<100MB recommended)
   ```

4. For large files (>4MB), verify Vercel Blob token:
   ```bash
   grep LATE_VERCEL_TOKEN .env
   # Should be set for files >4MB (get from Vercel Dashboard → Storage → Blob)
   ```

5. Check Late.dev status:
   - Visit https://late.dev/status
   - Check for ongoing incidents

</details>

<details>
<summary><strong>Missing Metadata</strong></summary>

**Problem**: `No metadata found for <platform>`

**Solutions:**
1. Verify metadata files exist:
   ```bash
   ls -la outputs/B0ABC/
   # Should contain metadata_youtube.json, metadata_tiktok.json, etc.
   ```

2. Regenerate metadata:
   ```bash
   poetry run python -m src.video.producer \
     outputs/B0ABC/data.json \
     slideshow_images1 \
     --target-platform multi \
     --debug
   ```

3. Check UPLOAD_INSTRUCTIONS.txt fallback:
   ```bash
   cat outputs/B0ABC/UPLOAD_INSTRUCTIONS.txt
   ```

4. Verify JSON format:
   ```bash
   python -m json.tool outputs/B0ABC/metadata_youtube.json
   ```

</details>

<details>
<summary><strong>Platform Connection Issues</strong></summary>

**Problem**: `No connected account for <platform>`

**Solutions:**
1. Verify connected accounts:
   ```bash
   poetry run python -m src.publisher.late list-accounts
   ```

2. Connect platform:
   - Visit https://late.dev/dashboard/accounts
   - Click "Connect Account" for the platform
   - Complete OAuth authorization
   - Verify account appears in dashboard

3. Check account status:
   - Ensure account is not expired
   - Reauthorize if needed

4. Verify platform is supported:
   - Supported: YouTube, TikTok, Instagram, Facebook, Twitter, LinkedIn
   - Check Late.dev documentation for platform-specific requirements

</details>

<details>
<summary><strong>Debug Mode</strong></summary>

Enable verbose debug logging for troubleshooting:

```bash
poetry run python -m src.publisher.late single \
  --video outputs/B0ABC/video.mp4 \
  --platform youtube \
  --immediate \
  --debug
```

**Debug output includes:**
- API request/response details
- Retry attempts and backoff timing
- Metadata loading process
- Upload progress
- Error stack traces

**Log file location:**
```
outputs/logs/publisher.log
```

</details>

---

## 📚 API Reference

<details>
<summary><strong>Python API Usage</strong></summary>

The publisher can also be used programmatically:

```python
import asyncio
from pathlib import Path
import aiohttp
from src.publisher import create_publisher, PublisherProvider
from src.publisher.models import Platform

async def publish_video():
    """Example: Publish a video programmatically."""

    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Create publisher instance
        publisher = create_publisher(
            provider=PublisherProvider.LATE,
            api_key="sk_live_your_key",
            session=session,
            vercel_token="your_vercel_token",  # Optional
            timeout=30.0,
            max_retries=3,
        )

        # Authenticate
        is_authenticated = await publisher.authenticate()
        if not is_authenticated:
            print("Authentication failed")
            return

        # Get connected accounts
        accounts = await publisher.get_accounts()
        youtube_account = next(
            acc for acc in accounts
            if acc["platform"] == "youtube"
        )

        # Upload video
        video_path = Path("outputs/B0ABC/video.mp4")
        media_id = await publisher.upload_media(video_path)

        # Publish
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

# Run
asyncio.run(publish_video())
```

**BasePublisher Interface:**

```python
from abc import ABC, abstractmethod

class BasePublisher(ABC):
    """Base interface for all publisher implementations."""

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the publishing service."""

    @abstractmethod
    async def get_accounts(self) -> list[dict]:
        """Get list of connected social media accounts."""

    @abstractmethod
    async def upload_media(
        self,
        file_path: Path,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> str:
        """Upload media file and return media ID."""

    @abstractmethod
    async def publish(
        self,
        media_id: str,
        platforms: list[dict],
        content: str,
        scheduled_time: datetime | None = None,
    ) -> dict:
        """Publish media to platforms."""
```

</details>

---

## 🔗 External Resources

- **Late.dev Documentation**: https://docs.late.dev
- **Late.dev API Reference**: https://docs.late.dev/api
- **Late.dev Dashboard**: https://late.dev/dashboard
- **Late.dev Pricing**: https://late.dev/pricing
- **Late.dev Status**: https://late.dev/status

---

## 📝 Version History

**v0.17.0** (2025-01-15)
- Initial publisher module release
- Late.dev integration
- Multi-platform publishing support
- Batch publishing with rate limiting
- Platform-specific metadata integration
- CLI interface for all operations

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details
