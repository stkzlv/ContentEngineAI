# Publisher Module - Social Media Publishing

[![Zernio Integration](https://img.shields.io/badge/Zernio-Integrated-brightgreen)](https://zernio.com)

**Automatically publish your generated videos to social media platforms via Zernio (published via the legacy Late SDK)**

The Publisher module provides a complete solution for distributing your AI-generated product videos across YouTube, TikTok, Instagram, Facebook, Twitter, and LinkedIn. It uses [Zernio](https://zernio.com) (still integrated via the legacy `late-sdk` package and `LATE_API_KEY` env var) for multi-platform publishing with scheduling, metadata management, and batch processing. Zernio was formerly named Late; old `getlate.dev` / `late.dev` URLs redirect to `zernio.com`.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Setup](#-setup)
- [CLI Reference](#-cli-reference)
- [CLI Usage](#-cli-usage)
- [Configuration](#-configuration)
- [Platform Metadata](#-platform-metadata)
- [Batch Publishing](#-batch-publishing)
- [Retry Queue](#-retry-queue)
- [Publishing Schedule & Calendar](#-publishing-schedule--calendar)
- [Webhooks](#-webhooks)
- [Post-Publication Cleanup](#-post-publication-cleanup)
- [Link-in-Bio Integration](#-link-in-bio-integration)
- [Affiliate Disclosure](#-affiliate-disclosure)
- [First Comment](#-first-comment)
- [Blob Store Retention](#-blob-store-retention)
- [Published Products Registry](#-published-products-registry)
- [Error Handling](#-error-handling)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)
- [Common Workflows](#-common-workflows)

---

## ✨ Features

- **🌐 Multi-Platform Publishing**: YouTube, TikTok, Instagram, Facebook, Twitter, LinkedIn
- **📅 Auto-Scheduling**: Automatically finds first available unoccupied slot in recurring schedule
- **📆 Calendar Management**: View and filter all scheduled posts by platform, date, and status
- **🔄 Batch Publishing**: Upload multiple videos via `schedule --immediate` with rate limiting
- **🔁 Retry Queue**: Resume failed batch items without reprocessing successes
- **📡 Webhooks**: Real-time status updates without polling
- **🗑️ Auto-Cleanup**: Automatically remove published products from outputs directory
- **📝 Platform-Specific Metadata**: Auto-loads AI-generated titles, descriptions, hashtags
- **⚡ Smart Uploads**: Large files (>4MB) automatically routed through Vercel CDN
- **🔁 Retry Logic**: Exponential backoff for rate limits and network errors
- **✅ Progress Tracking**: Real-time upload progress with callbacks
- **🔗 Link-in-Bio**: Auto-add affiliate links to bio page after publishing (Lnk.Bio, etc.)
- **💬 First Comment**: Post affiliate links as first comment on YouTube/Instagram to avoid algorithm penalties
- **🛡️ Affiliate Disclosure**: Can render the Amazon Associates literal phrase in every post's caption body, configurable for non-Amazon programs; off by default, since the phrase asserts active program membership
- **🎯 CLI Interface**: Simple command-line interface for all operations

For the disclosure stack the publisher produces (FTC, Amazon Associates, platform policy) and the per-video manual steps creators are expected to take, see [Compliance](compliance.md).

---

## 🚀 Quick Start

```bash
# 1. Get Zernio API credentials
# Sign up at https://zernio.com and get your API key from Dashboard -> Developers

# 2. Configure credentials in .env
echo "LATE_API_KEY=sk_live_your_key_here" >> .env
echo "LATE_VERCEL_TOKEN=vercel_blob_rw_xxx" >> .env  # Optional, for large files >4MB

# 3. Connect your social media accounts
# Visit https://zernio.com/dashboard/accounts and connect platforms

# 4. Verify setup - list connected accounts
poetry run python -m src.publisher.late list-accounts --debug

# 5. Publish a video (auto-discovers video and next available slot)
poetry run python -m src.publisher.late single B0BTYCRJSS --debug

# Or publish immediately to specific platforms
poetry run python -m src.publisher.late single B0BTYCRJSS \
  --platform youtube --platform tiktok --immediate --debug

# 6. Batch publish all videos in outputs directory
poetry run python -m src.publisher.late schedule --immediate \
  --platform youtube --platform tiktok --platform instagram \
  --debug

# 7. Verify first comments landed on recent published posts
poetry run python -m src.publisher.late verify-comments --limit 25 --debug
```

---

## 🔧 Setup

### 1. Zernio Account Setup

1. **Create Account**: Sign up at [https://zernio.com](https://zernio.com)
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
# Required: Zernio API Key
LATE_API_KEY=sk_live_your_api_key_here

# Optional: Vercel Blob Token for large file uploads (>4MB)
# Required if publishing videos larger than 4MB
# Get from: Vercel Dashboard → Storage → Create Blob → Settings → Token
LATE_VERCEL_TOKEN=vercel_blob_rw_your_token_here

# Optional: Override default settings
LATE_TIMEOUT=120.0
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

**Multi-Account Configuration** (optional):

```yaml
# Define multiple Zernio accounts
accounts:
  production:
    api_key: sk_live_prod_key_12345
    vercel_token: vercel_prod_token
    description: Production account
  staging:
    api_key: sk_live_staging_key_123
    description: Staging/test account
    default_platforms: [youtube]
default_account: production  # Account to use by default
```

Use `--account NAME` CLI flag to switch accounts at runtime:
```bash
poetry run python -m src.publisher.late single B0ABC123 --account staging
```

> **Note:** The `backoff_multiplier` key is deprecated and silently ignored. Use `retry_delay` and `retry_max_attempts` instead.

See [Configuration](#-configuration) for full options.

### 4. Verify Setup

Test your configuration by listing connected accounts (see [CLI Usage](#-cli-usage) for details).

---

## 📖 CLI Reference

Quick reference for all publisher commands and options.

### Commands Overview

| Command | Description | Example |
|---------|-------------|---------|
| `list-accounts` | List connected social accounts | `python -m src.publisher.late list-accounts` |
| `single` | Publish single video by product ID | `python -m src.publisher.late single B0ABC123 --immediate` |
| `schedule` | Auto-schedule or batch publish videos | `python -m src.publisher.late schedule --debug` |
| `calendar` | View scheduled posts | `python -m src.publisher.late calendar list` |
| `cleanup` | Remove published products | `python -m src.publisher.late cleanup --all --confirm` |
| `delete` | Delete a post from Zernio | `python -m src.publisher.late delete POST_ID` |
| `verify-comments` | Check first comments landed on recent posts | `python -m src.publisher.late verify-comments --limit 25` |
| `verify-delivery` | Sweep recent posts for silently-failed platform legs | `python -m src.publisher.late verify-delivery --limit 25` |
| `analytics` | Capture day-N views and rank posts by durability | `python -m src.publisher.late analytics` |

### Global Options

| Option | Description |
|--------|-------------|
| `--account NAME` | Use specific account (multi-account mode) |
| `--debug` | Enable verbose debug logging |

### Command: `single`

```
python -m src.publisher.late single <product_id> [options]
```

| Option | Required | Description |
|--------|----------|-------------|
| `product_id` | Yes | Product ID (e.g., B00TF9E6XE) - video auto-discovered from outputs/ |
| `--platform PLATFORM` | No | Target platform (youtube, tiktok, instagram, facebook, twitter, linkedin). Defaults to all 3 if not specified |
| `--immediate` | No | Publish immediately |
| `--schedule DATETIME` | No | Schedule for later (format: `2025-01-20 14:00:00`) |
| `--force` / `--no-force` | No | Republish even if already published. Default off: the publisher skips a product already published to the target platform (logs "already published"). Pass `--force` to bypass that guard and create a new post anyway. |
| `--no-cleanup` | No | Disable automatic cleanup after success |
| `--platform-specific` | No | Create separate posts per platform with optimized metadata |
| `--link-in-bio` | No | Enable link-in-bio update (overrides config) |
| `--no-link-in-bio` | No | Disable link-in-bio update (overrides config) |

*If neither `--immediate` nor `--schedule` is provided, auto-discovers next available slot from recurring schedule.

**`--force`**: by default the publisher refuses to re-post a product already published to a platform (the duplicate guard), so reruns are safe. Use `--force` to deliberately republish, e.g. after re-rendering a video you already posted:

```bash
# Re-schedule an already-published product (creates a new post)
poetry run python -m src.publisher.late single B0ABC123 --force --debug

# Schedule the whole backlog including products already published
poetry run python -m src.publisher.late schedule --force --debug
```

### Command: `batch` (Removed)

The `batch` command has been consolidated into `schedule`. Use `schedule --immediate` instead:

```
# Old: python -m src.publisher.late batch --platform youtube --immediate
# New: python -m src.publisher.late schedule --immediate --platform youtube
```

All batch options (`--fail-fast`, `--outputs-dir`, `--no-cleanup`, `--retry-failed`) work with `schedule --immediate`.

### Command: `schedule`

```
python -m src.publisher.late schedule [auto] [options]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--platform PLATFORM` | No | Target platform, can repeat (defaults to youtube, tiktok, instagram) |
| `--immediate` | No | Publish immediately (replaces old batch command) |
| `--force` | No | Bypass already-published checks |
| `--fail-fast` | No | Stop on first failure (immediate mode) |
| `--outputs-dir PATH` | No | Directory to scan (default: `outputs`) |
| `--dry-run` | No | Preview without making changes |
| `--auto-resolve` | No | Auto-resolve conflicts with first alternative |
| `--no-cleanup` | No | Disable cleanup after scheduling |

### Command: `calendar`

```
python -m src.publisher.late calendar list [options]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--platform PLATFORM` | No | Filter by platform |
| `--status STATUS` | No | Filter by status (pending, scheduled, published, failed, partial) |
| `--date-from DATE` | No | Start date filter (YYYY-MM-DD) |
| `--date-to DATE` | No | End date filter (YYYY-MM-DD) |

### Command: `cleanup`

```
python -m src.publisher.late cleanup [options]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--product-id ID` | Yes* | Clean specific product |
| `--all` | Yes* | Clean all published products |
| `--platform PLATFORM` | No | Filter by platform |
| `--outputs-dir PATH` | No | Directory to scan (default: `outputs`) |
| `--dry-run` | No | Preview without deleting |
| `--confirm` | No | Required for `--all` mode |

*One of `--product-id` or `--all` is required.

### Command: `delete`

```
python -m src.publisher.late delete <post_id>
```

| Option | Required | Description |
|--------|----------|-------------|
| `post_id` | Yes | Zernio post ID to delete |

### Command: `analytics`


```bash
# Measure recent published posts and store the figures
python -m src.publisher.late analytics

# Re-rank what is already stored, without touching the network
python -m src.publisher.late analytics --rank-only
```

**Run this on a schedule, not once at the end of a comparison.** The provider's
timeline has a retention horizon of roughly five weeks: past it a post's rows
start at a recent date rather than at its publication, so day-2 and day-7 are
no longer reachable and the durability ratio cannot be computed. Nothing can be
passed to widen the window — `from_date` makes no difference at any post age.

Figures already captured are safe. Each post's row is merged field by field, so
a later, shorter reading never replaces a measured value with an absent one;
what it cannot do is recover a figure that was never taken in time. The one
exception is a day-N figure a later sweep finds was measured before every
platform had started reporting — see below, where the point is that such a
figure was never true rather than merely stale.

**A day-N figure counts every platform or none.** Platforms start reporting on
their own lag — one commonly takes days — and a leg's first row carries its
whole lifetime total to that date rather than that day's increment. A figure
taken before a leg started would therefore count only part of the post, while
`views_total` counts all of it, so the two would describe different things.

A format comparison that ranks arms on median day-7 views would rank a post
understated that way below an identical one, for a reason that is reporting lag
rather than reach. So `views_day_2`, `views_day_7` and `durability_ratio` are reported
as unknown when a platform that appears later in the series had not reported by
the cutoff — the same rule already applied to a window the timeline has not
reached: unknown, not a small number.

The cost is coverage, and it is not small. Measured against the live API on
2026-08-25 over the 60 most recent published posts, 57 of them multi-platform:

| | Counting a lagging leg's absence as zero | Reporting it unknown |
|---|---|---|
| day-2 available | 56 | 37 |
| day-7 available | 51 | 33 |

So 18 of 51 day-7 figures — a third — were understated by a leg that had not
started reporting. That is the size of the bias the old rule carried into a
comparison, and roughly a third of posts is the coverage the new rule gives up
to remove it.

The trade is deliberate. A missing figure is visible and can be excluded; a
quietly understated one is neither. It does mean a comparison needs more posts
than its target sample to end up with that many usable ones.

Three causes produce a blank, and `timeline_end` alone does not separate them:
a post that aged past the retention horizon before its first sweep looks the
same as one with a silent leg. Read `lagged_cutoff_days` in
`outputs/post_metrics.json` first — it names the cutoffs a leg was silent for.
Failing that, `timeline_end` earlier than the cutoff means the window had not
closed when the sweep ran, and at or past it with no marker means the retained
rows begin after the cutoff.

A sweep still withholds its own figure wherever the retained window covers the
cutoff and the legs disagree — it just does not mark the post, because marking
withdraws figures other sweeps took. A figure is only withdrawn on the evidence
of a sweep whose record still reaches back to publication. Past the retention horizon every leg's rows begin
at the window edge, so a leg absent from that first date looks identical to one
that started late — and a ratio measured while the record was whole must not be
discarded on that reading, because no later sweep can recompute it. For the same
reason the merge keeps a durability ratio taken from a full record over one
computed from a truncated window, which divides by a partial figure and reads
higher.

**The sweep that stores a figure is usually not the one that can tell it was
biased.** A daily run reaches a young post while the slow platform has no rows
at all, so one leg looks like the whole post. The marker is what a later sweep
uses to withdraw a number already stored, and it is never unset: a sweep whose
rows all begin past the lag sees no disagreement between legs, which says
nothing about whether the figure was biased when it was taken. Figures captured
before this rule existed are corrected the same way, on the next sweep that
observes the lag.

**Running it on a schedule.**

`outputs/` is local and gitignored, and the API key comes from `.env`, so the
capture belongs on the machine that owns the data rather than in CI.

```bash
make install-analytics-timer
```

That renders a systemd user timer, installs it, enables it, runs one sweep, and
checks that `outputs/post_metrics.json` actually changed. It needs no root, and
with lingering enabled it runs whether or not you are logged in. Check on it
later with `make analytics-timer-status`, and remove it with
`make uninstall-analytics-timer`.

**Daily is the right default, and weekly is not.** Most of a short-form post's
views arrive in the first day or two, and one platform's analytics rows take
48-72 hours to finalise, so a same-day reading of a fresh post is still
settling; the merge corrects it on the next run. The durability ratio is the
binding constraint: it needs a post past day 30 while its rows still reach back
to publication, and retention is about five weeks, so the window is roughly five
days wide. A weekly sweep can step over a post's only window and never produce a
ratio for it at all.

Repeat runs are safe. Readings merge per field, and a later, better figure
replaces an earlier partial one. They are not free, though: a sweep costs one
timeline call per measured post plus the paging to list them, so the shipped
size is roughly 53 requests. Daily sits comfortably inside the documented hourly
cap; several times an hour does not.

**What to configure, and where.** Two files, split by what reads them:

| Setting | Lives in | Why there |
|---|---|---|
| How many posts a sweep measures | `config/publisher.yaml::analytics.limit` | Behaviour, read by both the manual and the scheduled run |
| Schedule, timeouts, failure reporting, paths | `deploy/schedule.env` | Shapes the unit files, which systemd reads before any of this project's code runs |

Copy `deploy/schedule.env.example` to `deploy/schedule.env` and edit what you
need; the copy is gitignored and every key is optional, so the installer works
before you write it at all.

Re-run the installer after editing it, because systemd does not pick up a
changed `OnCalendar` on its own. Use `./deploy/install-timer.sh --no-run` when
you changed only the schedule: it re-renders and re-arms the timer without
spending a sweep's worth of API calls to prove something you already proved.

The unit deliberately passes no `--limit`. The size lives in
`config/publisher.yaml` and nowhere else, so editing the YAML takes effect on
the next run with no reinstall, and the scheduled sweep cannot drift from
`make analytics`.

**Name the interpreter; do not go through `poetry run` or `make`.** A user
service does not inherit your login shell's environment, so its `PATH` has no
pyenv shims -- and because `poetry.toml` sets `virtualenvs.create = false`,
`poetry run python` then resolves the *base* interpreter rather than the project
environment. The service would fail at the first import, daily, while the
figures it was meant to capture age out. This is the same trap the `*-lowpri`
targets work around for `systemd-run`. The installer resolves the interpreter
for you and refuses to install a unit whose interpreter cannot import the
project, because a file that merely exists is not evidence of anything.

Two more properties of the generated unit are worth knowing, because both
failures are silent:

- `Persistent=true` runs a window the machine slept through on the next boot
  rather than skipping it. Retention is finite, so a missed sweep is a permanent
  hole in the record, not a late reading. Cron simply misses a machine that was
  asleep, which is why the timer is the better of the two.
- `TimeoutStartSec=` is set explicitly. systemd disables the start timeout by
  default for `Type=oneshot` units, and left disabled a hung request would leave
  the unit activating forever; systemd then refuses to start a second instance,
  so every later firing is dropped and the unit never reaches `failed` -- so the
  failure handler never runs either. A stuck sweep would look exactly like a
  working one.

A sweep that measured posts and captured none of them exits non-zero, so the
failure channels below actually fire. Every timeline call failing is a broken
sweep, not a quiet one, and exiting 0 there would keep the timer green while
the figures expire. A single post failing stays a warning, because a partial
reading is still worth storing, and an account with no published posts is not
an error at all.

A sweep whose timelines all come back *empty* is a different case and is not
treated as a failure: it is indistinguishable, in that sweep, from an account
whose posts are too young to have rows, and failing there would fail a new
account daily. What is reported instead is every post with a stored view count
returning none at once. Posts age out of the provider's timeline one at a time,
so all of them going quiet in the same sweep is the reader, not age.

It is recorded both as a warning and in `outputs/logs/analytics-failures.log`,
so `make analytics-timer-status` shows it. A warning alone would sit behind one
line per measured post and never reach the last few journal lines that command
prints. Stored figures are untouched.

When a sweep fails it is recorded three ways: in the journal, appended to
`outputs/logs/analytics-failures.log`, and as a desktop notification if a
session is there to receive one. The log file is the durable one, and
`make analytics-timer-status` surfaces it. Set `NOTIFY_ON_FAILURE=0` to install
no handler at all.

Cron works too, but has no equivalent of `Persistent=true`:

```cron
@daily cd /path/to/ContentEngineAI && ~/.pyenv/versions/ContentEngineAI/bin/python -m src.publisher.late analytics
```

| Option | Required | Description |
|---|---|---|
| `--limit N` | No | How many recent published posts to measure. Defaults to `analytics.limit` in `config/publisher.yaml`, shipped as 50. The other subcommands' `--limit` flags are not config-backed |
| `--rank-only` | No | Rank stored metrics without fetching. Makes no network call, but publisher config still loads first, so an API key must be configured |
| `--outputs-dir PATH` | No | Where `post_metrics.json` lives (default: `outputs`) |
| `--debug` | No | Enable debug logging |

Ranking by durability answers a different question from ranking by total views.
A post that spiked and stopped can outrank one still earning months later on
totals, and at day 7 the two are indistinguishable.

---

### Command: `registry`

```
python -m src.publisher.late registry --rebuild --outputs-dir outputs
```

| Option | Required | Description |
|--------|----------|-------------|
| `--rebuild` | Yes* | Rebuild registry from all `data.json` files |
| `--summary` | Yes* | Count published products per content-format arm |
| `--outputs-dir` | No | Directory to save registry files (default: `outputs`) |
| `--scan-dir` | No | Directory to scan for product data (default: same as `--outputs-dir`) |

*One of `--rebuild` or `--summary` is required. `--rebuild` wins if both are given.

---

## 💻 CLI Usage

The publisher provides three main commands: `list-accounts`, `single`, and `schedule`.

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

Publish a single video to one or more platforms. Video is auto-discovered from the product directory.

```bash
poetry run python -m src.publisher.late single <product_id> \
  [--platform <platform> ...] \
  [--immediate | --schedule <datetime>] \
  [--force] [--no-cleanup] [--debug]
```

**Required Arguments:**
- `<product_id>`: Product ID (e.g., B0BTYCRJSS) - video is auto-discovered from `outputs/<product_id>/`

**Optional Arguments:**
- `--platform <name>`: Target platform (can be specified multiple times)
  - Valid platforms: `youtube`, `tiktok`, `instagram`, `facebook`, `twitter`, `linkedin`
  - Defaults to all 3 (youtube, tiktok, instagram) if not specified

**Publishing Options:**
- `--immediate`: Publish immediately
- `--schedule <datetime>`: Schedule for future publishing
  - Format: `YYYY-MM-DD HH:MM:SS` or `YYYY-MM-DDTHH:MM:SS`
  - Example: `2025-01-20 14:00:00` or `2025-01-20T14:00:00`
- If neither is specified, auto-discovers next available slot from recurring schedule

**Other Options:**
- `--force`: Force republish even if already published to platform
- `--no-cleanup`: Disable automatic cleanup after successful publish
- `--debug`: Enable verbose debug logging

**Examples:**

```bash
# Auto-schedule to next available slot (recommended)
poetry run python -m src.publisher.late single B0BTYCRJSS --debug

# Publish immediately to YouTube
poetry run python -m src.publisher.late single B0BTYCRJSS \
  --platform youtube --immediate

# Publish to multiple platforms immediately
poetry run python -m src.publisher.late single B0BTYCRJSS \
  --platform youtube --platform tiktok --platform instagram \
  --immediate --debug

# Schedule for specific time
poetry run python -m src.publisher.late single B0BTYCRJSS \
  --platform youtube \
  --schedule "2025-01-20 14:00:00" \
  --debug

# Force republish (even if already published)
poetry run python -m src.publisher.late single B0BTYCRJSS \
  --platform youtube --immediate --force
```

**Metadata Loading:**

The `single` command automatically loads platform-specific metadata from:
1. `outputs/<PRODUCT_ID>/metadata_<platform>.json` (preferred)
2. `outputs/<PRODUCT_ID>/UPLOAD_INSTRUCTIONS.txt` (fallback)

If no metadata is found, it uses a basic content template.

---

### Command: `schedule` (Batch Mode)

Publish all videos immediately using `schedule --immediate`:

```bash
poetry run python -m src.publisher.late schedule --immediate \
  [--platform <platform> ...] \
  [--outputs-dir <path>] \
  [--fail-fast] \
  [--debug]
```

**Examples:**

```bash
# Batch publish all videos to all platforms
poetry run python -m src.publisher.late schedule --immediate --debug

# Publish to specific platforms with fail-fast
poetry run python -m src.publisher.late schedule --immediate \
  --platform youtube --platform tiktok --fail-fast --debug

# Force re-publish already-published products
poetry run python -m src.publisher.late schedule --immediate --force --debug
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
poetry run python -m src.publisher.late single B0ABC123 \
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

# === Batch Settings ===
stagger_delay_min: 30              # Min delay between batch uploads (seconds)
stagger_delay_max: 60              # Max delay between batch uploads (seconds)

# === Privacy Settings ===
privacy_settings:
  youtube: public                  # public, unlisted, private
  tiktok: public                   # public, friends, private
  instagram: everyone              # everyone, followers, close_friends
  facebook: public                 # public, friends
  twitter: public                  # public
  linkedin: public                 # public, connections

# === Analytics Capture ===
analytics:
  limit: 50                        # Posts measured per sweep; must exceed the
                                   # number published inside the ~5-week
                                   # retention horizon or the oldest expire

# === Affiliate Disclosure ===
affiliate_disclosure:
  enabled: false                                      # Opt-in; off by default
  phrase: "As an Amazon Associate I earn from qualifying purchases"
  program: "amazon"                                   # Override for non-Amazon programs
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
- Caption hard cap: 2200 characters (includes hashtags). 150 is only a soft prompt target for punchier captions, not the platform limit.
- **Content Disclosure** (required for commercial accounts):
  - `commercial_content_type`: `"brand_organic"` (Your Brand), `"brand_content"` (Branded Content -- note the spelling, not `branded_content`), or `"none"` (not commercial content)
  - `is_brand_organic_post`: `true` for Your Brand posts
  - Both are per render, not per config: `TikTokContentSettings.for_render()` returns `"none"` / `false` for a render with no material connection, such as a topic video with no affiliate link
  - `content_preview_confirmed`: `true` (user confirmed preview)
  - `express_consent_given`: `true` (user gave consent)
  - These are configured in `TikTokContentSettings` dataclass (`src/publisher/models.py`)

**Instagram:**
- Privacy: `everyone`, `followers`, `close_friends`
- Caption length: 2200 characters
- Supports scheduled publishing (Reels)

Title and description are trimmed on a word boundary with an ellipsis before reaching the publisher when either exceeds the per-platform limit. Hashtag-count violations are logged as warnings; auto-fixing those would invent or drop tags.

</details>

---

## 📝 Platform Metadata

### Overview

The publisher automatically loads platform-specific metadata generated by the video producer. This metadata includes AI-optimized titles, descriptions, hashtags, and formatting for each platform.

### Multi-Platform Publishing

The publisher supports two publishing modes controlled by `use_platform_specific_content` in `publisher.yaml` or the `--platform-specific` CLI flag:

**Unified mode** (default, `use_platform_specific_content: false`): Creates **one post per product** published to all platforms simultaneously. Uses a single API call with unified metadata. Even though the post is shared, each platform still gets its own `platformSpecificData` block (YouTube title, TikTok disclosure settings, first comment text, etc.), so per-platform behavior works in both modes.

```bash
# 1 post published to all 3 platforms (same content)
poetry run python -m src.publisher.late single B0ABC \
  --platform youtube --platform tiktok --platform instagram \
  --immediate
```

**Platform-specific mode** (`use_platform_specific_content: true` or `--platform-specific`): Creates **separate posts per platform** with per-platform optimized metadata from `metadata_<platform>.json`. Better for engagement and discoverability.

```bash
# 3 separate posts, each with platform-optimized metadata
poetry run python -m src.publisher.late single B0ABC \
  --platform youtube --platform tiktok --platform instagram \
  --platform-specific --immediate

# Or via global batch pipeline
poetry run python -m src.pipeline.global_batch --keywords "earbuds" \
  --max-products 1 --random-profile --platform-specific --debug
```

**Scheduling**: In unified mode, all platforms share the same post ID. In platform-specific mode, each platform gets its own post ID.

### Per-Platform Profile Routing

Maps each platform to a video profile from `config/video_production.yaml::video_profiles`. The publisher prefers the matching `video_<asin>_<profile>.mp4` render when present and falls back to the first `video_<asin>_*.mp4` found otherwise. Leave the block commented to keep the first-match behaviour.

```yaml
# config/publisher.yaml
profiles:
  youtube: slideshow_short_20s     # Use the 15-30s cut for Shorts
  tiktok: slideshow_images1        # Keep the 30-45s cut for TikTok
  instagram: slideshow_images1     # Same for Reels
```

The producer must render the matching profile per ASIN before the publisher runs. With unified upload (the default mode), the publisher uploads one file shared across platforms — when `profiles` is set, the file picked is the render for the first platform in the post's target list. True per-platform uploads (different files to different platforms) are a follow-up.

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
2. **Video Detection**: Finds one render per product (`video_<asin>_*.mp4`, honouring `profiles` when configured), so a product rendered under two profiles is published once rather than twice
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
poetry run python -m src.publisher.late schedule --immediate \
  --platform youtube \
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

## 🔁 Retry Queue

Failed batch items are automatically added to a retry queue, allowing you to resume publishing without reprocessing successful items.

<details>
<summary><strong>How It Works</strong></summary>

When a batch publish fails for some products:
1. Failed product IDs are stored in `outputs/publish_history.json`
2. Original scheduled times are preserved
3. Retry count is tracked per product
4. Successful items are removed from the queue

**Retry Queue Entry:**
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

</details>

<details>
<summary><strong>CLI Usage</strong></summary>

```bash
# Normal batch publish (failures automatically queued)
poetry run python -m src.publisher.late schedule --immediate \
  --platform youtube --platform tiktok --debug

# Retry only failed items
poetry run python -m src.publisher.late schedule --immediate \
  --platform youtube --platform tiktok \
  --retry-failed --debug
```

**Retry Mode Behavior:**
- Only processes items in the retry queue
- Preserves original scheduled times
- Removes items on success (idempotent)
- Increments retry count on repeated failures
- Reports "Retry queue is empty" if nothing to retry

</details>

<details>
<summary><strong>Python API</strong></summary>

```python
from src.publisher.tracking import (
    get_retry_queue,
    get_retry_queue_count,
    clear_retry_queue,
)

# Check retry queue
items = get_retry_queue(outputs_dir)
print(f"Failed items: {len(items)}")

# Clear retry queue
cleared = clear_retry_queue(outputs_dir)
print(f"Cleared {cleared} items")
```

</details>

---

## 📡 Webhooks

Receive real-time status updates from Zernio without polling.

<details>
<summary><strong>Webhook Events</strong></summary>

Zernio sends webhooks for these events:

| Event | Description |
|-------|-------------|
| `post.scheduled` | Post successfully scheduled |
| `post.published` | Post successfully published |
| `post.failed` | Post failed on all platforms |
| `post.partial` | Post succeeded on some platforms |
| `account.disconnected` | Social account token expired |

</details>

<details>
<summary><strong>Setting Up Webhooks</strong></summary>

1. **Create webhook endpoint** in your application
2. **Configure webhook** in Zernio dashboard:
   - URL: `https://your-app.com/webhooks/late`
   - Secret: Generate a secure random string
   - Events: Select events to receive

3. **Set up handler:**

```python
from src.publisher import WebhookHandler

handler = WebhookHandler(
    secret="your-webhook-secret",
    outputs_dir=Path("outputs")
)
```

</details>

<details>
<summary><strong>Flask Example</strong></summary>

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

</details>

<details>
<summary><strong>FastAPI Example</strong></summary>

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

</details>

<details>
<summary><strong>Security</strong></summary>

**Signature Verification:**

Webhooks are signed with HMAC-SHA256. The signature is sent in the `X-Late-Signature` header.

```python
# Signature is computed as:
import hmac, hashlib
signature = hmac.new(
    key=secret.encode("utf-8"),
    msg=payload_bytes,
    digestmod=hashlib.sha256
).hexdigest()
```

**Idempotency:**

The handler automatically tracks processed events to prevent duplicate processing:
- Events are tracked by `event_id`
- Duplicate events are skipped
- History is pruned to last 1000 events

**Best Practices:**
- Always verify signatures in production
- Return 200 quickly, process asynchronously if needed
- Handle duplicate events gracefully
- Log webhook errors for debugging

</details>

<details>
<summary><strong>Querying Webhook Status</strong></summary>

```python
from src.publisher.webhooks import (
    get_post_status,
    get_disconnected_accounts,
)

# Get post status from webhook updates
status = get_post_status("post_123", outputs_dir)
if status:
    print(f"Status: {status['status']}")
    print(f"URLs: {status['published_urls']}")

# Check for disconnected accounts
disconnected = get_disconnected_accounts(outputs_dir)
for acc in disconnected:
    print(f"Account {acc['account_id']} disconnected")
```

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
      Product: B0EXAMPLE1
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
  --platform youtube --platform tiktok --platform instagram \
  --debug

# Preview schedule without publishing
poetry run python -m src.publisher.late schedule auto \
  --platform youtube --platform tiktok --platform instagram \
  --dry-run \
  --debug

# Auto-resolve conflicts by using first available alternative
poetry run python -m src.publisher.late schedule auto \
  --platform youtube \
  --auto-resolve \
  --debug
```

**Auto-Scheduling Behavior:**
1. Loads recurring schedule from configuration
2. Scans outputs directory for unpublished videos
3. **Queries the Zernio API** to find occupied slots (8-week lookahead)
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

<details>
<summary><strong>Conflict Resolution</strong></summary>

When scheduling conflicts occur, the publisher suggests alternative time slots:

**Automatic Conflict Resolution:**

```bash
# Auto-resolve conflicts by using first available alternative
poetry run python -m src.publisher.late.cli schedule auto \
  --platform youtube --auto-resolve

# Without --auto-resolve, alternatives are suggested in logs
poetry run python -m src.publisher.late.cli schedule auto --platform youtube
# Output: "Suggested alternatives: 2026-01-20T14:00:00, 2026-01-22T10:00:00..."
```

**How It Works:**
1. When a slot is occupied or validation fails, `find_alternatives()` is called
2. Searches for next N available slots starting from preferred time
3. Alternatives are sorted by proximity to user's preferred time
4. With `--auto-resolve`, automatically uses first available alternative
5. Resolution decisions are logged for traceability

**Configuration** (`config/publisher.yaml`):

```yaml
# === Conflict Resolution ===
recurring_schedule:
  conflict_alternatives_count: 5  # Number of alternatives to suggest
```

**ConflictResolution Response:**
- `original_time`: User's originally requested time
- `conflict_reason`: Why the original time failed
- `alternatives`: List of available slots sorted by proximity
- `auto_resolved`: Whether conflict was auto-resolved
- `resolved_time`: The time actually used (if auto-resolved)

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
poetry run python -m src.publisher.late single B0ABC \
  --platform youtube \
  --immediate \
  --no-cleanup

# Disable cleanup for batch publish
poetry run python -m src.publisher.late schedule --immediate \
  --platform youtube --platform tiktok \
  --no-cleanup --debug
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
   - B0EXAMPLE1 (youtube, tiktok, instagram)
   - B0EXAMPLE2 (youtube, instagram)
   - B0EXAMPLE3 (youtube, tiktok)
   ...
⏭️  Skipped: 3 products
   - B0ABC123 (not published to all platforms)
   - B0DEF456 (publication failed)
   - B0GHI789 (cleanup disabled for platform)
💾 Disk space freed: 1.2 GB
================================================================================
```

</details>

<details>
<summary><strong>Safety Guidelines</strong></summary>

Follow these best practices to prevent accidental data loss:

**Before Cleanup:**

| Step | Command | Purpose |
|------|---------|---------|
| 1. Preview | `cleanup --all --dry-run` | See what will be deleted |
| 2. Verify | `calendar list --status published` | Confirm posts are live |
| 3. Check logs | View `outputs/logs/publisher.log` | Review audit trail |
| 4. Execute | `cleanup --all --confirm` | Run with confirmation |

**Recommended Configuration:**

```yaml
cleanup:
  enabled: true
  verify_before_delete: true      # ALWAYS keep enabled
  require_all_platforms: true     # Only cleanup if ALL platforms succeeded
  archive_before_delete: true     # Backup before deletion
  archive_dir: "outputs/archives"
  keep_published_days: 7          # Keep 7 days before cleanup
```

**Safety Checklist:**

- [ ] Always use `--dry-run` first on production data
- [ ] Verify `require_all_platforms: true` is set if publishing to multiple platforms
- [ ] Enable `archive_before_delete` for valuable content
- [ ] Check Zernio dashboard to confirm posts are live before cleanup
- [ ] Review `outputs/logs/cleanup_audit.log` after cleanup
- [ ] Keep backups for at least 7 days (`keep_published_days: 7`)

**Recovery Options:**

If cleanup runs accidentally:
1. Check archive directory: `outputs/archives/`
2. Review audit log for deleted products: `outputs/logs/cleanup_audit.log`
3. Re-scrape and reproduce videos if no archive exists

**Never Do:**

- ❌ Run `cleanup --all --confirm` without `--dry-run` preview first
- ❌ Disable `verify_before_delete` in production
- ❌ Set `keep_published_days: 0` without archiving enabled
- ❌ Clean up before verifying posts are actually published (not just scheduled)

</details>

---

## 🔗 Link-in-Bio Integration

After publishing a video, the publisher can automatically add the product's Amazon affiliate link to a link-in-bio page (e.g., Lnk.Bio, Linktree).

On the `single` path this also runs when nothing is published: if every requested platform is already done, the command refreshes the bio link and exits without uploading. That is deliberate (a fully-published product still wants a current bio link), so a `single` rerun on an already-published product is not a no-op — it touches the bio. Pass `--no-link-in-bio` if you want a genuinely inert rerun.

<details>
<summary><strong>Configuration</strong></summary>

**YAML** (`config/publisher.yaml`):

```yaml
link_in_bio:
  enabled: true           # Toggle on/off
  provider: lnkbio        # Provider name (lnkbio supported)
  max_links: 0            # Max links on bio page (0 = unlimited, >0 = oldest rotated out)
  max_title_length: 80    # Truncate link titles beyond this length
```

**CLI Overrides**:

```bash
# Enable link-in-bio for this publish (overrides config)
poetry run python -m src.publisher.late single B0ABC --link-in-bio

# Disable link-in-bio for this publish (overrides config)
poetry run python -m src.publisher.late single B0ABC --no-link-in-bio
```

**Environment Variables** (required when enabled):

```bash
export LNKBIO_CLIENT_ID=your_client_id
export LNKBIO_CLIENT_SECRET=your_client_secret
```

</details>

<details>
<summary><strong>How It Works</strong></summary>

1. After a successful `single` publish, the manager reads product info from `outputs/<product_id>/data.json`
2. Adds a link to the bio page with the product title, affiliate URL, and thumbnail
3. If `max_links` is exceeded, the oldest link is automatically removed
4. Failures are logged as warnings and never block publishing

**Data Source** (`outputs/<product_id>/data.json`):
- `title` → link title (truncated to `max_title_length`)
- `affiliate_link` → destination URL (falls back to `url` if unavailable)
- `images[0]` → thumbnail URL (falls back to `downloaded_images[0]` local file)

Protocol-level notes (auth, Cloudflare gate, the 50-link list ceiling, the undocumented `/lnk/edit` endpoint, dashboard escape hatch): [lnkbio-api.md](lnkbio-api.md).

</details>

<details>
<summary><strong>Adding New Providers</strong></summary>

Implement `BaseLinkInBioProvider` from `src/publisher/link_in_bio/base.py`:

```python
class BaseLinkInBioProvider(ABC):
    async def authenticate(self) -> bool: ...
    async def add_link(self, title: str, url: str, image: str | None = None) -> dict: ...
    async def list_links(self) -> list[dict]: ...
    async def delete_link(self, link_id: str) -> bool: ...
```

Register the new provider in `link_in_bio/manager.py:create_link_in_bio_manager()`.

</details>

---

## 🛡️ Affiliate Disclosure

Can render the affiliate program's required literal identification phrase in every post's caption body, between the `#ad` disclosure line and the description. Enabled, this satisfies the Amazon Associates Operating Agreement; it is off by default, so a stock install renders no phrase, and non-Amazon programs can override the phrase and program name.

<details>
<summary><strong>Configuration</strong></summary>

**YAML** (`config/publisher.yaml`):

```yaml
affiliate_disclosure:
  enabled: false                                      # Opt-in; off by default
  phrase: "As an Amazon Associate I earn from qualifying purchases"
  program: "amazon"                                   # Override for non-Amazon programs
```

**Behavior**:
- **Off by default, and opt-in on purpose.** The phrase asserts membership of the named program, so it must only be emitted while that membership is active. Enable it when you join, and disable it again if the account closes or you leave, because claiming it otherwise misstates a material connection. The loader also falls back to the disabled default when the section is missing or empty, so an unconfigured install publishes no claim.
- Works in both unified and platform-specific publishing modes.
- When `enabled: true`, the phrase is injected after `#ad` and before the description.
- When `enabled: false`, captions carry only `#ad` and the description.
- `phrase` and `program` default to the Amazon Associates values, so joining that program needs only `enabled: true`.
- The `program` key is metadata only; it does not change how the phrase is rendered.
- This is separate from the `#ad` disclosure, which leads the caption on any render carrying a material connection and is required for any affiliate relationship regardless of program. Both are gated on the same recorded decision, so a caption and a frame cannot disagree about whether a render is promotional.

</details>

<details>
<summary><strong>How It Works</strong></summary>

1. `load_publisher_config()` parses `affiliate_disclosure` into `AffiliateDisclosureConfig`.
2. `cmd_single()` and the global batch pass `disclosure_phrase` to `publish_product()` only when enabled.
3. `publish_product()` sets `PublishMetadata.affiliate_disclosure`, and `format_content()` places it between the disclosure line and the description.
4. The phrase is included in `PublishMetadata.to_dict()` and `PublisherConfig.to_dict()` for serialization.

</details>

---

## 💬 First Comment

Post affiliate links as the first comment instead of embedding them in captions. Meta's algorithm deprioritizes posts with outbound links in descriptions, so moving them to a comment keeps captions clean and avoids the penalty.

Supported on YouTube and Instagram. TikTok is skipped (the Zernio API doesn't support `firstComment`).

<details>
<summary><strong>Configuration</strong></summary>

**YAML** (`config/publisher.yaml`):

```yaml
first_comment:
  enabled: true
  move_hashtags_to_comment: false  # Move Instagram hashtags to comment too
  platforms:
    youtube: "Get it here: {affiliate_link}\n\nLike & subscribe for more deals!"
    instagram: "{product_title}\n🔗 Link in bio!"
```

**Template placeholders:**

| Placeholder | Source |
|-------------|--------|
| `{affiliate_link}` | `shortened_affiliate_link` or `affiliate_link` from data.json |
| `{product_title}` | `title` from data.json |
| `{hashtags}` | Hashtags from metadata (only when `move_hashtags_to_comment: true`, Instagram only) |

</details>

<details>
<summary><strong>How It Works</strong></summary>

1. After metadata is loaded, `build_first_comment()` renders the platform template with product data from `outputs/<product_id>/data.json`
2. Each platform gets its own rendered comment based on its template in `first_comment.platforms`
3. The rendered comments are passed via `platform_contents` to `publisher.publish()`, alongside that platform's caption and title. The dict is the authoritative per-platform payload, not a comment-only side channel: the client reads `content` and `title` from the same entry, so an entry carrying only a comment blanks the caption and sends no title.
4. The Zernio API receives each as `firstComment` inside that platform's `platformSpecificData`
5. If data.json is missing or has no affiliate link, the comment is silently skipped (warning logged)
6. TikTok entries are always skipped regardless of config

This works in both publishing modes. In unified mode (default), one post goes to all platforms, but each platform entry still carries its own `platformSpecificData` with a different `firstComment`. YouTube might get a subscribe CTA while Instagram gets just the link. In platform-specific mode, each platform is a separate post and the behavior is the same.

The feature is additive: post descriptions stay as-is, the first comment is extra.

</details>

---

## 🗑️ Blob Store Retention

Videos over 4 MB are staged in your Vercel Blob store (`LATE_VERCEL_TOKEN`); Zernio fetches them from the blob URL when a scheduled post goes live. After that the blob is dead weight, and without retention the store fills the free tier (1 GB) until Vercel pauses access, breaking every large upload.

Retention runs once after each publish run (`single`, `schedule`, and the global batch) and applies two policies in order: delete blobs older than `max_age_days`, then trim oldest-first until the store total is under `max_total_mb`. Blobs referenced by posts that aren't fully published yet are always kept, regardless of policy. Failures log a warning and never affect publishing; the step skips silently when disabled or when no Blob token is set.

```yaml
blob_retention:
  enabled: true
  max_age_days: 30    # delete blobs older than this
  max_total_mb: 500   # then trim oldest-first under this total
```

---

## 📋 Published Products Registry

The publisher maintains a persistent registry of all published products in both JSON and CSV formats. Entries are automatically added after each successful publish (single or batch).

**Registry files** (in outputs directory):
- `published_products.json` — machine-readable array of product objects
- `published_products.csv` — spreadsheet-friendly with header row

**Fields**: product ID (ASIN), title, canonical Amazon URL, affiliate URL.

<details>
<summary><strong>Rebuild from Existing Data</strong></summary>

Scan all `<product_id>/data.json` files and rebuild the registry:

```bash
# Rebuild from outputs/ directory
python -m src.publisher.late registry --rebuild --outputs-dir outputs

# Scan from one directory, save to another
python -m src.publisher.late registry --rebuild --scan-dir tmp --outputs-dir outputs
```

Running rebuild merges scanned entries into the existing registry; rows whose product directories were cleaned up after publishing stay in the registry. Each existing JSON/CSV file is renamed to `<name>.bak` before the new copy is written, so a write that drops or corrupts entries can be recovered from the backup.

</details>

<details>
<summary><strong>How It Works</strong></summary>

1. After a successful publish, `add_to_registry()` reads `data.json` for the product
2. Extracts title, URL (normalized to `https://www.amazon.com/dp/<ASIN>`), and affiliate URL
3. Skips if product already exists in registry (dedup by product ID)
4. Writes updated registry to both JSON and CSV, renaming each existing file to `<name>.bak` first
5. Failures are logged as warnings and never block publishing

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

**Zernio Rate Limits** (as of v0.17.0):
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
- Upgrade Zernio tier for higher limits

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
**Solution**: Connect platforms at https://zernio.com/dashboard/accounts

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
   - Go to https://zernio.com/dashboard/developers
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

4. Upgrade Zernio tier:
   - Standard: 100 req/hour
   - Pro: 1000 req/hour
   - Visit https://zernio.com/pricing

</details>

<details>
<summary><strong>Upload Failures</strong></summary>

**Problem**: `Upload failed: Network timeout`

**Solutions:**
1. Check internet connection:
   ```bash
   ping zernio.com
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

5. Check Zernio status:
   - Visit https://zernio.com/status
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
   - Visit https://zernio.com/dashboard/accounts
   - Click "Connect Account" for the platform
   - Complete OAuth authorization
   - Verify account appears in dashboard

3. Check account status:
   - Ensure account is not expired
   - Reauthorize if needed

4. Verify platform is supported:
   - Supported: YouTube, TikTok, Instagram, Facebook, Twitter, LinkedIn
   - Check Zernio documentation for platform-specific requirements

</details>

<details>
<summary><strong>TikTok Content Disclosure Errors</strong></summary>

**Problem**: `TikTok UX validation failed: Commercial content disclosure is enabled but no option selected. Please select "Your Brand" or "Branded Content" or both.`

**Cause**: TikTok requires explicit content disclosure settings for commercial accounts. If `commercial_content_type` and `is_brand_organic_post` are not set in the post's `tiktokSettings`, TikTok rejects the publish.

**Solutions:**

1. **For new posts** — the fix is built-in (v0.28.0+). `TikTokContentSettings` defaults include:
   ```python
   commercial_content_type = "brand_organic"
   is_brand_organic_post = True
   content_preview_confirmed = True
   express_consent_given = True
   ```

2. **For existing failed posts** — use the Late SDK `update` method to set correct settings (see Workflow 5 above). The update automatically triggers re-publish.

3. **Verify settings are being sent** — run with `--debug` and check for `tiktokSettings` in the API request payload.

</details>

<details>
<summary><strong>Instagram Container Errors &amp; Partial Posts</strong></summary>

**Problem**: A scheduled post publishes to some platforms but its top-level status is `partial`, and the Instagram leg shows:
```
instagram container error: ERROR
```
(per-platform `status: failed`, `errorCategory: platform_rejected`, no `platformPostId`, no `publishedAt`).

**Cause**: Instagram's Graph API publishes Reels in two steps — create a media container, then publish it. A `container error: ERROR` means the container stage failed with no sub-detail. When the **same** media published fine to YouTube and TikTok, the cause is almost always a **transient** Instagram-side container failure, not a bad video file.

**Solutions:**

1. **Retry the failed leg.** Zernio keeps the uploaded media on its own CDN (`media.zernio.com`), independent of the Vercel Blob store, so no re-render or re-upload is needed:
   ```python
   from late import Late
   client = Late(api_key=...)
   client.posts.retry(post_id)   # failed platform → processing, then auto-republishes
   ```
   Use `retry()` (not `update()`) for a transient failure with no settings change — `update()` is for cases like the TikTok disclosure fix where the payload itself must change. Already-published platforms are untouched, so there are no duplicate posts. The IG container usually resolves to `published` within ~1 minute; poll `posts.get(post_id)` to confirm.

2. **Confirm it actually went live.** Read `platforms[*].status` / `platformPostUrl` from `posts.get(post_id)` (dump with `model_dump(by_alias=True, mode="json")["post"]`). Do **not** trust `publish_history.json` for this — its `published_at` is **queue time**, not on-platform publish time, so "the Jun 11 post" by queue time and by publish time can be two different posts.

3. **If retry fails again**, the video likely violates a Reels spec (aspect ratio, duration, codec, frame rate). Re-render and re-upload.

**Finding these posts**: run `verify-delivery` to sweep recent posts for incomplete delivery. It WARNs on every post whose top status is `partial` or `failed`, names the failing platform and its error, and points at the `posts.retry` fix. Run it after a batch goes live, the same way `verify-comments` sweeps first comments.
```bash
python -m src.publisher.late verify-delivery --limit 25
```

</details>

<details>
<summary><strong>Debug Mode</strong></summary>

Enable verbose debug logging for troubleshooting:

```bash
poetry run python -m src.publisher.late single B0ABC \
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

For detailed instructions on using the Zernio SDK directly — listing posts, retrying failed legs, raw REST calls, and common SDK workarounds — see [Zernio Client Guide](zernio-client.md).

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

<details>
<summary><strong>Module Reference</strong></summary>

**`src/publisher/constants.py`** — Shared constants used across the publisher module:

| Constant | Value | Usage |
|----------|-------|-------|
| `SDK_LIST_PAGE_SIZE` | `100` | Page size for Late SDK list operations |
| `MAX_CONCURRENT_CLEANUPS` | `3` | Concurrent cleanup operations limit |
| `DEFAULT_OUTPUTS_DIR` | `Path("outputs")` | Default outputs directory path |

**`src/publisher/publish_modes.py`** — Shared publish orchestration used by CLI, global batch, and scheduler:

```python
from src.publisher.publish_modes import publish_product

results = await publish_product(
    publisher=publisher,
    media_id="media_123",
    product_id="B0ABC",
    platforms=[{"platform": "youtube", "account_id": "acc_123"}],
    outputs_dir="outputs",
    platform_specific=False,  # True for per-platform metadata
    schedule_time=None,       # datetime for scheduled posts
)
```

**`src/publisher/tracking.py`** — Publish history tracking with atomic writes (temp-file + rename) to prevent corruption on crash. Key functions: `record_publish()`, `is_already_published()`, `load_tracking()`, `save_tracking()`.

**`src/publisher/webhooks.py`** — Zernio webhook event handling for post-publish status updates, failure notifications, and retry triggers.

</details>

---

## 🔄 Common Workflows

### Workflow 1: First-Time Setup to First Publish

```bash
# Step 1: Configure credentials
echo "LATE_API_KEY=sk_live_your_key_here" >> .env
echo "LATE_VERCEL_TOKEN=vercel_blob_rw_xxx" >> .env

# Step 2: Verify connection
poetry run python -m src.publisher.late list-accounts --debug

# Step 3: Generate a video with metadata
poetry run python -m src.scraper.amazon.scraper --keywords B0BTYCRJSS --debug
poetry run python -m src.video.producer outputs/B0BTYCRJSS/data.json slideshow_images1 \
  --target-platform multi --debug

# Step 4: Publish immediately (for testing)
poetry run python -m src.publisher.late single B0BTYCRJSS \
  --platform youtube --immediate --debug

# Step 5: Verify in Zernio dashboard
# Visit https://zernio.com/dashboard/posts
```

### Workflow 2: Weekly Content Pipeline

```bash
# Monday: Scrape and produce videos
poetry run python -m src.pipeline.global_batch \
  --keywords "wireless earbuds" --max-products 7 \
  --profile slideshow_images1 --debug

# Monday: Schedule all videos for the week (one per day)
poetry run python -m src.publisher.late schedule auto \
  --platform youtube --platform tiktok --platform instagram \
  --debug

# View the schedule
poetry run python -m src.publisher.late calendar list --debug

# If conflicts occur, use auto-resolve
poetry run python -m src.publisher.late schedule auto \
  --platform youtube --auto-resolve --debug
```

### Workflow 3: Multi-Account Brand Management

```yaml
# config/publisher.yaml
accounts:
  brand_a:
    api_key: sk_live_brand_a_key
    default_platforms: [youtube, tiktok]
  brand_b:
    api_key: sk_live_brand_b_key
    default_platforms: [instagram]
default_account: brand_a
```

```bash
# Publish to Brand A (default)
poetry run python -m src.publisher.late single B0ABC123 --immediate

# Publish to Brand B
poetry run python -m src.publisher.late single B0ABC123 --account brand_b --immediate
```

### Workflow 4: Recover from Batch Failures

```bash
# Run batch publish (some may fail)
poetry run python -m src.publisher.late schedule --immediate \
  --platform youtube --platform tiktok --debug

# Check what failed
poetry run python -m src.publisher.late calendar list --status failed

# Retry only failed items
poetry run python -m src.publisher.late schedule --immediate \
  --platform youtube --platform tiktok --retry-failed --debug
```

### Workflow 5: Fix Failed TikTok Posts (Disclosure Error)

If TikTok fails with "Commercial content disclosure is enabled but no option selected":

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
                }
            },
        },
        {"platform": "instagram", "accountId": "<instagram_account_id>"},
    ]

    # Update triggers automatic re-publish (no retry() needed)
    result = await client.posts.aupdate(post_id, platforms=platforms)
    print(f"TikTok status: {result.post.platforms[1].status}")
    # Status changes: failed → pending → processing → published

asyncio.run(fix_tiktok("your_post_id"))
```

**Important**: Calling `aupdate()` with corrected platform settings automatically
triggers re-publish for the failed platform. Do NOT call `retry()` after — it will
return 409 "Post is currently publishing". Wait ~30s and check status with `aget()`.

### Workflow 6: Safe Cleanup After Publishing

```bash
# Preview what would be cleaned (dry run)
poetry run python -m src.publisher.late cleanup --all --dry-run --debug

# Verify posts are published
poetry run python -m src.publisher.late calendar list --status published

# Execute cleanup with confirmation
poetry run python -m src.publisher.late cleanup --all --confirm --debug
```

---

## 🔗 External Resources

- **Zernio Documentation**: https://docs.zernio.com
- **Zernio API Reference**: https://docs.zernio.com/api
- **Zernio Dashboard**: https://zernio.com/dashboard
- **Zernio Pricing**: https://zernio.com/pricing
- **Zernio Status**: https://zernio.com/status

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details
