# Working with the Late / Zernio Client

This guide covers common operations against the Zernio publishing API from the project codebase. The platform rebranded from Late to Zernio; the codebase still uses the `late-sdk` package and the `LATE_API_KEY` environment variable, but the live API endpoint is `https://zernio.com/api`.

---

## Table of Contents

- [Setup](#setup)
- [Instantiating the client](#instantiating-the-client)
- [Common operations](#common-operations)
  - [List connected accounts](#list-connected-accounts)
  - [List posts](#list-posts)
  - [Get a single post](#get-a-single-post)
  - [Retry failed or partial posts](#retry-failed-or-partial-posts)
  - [Update post content](#update-post-content)
  - [Delete a post](#delete-a-post)
- [Using the raw REST API](#using-the-raw-rest-api)
- [Proxy and environment gotchas](#proxy-and-environment-gotchas)
- [Known SDK issues and workarounds](#known-sdk-issues-and-workarounds)
- [Troubleshooting errors](#troubleshooting-errors)

---

## Setup

Add the API key to the project `.env`:

```bash
LATE_API_KEY=sk_live_your_key_here
LATE_VERCEL_TOKEN=vercel_blob_rw_xxx  # Optional, only for videos > 4 MB
```

The publisher CLI loads `.env` automatically. One-off scripts should load it explicitly:

```python
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path("/path/to/repo/.env"))
```

`load_dotenv()` with no arguments resolves the `.env` relative to the **calling script**, not the repo root, so always pass the absolute path when the script is outside the project directory.

---

## Instantiating the client

The high-level wrapper is `LatePublisher`:

```python
import asyncio
from src.publisher.late.client import LatePublisher

async def main():
    publisher = LatePublisher(api_key="sk_live_your_key")
    accounts = await publisher.get_accounts()
    print(accounts)

asyncio.run(main())
```

You can also pass `vercel_token`, `timeout`, `max_retries`, and `session` (an `aiohttp.ClientSession`). For most status / retry operations a session is not required.

---

## Common operations

### List connected accounts

```python
accounts = await publisher.get_accounts()
for acc in accounts:
    print(acc["platform"], acc["account_id"], acc["username"])
```

The SDK returns `Platform5` enum values; `get_accounts()` already normalizes them to lowercase strings.

### List posts

```python
# All posts
posts = await publisher.list_posts()

# Filter by top-level status
failed = await publisher.list_posts(status="failed")
partial = await publisher.list_posts(status="partial")
scheduled = await publisher.list_posts(status="scheduled")

for post in posts:
    print(post["id"], post["status"], post["scheduledFor"])
```

`list_posts()` returns a minimal normalized dict: `id`, `status`, `scheduledFor`, `platforms` (platform names only). It is safe to use through the empty `platformPostUrl` SDK bug.

### Get a single post

```python
platforms = await publisher.get_post_platforms(post_id)
for p in platforms:
    print(
        p["platform"],
        p["status"],
        p["platform_post_id"],
        p["error_message"],
        p["error_category"],
    )
```

This is the most reliable way to read per-platform status and error messages. It also tolerates the empty `platformPostUrl` bug.

### Retry failed or partial posts

```python
post_id = "6a46ef77672e4dd57f7df210"
await publisher.client.posts.retry(post_id)
```

`retry()` re-publishes **only the failed legs** and leaves already-published platforms untouched. It is the right tool for transient failures (`TikTok upload failed or timed out`, `Instagram container error: ERROR`). It is the wrong tool for payload errors like missing TikTok disclosure settings or invalid YouTube descriptions; fix the payload first, then retry.

Poll the post after retry:

```python
import time

for _ in range(10):
    platforms = await publisher.get_post_platforms(post_id)
    print([p["status"] for p in platforms])
    if any(p["status"] == "failed" for p in platforms):
        break
    time.sleep(20)
```

### Update post content

Use `update()` when the failure is due to the caption or title, e.g. YouTube rejecting angle brackets in the description:

```python
post_id = "6a46ef77672e4dd57f7df210"

# Fetch current content
post = await publisher.client.posts.get(post_id)
content = post.post.content  # or read from the dict

# Sanitize
sanitized = content.replace("<10ms", "sub-10ms").replace("<", "").replace(">", "")

# Update and retry
await publisher.client.posts.update(post_id, content=sanitized)
await publisher.client.posts.retry(post_id)
```

Updating the unified `content` field changes the caption used by any platform that does not have its own `customContent`. Already-published platforms are usually not affected, but verify the post after the update.

### Delete a post

```python
await publisher.client.posts.delete(post_id)
```

Published posts cannot be deleted through the SDK; use the platform's native UI for that.

---

## Using the raw REST API

The SDK's strict Pydantic models can crash on real-world responses (for example, a published TikTok leg with `platformPostUrl: ""`). For one-off scripts or debugging, use the raw REST API directly:

```python
import os
import httpx
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path("/path/to/repo/.env"))

client = httpx.Client(
    base_url="https://zernio.com/api",
    headers={
        "Authorization": f"Bearer {os.environ['LATE_API_KEY']}",
        "Accept": "application/json",
        "User-Agent": "late-python-sdk/0.0.1",
    },
)

# List posts
r = client.get("/v1/posts", params={"page": 1, "limit": 50})
posts = r.json()["posts"]

# Get a post
r = client.get(f"/v1/posts/{post_id}")
post = r.json()["post"]

# Retry a post
r = client.post(f"/v1/posts/{post_id}/retry")
print(r.json())

# Update post content
r = client.put(
    f"/v1/posts/{post_id}",
    json={"content": "sanitized caption without angle brackets"},
)
print(r.json())
```

The raw API returns the full post payload, including `mediaItems`, `platformSpecificData`, and per-platform error messages.

---

## Proxy and environment gotchas

The Zernio SDK (via `httpx`) reads proxy environment variables. If `ALL_PROXY` or `all_proxy` is set to a `socks://` URL, `httpx` raises:

```
ValueError: Unknown scheme for proxy URL URL('socks://127.0.0.1:2080')
```

Unset those variables before running the publisher or any client script:

```bash
env -u ALL_PROXY -u all_proxy poetry run python -m src.publisher.late schedule --debug
```

When the HTTP proxy is needed for routing, `HTTP_PROXY`/`HTTPS_PROXY` can stay set. The raw API can hit a Vercel WAF challenge through a proxy (HTTP 403 with `X-Vercel-Mitigated: challenge`); retrying usually succeeds, or you can bypass the proxy entirely if the network is fast enough.

---

## Known SDK issues and workarounds

| Issue | Symptom | Workaround |
|-------|---------|------------|
| Empty `platformPostUrl` | `Input should be a valid URL, input is empty` | Use `LatePublisher.list_posts()` / `get_post_platforms()` or the raw REST API |
| SOCKS proxy | `ValueError: Unknown scheme for proxy URL` | `unset ALL_PROXY all_proxy` |
| Vercel WAF challenge | HTTP 403 with `X-Vercel-Mitigated: challenge` | Retry; or bypass the HTTP proxy |
| `Video description is invalid` (YouTube) | Caption contains `<` or `>` characters | Sanitize the content and update + retry |
| TikTok account restriction | `This TikTok account has been restricted from posting` | Resolve the restriction in the TikTok app; retrying will not help |

---

## Troubleshooting errors

### `Post is currently publishing; please try again in a moment.` (HTTP 409)

The post is in the `publishing` state. Wait for the leg to finish or fail, then retry.

### `YouTube upload failed: Video description is invalid`

The description likely contains literal `<` or `>` characters. Fetch the post, sanitize the unified content, `update` the post, and `retry`.

### `This TikTok account has been restricted from posting`

A TikTok-side account restriction. Retrying will produce the same error. Fix the account in TikTok first.

### `Instagram container error: ERROR`

A transient IG Graph container failure. `retry()` is the correct fix.

### `Commercial content disclosure is enabled but no option selected` (TikTok)

The TikTok disclosure settings are missing. Use `posts.update()` with the correct `platformSpecificData.tiktokSettings` (or the project-level `TikTokContentSettings`) so the post auto-republishes.

---

For the full publisher workflow, scheduling, and compliance details, see [publisher.md](publisher.md) and [compliance.md](compliance.md).
