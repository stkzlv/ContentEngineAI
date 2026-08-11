# Lnk.Bio API notes

Operational notes about the Lnk.Bio OAuth API. Lnk.Bio's official documentation is thin and several behaviors were discovered the hard way. Capture them here so the next change doesn't re-learn them.

Configuration, env vars, and the `link_in_bio` block live in [publisher.md](publisher.md#-link-in-bio-integration); this doc covers the protocol, not the wiring.

## Auth

- OAuth2 `client_credentials` grant on `POST https://lnk.bio/oauth/token`.
- Credentials go in **HTTP Basic Auth** (`Authorization: Basic <base64(client_id:client_secret)>`), not in the form body. Form-encoded `client_id`/`client_secret` returns 401.
- Scope is hard-capped at `basic`. Nineteen alternative scope strings (`full`, `read_write`, `all`, `links`, `links.write`, `links:full`, etc.) all return `unsupported scope`. There is no premium scope to unlock.
- Tokens are short-lived. The provider re-auths automatically on a 401 from any call.

## Cloudflare gate

Lnk.Bio fronts everything with Cloudflare. Default library user agents (httpx, requests) get a 403 challenge page. Send a real `User-Agent` header on every request, including the token endpoint. The provider ships `User-Agent: ContentEngineAI/1.0` — change it only if you mean to.

## Endpoints

Base: `https://lnk.bio/oauth/v1`.

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/lnk/add` | `title`, `link`, optional `image` (URL) or multipart `image` (file) | Returns `{data: {id, url}}`. New link appears at the **top** of the bio. |
| GET | `/lnk/list` | — | Returns at most **50** links per call, see below. |
| POST | `/lnk/edit` | `link_id`, `title`, optional `link` | **Undocumented but real.** In-place edit; same `id`, same position in the bio. Verified working. |
| POST | `/lnk/delete` | `link_id` | Returns `{status: true}` on success. |

`/lnk/edit` changes the **destination URL** as well as the title: pass `link` alongside `link_id` and `title` and the link keeps its id, position, image, and `created_at`. Reach for delete + re-add only when you need to change the image or deliberately move the link to the top.

`title` is not optional and is not merged. Whatever you send replaces the stored title, so read the current one and send it back unchanged when you only mean to rewrite the URL.

Verified by rewriting 300 links in one sweep (tag stripping): 299 edits, no failures, and a re-saved copy of the bio page confirmed all 300 ids, titles, and positions intact. Sequential requests spaced ~0.35s hit no rate limiting; the ~150-request ceiling documented below applies to the separate dashboard API, not this endpoint.

## The 50-link cap is an API page size, not a bio cap

`/lnk/list` always returns at most 50 entries. Pagination is not exposed: nineteen parameter combinations (`page`, `offset`, `limit`, `start`, `cursor`, `per_page`, `from`, `after`, `since_id`, etc., singly and combined) all return the same first 50.

The bio itself has no link quota on the free plan. The 50 ceiling is purely a list-API constraint. Implications:

- To enumerate a bio with more than 50 links via OAuth, you can't.
- Paid Lnk.Bio tiers buy customization (themes, custom domain, analytics), not extra links.

### Fetching the public page is not a workaround

`curl https://lnk.bio/<slug>` returns only the newest ~48 links. The rest are rendered client-side by external JS, and the saved markup carries no endpoint to page through. So the two obvious sources cap out at 50 and 48 on a bio that actually held 300 links.

**Treat that agreement as a shared blind spot, not corroboration.** Both truncate newest-first, so tests that try to distinguish "truncated" from "complete" using only these two sources give confident wrong answers. Reasoning that the returned set spans the bio's full date range, or that the apparent gaps are interleaved rather than forming a contiguous oldest block, proves nothing here.

To enumerate every link, open the bio in a browser, let it finish loading, and save the page. Each anchor carries what an edit sweep needs:

```
<a href="<destination>" data-id="<link_id>" title="<full title>" data-type="TYPE_BIOLINK" ...>
```

Parse `(data-id, href, title)` per anchor, and validate the title parsing against the ids that overlap `/lnk/list` before any mass edit, since `/lnk/edit` overwrites the title with whatever you send. `link_id` values from the saved page work directly against `/lnk/edit` and `/lnk/delete`, so the 50-link list cap does not limit what you can modify, only what you can discover.

## `created_at` is link-add time, not video publish time

Each bio link carries a `created_at` set when `/lnk/add` succeeds. In this pipeline, the link is added immediately after Zernio accepts the post into its scheduler. The actual YouTube / TikTok / Instagram post goes live later, when Zernio fires `scheduledFor`. Don't sort `created_at` to find "the latest live videos" — query Zernio (`client.posts.get`) for true publish state.

## Internal dashboard API (last resort)

The `lnk.bio/manage` dashboard talks to an unauthenticated-looking endpoint at `POST https://lnk.bio/api/` with form fields `ACTION`, `link_id`, and a per-session `token` parsed from the dashboard HTML (regex `TOKEN = "..."`, uppercase). This route exposes operations the OAuth API doesn't (reorder, full edit, hide), but:

- It needs a logged-in browser session — cookies, CSRF token, the works.
- It is rate-limited around 150 requests. In one observed sweep, 117 of 150 came back with `Slow down, too many requests`.

Treat it as a manual-tool escape hatch, not as an automation surface. The OAuth `/lnk/edit` endpoint covers the common case and is what the pipeline uses.

## Failure policy

Every link-in-bio call is wrapped in try/except in `LinkInBioManager` and logs at warning level on failure. A bio failure never blocks the corresponding Zernio publish. The video shipping is the load-bearing step; the bio entry is decoration.
