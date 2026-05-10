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
| POST | `/lnk/edit` | `link_id`, `title` | **Undocumented but real.** In-place edit; same `id`, same position in the bio. Verified working as of 2026-05. |
| POST | `/lnk/delete` | `link_id` | Returns `{status: true}` on success. |

`/lnk/edit` is the right tool when you only need to change the title; reach for delete + re-add only if you also need to change the image or move the link to the top.

## The 50-link cap is an API page size, not a bio cap

`/lnk/list` always returns at most 50 entries. Pagination is not exposed: nineteen parameter combinations (`page`, `offset`, `limit`, `start`, `cursor`, `per_page`, `from`, `after`, `since_id`, etc., singly and combined) all return the same first 50.

The bio itself has no link quota on the free plan. The 50 ceiling is purely a list-API constraint. Implications:

- To enumerate a bio with more than 50 links via OAuth, you can't. Use the public bio page HTML (`https://lnk.bio/<slug>`) or a saved dashboard export instead.
- Paid Lnk.Bio tiers buy customization (themes, custom domain, analytics), not extra links.

## `created_at` is link-add time, not video publish time

Each bio link carries a `created_at` set when `/lnk/add` succeeds. In this pipeline, the link is added immediately after Zernio accepts the post into its scheduler. The actual YouTube / TikTok / Instagram post goes live later, when Zernio fires `scheduledFor`. Don't sort `created_at` to find "the latest live videos" — query Zernio (`client.posts.get`) for true publish state.

## Internal dashboard API (last resort)

The `lnk.bio/manage` dashboard talks to an unauthenticated-looking endpoint at `POST https://lnk.bio/api/` with form fields `ACTION`, `link_id`, and a per-session `token` parsed from the dashboard HTML (regex `TOKEN = "..."`, uppercase). This route exposes operations the OAuth API doesn't (reorder, full edit, hide), but:

- It needs a logged-in browser session — cookies, CSRF token, the works.
- It is rate-limited around 150 requests. In one observed sweep, 117 of 150 came back with `Slow down, too many requests`.

Treat it as a manual-tool escape hatch, not as an automation surface. The OAuth `/lnk/edit` endpoint covers the common case and is what the pipeline uses.

## Failure policy

Every link-in-bio call is wrapped in try/except in `LinkInBioManager` and logs at warning level on failure. A bio failure never blocks the corresponding Zernio publish. The video shipping is the load-bearing step; the bio entry is decoration.
