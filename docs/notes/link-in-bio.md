# Link-in-Bio Module Notes

<!--
Moved out of CLAUDE.md (#454), which had grown to 224KB -- about 50k tokens
loaded at the start of every assisted session, most of it per-entry history
rather than rules that must be in front of you at all times. The text is
unchanged; CLAUDE.md points here from the section each entry left.

Each entry records a defect and what it cost, so the shape that produced it is
recognisable the next time. Add to it the same way: what broke, why it was
invisible, and what now catches it.
-->

- **CLI flags**: `--link-in-bio` and `--no-link-in-bio` override `link_in_bio.enabled` config for single publish
- **Affiliate URL fallback**: Uses `affiliate_link` field first, falls back to `url` if unavailable
- **Image fallback**: Uses `images[0]` URL first, falls back to `downloaded_images[0]` local file upload
- **Lnk.Bio auth**: Requires HTTP Basic Auth (not form-encoded), plus `User-Agent: ContentEngineAI/1.0` header to bypass Cloudflare
- **Lnk.Bio API endpoints**: Auth: `POST /oauth/token`, Add: `POST /oauth/v1/lnk/add`, List: `GET /oauth/v1/lnk/list`, Edit: `POST /oauth/v1/lnk/edit` (undocumented; in-place edit of title AND destination URL via an optional `link` param, same id, same position, same image and `created_at`; `title` is replaced by whatever you send, so echo the current one back when only rewriting the URL), Delete: `POST /oauth/v1/lnk/delete`. `LnkBioProvider` exposes `add_link` / `list_links` / `delete_link` but no `update_link` — reach the edit endpoint directly until that gap closes. Full protocol notes in `docs/lnkbio-api.md`.
- **Lnk.Bio OAuth scope is hard-capped at `basic`**: nineteen alternative scope strings (`full`, `read_write`, `all`, `links.write`, ...) all return `unsupported scope`. There is no premium scope. Don't try to widen access via OAuth.
- **`/lnk/list` 50 ceiling is an API page size, not a bio cap**: pagination is not exposed (page/offset/cursor/etc. all return the same first 50). The bio itself has no link quota on the free plan. Fetching the public page with curl is NOT the workaround: it returns only the newest ~48, the rest being rendered client-side, so both automated sources truncate newest-first on a bio that really held 300 links. Their agreement is a shared blind spot, not corroboration, and any "is the list complete" test built on the two of them gives a confident wrong answer. Enumerate by opening the bio in a browser, letting it load, and saving the page; each anchor carries `data-id`, `href`, and the full `title`. Those ids drive `/lnk/edit` and `/lnk/delete` directly, so the cap limits discovery, not modification.
- **Non-blocking**: Failures never block video publishing; logged as warnings
- **`created_at` is link-add time, not platform publish time**: The bio link is added right after `posts.create` (queue time), but YouTube/TikTok/Instagram only go live when Zernio's scheduler fires `scheduledFor`. The bio link can be clickable for days before the corresponding video is up. Don't use lnk.bio `created_at` to verify a video is actually published; use Zernio's `platforms[*].status` instead.
- **Free plan has unlimited links**: lnk.bio's free tier has no link quota (their headline differentiator vs Linktree). Paid tiers unlock customization (themes, custom domain, analytics), not link count. Safe to keep adding without worrying about a cap.
