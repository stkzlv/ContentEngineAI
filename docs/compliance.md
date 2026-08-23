# Compliance

How the pipeline handles affiliate-content disclosure across the regulators that apply to short-form promotional video.

This doc grows as Phase 0 items in `docs/roadmap.md` land. Sections marked **(planned)** describe behavior that's not yet shipped; treat them as a contract for the implementation, not a description of current state.

## Why three regulators apply at once

Affiliate creators using this pipeline are in scope for:

1. **FTC Endorsement Guides (US)** — apply to any post that's reasonably foreseeable to reach US consumers. English audio + US Amazon links + a US-targeted persona clears that bar; physical residency outside the US is not a shield.
2. **Amazon Associates Operating Agreement** — contract, not regulation. Applies to anyone holding an Associates account, anywhere in the world.
3. **Spain CNMC + EU DSA + Royal Decree 444/2024** — apply to creators domiciled in the EU. Below the User of Special Relevance threshold (€300K/yr or 1M followers single-platform or 2M cross-platform) the form is flexible; the substance ("unambiguous to the average user that this is advertising") is not.

Disclosure is a default render output, not a per-video manual checklist. Where the pipeline can't produce a disclosure layer (because of an SDK or platform-API gap), the gap is documented under [Manual steps](#manual-steps) below and the creator is expected to handle it per video.

## What the pipeline produces today

| Layer | Status | What it satisfies |
|---|---|---|
| Persistent on-frame disclosure overlay (`#ad` burned in a fixed corner, full-clip duration) | Shipped | FTC two-punch (overlay component). Configurable via `video_settings.disclosure_overlay` in `config/video_production.yaml` (text, position, size, color, background). Sized at 45% of subtitle font by default (slightly under FTC's 50-60% band, tuned for tighter corner placement against the platform UI corridor). Survives the pycaps subtitle pass because pycaps burns its captions onto the assembler's output rather than replacing it, and applies on the FFmpeg subtitle paths including content-aware positioning, whose filter chain the assembler normalizes before drawing the overlay. |
| First-line caption disclosure (`#ad` leads the caption on TikTok / Instagram / YouTube) | Shipped | FTC two-punch (caption component) and Spain CNMC. Configurable per render via the `disclosure` field on `PublishMetadata`; defaults to `#ad`. Both this and the overlay are conditional on the render carrying a material connection: a topic render with no affiliate link emits neither, since a disclosure on content with no such connection is a false statement about one. (It is not primarily a reach question: TikTok's guidance says correctly disclosed branded content performs as well as or better than undisclosed, so the case for gating rests on accuracy, not on algorithmic penalty.) The gate defaults to disclosing, so only a record that positively shows there is nothing to disclose suppresses it; a product whose affiliate link failed to build still discloses. The producer records the decision in `metadata.json` and the publisher reads it, rather than the two deriving it separately and disagreeing. The string-only escape hatch lets language-aware variants inject `#publi` for Spanish renders without further code change. |
| Affiliate program literal phrase (`As an Amazon Associate I earn from qualifying purchases` rendered in the caption body after `#ad`) | Shipped, **opt-in** | Amazon Associates Operating Agreement. Configurable via `config/publisher.yaml::affiliate_disclosure` (enabled, phrase, program); off by default. Enable it only while membership of the named program is active, since the phrase asserts a material connection. Non-Amazon programs can override the phrase. |
| TikTok commercial-content disclosure (`commercialContentType`, `isBrandOrganicPost`) | Shipped | TikTok platform policy. Without it, TikTok rejects affiliate posts ("Commercial content disclosure is enabled but no option selected"). Conditional on the render carrying a material connection, on the same recorded decision as the overlay and the caption: an affiliate render declares `brand_organic`, a topic render with no affiliate link declares `none`. `none` is sent explicitly rather than omitting the settings, since an absent block reads as a payload that forgot them. The settings ride in two places on the payload: per-platform, carrying both fields, and top-level, carrying `commercialContentType` only. Both are built from the same per-render value, so they cannot disagree. |
| YouTube altered-or-synthetic-content disclosure (`containsSyntheticMedia`) | Shipped, **opt-in** | YouTube platform policy. Configurable via `config/publisher.yaml::synthetic_media_disclosure`; **off by default**. The policy targets realistic material that could mislead about real people or events, and its published examples explicitly exclude AI narration, AI-written scripts, faceless content and stock footage — which is what this pipeline renders, so nothing in the current feature set meets the bar. Declaring it anyway applies a viewer-facing label the policy does not ask for, and makes the flag useless internally by flagging every render alike. Two of YouTube's listed examples could apply later without any change here: AI-generated music, and AI-generated footage of a real place. Both are properties of what a music or stock provider returns, which is why the flag is gated rather than removed. |

## What's still planned

| Layer | Status | What it satisfies |
|---|---|---|
| Localized disclosure variants | **(planned)** | FTC same-language rule + Spain Royal Decree 444/2024. Spanish renders emit `#publi` or `#publicidad`; English renders emit `#ad`. The plumbing is already in place via the `disclosure` field on `PublishMetadata` and `DisclosureSettings`; remaining work wires script language to the field value. |

## Manual steps

Some disclosure layers can't ship as a pipeline default because the publishing API doesn't expose the relevant field. Each item below is a per-video manual action a creator takes after the pipeline publishes.

### YouTube paid-promotion checkbox

YouTube's "Includes paid promotion" platform label is set via YouTube Studio. The publisher SDK we use (Zernio) does not expose the `containsPaidPromotion` field. Until that gap closes, do this manually for every published video:

1. Open YouTube Studio, navigate to the published video.
2. Click the video's "Details" pencil.
3. Expand "Show more" at the bottom.
4. Under "Paid promotion", check "Yes, video contains paid promotion like a product placement, sponsorship, or endorsement."
5. Save.

This is a platform-policy layer (ranking impact if missed), not the FTC regulatory floor. The on-frame overlay and first-line caption disclosure cover the regulatory side independently.

### Instagram paid-partnership label

Instagram's "Paid partnership" label is set in the post-edit flow. The publisher SDK does not expose the branded-content tagging endpoint. Until that gap closes, do this manually for every published Reel:

1. Open Instagram, find the published Reel on the profile.
2. Tap the three-dot menu → Edit.
3. Tap "Add Paid Partnership Label".
4. Optionally tag a brand partner (skip for affiliate-only content; the label still applies).
5. Save.

Same regulatory framing as YouTube above: this is a platform-policy layer, not the FTC floor.

### Profile bio: Amazon Associates identification (optional)

The pipeline can render the required literal phrase in the caption body of every post, but the setting is off by default (`config/publisher.yaml::affiliate_disclosure`), because the phrase asserts membership of the named program. With it enabled the per-account bio step below is optional; with it off, the bio is the only place the phrase appears. If you want extra coverage, add it to the profile bio of every social account that links to Amazon affiliate URLs:

- TikTok bio: tight character cap (80 chars). Use the shortened form: "Amazon Associate. I earn from qualifying purchases."
- Instagram bio: 150-char cap. Same shortened form fits.
- YouTube channel description (About tab): no character pressure; use the full literal phrase.
- Link-in-bio page header: same as TikTok.

This is a one-time manual setup per account. A closing-frame render of the same phrase is deferred as a future belt-and-suspenders measure.

## Tracked SDK gaps

Tracked as GitHub issues, not blocking Phase 0:

- YouTube `paid_promotion` flag not exposed by Zernio SDK (issue [#119]).
- Instagram paid-partnership / branded-content tagging not exposed by Zernio SDK (issue [#120]).

The manual workarounds above stay in place until the SDK exposes the fields. When that happens, the pipeline auto-sets the flags and the manual steps drop from this doc.

[#119]: https://github.com/stkzlv/ContentEngineAI/issues/119
[#120]: https://github.com/stkzlv/ContentEngineAI/issues/120

## Penalty surface

| Source | Penalty | Enforcement speed | What's at risk |
|---|---|---|---|
| FTC | Up to $53,088 per violation, per post (2025) | Months; complaint-driven plus active monitoring | Civil fines |
| Amazon Associates | Account termination, affiliate ID lost | Days; algorithmic + manual review | Affiliate revenue path |
| TikTok / IG / YouTube policy | Post downrank or removal | Hours-days; algorithmic | Reach for that post |
| Spain CNMC (Royal Decree 444/2024) | Administrative warning, then fines | Months-years; under-resourced | Regulatory action against the creator |

The pipeline's Phase 0 work targets the FTC and Amazon layers. Platform policy is additive; Spain enforcement is currently the lowest-likelihood path.
