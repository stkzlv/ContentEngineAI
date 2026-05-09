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
| TikTok branded-content disclosure (`commercialContentType=brand_organic`, `isBrandOrganicPost=true`) | Shipped | TikTok platform policy. Without it, TikTok rejects affiliate posts. |
| YouTube AI-content disclosure (`containsSyntheticMedia=true`) | Shipped | YouTube platform policy. Required for AI-generated video. |
| First-line caption disclosure (`#ad` leads every caption on TikTok / Instagram / YouTube) | Shipped | FTC two-punch (caption component) and Spain CNMC. Configurable per render via the `disclosure` field on `PublishMetadata`; defaults to `#ad`. The string-only escape hatch lets language-aware variants (Phase 0.4) inject `#publi` for Spanish renders without further code change. |

## What's planned in Phase 0

| Layer | Status | What it satisfies |
|---|---|---|
| Persistent on-frame disclosure overlay | **(planned)** Phase 0.1 | FTC two-punch (overlay component). Burns `#ad` (or localized) into every render, full-clip duration, fixed corner. |
| Affiliate program literal-phrase rendering | **(planned)** Phase 0.3 | Amazon Associates Operating Agreement. The literal "As an Amazon Associate I earn from qualifying purchases" phrase rendered in at least one of bio / on-frame / caption. |
| Localized disclosure variants | **(planned)** Phase 0.4 | FTC same-language rule + Spain Royal Decree 444/2024. Spanish renders emit `#publi` or `#publicidad`; English renders emit `#ad`. The plumbing is already in place via the `disclosure` field on `PublishMetadata`; Phase 0.4 wires script language to the field value. |
| Disclosure test suite | **(planned)** Phase 0.6 | CI regression coverage on every render layer. |

## Manual steps

Some disclosure layers can't ship as a pipeline default because the publishing API doesn't expose the relevant field. Each item below is a per-video manual action a creator takes after the pipeline publishes.

### YouTube paid-promotion checkbox

YouTube's "Includes paid promotion" platform label is set via YouTube Studio. The publisher SDK we use (Zernio) does not expose the `containsPaidPromotion` field. Until that gap closes, do this manually for every published video:

1. Open YouTube Studio, navigate to the published video.
2. Click the video's "Details" pencil.
3. Expand "Show more" at the bottom.
4. Under "Paid promotion", check "Yes, video contains paid promotion like a product placement, sponsorship, or endorsement."
5. Save.

This is a platform-policy layer (ranking impact if missed), not the FTC regulatory floor. The on-frame overlay (Phase 0.1, planned) and first-line caption (Phase 0.2, planned) cover the regulatory side independently.

### Instagram paid-partnership label

Instagram's "Paid partnership" label is set in the post-edit flow. The publisher SDK does not expose the branded-content tagging endpoint. Until that gap closes, do this manually for every published Reel:

1. Open Instagram, find the published Reel on the profile.
2. Tap the three-dot menu → Edit.
3. Tap "Add Paid Partnership Label".
4. Optionally tag a brand partner (skip for affiliate-only content; the label still applies).
5. Save.

Same regulatory framing as YouTube above: this is a platform-policy layer, not the FTC floor.

### Profile bio: Amazon Associates identification

Amazon's Operating Agreement requires the literal phrase "As an Amazon Associate I earn from qualifying purchases" (or a substantially similar pre-approved statement) wherever Program Content is displayed. Render it in the profile bio of every social account that links to Amazon affiliate URLs:

- TikTok bio: tight character cap (80 chars). Use the shortened form: "Amazon Associate. I earn from qualifying purchases."
- Instagram bio: 150-char cap. Same shortened form fits.
- YouTube channel description (About tab): no character pressure; use the full literal phrase.
- Link-in-bio page header: same as TikTok.

This is a one-time manual setup per account, not per-video. The phrase doesn't need to appear on every video as long as the account that posts the video carries it in the bio. Phase 0.3 (planned) adds a closing-frame render of the same phrase as a belt-and-suspenders measure.

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
