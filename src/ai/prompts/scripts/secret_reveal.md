# Script Template: Secret Reveal

Write a voiceover script for a short product promo video using an insider knowledge angle.

Open with the "secret" or "hack" framing AND embed the audio-keyword hook in the first sentence (see Rules below for the full spec). The secret is about a specific product category, price band, or audience-specific use case — not a vague "most people don't know." Examples: "There's a $25 USB-C hub that does what the $80 ones do and almost nobody talks about it." / "Most reviews skip this: a $15 ergonomic mouse pad fixes wrist pain better than the $50 brands." Specific category and price, not vague hype. You're letting the viewer in on something good, not posturing.

Then reveal 2-3 features from the product description, but frame them as discoveries or little-known benefits rather than a spec sheet. Think "the thing nobody tells you is..." or "what most reviews skip over is..." Each point should feel like you're sharing something genuinely useful, not reading a product page.

End with one CTA per the narrator profile.

## Rules

- Don't make the "secret" angle feel forced. If the product is a common item, the secret can be about a specific feature or use case people overlook, not about the product's existence.
- Sound like someone who genuinely found something good and wants to share it. Not like an infomercial host.
- Be specific, not vague.
- **Open with a natural conversational hook that carries the audio keyword** (product category, price band, audience cue, pain point) in speech a person would actually say, never as a search-bar query. TikTok 2026 indexes ASR transcripts as a primary search signal, so the keyword must land in the first 5 seconds of TTS. Pick one of six proven shapes from `docs/promotional-video-best-practices.md` §1: price-first reveal, regret/contrarian, POV, outcome-first, numbered teardown, comparison. Word budget ≤3 seconds, ~5-12 words. Anti-pattern: "Best [X] under $[N] for [Y]" reads as a Google query. Name the brand and a concrete spec by the second sentence at the latest. Read it out loud; if it sounds like a search bar, rewrite. Anti-setup: line 1 states a concrete fact, result, or observation about the product or its use, not a description of what is about to be shown. Avoid "Today I'll", "Let me tell you", "In this video", "I want to share" framings; they read as setup and burn the 1.5-second decision window.
- **Close with a two-option opinion question right before the CTA. This is a SEPARATE beat from the trade-off rule below — not a downside, but a real fork that invites a one-tap pick.** Concrete examples: "USB-C or Lightning - which still annoys you more?" / "Team magnetic or team plug-in?" / "Quick charge or long battery - which matters more for you?" One short line, under 12 words. Both options must be pickable; rhetorical questions and yes/no asks don't count. NOT a trade-off, NOT a sponsor pitch — a real question with two clear sides.
{CTA_RULE}
- Include one short trade-off or limitation, one sentence max.

## Product Data

- **Product name (full):** {FULL_PRODUCT_NAME}
- **Suggested short alias:** {SHORT_PRODUCT_NAME} (auto-trimmed from the title, so it may be a fragment or carry a model code. Use it only if it reads like a name a person would say out loud; otherwise call the product by its plain category noun, e.g. "this 3D pen".)
- **Description:** {PRODUCT_DESCRIPTION}
- **Target audience:** {AUDIENCE}
