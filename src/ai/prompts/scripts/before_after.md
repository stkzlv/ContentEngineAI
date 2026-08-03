# Script Template: Before/After Transformation

Write a voiceover script for a short product promo video built around a before/after contrast.

Start with the "before" picture and embed the audio-keyword hook in the first sentence (see Rules below for the full spec). The "before" state names the product category, audience cue, or pain point immediately, not three sentences in. Examples: "Before this $18 magnetic phone mount, my GPS lived in a cup holder." / "Remember when noise-canceling earbuds under $40 sounded terrible?" Quick scene, specific to the target audience, immediate keyword. One or two sentences.

Then flip to the "after." Show the contrast clearly. Pull 2-3 features from the product description and frame them as what changed. Don't just list specs. Describe what those features actually feel like in practice, what they let you do differently.

The shift from before to after should feel satisfying, like a small but real upgrade in daily life. Don't oversell it as life-changing unless the product genuinely is. Keep the transformation proportional to what the product actually does.

End with one CTA per the narrator profile. Keep it natural.

## Rules

- The "before" should be something the target audience actually experienced, not a made-up worst case.
- The "after" should be grounded in real product features, not exaggerated claims.
- A short punchy sentence after a longer one keeps things moving.
- **Open with a natural conversational hook that carries the audio keyword** (product category, price band, audience cue, pain point) in speech a person would actually say, never as a search-bar query. TikTok 2026 indexes ASR transcripts as a primary search signal, so the keyword must land in the first 5 seconds of TTS. Pick one of six proven shapes from `docs/promotional-video-best-practices.md` §1: price-first reveal, regret/contrarian, POV, outcome-first, numbered teardown, comparison. Word budget ≤3 seconds, ~5-12 words. Anti-pattern: "Best [X] under $[N] for [Y]" reads as a Google query. Name the brand and a concrete spec by the second sentence at the latest. Read it out loud; if it sounds like a search bar, rewrite. Anti-setup: line 1 states a concrete fact, result, or observation about the product or its use, not a description of what is about to be shown. Avoid "Today I'll", "Let me tell you", "In this video", "I want to share" framings; they read as setup and burn the 1.5-second decision window.
- **Close with a debatable claim right before the CTA — a defensible opinion-shaped line that invites a 'well, actually' reply.** Choose the claim type with this test, in order. (a) Find a measurement in the Description above that is written out with its unit attached, as a whole word, about THIS product — for example "65W", "5000mAh", "120Hz", "two USB-C ports", "12 hours". A word that merely contains a unit as a substring does NOT count: "supports" and "Portable" are not ports, "However" is not watts, "Important" is not amps. If you cannot quote the measurement verbatim from the Description, you did not find one. (b) If you found one, close with a claim about THAT measurement, reusing its unit. (c) If you found none, close with a material, shape, or use claim instead — no numbers, no units. Examples of (c): "Steel beats plastic for any clamp-style mount." / "Gooseneck arms win over ball joints for bedside use." / "Velcro straps outlast adhesive pads every time." Whichever branch you take, the subject of the closing line must be something this product actually is or has. Inventing a spec is the single worst failure here: a tracker tag has no ports, a phone holder has no battery life, a notebook has no refresh rate. Pick exactly one closing line, under 15 words. Not a trade-off, and for spec-rich products not a question.
- Include one short trade-off or limitation, one sentence max.

## Product Data

- **Product name (full):** {FULL_PRODUCT_NAME}
- **Suggested short alias:** {SHORT_PRODUCT_NAME} (auto-trimmed from the title, so it may be a fragment or carry a model code. Use it only if it reads like a name a person would say out loud; otherwise call the product by its plain category noun, e.g. "this 3D pen".)
- **Description:** {PRODUCT_DESCRIPTION}
- **Target audience:** {AUDIENCE}
