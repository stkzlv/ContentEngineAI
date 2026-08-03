# Script Template: Comparison

Write a voiceover script for a short product promo video where the narrator has tested several alternatives in the same category and this product came out on top.

Open with something like "I went through five different [category] before I found one that actually worked" or "I've tried the cheap ones, the expensive ones, and the hyped ones. This is the one I kept." The framing should make it clear you've done the legwork so the viewer doesn't have to.

Don't name competitors or other brands. Keep it vague on what lost and specific on what won. Pick 2-3 standout features from the product description below and frame each one as the thing other products got wrong. For example, "Every other one I tried had terrible battery life. This one goes three days without charging." Each comparison should feel like a real frustration that this product solved.

Wrap up by reinforcing the verdict (this is the one worth buying), then end with one CTA per the narrator profile.

## Rules

- Never name competing brands or products. Say "the other ones", "everything else I tried", etc.
- Sound like someone who's genuinely frustrated with bad purchases and relieved to find a good one. Not angry, just honest.
- **Open with a natural conversational hook that carries the audio keyword** (product category, price band, audience cue, pain point) in speech a person would actually say, never as a search-bar query. TikTok 2026 indexes ASR transcripts as a primary search signal, so the keyword must land in the first 5 seconds of TTS. Pick one of six proven shapes from `docs/promotional-video-best-practices.md` §1: price-first reveal, regret/contrarian, POV, outcome-first, numbered teardown, comparison. Word budget ≤3 seconds, ~5-12 words. Anti-pattern: "Best [X] under $[N] for [Y]" reads as a Google query. Name the brand and a concrete spec by the second sentence at the latest. Read it out loud; if it sounds like a search bar, rewrite. Anti-setup: line 1 states a concrete fact, result, or observation about the product or its use, not a description of what is about to be shown. Avoid "Today I'll", "Let me tell you", "In this video", "I want to share" framings; they read as setup and burn the 1.5-second decision window.
- **Close with a debatable claim right before the CTA — a defensible opinion-shaped line that invites a 'well, actually' reply.** Choose the claim type with this test, in order. (a) Find a measurement in the Description above that is written out with its unit attached, as a whole word, about THIS product — for example "65W", "5000mAh", "120Hz", "two USB-C ports", "12 hours". A word that merely contains a unit as a substring does NOT count: "supports" and "Portable" are not ports, "However" is not watts, "Important" is not amps. If you cannot quote the measurement verbatim from the Description, you did not find one. (b) If you found one, close with a claim about THAT measurement, reusing its unit. (c) If you found none, close with a material, shape, or use claim instead — no numbers, no units. Examples of (c): "Steel beats plastic for any clamp-style mount." / "Gooseneck arms win over ball joints for bedside use." / "Velcro straps outlast adhesive pads every time." Whichever branch you take, the subject of the closing line must be something this product actually is or has. Inventing a spec is the single worst failure here: a tracker tag has no ports, a phone holder has no battery life, a notebook has no refresh rate. Pick exactly one closing line, under 15 words. Not a trade-off, and for spec-rich products not a question.
- Include one short trade-off or limitation, one sentence max.

## Product Data

- **Product name (full):** {FULL_PRODUCT_NAME}
- **Suggested short alias:** {SHORT_PRODUCT_NAME} (auto-trimmed from the title, so it may be a fragment or carry a model code. Use it only if it reads like a name a person would say out loud; otherwise call the product by its plain category noun, e.g. "this 3D pen".)
- **Description:** {PRODUCT_DESCRIPTION}
- **Target audience:** {AUDIENCE}
