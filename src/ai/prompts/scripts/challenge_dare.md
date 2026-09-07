# Script Template: Challenge Dare

Write a voiceover script for a short product promo video that poses a quiet challenge to the viewer. The tone is confident and steady, not loud or cocky. Think "I've done the research, here's what I found" rather than "prove me wrong."

Open with a calm challenge that names what you're challenging the viewer ON — embed the audio-keyword hook in the first sentence (see Rules below for the full spec). The challenge anchors in a specific product category, price band, or audience cue, not a vague "try this for a week." Examples: "Find me a 65W GaN laptop charger under $25 that does USB-C and pass-through. I'll wait." / "Use this $30 silicone splatter screen for a week. See if you go back to oil shields." Conversational tone, immediate keyword.

Back up the challenge with 2-3 specific features or specs from the product description below. Present them matter-of-factly: "It does X, which on its own is solid. But it also handles Y." Let the specs speak for themselves without overselling.

Close by restating the challenge in a relaxed way. End with one CTA per the narrator profile.

## Rules

- Confident but not aggressive. You're inviting the viewer to compare, not daring them.
- Every claim needs a specific detail backing it up. Don't say "find better quality" without saying what makes this one stand out.
- Challenge on concrete specs, features, or value. Things the viewer can actually verify.
- Keep the tone steady throughout. No volume spikes, no hype buildup. Calm conviction from start to finish.
- Mix in the occasional rhetorical question, but keep it measured, not rapid-fire.
- **Open with a natural conversational hook that carries the audio keyword** (product category, price band, audience cue, pain point) in speech a person would actually say, never as a search-bar query. TikTok 2026 indexes ASR transcripts as a primary search signal, so the keyword must land in the first 5 seconds of TTS. Pick one of six proven shapes from `docs/promotional-video-best-practices.md` §1: price-first reveal, regret/contrarian, POV, outcome-first, numbered teardown, comparison. Word budget ≤3 seconds, ~5-12 words. Anti-pattern: "Best [X] under $[N] for [Y]" reads as a Google query. Name the brand and a concrete spec by the second sentence at the latest. Read it out loud; if it sounds like a search bar, rewrite. Anti-setup: line 1 states a concrete fact, result, or observation about the product or its use, not a description of what is about to be shown. Avoid "Today I'll", "Let me tell you", "In this video", "I want to share" framings; they read as setup and burn the 1.5-second decision window.
- **Close with a debatable claim right before the CTA — a defensible opinion-shaped line that invites a 'well, actually' reply.** Choose the claim type with this test, in order. (a) Find a measurement in the Description above that is written out with its unit attached, as a whole word, about THIS product — for example "65W", "5000mAh", "120Hz", "two USB-C ports", "12 hours". A word that merely contains a unit as a substring does NOT count: "supports" and "Portable" are not ports, "However" is not watts, "Important" is not amps. If you cannot quote the measurement verbatim from the Description, you did not find one. (b) If you found one, close with a claim about THAT measurement, reusing its unit. (c) If you found none, close with a material, shape, or use claim instead — no numbers, no units. Examples of (c): "Steel beats plastic for any clamp-style mount." / "Gooseneck arms win over ball joints for bedside use." / "Velcro straps outlast adhesive pads every time." Whichever branch you take, the subject of the closing line must be something this product actually is or has. Inventing a spec is the single worst failure here: a tracker tag has no ports, a phone holder has no battery life, a notebook has no refresh rate. Pick exactly one closing line, under 15 words. Not a trade-off, and for spec-rich products not a question.
{CTA_RULE}
- Include one short trade-off or limitation, one sentence max.

## Product Data

- **Product name (full):** {FULL_PRODUCT_NAME}
- **Suggested short alias:** {SHORT_PRODUCT_NAME} (auto-trimmed from the title, so it may be a fragment or carry a model code. Use it only if it reads like a name a person would say out loud; otherwise call the product by its plain category noun, e.g. "this 3D pen".)
- **Description:** {PRODUCT_DESCRIPTION}
- **Target audience:** {AUDIENCE}
