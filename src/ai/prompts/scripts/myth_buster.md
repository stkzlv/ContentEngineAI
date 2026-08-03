# Script Template: Myth Buster

Write a voiceover script for a short product promo video that busts a common misconception about the product's category. The angle: most people believe something wrong about this type of product, and this one proves them wrong.

Open by stating the myth directly. Something like "Everyone thinks you need to spend a fortune on [category] to get decent quality. Wrong." or "There's this idea that [category] products can't do X and Y at the same time. Let me show you why that's outdated." The myth should be something the target audience has probably heard or assumed. Make it specific to the category, not generic.

Then dismantle it. Pull 2-3 features from the product description below and use each one as evidence that the myth doesn't hold up. Don't just say "this product is different." Show exactly how each feature contradicts the assumption. Frame it as "here's what actually matters" rather than attacking anyone who believed the myth.

Close with a one-liner that reframes the category for the viewer. They should walk away thinking about the product type differently. End with one CTA per the narrator profile.

## Rules

- The myth must be something people in the target audience actually believe. Don't invent a straw man nobody holds.
- Bust the myth with specifics, not opinions. Specs, design choices, real-world behavior from the product description.
- Tone is calm confidence, not condescension. You're correcting a misunderstanding, not dunking on people.
- Don't name competitors. The myth is about the category, not a specific brand.
- **Open with a natural conversational hook that carries the audio keyword** (product category, price band, audience cue, pain point) in speech a person would actually say, never as a search-bar query. TikTok 2026 indexes ASR transcripts as a primary search signal, so the keyword must land in the first 5 seconds of TTS. Pick one of six proven shapes from `docs/promotional-video-best-practices.md` §1: price-first reveal, regret/contrarian, POV, outcome-first, numbered teardown, comparison. Word budget ≤3 seconds, ~5-12 words. Anti-pattern: "Best [X] under $[N] for [Y]" reads as a Google query. Name the brand and a concrete spec by the second sentence at the latest. Read it out loud; if it sounds like a search bar, rewrite. Anti-setup: line 1 states a concrete fact, result, or observation about the product or its use, not a description of what is about to be shown. Avoid "Today I'll", "Let me tell you", "In this video", "I want to share" framings; they read as setup and burn the 1.5-second decision window.
- **Close with a debatable claim right before the CTA — a defensible opinion-shaped line that invites a 'well, actually' reply.** First scan the product description for a contestable performance number (W, mAh, Hz, GHz, MP, GB, ports, hours of battery, dB, Mbps, lumens, Nm, PSI, ANC, refresh rate). If one appears, close with a spec claim: "Most people only need two ports, but three is usually better." / "65W is the sweet spot for laptop charging." / "Eight hours is the right battery target for daily HDR use." If none do (passive product — mount, hook, organizer, kitchen tool, bracket, holder, decor, manual gadget), close with a material-or-use claim instead: "Steel beats plastic for any clamp-style mount." / "Gooseneck arms win over ball joints for bedside use." / "On the desk or on the nightstand - which would you actually grab?" Never invent a spec the product doesn't have (e.g. don't claim battery life for a phone holder). Pick exactly one closing line, under 15 words. Not a trade-off, and for spec-rich products not a question.
- Include one short trade-off or limitation, one sentence max.

## Product Data

- **Product name (full):** {FULL_PRODUCT_NAME}
- **Suggested short alias:** {SHORT_PRODUCT_NAME} (auto-trimmed from the title, so it may be a fragment or carry a model code. Use it only if it reads like a name a person would say out loud; otherwise call the product by its plain category noun, e.g. "this 3D pen".)
- **Description:** {PRODUCT_DESCRIPTION}
- **Target audience:** {AUDIENCE}
