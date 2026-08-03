# Script Template: Question Driven

Write a voiceover script for a short product promo video built around provocative rhetorical questions that the product answers.

Open with a question that names the audio keyword inside it — product category, price band, or audience cue (see Rules below for the full spec). The question and the hook land together; vague "what if I told you" without category doesn't carry the search signal. Examples: "Ever wondered why $20 wireless earbuds keep dying after six months?" / "Why does a $40 mechanical keyboard feel better than a $200 one?" The question should still feel like it's about to reveal something the viewer hasn't considered — just with the keyword built in.

Follow with 2-3 more questions, each one targeting a different pain point or desire from the target audience. After each question, deliver the answer in one tight sentence tied to a real feature from the product description below. The pattern is: question, answer, question, answer. Don't stack all the questions at the start and all the answers at the end.

Close with a short statement that ties the answers together, then end with one CTA per the narrator profile.

## Rules

- Questions should feel genuine, not gimmicky. "What if I told you" is fine as an opener but don't use it for every question. Vary the question format.
- Don't ask yes/no questions. Ask questions that make the viewer think for a second.
- Answers should be specific, not vague. Connect directly to product features from the description below.
- Keep the overall tone curious and confident, like someone who figured something out and wants to share.
- **Open with a natural conversational hook that carries the audio keyword** (product category, price band, audience cue, pain point) in speech a person would actually say, never as a search-bar query. TikTok 2026 indexes ASR transcripts as a primary search signal, so the keyword must land in the first 5 seconds of TTS. Pick one of six proven shapes from `docs/promotional-video-best-practices.md` §1: price-first reveal, regret/contrarian, POV, outcome-first, numbered teardown, comparison. Word budget ≤3 seconds, ~5-12 words. Anti-pattern: "Best [X] under $[N] for [Y]" reads as a Google query. Name the brand and a concrete spec by the second sentence at the latest. Read it out loud; if it sounds like a search bar, rewrite. Anti-setup: line 1 states a concrete fact, result, or observation about the product or its use, not a description of what is about to be shown. Avoid "Today I'll", "Let me tell you", "In this video", "I want to share" framings; they read as setup and burn the 1.5-second decision window.
- **Close with a debatable claim right before the CTA — a defensible opinion-shaped line that invites a 'well, actually' reply.** Choose the claim type with this test, in order. (a) Find a measurement in the Description above that is written out with its unit attached, as a whole word, about THIS product — for example "65W", "5000mAh", "120Hz", "two USB-C ports", "12 hours". A word that merely contains a unit as a substring does NOT count: "supports" and "Portable" are not ports, "However" is not watts, "Important" is not amps. If you cannot quote the measurement verbatim from the Description, you did not find one. (b) If you found one, close with a claim about THAT measurement, reusing its unit. (c) If you found none, close with a material, shape, or use claim instead — no numbers, no units. Examples of (c): "Steel beats plastic for any clamp-style mount." / "Gooseneck arms win over ball joints for bedside use." / "Velcro straps outlast adhesive pads every time." Whichever branch you take, the subject of the closing line must be something this product actually is or has. Inventing a spec is the single worst failure here: a tracker tag has no ports, a phone holder has no battery life, a notebook has no refresh rate. Pick exactly one closing line, under 15 words. Not a trade-off, and for spec-rich products not a question.
- Include one short trade-off or limitation, one sentence max.

## Product Data

- **Product name (full):** {FULL_PRODUCT_NAME}
- **Suggested short alias:** {SHORT_PRODUCT_NAME} (auto-trimmed from the title, so it may be a fragment or carry a model code. Use it only if it reads like a name a person would say out loud; otherwise call the product by its plain category noun, e.g. "this 3D pen".)
- **Description:** {PRODUCT_DESCRIPTION}
- **Target audience:** {AUDIENCE}
