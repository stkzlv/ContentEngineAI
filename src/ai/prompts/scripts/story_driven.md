# Script Template: Story Driven

Write a voiceover script for a short product promo video built around a mini personal story.

Open with a first-person anecdote that carries the audio-keyword hook in the first sentence (see Rules below for the full spec). The anecdote and the hook are one beat — pack the product category, price band, or audience cue into the opening so the story names the product world immediately, not three sentences in. Examples: "Spent $25 on this USB-C hub after my third dongle broke at the coffee shop." / "Found this 4K dashcam for under $80 on a whim and now I keep one in each car." One or two sentences. Make it feel like an actual moment, not a setup.

Then bridge into the product discovery. How did you come across it, and what made you try it? This should feel natural, like you're telling a friend about something you bought. Not a testimonial, just a story.

Weave in 2-3 product features from the description below, but embed them in the narrative. Don't pause the story to list specs. Instead of "it has 40-hour battery life", say something like "I charged it on Monday and forgot about it until Friday." Let the features live inside the experience.

End with one CTA per the narrator profile. Quick and casual.

## Rules

- The story should feel plausible for someone in the target audience. Match the tone and situation to who'd actually buy this.
- Don't make the story too polished or cinematic. Real stories meander a little, have small details that don't matter but make it feel true.
- Not scripted, not rehearsed.
- **Open with a natural conversational hook that carries the audio keyword** (product category, price band, audience cue, pain point) in speech a person would actually say, never as a search-bar query. TikTok 2026 indexes ASR transcripts as a primary search signal, so the keyword must land in the first 5 seconds of TTS. Pick one of six proven shapes from `docs/promotional-video-best-practices.md` §1: price-first reveal, regret/contrarian, POV, outcome-first, numbered teardown, comparison. Word budget ≤3 seconds, ~5-12 words. Anti-pattern: "Best [X] under $[N] for [Y]" reads as a Google query. Name the brand and a concrete spec by the second sentence at the latest. Read it out loud; if it sounds like a search bar, rewrite. Anti-setup: line 1 states a concrete fact, result, or observation about the product or its use, not a description of what is about to be shown. Avoid "Today I'll", "Let me tell you", "In this video", "I want to share" framings; they read as setup and burn the 1.5-second decision window.
- **Close with a two-option opinion question right before the CTA. This is a SEPARATE beat from the trade-off rule below — not a downside, but a real fork that invites a one-tap pick.** Concrete examples: "USB-C or Lightning - which still annoys you more?" / "Team magnetic or team plug-in?" / "Quick charge or long battery - which matters more for you?" One short line, under 12 words. Both options must be pickable; rhetorical questions and yes/no asks don't count. NOT a trade-off, NOT a sponsor pitch — a real question with two clear sides.
- Include one short trade-off or limitation, one sentence max.

## Product Data

- **Product name (full):** {FULL_PRODUCT_NAME}
- **Suggested short alias:** {SHORT_PRODUCT_NAME} (auto-trimmed from the title, so it may be a fragment or carry a model code. Use it only if it reads like a name a person would say out loud; otherwise call the product by its plain category noun, e.g. "this 3D pen".)
- **Description:** {PRODUCT_DESCRIPTION}
- **Target audience:** {AUDIENCE}
