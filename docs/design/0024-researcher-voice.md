# 0024. Product scripts in the researcher's voice

- **Status:** Implemented
- **Issue:** #700
- **Requirements:** REQ-CNT-156, REQ-CNT-157, REQ-CNT-158, REQ-CNT-159, REQ-CNT-160

## Context

The channel's narrator is a virtual creator: one voice, no face, and no hands to hold a product. A review generated 30 product scripts, one per product template on two of four scraped products (smartwatch, cat water fountain, smart plug, USB cable), through the producer's own script step with the shipped config and the fact check on. Measured:

- **20 of 30 claim the narrator owned, bought, received or used the product**: "I picked up this smartwatch", "I've tried five different smartwatches", "Spent $40 on this", "Just got this cable in the mail". Six templates ask for an experience by design: `unboxing_reaction`, `story_driven`, `before_after`, `comparison`, `skeptic_converted` and `lifestyle_flex`; `social_proof` invents buzz ("people keep talking about it"). The narrator profile's own voice example does the same ("So I picked this up last month ... Took it on a hike and never lost signal").
- **4 speak a price, 3 of them wrong.** The smart plug costs $10.92; its scripts said "a $25 smart plug that does what the $80 ones do" and "Spent $40". The narrator profile forbids a price; the hook rule lists a price-first reveal (REQ-CNT-017), and the story and lifestyle templates quote prices in their example openers.
- **Prompt examples are reused.** "Steel beats plastic for any clamp-style mount." closed three smartwatch scripts; "USB-C or Lightning - which still annoys you more?" closed a cat fountain script; the voice example's "actually grips" reached a smart plug and a cable. The guard for copied examples (REQ-CNT-155) is on and exempts questions, so it would not stop the second.
- **Spec lists, not help.** "140 sports modes" appears in every smartwatch script. About a third carry the trade-off REQ-CNT-023 asks for, and none says who the product suits or who should skip it.

The fact check reads product scripts against the listing, so it confirms that a smartwatch has 140 sports modes and cannot see that nobody wore it.

Evidence ([evidence grades](README.md#evidence-grades)):

- "Endorsements must reflect the honest opinions, findings, beliefs, or experience of the endorser", and "when the advertisement represents that the endorser uses the endorsed product, the endorser must have been a bona fide user of it at the time the endorsement was given." [A] [16 CFR 255.1](https://www.law.cornell.edu/cfr/text/16/255.1)
- Meta demotes engagement bait, and YouTube's inauthentic-content policy names "content that looks like it's made with a template". [A] ([0008](0008-bait-free-closing-lines.md), [0022](0022-signature-lines.md))

## Goals

- A product script never claims the narrator owned, bought, received, used or tested the product, nor that other people talk about it; the first person is for research and opinion ("I went through the listing", "here's what I'd check").
- No price is spoken, and no template or rule models one.
- No prompt quotes a whole example sentence a script could reuse; an example shows a shape.
- Each product script says who the product suits or who should skip it, beside its one trade-off.

## Non-goals

- **The call-to-action pools.** Their bait lines are [0008](0008-bait-free-closing-lines.md) and #549.
- **Topic scripts.** They already speak as someone who fixed the problem, which is honest for a procedure the narrator can describe exactly.
- **Rendered visuals.** A product shown on stills and stock footage claims no use.

## Design

- **Narrator profile.** Replace the voice example with one in the researcher's voice that carries no ownership claim, and replace "the tone you'd use telling a friend about something you bought" with research language. Add a rule: never claim to own, buy, receive, wear, use or test the product, and never claim other people talk about it.
- **Templates.** Rewrite the six experience templates around research rather than use: `unboxing_reaction` becomes a listing walk-through ("what's actually in the box, per the listing"), `story_driven` and `lifestyle_flex` tell the viewer's scenario in the second person, `before_after` contrasts the problem and the product rather than the narrator's life, `comparison` compares against the category rather than "five I tried", `skeptic_converted` keeps the doubt and resolves it from the listing and its stated figures. `social_proof` draws on the listing's rating and review count only when the data carries them, and is otherwise left out of the pool.
- **Prices.** Drop the price-first reveal from the hook patterns (REQ-CNT-017) and every price from the template examples.
- **Examples as shapes.** Rewrite each quoted example sentence in the templates and profiles as a shape with a placeholder ("[material] beats [material] for [use]"), so there is nothing to copy whole.
- **Who it suits.** Ask for one sentence naming who the product suits or who should skip it, beside the trade-off.
- **A check.** The research module's measured checks gain an ownership-claim count and a price count per script, so the next sample shows the rate before and after.

**Tests.** No template or profile carries an ownership phrase, a price or a quoted whole-sentence example; the research checks count ownership claims and prices on recorded scripts.

## As built

Built as designed after the reach-test hold ended ([decision 0014](../decisions/0014-the-reach-test-hold-ends-when-its-posts-are-queued.md)). The research checks count claims of use (owning, buying, receiving, using or testing the product, handling remarks such as "heavier than I expected", and claims that others talk about it) and spoken prices. One script per product template for two scraped products, 30 in all: 20 claimed use and 1 spoke a price before; after the rewrite 4 still said a weight was "heavier than I expected", and a profile line asking for the listing's figure instead brought it to 0 claims and 0 prices, with every script naming who the product suits or should skip it. `social_proof` is written from the listing's rating and ratings count (the page's count is of ratings, not written reviews), and a product missing either never draws it, as a fixed template either. The value pillar's preamble no longer leans on price.

## Alternatives considered

- **Disclose the narrator as AI and keep the anecdotes.** An AI label does not make a claim of use true; the FTC treats the ad disclosure and the AI disclosure as separate duties.
- **Drop the six experience templates.** Simpler, but it cuts the template pool from fifteen to nine and with it the variety the pool exists for; the templates' shapes survive the rewrite.
- **Rely on the copied-example guard.** It retries after the fact, misses reworded and question-shaped copies, and costs a model call per retry; removing the examples removes the cause.

## Rollout

The test posts are rendered and scheduled, so new renders reach no test post. The rewrite ships on: a claim of use is false on every render that carries it. Each template is checked by one live script before the release.

Remove the switch when: not applicable; there is no switch.

## Open questions

- **Whether the brand's own example lines change too.** The channel's tone-of-voice examples outside this repository use first-person use claims; this design covers the pipeline only.
