# 0007. Script lint and search-phrase placement

- **Status:** Held
- **Issue:** #548
- **Requirements:** REQ-CNT-053, REQ-CNT-054, REQ-CNT-055

## Context

`validate_script_completeness` checks truncation, length floors and the CTA; the narrator profiles carry banned-phrase prose, which the model may ignore.

Evidence ([evidence grades](README.md#evidence-grades)):

- Machine-written phrasing has well-catalogued tells ("it's not X, it's Y", reflexive lists of three, "delve", "game-changer", "seamless"); whether they cost engagement is unmeasured. [C] [Pangram](https://www.pangram.com/signs-of-ai-writing)
- Superficial competence, a fluent veneer over little specific, checkable content, is one of three defining features of AI slop. [B] [arXiv 2601.06060](https://arxiv.org/abs/2601.06060)
- Filler and fake urgency ("in today's video", "you won't believe") read as generic. [C]
- A specific, verifiable fact or opinion is the "added information" every platform's originality criteria ask for. [A]
- A registered meta-analysis of 8,977 headline tests found concreteness follows an inverted U: name the product, number or situation and withhold only the answer. [A] [Scientific Reports](https://www.nature.com/articles/s41598-024-81575-9)
- "But" and "therefore", never "and then": each sentence causes or complicates the next, a screenwriting rule that suits a 90-word script. [C] [The Script Lab](https://thescriptlab.com/features/screenwriting-101/13636-how-south-park-creators-plot-better-scripts/)
- No study sets an optimal words-per-minute for Shorts; 150-170 WPM is common practice, and lab work found fast speech reduced listeners' ability to judge arguments. [weak, ungraded] [Personality and Social Psychology Bulletin](https://journals.sagepub.com/doi/10.1177/01461672952110006)
- TikTok reads captions, on-screen text and transcribed speech, so the search phrase goes in the first spoken line, the on-screen text and the start of the caption. [A for the mechanism, C for effect sizes]
- Not supported: "300-500% more search views" from search-phrase placement; the figures are unsourced.

## Goals

- Reject a script that uses common machine-writing phrases, exceeds a sentence-length cap or exceeds a word count derived from the target duration, and retry.
- Lead the hook headline and every platform caption with the search phrase.
- Report, per render, where the search phrase appears.

## Non-goals

- Losing a render to the lint. A script that fails only the lint ships with a warning.

## Design

- `script_validation.lint`: `enabled` (default false), `banned_phrases` (a list: "it's not X, it's Y" shapes as regexes, "game-changer", "say goodbye to", "elevate", "seamless", "delve", "whether you're"), `max_sentence_words` (default 16), `max_words_per_sec` (default 2.8, applied to the profile's target duration).
- Run inside the existing `_validate` closure, so a failing script re-enters the retry loop like a missing CTA, with its own reason string. The last-resort fallback today rescues only a script whose sole defect is the CTA; extend it so a script that fails only the lint also ships (with a warning) rather than losing the render.
- Prompt rules for the hook's concreteness and the "but/therefore" chain render into `{CTA_RULE}` after the existing rules, like naturalism, behind `script_templates.hook_rules.enabled` (default false).
- Search-phrase placement: a prompt rule for the hook headline and each platform caption prompt to lead with the search phrase (the product keyword or the topic keyword), behind the same switch as the hook rules. The first spoken sentence already carries it through the existing audio-keyword rule.
- Search-phrase report: the phrase checked against the first spoken sentence, the hook headline and the first 60 characters of each platform caption, counted per render in the [0006](0006-render-choices-and-variety-report.md) report.

**Tests.** Each banned shape rejects a fixture script; the length caps reject an over-long script; off leaves `_validate` unchanged; the report counts a fixture where the phrase is missing from the caption.

## As built

- The lint (`REQ-CNT-053`) is built and held off. `script_validation.lint` carries `enabled`, `banned_phrases` (regular expressions, matched case-insensitively, checked at config load), `max_sentence_words`, `max_words_per_sec` and `target_duration_sec`. No profile has a target duration, so the word cap is `max_words_per_sec` times a configured `target_duration_sec` of 40, the top of the narrator profiles' 30-40 seconds (`REQ-CNT-142`). A tutorial written from a step list skips the word cap, since its step count sets its length.
- The bundled list adds "in today's video" and "you won't believe" to the design's phrases, from the Context's filler evidence. "It's not X, but Y" is an ordinary concession and passes; only "it's not X, it's Y" fails, and "elevate" fails only as "elevate your/the". Curly apostrophes are read as straight ones, and a match that also occurs in the product's own title or keyword is no tell, so a product named "Seamless" can be called by its name.
- A script failing only the lint is kept as a last resort ahead of a script missing its call to action, since it is the more complete of the two.
- The search-phrase report (`REQ-CNT-055`) shipped: each render's row in `state/render_choices.jsonl` records where the phrase appears (captions read the way the publisher reads them, unified file first, and measured before it puts the disclosure in front), and the variety report prints the share per place. The phrase is the product keyword, or a topic's title rather than its stock keywords, since the title is phrased as searched. A place counts when every word of three or more letters, in a topic's title question and filler words aside (a spoken answer to "Why your laptop fan runs" drops "why"; a product keyword such as "can opener" keeps every word), appears there, or the phrase appears with its spaces closed up, since the product keyword "smart watch" was written "smartwatch" in a real script and headline.
- The hook and placement rules (`REQ-CNT-054`) are built and held off behind `script_templates.hook_rules.enabled`. The script rules render into `{CTA_RULE}` after the signature rules. The search-phrase rule is appended after the formatted prompt, so no template changes and every prompt is unchanged when off; for a topic it names the title's key words (question and filler words aside) rather than the whole title, which runs longer than a headline holds. It goes to the hook headline, the unified description, and the YouTube, TikTok and Instagram caption prompts, and not to the fact check or the stock search phrases, which share the same LLM helper.

## Alternatives considered

None recorded.

## Rollout

- The search-phrase report is measurement only and can land before the reach-test readout (#540).
- The lint ships off: `script_validation.lint.enabled` defaults to false. Set it after the readout, once rejection rates on a batch stay low (the retry loop is paid per call) and the scripts read better on review.
- The hook and placement rules ship off: `script_templates.hook_rules.enabled` defaults to false. Set it after the readout; the placement rules wait for the report.

Remove the switch when: each of `script_validation.lint.enabled` and `script_templates.hook_rules.enabled` has been on in the bundled config for two weekly batches, the lint rejecting no more than one script in five on its first attempt and the search-phrase report showing the phrase in the hook headline and caption openings at least as often as before; each key and its off path then go, separately, in a minor release with a `**Breaking**:` CHANGELOG entry, and the banned-phrase list and length caps stay.

## Open questions

None recorded.
