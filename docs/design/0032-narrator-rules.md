# 0032. Narrator rules apart from the narrator profile

- **Status:** Accepted
- **Issue:** #804
- **Requirements:** REQ-CNT-171, REQ-CNT-172

## Context

Each narrator profile (`narrator_profile`, `narrator_profile_topic`, `narrator_profile_commentary`) is one string holding two kinds of text:

- who the narrator is: the "Who", "Voice" and "Specificity" sections;
- rules that every script of the kind follows: the call-to-action rule, the phrases that read as machine-written, and the universal rules, which carry the spoken-length target's `{TARGET_SECONDS}` and `{TARGET_WORDS}` placeholders (REQ-CNT-163) and the no-invention rule.

An installation sets its own narrator in an overlay (decision 0017), and an overlay key replaces the whole string. So the overlay has to copy the rules by hand. If it doesn't, the length target, the anti-AI list and the no-invention rule disappear, and nothing warns: the prompt still reads as complete. A commentary narrator written in an overlay without the universal rules would have produced scripts with no length target at all.

A probe render with an overlay narrator found a second gap. The profile quoted short examples (a four-word comparison and a five-word closing question), and the script spoke both verbatim. The copied-example retry (REQ-CNT-155) checks only quoted examples of three content words or more, and skips questions, so neither was caught. The shipped profiles already quote no whole sentence (REQ-CNT-159); nothing checks an overlay's.

## Goals

- An overlay that sets only a narrator profile keeps the kind's rules.
- With the bundled config, every assembled prompt is byte-identical to the one before this change.
- A narrator text with no length target, or with a quoted example a script could reuse, is reported at config load.

## Non-goals

- Checking that a script meets the length target. REQ-CNT-166's Gap stays.
- Refusing a config. The checks warn, because a deliberate narrator without a length target is allowed.

## Design

- **Three keys.** `script_templates.narrator_rules`, `narrator_rules_topic` and `narrator_rules_commentary` in `config/ai_services.yaml`, each a string. The shipped profiles are split where their first rule section starts (the "CTA:" line). The profile keeps everything above it and the rules key takes the rest, including the blank line between them, so `profile + rules` is the old string byte for byte. The topic rules carry the voice example, because it sits between the call-to-action rule and the universal rules, and moving it would change the prompt.
- **One assembly point.** `ScriptTemplateConfig.narrator_for` already returns the profile with its length filled, for the script and for the platform captions (REQ-CNT-115). It returns `target.fill(profile + rules)` instead, picking the topic or product pair the way it picks the profile today. An empty topic rules key falls back to the product rules, as an empty topic profile falls back today.
- **Commentary.** `config_for_record` copies `narrator_rules_commentary` over `narrator_rules_topic` when it is set, beside the profile copy it already makes (design 0030).
- **The warnings.** A model validator on `ScriptTemplateConfig`, run after the overlays merge, assembles each kind's text and logs one warning per kind:
  - with neither `{TARGET_SECONDS}` nor `{TARGET_WORDS}` in it;
  - with a quoted span that is a sentence or a question of four words or more, the shape REQ-CNT-159's test already refuses in the shipped prompts. The validator reuses that test's rule, moved into the config module so both read one definition.
- **Tests.** The bundled assembled text per kind equals a snapshot of the pre-change profile; an overlay that sets only `narrator_profile_commentary` still gets the commentary rules; each warning fires on a crafted overlay and not on the bundled config.

## Alternatives considered

- **A persona field on the narrator identity**, filled into a placeholder in the shipped profile. The identity lives in the video config and the prompts in the AI config, so the script step would need both, and the topic profile's voice example would still sit inside the rules.
- **Documenting that an overlay must copy the rules.** That is today's state, and it failed silently.
- **Moving the voice example into the profile.** It changes the topic prompt, which needs its own measurement.

## Rollout

No switch: the bundled prompts don't change. An overlay that already copied the rules into its profile gets them twice until it drops its copy, which the release notes ask for.

## Open questions

None recorded.
