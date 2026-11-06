# 0030. Commentary topics

- **Status:** Implemented
- **Issue:** #782
- **Requirements:** REQ-CNT-166, REQ-CNT-167

## Context

A topic record renders with no product, so it carries no affiliate link and no disclosure overlay (REQ-CMP-004). Its templates are problem-first tutorials (REQ-CNT-013). A topic render also runs the tutorial-only parts: the sourced step list and the topic filter, then the step cards and step screens built from that list.

A short commentary video is an opinion on a stated subject. It has no steps, so none of those parts apply, and none of the tutorial templates fit it.

Every branch in the pipeline tests the same thing: whether the record is a topic. There are only two arms.

## Goals

- A topic entry marked as commentary renders with a commentary template, narrator, call-to-action list and length target.
- It renders without step visuals, and publishes like any topic, with no disclosure overlay.
- The shipped templates stay generic, and an installation replaces them in its overlay (decision 0017).

## Non-goals

- A third analytics arm. A commentary post counts in the `topic` content format.
- Enforcing the length target. The prompt asks for it, as it does for the other families.

## Design

- **The record.** `TopicSpec.kind` is `tutorial` (the default) or `commentary`. A topics file or the batch's `topics:` block sets it with a `kind:` key, and the producer and the batch set it with `--topic-kind` beside `--topic`. The flag is refused without `--topic`, because a topics file sets the kind per entry. The record carries `topic_kind: commentary` in `data.json`; a tutorial topic and a product carry `topic_kind: null`, because the serialiser writes every field.
- **The swap.** `config_for_record` in the producer's orchestration runs per record, before the pipeline context is built. For a commentary record it works on a deep copy of the config:
  - `topic_templates` becomes `commentary_templates`;
  - `narrator_profile_topic`, `cta_options_topic` and `target_length.topic` become their commentary counterparts, where those are set;
  - `topic_routing` and `topic_scripts.step_list` are turned off.
  
  Every consumer that asks "is this a topic" then behaves as it should for disclosure, voice, captions and the hook headline, and draws the commentary family. The step cards and screens read the step list, so they don't appear. The shared config is untouched for the next record in a batch. A commentary record with no commentary templates is refused, because a tutorial template would write a how-to under a take's title.
- **Products.** The product pool already excluded `topic_templates`. It now also excludes `commentary_templates`, because the default pool is a glob over the templates directory.
- **Shipped defaults.** One generic template, `commentary_take`: a hook, a turn, an opinion and a closing question, using only facts the topic detail states. It comes with a commentary narrator, two commentary CTAs and a 35-45 s length target. Each CTA starts with an opener the publisher strips, so the YouTube first comment is the closing question.

## Alternatives considered

- **A `kind` argument threaded through every `is_topic` consumer.** That is about twenty call sites, and each one a future change could miss. Swapping the topic family on a per-record copy reuses the existing two-armed branches.
- **A separate record type.** It would need its own path through discovery, the batch and analytics for the same disclosure and voice behaviour as a topic.

## Rollout

No switch. The commentary path runs only for a record an operator marks `kind: commentary`, and every other record renders as before. Decision 0002's switch is the record's kind.

## As built

A real commentary render drew `commentary_take`, built no step list, and closed on a question followed by the CTA. The publisher's closing-line extractor took the question. The script came in at 71 words, about 28 s, under the 90-115 word target, which REQ-CNT-166's Gap records.

## Open questions

None recorded.
