# 0017. Sourced step list

- **Status:** Implemented
- **Issue:** #559
- **Requirements:** REQ-VID-121, REQ-VID-122, REQ-VID-151, REQ-VID-161, REQ-CNT-146

## Context

The detailed design is in the issue body, from "Length", "What makes a short tutorial useful", "Visuals that show the spoken step" and "Explanatory graphics" in [the tutorials explanation](../explanation/tutorials.md). This doc records how it fits the [rules every design follows](README.md#rules-that-apply-to-every-design).

Evidence ([evidence grades](README.md#evidence-grades)):

- Hallucinated facts give automated output away; the topic and product fact checks guard them, and a sourced step list extends that to each step. [C]

## Goals

- A topic script is written from a sourced step list, and its length follows the step count.

## Non-goals

None recorded.

## Design

- The topic script step first returns a structured list (action, exact UI path, expected result, source URL per step), validated like the other LLM outputs.
- The script is written from it and its length follows the step count.
- A step with no source is refused and a topic that cannot be sourced is dropped.
- Behind `llm_settings.topic_scripts.step_list.enabled`.

## As built

- The setting is `llm_settings.topic_scripts.step_list` (`enabled`, `model`, `max_steps`, `timeout_seconds`, `attempts`) in `config/ai_services.yaml`.
- One grounded call (`prompts/topic_step_list.md`) returns the list as JSON in text: the start screen, the platform, whether the steps fork, the steps, the source's common mistake, and a check of the topic itself. A step whose `source` is not an `http(s)` URL is refused, and its topic dropped: a tutorial with a step missing cannot be followed. The common mistake keeps its step through the renumbering, and is dropped with it when that step is refused.
- The topic check is the pool filter (REQ-VID-151): the topic must be specific (one device family or app, one outcome), searchable, demonstrable and non-default, and must not ask for health, financial or legal advice. A missing check, a criterion not answered `true`, or an `advice` answer other than the string "none" fails it. `python -m tools.check_topic_pool [topics.yaml]` runs the same call over the configured pool, prints `ok` or `drop` with the reason per topic, exits 1 when any would drop and 2 without the API key, so a failing topic can be taken out of the pool before its turn.
- The topic is dropped, as a skipped product, when the call fails, the topic fails its check, any step is unsourced, or the steps fork or exceed `max_steps`; this release sets such a topic aside rather than splitting it into a series.
- The script is written from the steps with `prompts/topic_from_steps.md`, a new prompt beside the existing templates, which stay untouched. Its length follows the tutorial research's length table (docs/explanation/tutorials.md, "Length"): 40-80 words for one or two steps and 110-200 for three to six. The script also names the device and version once and closes on a one-sentence path recap, which the same research ranks above demonstration alone. A draft at or above 80% of the band's floor is accepted. Below it, a draft is retried, and kept only when no attempt reaches it and it still clears `script_validation.min_words`; for one or two steps that minimum is lowered to 80% of the band's floor (32 words), so there such a draft is refused rather than kept. A short draft that also misses its closing line is never the CTA fallback.
- The list is written to `temp/step_list.json`, and the state carries a one-line `step_list` summary.
- On a live run (Background App Refresh on iPhone) the call returned three steps sourced to Apple's support page, and the script named the start screen and each step's result. The pool check passed that topic and dropped "screenshot anything on any device" (not specific) and a credit-card question (financial advice).
- A concept explainer is told apart by its title: a topic that starts "Why" is written from `prompts/topic_explainer_from_steps.md`, its cause first and the sourced steps as checks, in the research's 100-160 words whatever the step count (REQ-VID-161). The step-list call itself is unchanged, so the explainer reuses the same sourced steps. Its topic filter asks for one outcome, though, and live it dropped two explainer titles ("not specific", and no list at all), so explainers reach this path only once the filter admits them (#657).
- On since the reach-test hold ended ([decision 0014](../decisions/0014-the-reach-test-hold-ends-when-its-posts-are-queued.md)). Checking the 17-topic pool showed the grounded call taking 45-55 s against a 60 s timeout, so most pool topics timed out; the timeout is 120 s, and a failed call's log line names the error's type, since a timeout carries no message. The topic filter's `specific` rule admits a "why" question that names one device family or app and one symptom (#657). Even at 120 s about one call in five timed out, so a timed-out, dropped or 5xx call is tried again (`attempts`, default 2); a 4xx is not. The filter's verdict also varied between runs (16 of 17 pool topics passed one day, 7 of 17 three days later), so the pool was rebuilt from 15 topics that passed every one of three checks, or two with the third timing out, before the switch went on.

## Alternatives considered

None recorded.

## Rollout

On in the bundled config. The model default stays `false`, so a config without the block keeps the free-form topic script.

Remove the switch when: `llm_settings.topic_scripts.step_list.enabled` has been on in the bundled config for two weekly batches with topic-render completion (#551) no worse than without it; the key and the free-form script path then go in a minor release with a `**Breaking**:` CHANGELOG entry.

## Open questions

None recorded.
