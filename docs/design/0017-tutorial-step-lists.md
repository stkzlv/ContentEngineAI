# 0017. Sourced step list

- **Status:** Accepted
- **Issue:** #559
- **Requirements:** REQ-VID-121, REQ-VID-122

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
- Behind `topic_scripts.step_list.enabled` (default false).

## As built

- The setting is `llm_settings.topic_scripts.step_list` (`enabled`, `model`, `max_steps`, `timeout_seconds`) in `config/ai_services.yaml`.
- One grounded call (`prompts/topic_step_list.md`) returns the list as JSON in text: the start screen, the platform, whether the steps fork, the steps, and the source's common mistake. A step whose `source` is not an `http(s)` URL is refused.
- The topic is dropped, as a skipped product, when the call fails, no step is sourced, or the steps fork or exceed `max_steps`; this release sets such a topic aside rather than splitting it into a series.
- The script is written from the steps with `prompts/topic_from_steps.md`, a new prompt beside the existing templates, which stay untouched. Its length is 50-75 words for one or two steps and 100-185 for more; a draft under 80% of the floor is retried, and kept only when no attempt reaches it.
- The list is written to `temp/step_list.json`, and the state carries a one-line `step_list` summary.
- On a live run (Background App Refresh on iPhone) the call returned three steps sourced to Apple's support page, and the script named the start screen and each step's result.
- The topic pool filter (REQ-VID-151) is not part of this release.

## Alternatives considered

None recorded.

## Rollout

Ships off: `topic_scripts.step_list.enabled` defaults to false. Set it after the reach-test readout (#540). The switch gains a check in `tests/test_reach_test_holdout.py` when it lands.

Remove the switch when: `topic_scripts.step_list.enabled` has been on in the bundled config for two weekly batches with topic-render completion (#551) no worse than without it; the key, its holdout check and the free-form script path then go in a minor release with a `**Breaking**:` CHANGELOG entry.

## Open questions

None recorded.
