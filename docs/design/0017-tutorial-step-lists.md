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

## Alternatives considered

None recorded.

## Rollout

Ships off: `topic_scripts.step_list.enabled` defaults to false. Set it after the reach-test readout (#540). The switch gains a check in `tests/test_reach_test_holdout.py` when it lands.

Remove the switch when: `topic_scripts.step_list.enabled` has been on in the bundled config for two weekly batches with topic-render completion (#551) no worse than without it; the key, its holdout check and the free-form script path then go in a minor release with a `**Breaking**:` CHANGELOG entry.

## Open questions

None recorded.
