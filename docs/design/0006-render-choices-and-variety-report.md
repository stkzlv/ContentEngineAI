# 0006. Record render choices and report output variety

- **Status:** Accepted
- **Issue:** #547
- **Requirements:** REQ-PUB-083

## Context

`pipeline_state.json` records `script_template`, `cta`, the voice profile, `signoff` and some subtitle choices; the registry records `content_format`.

Every drawn choice in the other designs goes into `pipeline_state.json` (see [the rules every design follows](README.md#rules-that-apply-to-every-design)). This design turns those records into a variety report, and [0010](0010-first-seconds-metrics.md) segments metrics by them.

## Goals

- Every choice that shapes a render is recorded.
- A report shows their distribution over recent renders, with an alert when one value dominates or two scripts are near-identical.

## Non-goals

- Blocking a script for similarity. The check warns, it does not block.

## Design

- Record every choice that shapes a render in the state: hook archetype, caption template, music track id, motion moves, transitions, effect variants, voice chain. Carry them into the published-products registry at publish time, as `content_format` is.
- A `variety` report (a report type beside the existing analytics reports): for the last N published renders (default 14), the distribution per dimension, and an alert when one value exceeds a share threshold (default 60%) where the pool has more than one option.
- A script similarity check: character 5-gram Jaccard similarity between the generated script and each of the last N scripts, with a warning above a threshold (default 0.5) logged at generation time and counted in the report. Warn, do not block.

**Tests.** A render records every listed choice; the report flags a dimension dominated by one value in a fixture; two near-identical scripts cross the similarity threshold and two unrelated ones do not.

## Alternatives considered

None recorded.

## Rollout

Measurement only. It doesn't change rendered output, so it ships on and can land before the reach-test readout (#540).

## Open questions

None recorded.
