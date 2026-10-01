# 0002. Output-changing features ship off by default

- **Status:** Accepted
- **Date:** 2026-10-01

## Context and problem

The project compares content formats by their reach, which only means something if each arm's script, voice, visuals and packaging stay constant for the length of the comparison. Features that change what a render looks or sounds like keep arriving while a comparison runs.

## Options considered

- **Ship each feature on.** Fastest, but every change moves the baseline and the comparison can no longer attribute a difference.
- **Hold the features on branches until the comparison ends.** Keeps the baseline, but the branches drift from `main` and merge late.
- **Merge behind a switch that defaults to the existing behaviour.**

## Decision

Merge behind a switch that defaults to the existing behaviour. `tests/test_reach_test_holdout.py` checks that each such switch stays off in the bundled config. Measurement-only work can ship on at any time.

## Consequences

- Each held feature needs a written condition for turning it on, and another for removing the switch. Its design doc's rollout section carries both, and its requirement has the status `held`.
- Features are turned on in stages after the readout, so each one's effect can be measured on its own.
