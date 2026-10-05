# 0013. Do not reuse stock clips across recent renders

- **Status:** Held
- **Issue:** #555
- **Requirements:** REQ-VID-110

## Context

Stock candidates come from the provider search and the relevance judge with no memory of earlier renders.

Evidence ([evidence grades](README.md#evidence-grades)):

- YouTube's policy names "generic or unoriginal templates giving the impression of mass production"; identical zooms, mismatched stock and the same clips repeated are the specifics. [A for the policy, C for the specifics] [YouTube inauthentic-content policy](https://support.google.com/youtube/answer/1311392)

## Goals

- A stock clip used in a recent render is excluded while alternatives exist.
- The ids each render uses are recorded so the rule survives cleanup.

## Non-goals

None recorded.

## Design

- A small append-only store under `outputs/state/` of `(stock_id, product_id, used_at)`, written when a render finishes, outside the product directory so cleanup does not remove it.
- Before judging, drop candidates used within the last `window` renders (default 30). If fewer than the needed count remain, fill from the dropped set, least recently used first, and log it.
- `stock_reuse_guard.enabled` (default false).

**Tests.** A candidate used within the window is excluded; the fallback fills from the least recently used; the store survives a product directory's deletion; off records ids but leaves the candidate pool as today.

## As built

- The ids live in `state/render_choices.jsonl`, the per-render store of design 0006, as a `stock_ids` list on each row, rather than a second store. It is written when a render finishes, sits outside the product directories, and keeps one row per product, so a rerun replaces that product's ids instead of counting twice.
- The settings are `stock_media_settings.stock_reuse_guard` with `enabled` and `window` (default 30), rather than a separate `stock_reuse_window` key.
- An id is `<source>:<provider id>`, lowercased source.
- With the relevance judge on, the judge scores the whole page and the guard only orders the result: candidates at or above `min_score` come first, fresh ones by score and then reused ones least recently used first, and those below the floor only fill a shortfall in the same order. So a fresh irrelevant clip never displaces a reused relevant one (REQ-VID-108). Without the judge, recently used candidates are dropped before the random sample, with the least recently used filling a shortfall.
- The batch's background prefetch builds its fetcher with no secrets, so it never reaches the provider and is left without the guard.

## Alternatives considered

None recorded.

## Rollout

The guard changes which clips the topic arm shows, and the protocol holds each arm's visuals constant, so it ships off: `stock_reuse_guard.enabled` defaults to false and is set after the reach-test readout (#540). The id store records from day one, so the window is full when the guard is switched on.

Remove the switch when: `stock_reuse_guard.enabled` has been on in the bundled config for 30 days with the [0006](0006-render-choices-and-variety-report.md) report showing fewer repeated stock clips than before and no render short of clips; the key and the unguarded path then go in a minor release with a `**Breaking**:` CHANGELOG entry, and `window` stays as the tuning.

## Open questions

None recorded.
