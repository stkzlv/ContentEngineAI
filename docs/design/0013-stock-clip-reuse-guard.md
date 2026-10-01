# 0013. Do not reuse stock clips across recent renders

- **Status:** Accepted
- **Issue:** #555
- **Requirements:** REQ-VID-110

## Context

Stock candidates come from the provider search and the relevance judge with no memory of earlier renders.

The finding comes from [ai-slop-research.md](../ai-slop-research.md).

## Goals

- A stock clip used in a recent render is excluded while alternatives exist.
- The ids each render uses are recorded so the rule survives cleanup.

## Non-goals

None recorded.

## Design

- A small append-only store under `outputs/state/` of `(stock_id, product_id, used_at)`, written when a render finishes, outside the product directory so cleanup does not remove it.
- Before judging, drop candidates used within `stock_reuse_window` renders (default 30). If fewer than the needed count remain, fill from the dropped set, least recently used first, and log it.
- `stock_reuse_guard.enabled` (default false).

**Tests.** A candidate used within the window is excluded; the fallback fills from the least recently used; the store survives a product directory's deletion; off records ids but leaves the candidate pool as today.

## Alternatives considered

None recorded.

## Rollout

The guard changes which clips the topic arm shows, and the protocol holds each arm's visuals constant, so it ships off: `stock_reuse_guard.enabled` defaults to false and is set after the reach-test readout (#540). The id store records from day one, so the window is full when the guard is switched on.

## Open questions

None recorded.
