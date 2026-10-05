# 0012. A keyword belongs to one pillar

- **Status:** Accepted
- **Date:** 2026-10-05

## Context and problem

`batch.keywords` groups keywords by content pillar, and a scraped product carries the pillar of the keyword that found it. A keyword listed under two pillars kept only the last group's pillar, silently, and was searched once per listing (#584). `REQ-CNT-107` asked for the keyword to carry each pillar, which a product record cannot do: it has one pillar, and the template pool, the preamble and the audience all follow from it.

## Options considered

- **Let a product carry several pillars.** Matches the old requirement, and changes the product model and every place that reads a product's pillar.
- **Refuse the config.** A keyword sits under one pillar; listing it twice is an error to fix in the config.

## Decision

Refuse it. Config loading fails with an error naming the keyword and both pillars, matching keywords after normalization (case and spacing aside). A repeat under the same pillar is not a conflict.

## Consequences

The product model keeps one pillar. A config that listed a keyword twice no longer loads until one listing goes; the bundled config has none.
