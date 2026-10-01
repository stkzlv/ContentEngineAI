# 0001. Documentation structure by layer

- **Status:** Accepted
- **Date:** 2026-10-01

## Context and problem

The roadmap, requirements and specs outgrew their files (#562). The roadmap carried specification detail, the requirements had no ids or statuses and mixed shipped behaviour with planned, one spec file covered 21 features, and decisions were spread across notes, the CHANGELOG and the roadmap. A review of the requirements against the code had to infer each item's status by hand, and no test could be traced to the requirement it checks. The options are compared in `docs/documentation-practices-research.md`.

## Options considered

- **Keep the flat layout and fix it in place.** Cheapest, but the same mixing returns, because nothing separates a frozen spec from a living requirement.
- **One folder per feature holding its spec, plan and tasks.** Duplicates what issues already track.
- **A wiki.** Not reviewed in pull requests, so it can't change in the same pull request as the code.
- **Separate folders by layer**, as large open-source projects do (proposals, decision records, living architecture), with the lightest proposal process: issue first, design doc only past a threshold.

## Decision

Separate folders by layer, as mapped in `docs/README.md`. Requirements get stable ids (`REQ-<AREA>-NNN`) and statuses, design docs are numbered and frozen once implemented, and decisions are append-only records in this folder. User docs are split into guides, reference and explanation as each page is next changed.

## Consequences

- Every pull request that changes behaviour updates a requirement's status and cites its id.
- Moving pages breaks links, so a test checks that every relative link and every cited `docs/` path resolves.
- The requirements and the spec are split in one pass each; the large user guides move only when they are next edited.
