# 0001. Documentation structure by layer

- **Status:** Accepted
- **Date:** 2026-10-01

## Context and problem

The roadmap, requirements and specs outgrew their files (#562). The roadmap carried specification detail, the requirements had no ids or statuses and mixed shipped behaviour with planned, one spec file covered 21 features, and decisions were spread across notes, the CHANGELOG and the roadmap. A review of the requirements against the code had to infer each item's status by hand, and no test could be traced to the requirement it checks. The practices compared are summarised under Background.

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

## Background

Standards, large open-source projects and product practice use different names for the same layers. Each answers one question and links to the layers above and below it.

| Layer | Question | Typical artifact | Changes |
|---|---|---|---|
| Goals | Why build anything? | Vision, roadmap | Quarterly |
| Requirements | What must be true? | "Shall" statements with acceptance criteria | Per feature |
| Design | How, and why this way? | Design doc, RFC, KEP, PEP | Per change, then frozen |
| Decisions | What was chosen and rejected? | Architecture decision records | Append-only |
| Architecture | How does it fit together now? | arc42 description, C4 diagrams | Kept current |
| Work | Who does what, when? | Issues, milestones | Daily |
| Verification | How do we know it holds? | Tests traced to requirement ids | With the code |
| User docs | How do people use it? | Tutorials, how-tos, reference, explanation | With releases |

The common failures are one file doing several layers' jobs, a finished design doc edited as if it were the architecture, and nothing recording which requirement a test or a pull request serves.

What this repository takes from each source:

- **Requirements.** ISO/IEC/IEEE 29148:2018 lists the qualities of a good requirement (necessary, unambiguous, singular, feasible, verifiable, among others). EARS gives each requirement one of six fixed sentence forms ("When ...", "If ...", "While ...", "Where ..."), which people and tools can both check. Stable ids let a test or a pull request cite a requirement and let a script find one nothing cites.
- **Design docs.** Google's design docs carry context, goals and non-goals, the design and its trade-offs, and alternatives. Rust RFCs, Python PEPs, Kubernetes KEPs and Go proposals share one file per proposal, a status in a header, and a size threshold. Go's "issue first, design doc only when the discussion asks for one" is the lightest and fits a small project. A KEP's feature gates with graduation criteria are the formal form of "ships off until a condition holds".
- **Decisions.** Michael Nygard's architecture decision records, in the MADR minimal template, one per decision, superseded rather than edited.
- **Architecture.** arc42's sections for the living description, with C4 context and container diagrams.
- **User docs.** Diataxis separates tutorials, how-to guides, reference and explanation, which keeps requirements and design out of user docs.
- **Open source specifically.** The process is public and asynchronous, status is machine-readable, rejected and superseded proposals stay, public docs describe capabilities generically, and the roadmap is a direction rather than a promise.
- **Agents.** Spec-driven development (GitHub's Spec Kit) compresses the same layering per feature. Specs in files survive a lost session, and ids with statuses let an agent check coverage instead of inferring it.

Sources:

- IEEE Standards Association, [ISO/IEC/IEEE 29148-2018](https://standards.ieee.org/ieee/29148/6937/). The standard's text is paywalled; the list of qualities is from secondary material.
- Alistair Mavin, [EARS](https://alistairmavin.com/ears/).
- Malte Ubl, [Design docs at Google](https://www.industrialempathy.com/posts/design-docs-at-google/).
- Basecamp, [Shape Up: Write the pitch](https://basecamp.com/shapeup/1.5-chapter-06).
- [Rust RFCs](https://github.com/rust-lang/rfcs/blob/master/README.md), [PEP 1](https://peps.python.org/pep-0001/), [Kubernetes KEPs](https://github.com/kubernetes/enhancements/blob/master/keps/README.md), [Go proposal process](https://github.com/golang/proposal/blob/master/README.md).
- [adr.github.io](https://adr.github.io/) and [MADR](https://github.com/adr/madr).
- [arc42](https://arc42.org/overview) and the [C4 model](https://c4model.com/).
- [Diataxis](https://diataxis.fr/) and Write the Docs, [Docs as Code](https://www.writethedocs.org/guide/docs-as-code/).
- Open Source Guides, [Leadership and governance](https://opensource.guide/leadership-and-governance/).
- GitHub, [Spec Kit](https://github.com/github/spec-kit).
