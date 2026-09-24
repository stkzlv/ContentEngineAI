# Documentation practices: roadmap, requirements, design and decisions

How software projects, and open-source projects in particular, organize requirements, design documents, decisions and roadmaps, and how this repository's `docs/` compares. Researched September 2026 as input to #562, which holds the open questions for restructuring the roadmap, requirements and design docs. Primary sources were read directly; where a paywalled standard could not be read, that is noted.

## 1. The layers everyone converges on

Standards, big open-source projects and product practice use different names for the same stack. Each layer answers one question, and each links to the one above and below it.

| Layer | Question | Typical artifact | Changes |
|---|---|---|---|
| Goals | Why build anything? | Vision, business case, roadmap | Quarterly |
| Requirements | What must be true? | Requirements spec: stakeholder needs, then system "shall" statements with acceptance criteria | Per feature |
| Design / proposal | How will we do it, and why this way? | Design doc, RFC, KEP, PEP | Per significant change, then frozen |
| Decisions | What did we choose, and what did we reject? | Architecture Decision Records (ADRs) | Append-only |
| Architecture | How does the system fit together now? | Living architecture description (arc42, C4 diagrams) | Kept current |
| Work | Who does what, when? | Issues, milestones, project boards | Daily |
| Verification | How do we know it holds? | Tests and checks traced to requirement ids | With the code |
| User docs | How do people use it? | Tutorials, how-tos, reference, explanation (Diataxis) | With releases |

The failure modes are also consistent. One file ends up doing three layers' jobs, and a finished design doc keeps being edited as if it were the architecture. Or no one records which requirement a test or a PR serves.

## 2. Standards

- **ISO/IEC/IEEE 29148:2018** covers requirements engineering: processes, information items, and what makes a good requirement.
  - It is an active standard: IEEE approved it on 2018-10-23 and published it on 2018-11-30. It superseded the 2011 edition, which had itself replaced IEEE 830 (the classic SRS standard).
  - It separates requirements by level: business or mission (BRS), stakeholder (StRS), system (SyRS) and software (SRS). It also has concept-of-operations documents that describe how the system is used.
  - The characteristics of a good individual requirement, as commonly cited from the standard: necessary, appropriate, unambiguous, complete, singular, feasible, verifiable, correct, conforming. The standard itself is paywalled; this list is from secondary material.
- **ISO/IEC 25010** is the usual vocabulary for quality requirements: performance, reliability, security, maintainability and so on. Cited from general knowledge, not re-read.
- **IEEE 1016** covers software design descriptions. It is heavyweight and rarely used in open source; arc42 and C4 took its place in practice.

## 3. Writing requirements

- **EARS (Easy Approach to Requirements Syntax)**, by Alistair Mavin (Rolls-Royce), constrains each requirement to one of six patterns. The clauses always appear in the same order:
  - Ubiquitous: "The <system> shall <response>".
  - State-driven: "While <precondition>, the <system> shall <response>".
  - Event-driven: "When <trigger>, the <system> shall <response>".
  - Optional feature: "Where <feature is included>, the <system> shall <response>".
  - Unwanted behaviour: "If <trigger>, then the <system> shall <response>".
  - Complex: "While <precondition>, when <trigger>, the <system> shall <response>".

  Because the patterns are fixed, EARS lines are easy for people to review and easy for tools and agents to check.
- **User stories and job stories** ("As a..., I want..., so that...") state needs at the stakeholder level. They are not system requirements; each needs acceptance criteria before it is testable.
- **Acceptance criteria as examples** (BDD, Given/When/Then) turn a requirement into a test. Tools such as Gherkin can run them, but the format pays off even as plain text.
- **Stable ids** (for example `REQ-PUB-012`) make traceability possible: a test, a PR or a spec cites the id, and a script can find a requirement with nothing citing it.

## 4. Design documents and proposals

**Google design docs** (Malte Ubl's write-up):
- Sections: context and scope, goals and non-goals, the design with its trade-offs, alternatives considered, cross-cutting concerns (security, privacy, observability).
- When to write one: if three or more of five questions are "yes" (the design is unclear, senior review would help, it is contentious, cross-cutting concerns get missed, legacy needs explaining).
- When not to: the solution is obvious, or rapid prototyping is the point.
- Lifecycle: draft, review, implement (updating the doc as reality diverges), then keep it as the entry point to that part of the system.

**Shape Up pitches** (Basecamp) have five ingredients:
- **Problem.**
- **Appetite:** how much time it is worth. This is set first and constrains the solution.
- **Solution.**
- **Rabbit holes.**
- **No-gos:** what is explicitly out.

Appetite and no-gos are the parts most templates lack.

## 5. What large open-source projects do

| Project | Artifact | When required | Lifecycle | Where it lives |
|---|---|---|---|---|
| Rust | RFC | "Substantial" changes; not bugfixes, refactors or small additions | PR with a `0000-` number -> review -> Final Comment Period (10 days, disposition merge, close or postpone) -> merged (active) -> tracking issue | `rust-lang/rfcs`, one file per RFC |
| Python | PEP | New features (Standards Track), guidelines (Informational), process changes (Process) | Draft -> Accepted or Provisional -> Final; also Rejected, Withdrawn, Deferred, Superseded, Active. The Steering Council or a delegate decides | `peps` repo; a header preamble carries the status |
| Kubernetes | KEP | Most features, anything controversial or wide-ranging | Stages alpha -> beta -> stable behind feature gates, with graduation criteria and a production-readiness review | `kubernetes/enhancements`, a directory per KEP |
| Go | Proposal | Significant changes to the language, libraries or tools | Issue first. A design doc only if the discussion asks for one. Weekly review group: Incoming -> Active -> Likely Accept or Decline (one-week wait) -> Accepted or Declined | Issues, plus `golang/proposal/design/NNNN-name.md` |

What they share:
- **One file per proposal, numbered, in the repo.** It is reviewed as a pull request and never deleted. A superseded proposal gets a status and a pointer, not an edit.
- **Explicit status in a header**, so readers and tools can tell a draft from an accepted plan from shipped behaviour.
- **A size threshold.** Go's "issue first, design doc only when needed" is the lightest version and the best fit for a small project.
- **Specific sections:**
  - Kubernetes: Drawbacks, Alternatives, Implementation History, and graduation criteria per stage.
  - Python: Rejected Ideas, Backwards Compatibility, Security Implications, How to Teach This.
  - Rust: Unresolved Questions.
- **Feature gates tie the spec to rollout.** A KEP's alpha/beta/GA with graduation criteria is the formal version of "ships off by default until a condition holds".
- **A decision authority is named** (steering council, SIG, review group), so "accepted" means something.

## 6. Decisions and architecture

- **Architecture Decision Records** were proposed by Michael Nygard in 2011.
  - Each ADR records one decision with its context, the options and the consequences. ADRs are append-only: a later ADR supersedes an earlier one.
  - Templates: Nygard's original, MADR (Markdown ADRs, in full, minimal and bare variants), and Y-statements.
  - The recommended location is `docs/decisions/NNNN-title.md`.
- **arc42** is a free template (CC BY-SA 4.0, used since 2005) for the living architecture description, in twelve sections:
  - goals, constraints, context and scope, solution strategy;
  - building-block, runtime and deployment views;
  - cross-cutting concepts, decisions, quality requirements, risks and technical debt, glossary.

  Section 9 is where ADRs plug in.
- **The C4 model** (Simon Brown) gives four zoom levels for diagrams: system context, container, component, code. It is independent of notation and tooling, which makes it a good fit for text-based diagrams (Mermaid, Structurizr) kept in the repo.

## 7. User docs and process files

- **Diataxis** splits user documentation into four types: tutorials, how-to guides, reference and explanation. It is not a place for requirements or design, but it keeps those out of user docs.
- **Docs as code** (Write the Docs): documentation lives in the same repo, as plain-text markup, reviewed in pull requests and checked by CI. That is what makes "the spec changed in the same PR as the code" enforceable.
- **Community health files** GitHub checks on a public repo: README, CODE_OF_CONDUCT, LICENSE, CONTRIBUTING, a security policy, issue templates and a PR template.
- **GOVERNANCE.md** is recommended once roles exist, because decisions made "behind closed doors" erode trust (opensource.guide).

## 8. Specific to open source

1. **The process is public and asynchronous.** Proposals are pull requests, discussion is in the open, and a comment period (Rust's 10 days, Go's one-week "likely" state) gives anyone a fixed window to object.
2. **Issue first, document second.** Most ideas die or get settled in an issue; a design doc is only worth writing for what survives.
3. **Status is visible and machine-readable** (front matter or a header), because outside readers can't ask "is this real yet?".
4. **Rejected and superseded proposals stay.** They stop the same idea being re-proposed, and they preserve why.
5. **Public docs describe capabilities generically.** Private or commercial motivation lives elsewhere. This repository does this with gitignored `*.private.md` overlays (see `CLAUDE.md`).
6. **Licensing covers docs too.** Contributors must be able to reuse and change templates and specs; arc42's CC BY-SA is an example.
7. **The roadmap is a direction, not a promise.** Public roadmaps are usually short (themes, horizons, links to issues). The detail lives in issues and proposals.

## 9. Spec-driven development

- **GitHub's Spec Kit** is an open-source toolkit for "spec-driven development" with coding agents. Its sequence:
  1. A project **constitution**: principles, set once.
  2. Per feature: **specify** (what and why), **plan** (how), **tasks**, **implement**, and **converge** (validate, and repeat until done).
  3. Artifacts are stored as files under a `.specify/` directory.
- This is the same layering as above, compressed per feature. What agents add to the argument:
  - **Specs in files beat specs in chat.** A session can reload them after compaction.
  - **Stable ids and statuses** let an agent check coverage instead of inferring it.
  - **Small, singular requirements** (EARS) are what an agent can verify one by one.

## 10. This repository against these practices

| Layer | Today | Gap |
|---|---|---|
| Goals | `docs/roadmap.md` (513 lines) plus the private overlay | The roadmap carries specification detail. Best practice is a short themes-and-horizons page linking to issues and specs. |
| Requirements | `docs/requirements.md` (560 lines): behaviour bullets, planned items tagged `(planned, #N)` | No stable ids, no acceptance criteria, shipped and planned mixed in one list, no EARS-style form. Nothing can check coverage. |
| Design | `docs/creator-techniques-spec.md`, one file for 21 items | Should be one numbered design doc per feature with a status header (Draft / Accepted / Implemented / Superseded), goals and non-goals, alternatives, and rollout (off-by-default, then the graduation criterion). |
| Decisions | Scattered across `docs/notes/`, the CHANGELOG, the roadmap and the private business `decisions.md` | No ADRs. `docs/notes/` is closer to a defect and lessons log, which is valuable but a different thing. |
| Architecture | `docs/architecture.md` (968 lines) | Could follow arc42's sections, with C4 context and container diagrams in Mermaid. |
| Work | GitHub issues with a `follow-up` label; issue and PR templates exist | Issues don't cite requirement ids; there is no status field linking issue -> design doc -> requirement. |
| Verification | A large test suite, reach-test hold-out tests | Tests aren't traceable to requirements. A marker or naming convention would close that. |
| Rollout | Output-changing features ship off by default until the reach-test readout, guarded by a hold-out test | This is the KEP feature-gate pattern already; it only needs writing down as graduation criteria. |
| User docs | Guides mixed with research and best-practice pages in `docs/` | Diataxis would separate how-to (`installation`, `batch-processing`), reference (`configuration`), explanation (`architecture`) and research. |

**A possible shape.** This is a starting point for #562, not a decision:

```
docs/
  roadmap.md             short: themes, Now / Next / Later, links to issues and design docs
  requirements/          one file per area (publisher, video, scraper...): REQ ids, EARS lines, acceptance criteria, status
  design/NNNN-name.md    one per feature: status header, goals/non-goals, design, alternatives, rollout and graduation
  decisions/NNNN-name.md ADRs (MADR minimal)
  architecture.md        arc42 sections, C4 diagrams in Mermaid
  research/              evidence pages (creator, AI slop, best practices)
  guides/ reference/     user docs by Diataxis type
  notes/                 defect lessons per module (unchanged)
```

Workflow: an issue first; a design doc only past a threshold, as in Go's process. The PR cites the requirement id and the design doc. Tests carry the id. A CI check lists requirements that nothing references. The thresholds, the tracking tool and the migration order are open questions in #562.

## Sources

- IEEE Standards Association, [ISO/IEC/IEEE 29148-2018](https://standards.ieee.org/ieee/29148/6937/). The ISO page returned 403 and the standard's text is paywalled.
- Alistair Mavin, [EARS](https://alistairmavin.com/ears/).
- Malte Ubl, [Design docs at Google](https://www.industrialempathy.com/posts/design-docs-at-google/).
- Basecamp, [Shape Up: Write the pitch](https://basecamp.com/shapeup/1.5-chapter-06).
- [Rust RFCs README](https://github.com/rust-lang/rfcs/blob/master/README.md).
- [PEP 1](https://peps.python.org/pep-0001/).
- [Kubernetes KEPs](https://github.com/kubernetes/enhancements/blob/master/keps/README.md) and the [KEP template](https://github.com/kubernetes/enhancements/blob/master/keps/NNNN-kep-template/README.md).
- [Go proposal process](https://github.com/golang/proposal/blob/master/README.md).
- [adr.github.io](https://adr.github.io/) and [MADR](https://github.com/adr/madr).
- [arc42](https://arc42.org/overview).
- [C4 model](https://c4model.com/).
- [Diataxis](https://diataxis.fr/).
- Write the Docs, [Docs as Code](https://www.writethedocs.org/guide/docs-as-code/).
- GitHub Docs, [community profiles](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/about-community-profiles-for-public-repositories).
- Open Source Guides, [Leadership and governance](https://opensource.guide/leadership-and-governance/).
- GitHub, [Spec Kit](https://github.com/github/spec-kit).
