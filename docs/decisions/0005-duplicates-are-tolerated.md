# 0005. Duplicate posts from normal operation are tolerated

- **Status:** Accepted
- **Date:** 2026-10-01

## Context and problem

Some normal operations produce a second post or link for the same product: a forced republish, a re-render that is published again, or a link-in-bio entry older than the window the provider's list returns. Treating each of these as a defect would mean stopping to ask, or adding cleanup for outcomes that were asked for.

## Options considered

- **Prevent every duplicate.** It would need full publish history from the provider, which the provider's list endpoints don't return in one page.
- **Tolerate duplicates that an explicit action produces, and keep the guards against accidental ones.**

## Decision

A duplicate produced by an explicit action (`--force`, a knowing re-render) is an expected outcome. The guards against accidental duplicates stay on: the already-published filter before rendering, the publish check unless `--force` is passed, and the link-in-bio check.

## Consequences

- A guard that silently stops working is still a defect, because it is the only protection against an accidental second post.
- `CLAUDE.md` ("Duplicates are acceptable") carries the operating detail.
