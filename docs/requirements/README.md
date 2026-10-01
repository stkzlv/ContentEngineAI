# Requirements

What the system must do, one testable statement per id, split by area. How a requirement relates to design docs, decisions and issues is in [the documentation map](../README.md).

| Area | Prefix | Covers |
|---|---|---|
| [Operations](operations.md) | `REQ-OPS` | Configuration, logging, errors, outputs, resources and performance tracking |
| [Scraper](scraper.md) | `REQ-SCR` | Product discovery, search, media extraction and affiliate URLs |
| [Batch](batch.md) | `REQ-BAT` | The global pipeline and the producer's batch mode |
| [Video](video.md) | `REQ-VID` | Assembly, positioning, captions, profiles, stock visuals and topics |
| [Content](content.md) | `REQ-CNT` | Scripts, voices, music and content pillars |
| [Publisher](publisher.md) | `REQ-PUB` | Scheduling, metadata, link-in-bio, analytics and the published registry |
| [Compliance](compliance.md) | `REQ-CMP` | Disclosure on frame, in captions and in platform tags |

## Format

```markdown
- **REQ-PUB-024** `partial` When `--dry-run` is passed, the `schedule` command shows the slot each product would take and publishes nothing.
  - Gap: combined with `--immediate`, it publishes.
```

Each requirement is one bullet: the id in bold, the status in backticks, then one statement in the present tense. Where the behaviour depends on a trigger or a state, the statement follows the EARS forms: "When ...", "If ...", "While ...", "Where ...".

Up to four sub-bullets may follow, each one sentence:

- `Gap:` what a `partial` requirement is missing. Required for `partial`.
- `On when:` the setting that turns the feature on and the condition for doing it. Required for `held`; optional for `planned`.
- `Why:` the reason, only where it stops a likely wrong change.
- `Check:` an acceptance criterion, where the statement alone isn't testable.

## Statuses

| Status | Meaning |
|---|---|
| `shipped` | Implemented as written. |
| `partial` | Implemented, with the gap its `Gap:` line names. |
| `planned #N` | Specified and tracked by issue N, not built. |
| `planned (decision NNNN)` | Decided in a [decision record](../decisions/), not built. |
| `held` | Built behind a switch that ships off ([decision 0002](../decisions/0002-output-changes-ship-off-by-default.md)). |
| `deprecated` | No longer required. The id stays so old citations still resolve. |

## Rules for changing a requirement

- Change the requirement in the same pull request as the code, and cite its id in the pull request.
- Ids are permanent. Never renumber or reuse one: a new requirement takes the next free number in its file, and a removed one becomes `deprecated`.
- Requirements say what, not how. Class and function names, source paths and algorithms belong in the code, design docs and module notes. CLI flags, config keys and output names are user-visible and belong here.

## Tests and coverage

A test cites the requirements whose behaviour it checks with a marker:

```python
@pytest.mark.req("REQ-PUB-024")
def test_dry_run_publishes_nothing() -> None: ...
```

A test that checks a `held` switch stays off cites that requirement too, so the report shows the hold is guarded.

`python -m tools.requirements_coverage --uncited` lists the requirements no test cites. `tests/docs/test_requirements_format.py` checks the format, and fails when a test cites an id that doesn't exist.
