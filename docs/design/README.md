# Design docs

A design doc says how a feature works and why it works this way. It sits between a requirement, which says what must be true ([the requirements index](../requirements/README.md)), and the code. Where each kind of document belongs is in [the documentation map](../README.md).

## When to write one

Open an issue first; most work is specified in the issue alone. Write a design doc when the work:

- takes more than a few days,
- adds an external dependency, or
- changes config or a data schema.

## Format

One numbered file per feature, `NNNN-short-title.md`, with a status header (status, issue, requirement ids) and the sections Context, Goals, Non-goals, Design, Alternatives considered, Rollout and Open questions. A section with nothing to say reads "None recorded."

## Statuses

| Status | Meaning |
|---|---|
| `Draft` | The design itself is undecided, for example an evaluation. |
| `Accepted` | Specified and agreed, not built. |
| `Implemented` | The feature has shipped. |
| `Superseded by NNNN` | A later design doc replaces this one. |

A design doc is frozen once its feature ships: its status becomes `Implemented`, and a later change to the feature gets a design doc of its own that supersedes it.

## Rules that apply to every design

A feature that changes rendered output ships off by default ([decision 0002](../decisions/0002-output-changes-ship-off-by-default.md)). The designs below follow these rules on top of it:

- **Off by default when it changes output.** The format-vs-format reach test needs the script, voice and sound held constant until its readout (#540). Anything that changes what a render looks or sounds like ships behind a switch that defaults to today's behaviour, and `tests/test_reach_test_holdout.py` gains a check for it. Measurement-only work ([0006](0006-render-choices-and-variety-report.md), [0010](0010-first-seconds-metrics.md)) can land at any time.
- **Byte-identical when off.** Each design's off state must leave the FFmpeg command, the prompt or the payload exactly as it is today, pinned by a test.
- **Seeded variation.** A choice drawn per render uses a salted MD5 of the product id (`<product_id>:<purpose>`), the pattern the CTAs and pauses already use (fonts and voices hash the bare product id with different slices). A product renders the same way every time, and a batch varies.
- **Record what was chosen.** Every drawn choice goes into `pipeline_state.json` beside `script_template` and `cta`, and is mirrored into the step entry so a truncating resume keeps it. [0006](0006-render-choices-and-variety-report.md) turns these records into a variety report, and [0010](0010-first-seconds-metrics.md) segments metrics by them.
- **Measure before enabling.** Each production design lists, under Rollout, the check that decides whether to enable it after the readout. Most of the evidence is creator opinion or ad research, so the pipeline's own analytics decide.

## Evidence grades

Each design's Context lists the evidence behind it, and the best-practice guides cite theirs the same way. Most "technique X adds N% retention" figures online come from tool vendors and publish no method, so every claim carries a grade:

| Grade | Meaning |
|---|---|
| **A** | Peer-reviewed or registered research, or a platform's own policy or documentation |
| **B** | A large observational dataset with a stated method, a preprint or working paper, agency research, or a platform's first-party ad research. Correlation, not cause. |
| **C** | A creator's or vendor's claim, often with one data point or no method |

Two cautions apply throughout:

- **Ad research is not organic research.** TikTok's creative studies measure paid ads (recall, awareness), not how organic posts are distributed.
- **Several key sources predate 2025.** Each is flagged with its year where it is cited.

The evidence was gathered in September 2026. The tutorial designs draw on [the tutorials explanation](../explanation/tutorials.md).

## Index

| Number | Title | Issue | Status |
|---|---|---|---|
| [0001](0001-motion-on-every-still.md) | Motion on every still | #542 | Accepted |
| [0002](0002-end-on-the-peak.md) | End on the peak, with an optional seamless loop | #543 | Accepted |
| [0003](0003-sparse-sound-effects.md) | Sparse event sound effects | #544 | Accepted |
| [0004](0004-voice-processing-chain.md) | Optional voice processing chain | #545 | Accepted |
| [0005](0005-beat-snapped-cuts.md) | Snap visual cuts to music beats | #546 | Accepted |
| [0006](0006-render-choices-and-variety-report.md) | Record render choices and report output variety | #547 | Accepted |
| [0007](0007-script-lint.md) | Script lint and search-phrase placement | #548 | Accepted |
| [0008](0008-bait-free-closing-lines.md) | Remove engagement-bait lines from the CTA pools | #549 | Accepted |
| [0009](0009-product-titles.md) | YouTube titles for products | #550 | Accepted |
| [0010](0010-first-seconds-metrics.md) | First-seconds metrics in the analytics sweep | #551 | Accepted |
| [0011](0011-cover-frames.md) | Cover frames, including YouTube Shorts thumbnails | #552 | Accepted |
| [0012](0012-clean-product-images.md) | Prefer clean product images over seller infographics | #554 | Accepted |
| [0013](0013-stock-clip-reuse-guard.md) | Do not reuse stock clips across recent renders | #555 | Accepted |
| [0014](0014-tts-text-normalisation.md) | Normalise numbers, units and model names before TTS | #556 | Accepted |
| [0015](0015-narrator-voice-evaluation.md) | Evaluate a distinctive or owned narrator voice | #557 | Draft |
| [0016](0016-tiktok-ai-label.md) | Revisit the TikTok AI label | #558 | Accepted |
| [0017](0017-tutorial-step-lists.md) | Sourced step list | #559 | Accepted |
| [0018](0018-tutorial-step-visuals.md) | A visual per step | #560 | Accepted |
| [0019](0019-tutorial-graphics.md) | Explanatory graphics | #561 | Accepted |
| [0020](0020-analytics-history.md) | Analytics history | none | Accepted |
