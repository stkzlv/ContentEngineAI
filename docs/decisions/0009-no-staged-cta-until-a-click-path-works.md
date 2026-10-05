# 0009. The call to action is spoken, not staged, until a click path works

- **Status:** Accepted
- **Date:** 2026-10-05

## Context and problem

The promotional-video research names CTA staging as one of its six rules: a soft on-frame call to action at 3-5 s, and a hard one at the end, larger, in an accent colour and on screen for at least 1.5 s ([promotional videos](../explanation/promotional-videos.md)). The render speaks the call to action as the script's last sentence and captions it like the narration. #103 specified the staging and was closed without it, and only its closing comment said why, so the gap kept coming back.

## Options considered

- **Stage both calls to action now.** Follows the research. It optimises the step from a viewer to a link, and on the largest surface there is no such step: a YouTube Short cannot carry a clickable link on any per-video surface, and TikTok and Instagram captions link nowhere. It also changes every render during the reach test ([decision 0002](0002-output-changes-ship-off-by-default.md)).
- **Keep the spoken call to action and point it at the profile.** The link-in-bio page is the one destination every platform can reach. The pool rework in #549 moves the wording there.

## Decision

The call to action stays spoken and captioned, with no separate on-frame staging. Revisit when a click path from a video to a destination is shown to work, and then specify the staging as a design doc behind a switch.

## Consequences

Renders keep one caption style for the whole video. The research's soft and hard CTA, the first-quarter placement and the accent colour are not requirements. The end-card question is separate and held in [design 0002](../design/0002-end-on-the-peak.md).
