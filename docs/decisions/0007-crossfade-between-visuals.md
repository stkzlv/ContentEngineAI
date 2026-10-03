# 0007. Visuals join with a short crossfade, not a hard cut

- **Status:** Accepted
- **Date:** 2026-10-03

## Context and problem

The promotional-video research lists the hard cut as the default transition in short-form editing and keeps zooms and whips for emphasis. The render joins every pair of consecutive visuals with a 0.5 s crossfade instead (`REQ-VID-005`, `transition_duration_sec` and the profiles' `video_transition_duration`), and the same value also sets image segment durations and caption segment boundaries. Nothing recorded why, so the choice kept coming back as a gap.

## Options considered

- **Hard cut by default, crossfade per profile.** Matches the editing convention the research describes. It changes every render's look during the reach test, and a slideshow of stills cut hard reads as a slideshow, which the platforms' originality rules name.
- **Keep the crossfade, recorded.** TikTok's ad coding found seamless transitions gave about 14% more view time (grade B, paid ads, `docs/explanation/promotional-videos.md`). It also smooths the still-image formats the pipeline renders most.
- **Vary the transition per render.** Adds a variant dimension before the variety report (design 0006) can show whether it helps.

## Decision

Keep the 0.5 s crossfade as the default join between visuals. A profile can change its length.

## Consequences

- The research's hard-cut convention is recorded as considered and not adopted, with the reason.
- A hard-cut profile, or transitions as a variant dimension, is a candidate for the variety work (design 0006) and is measured against the hold-out before it becomes a default (decision 0002).
