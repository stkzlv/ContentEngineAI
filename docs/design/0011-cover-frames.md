# 0011. Cover frames, including YouTube Shorts thumbnails

- **Status:** Accepted
- **Issue:** #552
- **Requirements:** REQ-VID-014

## Context

No cover is produced. Since July 2026 YouTube accepts custom Shorts thumbnails from Partner Program channels; roadmap item 4.7 in [the roadmap](../roadmap.md) records this.

The technique comes from [creator-research.md](../creator-research.md).

## Goals

- Every render produces a cover image (the hero image and the hook headline inside the centred 3:4 area), set on each platform that accepts one.

## Non-goals

None recorded.

## Design

- After assembly, render `cover.jpg` at 1080x1920 from the frame 0 composition: the hero image and the hook headline, with the headline inside the centred 3:4 area Instagram's grid crops to.
- Pass the cover in the publish payload for every platform the provider accepts one for.

**Tests.** Every render writes a cover of the right size with the headline inside the 3:4 area; the payload carries it where supported.

## Alternatives considered

None recorded.

## Rollout

The cover does not change the feed video, but it changes the profile grid and search presentation for both reach-test arms equally. It ships without a switch and can land before the reach-test readout (#540), once the payload change is verified on one post.

## Open questions

- Whether the provider's API has a Shorts thumbnail field. If it has none, record the gap and keep frame 0 as the YouTube lever.
