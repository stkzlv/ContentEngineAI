# 0014. The reach-test hold ends when its posts are queued

- **Status:** Accepted
- **Date:** 2026-10-09

## Context and problem

[Decision 0002](0002-output-changes-ship-off-by-default.md) held every output change behind a switch until the reach test read out, and kept the prompt templates byte-identical, so each arm's script, voice and visuals stayed constant. The test's last post is rendered and scheduled for 2026-10-10, and the readout on 2026-10-17 reads only posts already queued. A render made now cannot enter the test, so the hold protects nothing and keeps about twenty finished features off. Tracked in #540.

## Options considered

- **Keep the hold until the readout.** Protects nothing the queue doesn't already fix, and costs another week of renders without the features.
- **End the hold and turn the features on in stages,** one at a time with a measured batch each. Attributes each effect, but takes months at one post a day.
- **End the hold and turn the features on now.** Ships them all; their effects can't be told apart in the analytics.

## Decision

End the hold now, and turn the held features on as each is verified on a real render. The owner accepts that their individual effects won't be separable. A queued test post is never re-rendered: that would put the new output into the test.

## Consequences

- Output changes, prompt edits and published-caption changes can ship on. A switch that exists stays in the config, so a feature can still be turned off.
- `tests/test_reach_test_holdout.py` loses each check as its feature is turned on.
- [Decision 0013](0013-tiktok-ai-label-goes-off-after-the-readout.md) applies now: the TikTok AI label goes off with the other switches.
- A feature that needs an input the project doesn't ship (an audio file, a table of misread words) stays `held` until it has one.
