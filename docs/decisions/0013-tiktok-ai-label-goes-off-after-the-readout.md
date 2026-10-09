# 0013. The TikTok AI label goes off after the reach-test readout

- **Status:** Amended by [0014](0014-the-reach-test-hold-ends-when-its-posts-are-queued.md)
- **Date:** 2026-10-05

## Context and problem

Every TikTok post carries the AI-generated-content label (`tiktok_settings.video_made_with_ai`, on by default). TikTok's 2026-H2 guidelines, effective 24 September 2026, do not require it for generic text-to-speech narration that is not a recognizable voice of a known individual, and require it for AI or edits showing realistic people or scenes, or audio mimicking a real person ([TikTok integrity and authenticity](https://www.tiktok.com/community-guidelines/en/integrity-authenticity)). These renders use a generic TTS voice over real product photos and stock footage. The label has a cost: viewers can turn AI content down, and a study of about a million posts found disclosure cut engagement by 7-8% ([design 0016](../design/0016-tiktok-ai-label.md)). #558 asked whether to keep it voluntarily.

## Options considered

- **Keep the label on voluntarily.** Costs reach the rule does not ask for.
- **Turn it off now.** Changes both reach-test arms mid-test ([decision 0002](0002-output-changes-ship-off-by-default.md)).
- **Turn it off after the readout.** Follows the rule without disturbing the test.

## Decision

Keep the label on until the reach-test readout (#540), so both arms carry it, then set `tiktok_settings.video_made_with_ai: false`. Turn it back on for any render that adds AI or edits showing realistic people or scenes, or a voice that mimics a real person.

## Consequences

Until the readout the label is a documented, voluntary choice (`REQ-CMP-018`). After it, TikTok disclosure follows the platform's rule (`REQ-CMP-017`). A future render type that meets TikTok's bar needs the label back on for those posts.
