# 0027. Instagram AI label

- **Status:** Held
- **Issue:** #787
- **Requirements:** REQ-CMP-025

## Context

The publisher can set an AI label on TikTok (`tiktok_settings.video_made_with_ai`) and YouTube (`synthetic_media_disclosure`), but sends nothing for Instagram, so an installation that labels its posts can't do it there.

Evidence ([evidence grades](README.md#evidence-grades)):

- Meta requires its disclosure "when they post organic content with a photorealistic video or realistic-sounding audio that was digitally created or altered", and may apply penalties when it's missing. [A] [Meta newsroom, February 2024](https://about.fb.com/news/2024/02/labeling-ai-generated-images-on-facebook-instagram-and-threads/)
- A self-disclosed post shows the "AI info" label. [A] [Meta newsroom, April 2024 with later updates](https://about.fb.com/news/2024/04/metas-approach-to-labeling-ai-generated-content-and-manipulated-media/)
- The publishing provider takes an Instagram `isAiGenerated` flag in the platform data, set at creation only. Its SDK documents the field in `InstagramPlatformData` from a release later than the one locked here. [B] The provider SDK's generated models.

## Goals

- An installation can send Instagram's AI label on every Instagram post.
- The label reaches every payload branch, beside the first comment and the cover.

## Non-goals

- Labelling a post that is already published: the provider can't.
- Deciding for an installation whether its renders meet Meta's bar.

## Design

A top-level boolean `instagram_ai_label` in `config/publisher.yaml`, default `false`, read into `PublisherConfig` and passed by `create_publisher_from_config` (the factory both the publisher CLI and the global batch use) to `LatePublisher`. `_build_sdk_platforms` sets `platformSpecificData.isAiGenerated` on every Instagram entry after the per-platform branch, so the shared-content path and the per-platform path both carry it, merged with any first comment, and the cover merges in after it. A value that isn't a boolean is refused at load, as every publisher switch is (REQ-PUB-160).

Tests in `tests/publisher/test_instagram_ai_label.py` cover the default, both values in both branches, the merge with the first comment and the cover, that other platforms don't carry it, the factory wiring, and the refusal.

## Alternatives considered

- Upgrade the provider SDK first, as the issue proposed. The SDK passes `platformSpecificData` through as a raw dict, so the locked release sends the field unchanged; an upgrade across many releases is a separate dependency change.
- Warn and fall back on an invalid value, as the TikTok section does. Rejected: top-level publisher switches are refused at load (REQ-PUB-160), and a silent fallback here would publish posts that can never be labelled.
- An `instagram_settings` section. Rejected for one field; the YouTube flag is a top-level key too.

## Rollout

Ships off. An installation turns it on with `instagram_ai_label: true` in its private publisher overlay, when its renders meet Meta's bar (a realistic-sounding TTS voice can) or when it labels its posts voluntarily.

Remove the switch when: never; whether a render needs the label depends on the installation's voice and footage, so the switch is a lasting option.

## Open questions

None recorded.

## As built

As designed. The field name matches the provider's documented `isAiGenerated`; the locked SDK has no model for it to check against, so a real post with the switch on is the check that Instagram applied the label.
