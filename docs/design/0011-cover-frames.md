# 0011. Cover frames, including YouTube Shorts thumbnails

- **Status:** Accepted
- **Issue:** #552
- **Requirements:** REQ-VID-014, REQ-PUB-154 to REQ-PUB-159

## Context

The render writes a cover per video (see "As built"); no platform is sent it. Each platform takes a cover differently, and the publishing provider exposes only part of what the platforms do.

Evidence ([evidence grades](README.md#evidence-grades)):

- The swipe feeds autoplay the video, so a cover matters for the profile grid, search and the Reels tab rather than feed reach. YouTube says Shorts thumbnails don't appear in the Shorts feed, only in search, hashtag and audio pages, the channel page and TV browse. [A, relayed] [PPC Land on Creator Insider](https://ppc.land/youtube-ends-2-year-wait-for-shorts-thumbnails-but-blocks-a-b-testing/)
- Instagram's profile grid shows 3:4 tiles since January 2025, cut from the centre of a 9:16 frame, so about 240 px is lost at the top and bottom of a 1080x1920 cover. [A for the change, C for the crop] [Business Today](https://www.businesstoday.in/technology/news/story/instagram-head-announces-big-changes-3-minute-reels-and-new-look-for-profile-grid-461527-2025-01-21), [Kapwing](https://www.kapwing.com/resources/instagrams-new-grid-layout-size-and-dimensions-2025/)
- TikTok's profile grid also crops near 3:4; TikTok publishes no crop spec. [C] [Moda](https://moda.app/resources/sizes/tiktok-thumbnail)
- A video must confirm at once what its thumbnail promised, so the cover and the opening match. YouTube removes thumbnails that mislead viewers about what is in the video. [C] [MrBeast production memo](https://simonwillison.net/2024/Sep/15/how-to-succeed-in-mrbeast-production/), [A] [YouTube thumbnail policy](https://support.google.com/youtube/answer/9229980)
- Instagram recommends against reels whose text covers most of the screen; that this also applies to a cover is an inference. [A] [Instagram creators blog](https://creators.instagram.com/blog/instagram-recommendations-eligibility-tips-creators)
- Text, faces and contrast on covers: only practitioner advice, no study with a method was found. [C]

What each platform and the provider accept:

| | Instagram Reels | TikTok | YouTube Shorts |
|---|---|---|---|
| Platform API | A cover image (`cover_url`) or a frame offset (`thumb_offset`, default the first frame); the image wins when both are set [A] | A frame only (`video_cover_timestamp_ms`; unset or invalid falls back to the first frame) [A] | `thumbnails.set` uploads an image; Google's docs don't say whether it applies to Shorts [A] |
| Image limits | JPEG, sRGB, 8 MB at most, 9:16 recommended, on a public server [A] | none (frame only) | JPEG or PNG, 50 MB at most since September 2026, 9:16 for Shorts [A] |
| Provider field | `instagramThumbnail` (alias `reelCover`) or `thumbOffset` | `videoCoverTimestampMs` (default 1 s), or `videoCoverImageUrl` | none: "Custom thumbnails for regular videos only" |
| Changing it later | not with the installed provider SDK, which edits a post's text, schedule and settings but not its platform data | the same | not for Shorts |

Sources: [Instagram IG User Media reference](https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/media), [TikTok Direct Post reference](https://developers.tiktok.com/doc/content-posting-api-reference-direct-post), [YouTube custom thumbnails](https://support.google.com/youtube/answer/72431), [YouTube Data API revision history](https://developers.google.com/youtube/v3/revision_history), and the provider's [Instagram](https://docs.zernio.com/platforms/instagram), [TikTok](https://docs.zernio.com/platforms/tiktok) and [YouTube](https://docs.zernio.com/platforms/youtube) pages and its SDK's platform data models.

The provider's TikTok image cover is not TikTok's: TikTok's API takes no image. For accounts connected through TikTok's developer app the provider "stitches it in as a single frame at the start of the video", which changes the feed video. Accounts connected through the TikTok for Business app, which every new connection uses, are given the image as the cover without stitching; an account keeps the lane it was connected on until it is reconnected. [A] [provider TikTok page](https://docs.zernio.com/platforms/tiktok)

## Goals

- Every render produces a cover: the frame viewers see while the hook headline is on screen, with the headline inside the centred 3:4 area.
- The publisher sets that cover on Instagram and TikTok, the same picture on both, without changing the feed video.

## Non-goals

- A YouTube Shorts cover: the provider can't set one, and the Shorts feed doesn't show it.
- A separately designed cover image (a different picture or text from the video's opening): a cover that differs from the opening breaks the promise the video has to confirm.
- Changing the cover of a post already scheduled or published.

## Design

**Render** (REQ-VID-014). As built, below.

**Publish** (REQ-PUB-154 to REQ-PUB-159), behind `covers.enabled` in the publisher config, bundled off:

- Read `<video>.cover.json` beside the video being published. No record, or no image, means no cover and a warning (REQ-PUB-159).
- **Instagram:** upload `<video>.cover.jpg` through the provider's media upload (a cover is about 0.3 MB, within the direct upload's 4 MB) and send the returned URL as `instagramThumbnail` in the Instagram platform data.
- **TikTok:** send the record's `at_ms` as `videoCoverTimestampMs`. Never `videoCoverImageUrl`: which lane an account is on isn't visible to the publisher, and on the developer-app lane the image becomes a frame of the video. The frame at `at_ms` is the same picture as the image, so both platforms show one cover.
- **YouTube:** nothing.
- Every path that builds a post sends it the same way (REQ-PUB-158). The platform data is assembled in one place, the provider client's platform builder, where Instagram's platform data carries only its first comment, so the cover merges into it beside that; the cover reaches it from the paths that upload a video: `single` and the batch through the shared publish modes, and the scheduler's unified and per-platform posts.

**Tests.** The payload carries `instagramThumbnail` and `videoCoverTimestampMs` from a fixture record on each path and in both metadata modes; YouTube's entry carries no cover; a missing record or a failed upload still publishes; with the switch off nothing changes. Driven at the call site that builds the post, not at a helper.

## Alternatives considered

- **TikTok image cover (`videoCoverImageUrl`).** Rejected: on developer-app connections it prepends a still frame to the video. The timestamp shows the same picture on every lane.
- **Instagram frame offset (`thumbOffset`) instead of the image.** Equivalent picture, one upload fewer, but it relies on Instagram's frame extraction matching ours. Kept as the fallback if the image upload proves unreliable.
- **YouTube thumbnail through the provider's post-publish metadata update.** Rejected: the provider documents custom thumbnails for regular videos only, and a Shorts thumbnail misses the feed where views come from.
- **A separate frame-0 composition rendered for the cover** (the original design). Rejected when built: a frame of the finished video carries the burned captions and the exact picture viewers see.

## Rollout

The switch ships off, since a cover changes how posts appear on the profile and in search. It turns on after one real post on each of Instagram and TikTok shows the rendered cover; the provider's field descriptions are the only evidence that it reaches the platform as described. A queued post is never changed.

## Open questions

- Whether the TikTok cover timestamp lands on the same frame as ffmpeg's seek: TikTok may round to a keyframe. The verification post answers it.
- Whether Instagram uses the uploaded cover for the profile grid as well as the Reels tab: the API names only the Reels tab.
- With the hook overlay off, the cover has no headline (REQ-VID-014's Gap); whether such renders should send a cover at all.

## As built

- The cover is a frame of the finished video, after the burned captions, taken halfway through the hook (0.75 s with the bundled 1.5 s hook), not a separate frame 0 composition: the hook headline is fully on screen there and the picture is the one viewers see.
- Each video gets its own cover beside it, `<video>.cover.jpg`, because a product renders once per profile and a platform may be sent any of those videos. `<video>.cover.json` records the image, the video and the frame time in milliseconds, which a timestamp-based platform can use.
- A whole render writes it, and so does a `--step` run of the profile's final step, which is how topic batches finish a video. `--clean` removes it with the video.
- A cover that can't be written is logged and skipped, and an earlier render's cover is removed first so it never outlives a failed write.
- The publish side is not built (REQ-PUB-154 to REQ-PUB-159).
