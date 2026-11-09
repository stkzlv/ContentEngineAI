# 0035. Correction topics

- **Status:** Implemented
- **Issue:** #807
- **Requirements:** REQ-CNT-175, REQ-PUB-166

## Context

A commentary video states facts (design 0030). When one turns out wrong, the fixes available today reach few of the viewers who saw it:

- a description edit or a pinned comment, which most viewers never open;
- YouTube's corrections card, which shows once, at the first correction's timestamp, and only on YouTube [B] ([Tubefilter](https://www.tubefilter.com/2022/06/15/youtube-creator-insider-corrections-copyright-transparency-vidcon/)).

A short follow-up video reaches the same audience on every platform. Owning a mistake in the open is also what an opinion channel's trust rests on: a public accuracy dispute between two large tech channels turned on corrections posted where viewers didn't see them [C] ([Tubefilter](https://www.tubefilter.com/2022/06/15/youtube-creator-insider-corrections-copyright-transparency-vidcon/)). Admitting a minor mistake raises liking for someone already seen as competent (the pratfall effect, Aronson 1966) [A].

The pipeline can render a commentary topic and pin a narrator identity to it (design 0031), and a post's `metadata.json` can set its own first comment (REQ-PUB-161). The publish history (`outputs/state/publish_history.json`) maps a record id and platform to the provider's post id, and the provider returns the platform URL for a post id.

## Goals

- A topic marked `correction` names the published record it corrects and renders as a short correction in the commentary identity.
- On every platform that takes a first comment, the correction's first comment names the original post and gives its URL.
- A correction that names a record with no row in the publish history is refused before any render cost.

## Non-goals

- Editing or deleting the original post, or adding YouTube's corrections card. Both are manual.
- Finding mistakes. An operator writes the correction's detail.
- A comment on TikTok, which takes no first comment (REQ-PUB-038).

## Design

- **The record.** `TopicSpec.kind` gains `correction`, with a `corrects:` field holding the corrected record's id. A topics file sets both per entry; the producer and the batch take `--topic-kind correction --corrects <record id>` beside `--topic`. `--corrects` without `--topic-kind correction` is refused. The record's `data.json` carries `topic_kind: correction` and `corrects`.
- **The check.** Where topic records are built, which the producer and the global batch share, the publish history is read for `corrects`. No row on any platform, queued or published, refuses the record with a message naming the id. A row records the queue time, not delivery, so an original that is only scheduled passes; the first comment handles that case below.
- **The render.** `content_kind` returns `commentary` for a correction, so the commentary identity, narrator, CTAs and length apply (design 0031). `config_for_record` then sets `topic_templates` to `correction_templates`. The shipped `correction` template asks for: what the earlier video said, in one sentence; the right fact, with the detail's source; what changes because of it; and no closing question, because a correction is not a debate. The detail must carry the right fact and its source; the template forbids anything else, as the commentary template does.
- **The first comment.** At publish time, for a correction record, the publisher resolves each platform's original post id from the publish history and asks the provider for its URL. The correction's first comment on that platform is the configured template `first_comment.correction_template` (default "Corrects: {original_title} {original_url}") filled with them. YouTube shows a URL in a Shorts comment as plain text (`config/publisher.yaml`, decision 0009), so there the title is what lets a viewer find the original; whether Instagram makes it clickable is unchecked. A platform with no URL yet (the original still scheduled) posts the template's fallback, the record's title, and logs a warning. A `first_comment` set in the record's own `metadata.json` still wins (REQ-PUB-161).
- **Analytics.** A correction counts in the `topic` content format, like a commentary post.
- **Tests.** The flag and file parsing; the refusal for an id with no publish-history row; the template swap on the config copy; the first comment per platform from a fake history and provider, the fallback, and the metadata override.

## Alternatives considered

- **A plain commentary topic with a hand-written first comment.** It works today, but nothing refuses a correction of the wrong record, and the link has to be looked up by hand per platform.
- **A reply on the original post.** Most viewers of the original have already scrolled past it, and the comment reaches only those who open it.

## Rollout

No switch: nothing changes unless a topic is marked `correction`, which needs a template that the bundled config ships but no record uses. An installation sets its own `correction_templates` in its overlay if it wants another shape.

## Open questions

None recorded.
