# 0036. Comment-reply videos

- **Status:** Accepted
- **Issue:** #808
- **Requirements:** REQ-CNT-176, REQ-PUB-167

## Context

A reply video answers one viewer comment with a new short video. TikTok has it as a native feature, which shows the comment as a sticker on the reply [A] ([TikTok newsroom](https://newsroom.tiktok.com/en-us/product-tutorial-reply-to-comments-with-video)); creators use the same shape on Shorts and Reels with the comment drawn on screen. Roadmap item H.10 tracks it.

Why it matters for a recurring narrator: in a study of an AI streamer's audience, fans' most valued part was interacting with the character, and 85% of their paid messages steered what it did next (arXiv 2509.10427, 334 survey responses and 550,000 chat messages) [B]. A reply video gives a short-form audience a smaller version of that part. No study measures its effect on reach; the pipeline's own analytics will.

What exists:

- the publisher reads a published post's comments from the provider, flagging the account's own (`get_post_comments`), today used only by `verify-comments` (REQ-PUB-044);
- the publish history maps a record id and platform to the provider's post id;
- a commentary topic renders an opinion with the commentary identity (designs 0030, 0031), and the hook overlay draws text on the opening seconds.

The publishing API has no way to attach TikTok's native reply sticker, so the comment has to be drawn into the video.

## Goals

- An operator can list recent viewer comments on a published record, per platform.
- The producer renders a reply to one comment, in the commentary identity, opening on a card that shows the comment.
- A reply lands in a drafts folder and is never published by the pipeline on its own.

## Non-goals

- Picking comments automatically, or replying to a comment without an operator.
- The native TikTok reply link. The API doesn't offer it.
- Publishing. The operator publishes a reviewed reply with the publisher's usual commands.

## Design

- **Listing.** `python -m src.publisher.late comments <record id> [--platform P] [--limit N]` resolves each platform's post id from the publish history and prints, per comment, its id, author handle and text, leaving out the account's own comments. Read-only.
- **The record.** The producer's reply mode takes `--reply-to-comment "<text>"`, `--reply-record <record id>` (the record the comment was left on) and an optional `--reply-author <handle>`. It builds a topic record whose title comes from the comment's first words, whose detail is the comment and the parent record's title and script, and whose `topic_kind` is `reply`. `content_kind` returns `commentary`, so the commentary narrator, CTAs, length and identity apply; `config_for_record` sets `topic_templates` to `reply_templates`.
- **The template.** The shipped `reply` template answers the comment in its first sentence, adds one reason or fact from the parent record's script, and closes on the CTA with no new question. It never quotes the comment back word for word in the voice, since the card shows it.
- **The card.** A new `video_settings.reply_card` block (enabled, duration in seconds, default 3.0, and `show_author`, default false) draws the comment, truncated to two lines, on a rounded panel in the opening seconds, through the same overlay path as the hook headline. With `show_author` off the card shows the text only, so a viewer's handle is never drawn unless the operator turns it on.
- **Drafts.** A reply renders into `<outputs root>/drafts/<record id>/` instead of the root. Publisher discovery scans only the root's top level, so nothing publishes a draft. After review, the operator moves the record's directory into the root, where the usual publish commands find it.
- **Tests.** The listing against a fake provider, the owner filter and the missing-history message; the record built from the flags; the template swap; the card overlay present for a reply and absent otherwise; the drafts path, and publisher discovery not finding it.

## Alternatives considered

- **A commentary topic written by hand from the comment.** It works today, but it carries no card and lands among the publishable outputs.
- **Burning the comment into the captions.** The captions follow the voice, and a comment that is never spoken would break their timing.

## Rollout

No switch for the mode itself: nothing changes unless the reply flags are passed. The card ships with `enabled: true`, since a reply without it makes no sense, and `show_author: false`.

Remove the switch when: never for `show_author`, which is an operator's lasting choice about drawing a viewer's handle; `enabled` can go once replies have rendered with the card for a month.

## Open questions

- Whether a reply should be allowed to name the commenter in the voice. The design keeps the voice off the handle, the same as the card's default.
- Whether the provider's comments endpoint returns TikTok comments; it has only been used for YouTube and Instagram. One call against a published TikTok post settles it before the listing is built.
