# 0039. Instagram's paid-partnership label from the material connection

- **Status:** Held
- **Issue:** #789
- **Requirements:** REQ-CMP-026

## Context

Instagram's "Paid partnership" label was a manual post-edit step because the publishing provider didn't expose it (REQ-CMP-020). The provider's SDK has `isPaidPartnership` in Instagram platform data: the label on feed posts, Reels and carousels, for an account connected through Facebook Login; an account on classic Instagram Login gets a 400 (`instagram_paid_partnership_requires_facebook_login`).

## Goals

- Send the label from the render's material connection, the way TikTok's commercial content type is set (REQ-CMP-014).
- Don't lose a post when the provider refuses the label.

## Non-goals

- Tagging a brand sponsor (`brandedContentSponsors`).
- YouTube's paid-promotion flag, which the SDK still doesn't expose.

## Design

- `instagram_paid_partnership` in `config/publisher.yaml` (default false). On, every Instagram payload carries `isPaidPartnership` equal to the render's material connection.
- On the Facebook Login refusal, drop the flag from the Instagram entries and send the post once more, with a warning to set the label by hand. The caption's disclosure still carries the material connection, so the post isn't held back for a platform label.

## As built

- The flag merges into the Instagram platform data beside the AI label, the first comment and the cover, in every payload branch.
- The refusal is recognised by the error code in the SDK error's text; any other error is raised as before. The retry sends the same post without the flag.

## Alternatives considered

- **Refuse the post with a clear message.** Keeps the label rule strict, but loses a post whose caption already discloses, over a platform label that can be set afterwards.

## Rollout

Ships off. Turn it on for an installation that publishes renders with a material connection from an Instagram account connected through Facebook Login.

Remove the switch when: never; whether an installation's posts carry a material connection, and how its account is connected, differ from installation to installation.

## Open questions

None recorded.
