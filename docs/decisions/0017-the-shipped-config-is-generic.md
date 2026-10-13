# 0017. The shipped config is generic

- **Status:** Accepted
- **Date:** 2026-10-13
- **Amends:** [0013](0013-tiktok-ai-label-goes-off-after-the-readout.md)

## Context and problem

The tracked config shipped one installation's operating choices as defaults: a daily posting slot in one timezone, an author's signature lines, a CTA and an Instagram first comment that point at a bio link, link-in-bio updates switched on, and the TikTok AI-content label switched off. Anyone else who ran the project inherited them, and the link-in-bio default failed without that provider's credentials. Private overlays ([design 0026](../design/0026-private-config-overlays.md)) give an installation somewhere else to keep them.

## Options considered

- **Keep the choices in the shipped config.** Nothing moves, and every other installation keeps inheriting one channel's schedule, voice and funnel.
- **Ship generic defaults and keep each installation's choices in its overlay.**

## Decision

The shipped config holds defaults that suit any installation. An installation's schedule, author voice, CTA funnel, link-in-bio setup and any label it is exempt from live in its `config/<name>.private.yaml`. Where a default decides a disclosure, it is the one that discloses: the TikTok AI-content label ships on, and an installation whose renders meet TikTok's exemption turns it off in its overlay, as decision 0013 allows.

## Consequences

- The bundled schedule is one example slot in UTC, the signature ships off with empty pools, no bundled CTA or first comment mentions a bio link, and link-in-bio updates ship off.
- An installation that relied on those defaults moves them into its overlay before upgrading; `config/publisher.private.yaml.example` and `config/ai_services.private.yaml.example` show the shape.
- A test pins each generic default (REQ-OPS-121).
- The YouTube description's "Link in bio." pointer (`youtube_profile_pointer`) and the metadata prompts that ask for it still assume a bio link. They change prompts and rendered metadata, so they ship separately (#756).
