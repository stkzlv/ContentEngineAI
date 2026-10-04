# Compliance requirements

Ids use the prefix `REQ-CMP`. The format and the statuses are described in [the requirements index](README.md).

## On-frame disclosure

- **REQ-CMP-001** `shipped` A render that carries a material connection has a persistent disclosure overlay burned in, on every subtitle engine and every subtitle positioning mode.
- **REQ-CMP-002** `shipped` The disclosure overlay sits in a fixed corner for the full clip, sized smaller than the narration captions.
- **REQ-CMP-003** `shipped` The overlay's text, corner, size, color, outline and background are configurable under `video_settings.disclosure_overlay`.
- **REQ-CMP-004** `shipped` If a record shows that a render has nothing to disclose (a topic with no affiliate link), the render carries no overlay.
- **REQ-CMP-005** `shipped` If a record does not positively show that a render has nothing to disclose, the render discloses.

## Caption disclosure

- **REQ-CMP-006** `shipped` The caption of every post that carries a material connection leads with the disclosure on its own line, ahead of the description and the hashtag block.
- **REQ-CMP-007** `shipped` The caption disclosure and the on-frame overlay follow the same decision, so a caption and a frame never disagree about whether a render is promotional.
- **REQ-CMP-008** `shipped` The producer records the material-connection decision in the render's metadata, and the publisher reads it from there.
- **REQ-CMP-009** `shipped` Metadata written without the decision gains it on the next render.
- **REQ-CMP-010** `shipped` If a metadata file lacks the decision, the publisher discloses.
- **REQ-CMP-011** `shipped` If the caption's hashtags include the disclosure token, the published caption shows the disclosure once, on the leading line.
- **REQ-CMP-012** `shipped` If a render has no material connection, the publisher removes `#ad` and the configured disclosure token from its description and hashtags.
- **REQ-CMP-013** `shipped` The disclosure text is configurable per render, so language-matched variants need no code change.
- **REQ-CMP-022** `shipped` The overlay and the caption disclosure are in the script's language, taken from the TTS `language_code`: `#ad` for English, `#publi` for Spanish, from `video_settings.disclosure_overlay.variants`.
- **REQ-CMP-023** `shipped` If `disclosure_overlay.language` differs from the script's language, or the script's language has no variant, the config load logs a warning naming both; the render falls back to `disclosure_overlay.text` for a language with no variant.
- **REQ-CMP-024** `shipped` At config load, every disclosure variant and the fallback text are checked for glyphs the overlay font can draw.

## Platform disclosure settings

- **REQ-CMP-014** `shipped` Every TikTok post declares its commercial content type from the material-connection decision: `brand_organic` for a render with a material connection, `none` for one without.
- **REQ-CMP-015** `held` YouTube posts declare altered or synthetic content.
  - On when: `synthetic_media_disclosure` is set to true for output that meets YouTube's bar, such as AI-generated music or AI-generated footage of a real place.
- **REQ-CMP-016** `shipped` TikTok posts carry the AI-generated-content label by default, and `tiktok_settings.video_made_with_ai: false` turns it off.
- **REQ-CMP-021** `shipped` If the `tiktok_settings` section is invalid, every TikTok setting falls back to its default with a warning, so the AI-generated-content label stays on.
- **REQ-CMP-017** `planned #558` AI disclosure on each platform follows that platform's current rule, recorded with its source.
- **REQ-CMP-018** `planned #558` A label beyond what a platform's rule requires is a documented, voluntary choice.
- **REQ-CMP-019** `planned #558` An optional statement says what AI did and what a person did.

## Manual steps

- **REQ-CMP-020** `shipped` The compliance guide documents the manual steps for the disclosure settings the publishing provider does not expose: the YouTube paid-promotion checkbox and the Instagram paid-partnership label.
