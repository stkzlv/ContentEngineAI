# 0004. pycaps with the CSS renderer is the caption engine

- **Status:** Accepted
- **Date:** 2026-10-01

## Context and problem

Short-form captions are expected to be animated word by word, with highlights and per-word styling. The FFmpeg engine burns static SRT or ASS captions. pycaps renders animated, CSS-styled captions as a step after assembly and offers two renderers: CSS, through a headless browser, and pictex, which needs no browser.

## Options considered

- **FFmpeg only.** Small install, but static captions.
- **pycaps with the pictex renderer.** No browser needed, but it drops the gaps between words (#565).
- **pycaps with the CSS renderer, falling back to FFmpeg.**

## Decision

pycaps with the CSS renderer is the bundled default. An install without pycaps falls back to the FFmpeg engine with a warning that names the install command. pictex is for previews only.

## Consequences

- The install adds Playwright and Chromium, about 1 GB.
- The two-part caption layout is available on the FFmpeg engine only.
- The engine a run used is recorded in the run state rather than re-read from config, because a fallback can change it mid-run (`docs/notes/subtitles.md`).
