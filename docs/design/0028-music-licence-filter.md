# 0028. Music licence filter

- **Status:** Implemented
- **Issue:** #786
- **Requirements:** REQ-CNT-165

## Context

The music chain records each track's licence (REQ-CNT-081) and never filters on it. Jamendo is sent no licence parameter, and Freesound's filter sets only a duration, so either provider can hand a render a NonCommercial (`by-nc`) or NoDerivatives (`by-nd`) track. Renders on disk carried both.

Evidence ([evidence grades](README.md#evidence-grades)):

- A live Freesound search for ambient music returned `by-nc` licences for 11 of 40 results; every licence came back as a Creative Commons URL. [B] Measured through the API.
- Jamendo's `license_ccurl` is a Creative Commons URL too, and a manual search for instrumental beds returned NC tracks among the top results. [B] Measured through the API.
- NonCommercial licences allow use for noncommercial purposes only, and NoDerivatives licences allow no adapted material; a track synced to video may count as adapted. [A] [Creative Commons licence list](https://creativecommons.org/licenses/list.en)

## Goals

- No render picks an NC or ND track from any provider with the bundled config.
- One rule for every provider, with the allowed licences in config.

## Non-goals

- Local stock files: their licences are checked by hand when they are added.
- Per-track licence attribution text in captions.

## Design

`src/audio/licence.py` reads a licence string as a code (`cc0`, `by`, `by-sa`, `by-nc`, `by-nd`, `by-nc-sa`, `by-nc-nd`, `fal`) from a Creative Commons URL, an artlibre.org URL or a short name, and returns None when it can't. `AudioManager._try_provider` drops every track whose code isn't in `audio_settings.allowed_music_licenses` before the duration and mood checks; a provider left with none returns no track, so the chain moves on (REQ-CNT-077). An unreadable licence is refused. The producer passes the configured list; `None` turns the filter off.

Jamendo also filters at the API: when the list has no NonCommercial or NoDerivatives code, the search sends `ccnc=false` and `ccnd=false` (`jamendo_licence_params`). Filtering only the results left one or two usable tracks of twenty, since most of Jamendo's catalogue is NonCommercial; with the API filter a page of 50 came back entirely under CC BY or CC BY-SA. The manager still checks every track.

Config: `allowed_music_licenses`, default `[cc0, by, by-sa, fal]`, in the model and the bundled YAML.

Tests: `tests/audio/test_music_licence_filter.py`.

## Alternatives considered

- Filter only in each provider's search request. Each provider has its own syntax (Jamendo's `ccnc`/`ccnd`, Freesound's `license:` filter term), and a provider added later would need its own; the manager check is one rule for every provider. Jamendo's API filter is added on top, for the size of its pool.
- Allow unknown licences. Rejected: terms that can't be read can't be shown to permit commercial use.

## Rollout

On by default: it's a licensing fix. Null turns it off; a list widens or narrows it.

Remove the switch when: never; which licences an installation accepts is its own choice.

## Open questions

None recorded.
