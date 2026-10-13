# 0026. Private config overlays

- **Status:** Implemented
- **Issue:** #754
- **Requirements:** REQ-OPS-116 to REQ-OPS-121

## Context

The tracked YAML under `config/` holds two kinds of setting. Most are software defaults that suit any installation. Some are one installation's operating choices: the posting schedule, the author signature pools, the CTA lists and first-comment templates, the link-in-bio provider. To change one, an installation edits the tracked file, so its choices either reach the public repository or sit as a local diff on every checkout.

Two private channels exist already. The environment carries secrets, machine settings and the operator's account values ([decision 0008](../decisions/0008-operator-account-values-in-the-environment.md)). The topic pool has its own gitignored file. Neither fits a nested behaviour setting such as a slot list or a signature pool: an environment variable per key doesn't scale, and the topic file mechanism is specific to one list.

## Goals

- An installation sets any config key without editing a tracked file.
- Every reader of a config file sees the same values: the standalone CLIs, the batch, the research tools and the validators.
- A broken overlay fails as loudly as a broken shipped file.
- The test suite is unaffected by a developer's overlays.

## Non-goals

- Changing any shipped default. That is the second step (REQ-OPS-121), after an installation's values are in its overlay.
- A new tier in the precedence order. The overlay is part of the YAML tier: CLI flags, the environment and profiles still win over it.

## Design

`config/<name>.private.yaml`, when present, is parsed and merged over `config/<name>.yaml` before any validation. A mapping merges key by key; a list, a scalar or `null` replaces the shipped value, so an overlay can shorten a list or clear a key. The overlay sits beside whichever file is read, so a loader pointed at another config directory reads that directory's overlays.

One helper parses every file under `config/`, and a test fails if any other module parses YAML outside an allowlist: the readers had drifted apart once (the batch and the standalone modules), and a second private channel that one reader misses would repeat that.

A file that is itself an overlay gets no overlay, so the existing `topics.private.yaml` is never treated as one. An overlay that doesn't parse or isn't a mapping raises an error naming it; the publisher reports it as an unreadable config, like a broken `publisher.yaml`. An empty overlay changes nothing. Each overlay applied is logged once per process at INFO.

`CONFIG_OVERLAYS=off` in the environment reads the shipped files alone. It is a machine setting under decision 0003, set in the shell: the circuit breakers and the video config singleton read config at import, before any `.env` loads. The test suite sets it at the top of the root conftest, before any test module imports config, since a fixture runs after collection has already cached what those modules read.

A parse error names the file, as the readers' own parsing did, because the helper parses from a stream. An overlay applied before any handler exists, at import time, is held and logged when the logging setup attaches its handlers, since that import may be the only read of its file in a run (the scraper reads `performance.yaml` only there).

`*.private.yaml` is already gitignored. `config/publisher.private.yaml.example` shows the shape.

## Alternatives considered

- **Environment variables per setting.** Works for a handful of scalar account values (decision 0008); it doesn't scale to slot lists, phrase pools or nested blocks, and each variable needs its own reader.
- **A whole private config directory selected by a variable.** Every installation would copy and maintain all nine files, and a new shipped default would never reach it.
- **One combined overlay file.** Its top-level keys would have to name the file they overlay, a second layout to learn; the per-file overlay mirrors the tree.

## Rollout

Ships on: no overlay exists until an installation writes one, so nothing changes for anyone else.

Remove the switch when: `CONFIG_OVERLAYS` is a lasting option, needed by the test suite and by a run that has to reproduce the shipped defaults.

The second step (REQ-OPS-121) moves the installation-specific values out of the shipped config into generic defaults, with a decision record stating the rule. It ships after an installation's own values are in its overlay, so its posts don't change.

## Open questions

- Whether an overlay should be able to delete a mapping key rather than set it to `null`. No setting needs it yet: every optional key reads `null` as unset.

## As built

- The loader shipped first (REQ-OPS-116 to REQ-OPS-120), then the generic defaults (REQ-OPS-121, [decision 0017](../decisions/0017-the-shipped-config-is-generic.md)). The installation's values went into its overlays before the defaults changed: the merged config with the overlays in place was compared key by key with the config before the change, and every value matched.
- The defaults that moved: the posting schedule (one example slot in UTC), the signature (off, empty pools), the product CTA that points at a bio link, the Instagram first comment (the closing line, like YouTube's), link-in-bio updates (off), and the TikTok AI-content label (on). `config/ai_services.private.yaml.example` shows the signature and CTA shape.
- A CTA in an overlay must still start with an opener the publisher strips before it reads the closing line; that list is hand-kept in the publisher, which doesn't load the video config.

