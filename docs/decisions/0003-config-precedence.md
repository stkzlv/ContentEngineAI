# 0003. Four configuration tiers; the environment holds machine settings and secrets

- **Status:** Amended by [0008](0008-operator-account-values-in-the-environment.md)
- **Date:** 2026-10-01

## Context and problem

The requirements describe three configuration tiers: CLI, profile and YAML. The code also reads an environment layer between the CLI and the profile, and through it `.env` carries behaviour settings that aren't secrets, such as the subtitle overrides, publisher privacy and retries, and the pipeline timeout. A run's behaviour then depends on a file that no review sees, and the documented precedence is wrong.

## Options considered

- **Document the environment layer as it is.** Keeps behaviour settings outside review.
- **Remove the environment layer.** Machine-specific values (output directory, FFmpeg threads, GPU, proxy) would then have to be committed or passed on every command.
- **Keep the layer, limited to secrets and machine-specific settings.**

## Decision

Four named tiers, highest first: CLI, machine environment, profile, YAML. The environment holds secrets and machine-specific settings only. Behaviour settings move to YAML or a profile. Each client keeps reading its own secret from the environment.

## Consequences

- Removing the behaviour variables from the environment layer is a breaking change and ships as a minor release.
- `.env.example` lists only secrets and machine settings.
