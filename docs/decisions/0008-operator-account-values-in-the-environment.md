# 0008. The operator's account values stay in the environment

- **Status:** Accepted
- **Date:** 2026-10-04
- **Amends:** [0003](0003-config-precedence.md)

## Context and problem

Decision 0003 limits the environment to secrets and machine settings. Carrying it out left a third kind of value in `.env`: the operator's own accounts. The affiliate tag and whether an affiliate program is live (`AMAZON_ASSOCIATE_TAG`, `AMAZON_AFFILIATE_LINKS_ENABLED`), the link-in-bio address (`LINK_IN_BIO_URL`, `SUBTITLE_BUSINESS_URL`) and the topics file (`PIPELINE_TOPICS_FILE`) aren't secrets and don't depend on the machine, but they belong to one operator, and the public YAML ships generic defaults.

## Options considered

- **Move them to the YAML.** The public config would carry one operator's accounts, or each operator would keep a modified tracked file.
- **Keep them in the environment as a named exception.**

## Decision

The environment holds secrets, machine settings, and the operator's account values listed above. Any other behaviour setting lives in the YAML files or a profile.

## Consequences

- `.env.example` lists these three kinds and nothing else; a test checks its variable names.
- `AMAZON_AFFILIATE_LINKS_ENABLED` keeps overriding `affiliate_links.enabled` from the YAML.
