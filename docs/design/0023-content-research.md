# 0023. Repeatable content research

- **Status:** Implemented
- **Issue:** #686
- **Requirements:** REQ-OPS-107, REQ-OPS-108, REQ-OPS-109, REQ-OPS-110, REQ-OPS-111, REQ-OPS-112

## Context

Which topics to render, which product keywords to scrape and how scripts are configured have so far rested on research done by hand. One October 2026 pass:

- pulled Google Trends and autocomplete by hand ([the tutorials explanation](../explanation/tutorials.md), "What viewers ask about");
- ran the topic filter over the pool, which kept 1 of 17 topics;
- generated 16 topic scripts text-only with the shipped config and checked each menu path against Apple, Samsung, Google and Microsoft support pages.

Seven of the 16 scripts had a wrong or unsupported statement. Four had wrong steps: iOS 26 renamed two settings the model still used, one path was wrong, and Samsung's battery labels were misnamed. The fact check caught several errors but missed every rename, and two of its rewrites made things worse (#685). Two of the three topic templates also invented a symptom or a mistake for topics that were tasks.

None of that is repeatable. The answers go stale as operating systems change and as search demand moves.

## Goals

- **Product keywords:** which scraper keywords to add or drop, by relative search demand.
- **Topics:** which topics to add or drop, by demand and by whether they pass the topic filter.
- **Script settings:** whether step lists, and a template chosen by topic shape, beat the shipped config, measured on the same sample.
- **Evidence:** every recommendation names the config key, the value and the evidence behind it.

## Non-goals

- **Editing configuration.** The report recommends; a person changes the YAML.
- **Background research.** Evidence grades and literature stay in the explanation pages, written by hand.
- **Scraping or rendering.** Product samples come from products already scraped; nothing is rendered.
- **Absolute search volumes.** Google Trends gives relative interest only, and no free source gives volumes with a stated method.

## Design

A package `src/research`, run as `python -m src.research <stage>` or `make research`. Settings are in `config/research.yaml`, read by its own Pydantic model, not merged into the video config. Each stage writes JSON under `outputs/reports/research-<date>/` so the report can be rebuilt without repeating calls.

| Stage | What it does | Calls |
|---|---|---|
| `demand` | Google Trends relative interest for the scraper keywords and the topic pool against one anchor term per side, 12 months and 5 years, in each configured country; autocomplete suggestions for configured stems | Trends (unofficial, rate-limited), Google suggest (unofficial) |
| `pool` | The topic filter (REQ-VID-151) over the pool and over candidates from autocomplete | One grounded call per topic |
| `sample` | Text-only script generation, in process, for N topics and N scraped products under each variant, into a scratch outputs root | The pipeline's own: script, fact check, headline |
| `check` | Measured checks per script, and one grounded verification per topic script that returns a verdict and a source URL per step or claim | One grounded call per topic script |
| `report` | Markdown and JSON with the answers and the recommended changes | None |

**Topic searches.** A pool title is often too long for Trends. A topic's optional `search` field, in the topics file, names the search it is measured by; without one, the title minus "How to" is used. The field lives with the topic, not in `config/research.yaml`, so a private pool's titles stay out of public configuration.

**Variants.** A variant is a named set of config overrides applied in memory to the same sample:

- `shipped`: the config as loaded.
- `step_lists`: `llm_settings.topic_scripts.step_list.enabled: true`.
- `task_answer_first`: a topic whose title starts "How to" is a task and uses `topic_answer_first` only; others keep the template pool.

**Measured checks**, per script:

- word count against the narrator's band (75-100 words; the step-list band when one applies);
- the search phrase in the first sentence;
- openings repeated across the sample;
- the lint's machine-writing tells;
- the CTA as the last sentence;
- fact-check flags and accepted rewrites;
- template fit: a task topic written with a symptom or mistake template.

**Verification** asks a Gemini model with Google Search grounding to judge each step and claim against official support pages, returning correct, wrong, outdated or unverified, with the source URL and a short quote. A verdict with no source counts as unverified.

**Recommendations.** A variant is recommended when it beats `shipped` on wrong-or-outdated steps without making length or fit worse, on the same sample. A scraper keyword is a drop candidate when its interest is below `drop_below` of the keywords' median in every country and it has not risen over the last quarter (its last quarter's mean is at most 1.1 times the year's). Rising related searches from the product seeds that are not already keywords are add candidates. Autocomplete suggestions that no pool topic covers are listed as uncovered searches, phrased as found.

**Dependencies.** `pytrends` in an optional `research` Poetry group. Without it the demand stage reports the source as unavailable. Suggestions come from `aiohttp`, already a dependency. When a source is rate-limited or empty, the report says so rather than reporting a zero.

**Tests** use recorded responses for Trends, suggest and the model calls. The sample stage is tested by driving `generate_script` with a stubbed model, as the script tests do.

## Alternatives considered

- **A script under `tools/`.** `tools/` holds one-off utilities the pipeline never imports. This one reuses the generator, the fact check, the step lists, the filter and the lint, and has its own config and tests.
- **Subprocess runs of `--step generate_script`.** The manual pass did this. In process is faster, keeps variants in memory without editing YAML, and needs no outputs tree per run.
- **A paid keyword-volume API.** It gives volumes, but costs money and publishes no method; relative interest answers the ranking questions asked here.

## Rollout

Nothing renders differently: the package only reads configuration and writes reports, so it ships on. It is built in three releases: `demand`, then `pool` and `sample` and the measured checks, then verification and the report. Each release documents its stage.

Remove the switch when: not applicable; there is no switch.

## Open questions

- **A stable anchor term.** The product anchor and topic anchor must stay comparable across runs; a term whose own interest moves shifts every figure.
- **Grounded verification against itself.** The verifier is a model with search, as the fact check is. Its agreement with a human check on one batch should be measured before its verdicts drive a recommendation alone.

## As built

Built in three releases (v0.176.0 demand, v0.177.0 sample and checks, v0.178.0 verification and recommendations), as designed, with these differences:

- **Topic searches** are an optional `search` field in the topics file, not a map in `config/research.yaml`, so a private pool's titles stay out of public configuration.
- **The comparison uses a share.** A variant is judged on the share of its verified scripts with a wrong or outdated step, not the total: a variant that drops topics has fewer scripts, and a total rewards it for that. Dropped topics are reported beside it.
- **Every failed request is missing, never zero:** a rate-limited Trends request, a failed five-year request and a failed rising-searches request each appear under "Missing data".
- **Reruns.** A sample rerun clears that variant's earlier samples first, because the script step reuses a script already on disk.
- **Verified means sourced.** Only a script with at least one verdict that cites a source counts toward the share; a script whose verdicts are all unverified is reported beside it as one the verifier could not judge, not as a clean one.
- **The verifier's agreement with a human check is still unmeasured** (Open questions). Until it is, the recommendations are advisory, and the guide asks for a spot check of the verdicts behind one before acting on it.
- **The lint check matches the producer:** the product's own name is exempt, and only a step-list script skips the word cap.

Measured on the first runs: Google Trends rate-limited this machine's IP after a day of manual pulls, and every request answered 429 until the next day; the run recorded each as missing and the report rendered. A sample of 2 topics under the three variants and 1 product produced every record. The step-list variant dropped one topic (its steps forked by device), and verification found two outdated steps in the answer-first scripts, each with an Apple or Google support page. Two topics are too few to decide a variant; the full run decides.
