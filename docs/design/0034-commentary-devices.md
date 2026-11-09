# 0034. A comparison and a contrast in commentary scripts

- **Status:** Implemented
- **Issue:** #806
- **Requirements:** REQ-CNT-174

## Context

The commentary template (design 0030) asks for a hook carrying the fact, a turn, one owned opinion and a closing question. It asks for nothing that makes the fact land, so a take can state a bare count and move on.

Research on speaker charisma names a set of verbal tactics that can be trained and that raise how charismatic a speaker is rated: metaphors, similes and analogies, stories, contrasts, rhetorical questions, three-part lists, and expressions of moral conviction (Antonakis, Fenley and Liechti 2011, Academy of Management Learning and Education; read through a secondary summary) [A]. A comparison turns a bare number into something the viewer can picture, and a contrast sets the common assumption against the fact, which is what a take's turn already is.

A probe render whose overlay narrator asked for a comparison with a quoted example spoke the example verbatim and invented the figure it implied. So the rule must quote no example (REQ-CNT-159) and tie the comparison to the detail's own numbers.

## Goals

- With a switch on, the commentary prompt asks for one comparison built from the detail's numbers and one contrast between the common assumption and the fact.
- Off, the prompt is byte-identical.

## Non-goals

- Stories and the other tactics. A 30-40 second take has room for two devices, and a story invites an invented anecdote, which REQ-CNT-156 forbids.
- Checking that the devices appear. Measured by reading samples, not enforced.

## Design

- **The switch.** `script_templates.commentary_devices.enabled` (bool, default false) in `config/ai_services.yaml`, on a small model beside `topic_routing`.
- **The rule.** Two bullet rules added to what `{CTA_RULE}` carries (the CTA rule, naturalism, the signature, the hook rules and the lint's limits are joined there today), only for a commentary record with the switch on; otherwise the bundle is unchanged, so no template file changes:
  - make the main fact land with one comparison built only from numbers the detail states, and never invent a figure for it;
  - set what most people assume against what the fact shows, once.
  Neither quotes an example.
- **Where it applies.** Only to a record whose kind is commentary. A topic or product prompt never carries the rules.
- **Tests.** Off, the commentary prompt equals the pre-change prompt; on, it carries both rules and no quoted sentence; a tutorial prompt is unchanged either way.

## Alternatives considered

- **Putting the devices in the narrator profile.** That is per installation, and the probe showed where an overlay's own wording goes wrong. A shipped, example-free rule is safer to share.
- **A separate commentary template with the devices.** It would split the commentary pool, and the variety report would then attribute a template difference to the devices.

## Rollout

Ships off. Turn it on after sampling ten commentary scripts with it on and ten off, on the same topics, and reading each for a comparison, a contrast, an invented figure and the length. Turn it on if most "on" scripts carry both devices with no invented figure and no longer than the "off" ones.

Measured on ten commentary topics with sourced details, one script per topic with the switch off and one with it on, through the producer's script step with the installation's narrator profile:

| | Off | On |
|---|---|---|
| A contrast between the common assumption and the fact | 9 of 10 | 9 of 10 |
| A comparison built from the detail's numbers | 2 of 10 | 3 of 10 |
| Both | 2 of 10 | 3 of 10 |
| A figure not in the detail | 0 | 0 |
| Words, mean (range) | 87 (69-103) | 90 (75-105) |

The rule to turn it on is not met: most "on" scripts carry the contrast but not the comparison, and they run slightly longer. The contrast was already there with the switch off because the installation's narrator profile asks for both devices itself, so on that installation the switch adds little. It stays off.

Remove the switch when: it has been on for a month of commentary renders with no invented figure found in review; then the rule becomes part of the template.

## Open questions

None recorded.
