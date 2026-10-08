# Explainer script from a sourced step list

Write a voiceover script for a short video that explains a cause, for
{AUDIENCE}.

TOPIC: {TOPIC_TITLE}

Explain the usual cause, then turn these sourced checks into what the viewer
can test, in this order, and add no check, label or claim they do not state:

<<STEP_LIST>>

## Rules

- Open by answering the question in one sentence: name the usual cause the
  way a viewer would search for it.
- Explain the cause in one or two plain sentences, no more.
- Say once which device and version the checks are for: <<PLATFORM>>.
- Before the first check, say where it starts: <<START_SCREEN>>.
- Give each check as something the viewer can do now, one instruction per
  sentence. Say a menu path with its labels in order.
- <<MISTAKE_RULE>>
- Close on the one test that tells the viewer whether the cause was theirs.
- **Length: <<WORD_RANGE>> words.** This overrides any target length given
  above: an explainer runs longer than a single setting and shorter than a
  long fix.
- Spoken text only: no headings, lists, emojis, stage directions or URLs.
{CTA_RULE}
