# 0022. Signature lines per video type

- **Status:** Implemented
- **Issue:** #440
- **Requirements:** REQ-CNT-045, REQ-CNT-046, REQ-CNT-152, REQ-CNT-153

## Context

The author signature (#440) shipped as one set of pools, empty, for every render. Filling them for a test showed three problems:

- **A how-to video signed off with an opinion.** One pool served product and topic renders, so a tutorial ended on "That's my honest take." A tutorial's value is the procedure, and it offers no take.
- **The opener split from the first sentence.** The rule joined the opener with a comma ("Quick find for you, ..."). The voice paused there and Whisper transcribed "Quick find for you." as a sentence of its own, so the first caption carried no search phrase.
- **No lines existed to turn on.** Turning the feature on meant writing the lines at the same moment, with no record of why each was chosen.

What the evidence says about recurring verbal lines:

- **No study measures catchphrases or sign-offs in short-form video.** The parasocial literature shows creators cultivate attachment and viewers adopt their language, but none of it isolates a recurring line. [A, indirect] [JSMS](https://www.thejsms.org/index.php/JSMS/article/view/304), [PMC6928007](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6928007/)
- **Branding research treats a recurring line as a distinctive asset.** An asset is judged on fame (how many people link it to the brand) and uniqueness (whether it points only to this brand), with about 50% on each as the bar. A generic line ("follow for more") fails uniqueness, and a line that echoes the tagline can earn both. This is Romaniuk's framework; the 50% figure comes from secondary accounts, not the book. [B for the method, C for the summaries] [Boringly Effective on distinctive assets](https://boringlyeffective.com/distinctive-assets/)
- **Varied wording beats repeating the same execution.** Varied ad executions improved brand-name memory over repeating one (Unnava and Burnkrant, 1991). Cosmetic variation had more effect on attitudes when motivation to process the ad was low (Schumann, Petty and Clemons, 1990). Both studies are about ads. [A] [Unnava and Burnkrant](https://doi.org/10.1177/002224379102800403), [Schumann, Petty and Clemons](https://fbaum.unc.edu/teaching/articles/Schuman_1990.pdf)
- **Templated sameness costs monetization.** YouTube's inauthentic-content policy names "content that looks like it's made with a template, or that may feel repetitive to viewers". [A] [YouTube](https://support.google.com/youtube/answer/1311392)
- **The first seconds carry the proposition.** TikTok's creative guidance says "Introduce your content proposition in the first 3 seconds". It is written for ads. [A] [TikTok](https://ads.tiktok.com/help/article/creative-best-practices)
- **Finishing a video counts.** TikTok weighs whether a viewer finishes a longer video as a strong signal of interest, so every second before the end is a chance to leave. [A] [TikTok](https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you)
- **TTS phrasing breaks at commas.** Five open-source TTS systems failed to render distinct prosodic boundaries for comma contrasts (Shim et al., Interspeech 2025). This matches the split opener above. [A] [ISCA](https://www.isca-archive.org/interspeech_2025/shim25_interspeech.pdf)
- **Creators describe catchphrases as habits, not tactics.** YouTube's own Shorts stories describe creators whose catchphrases grew out of their content. [C] [YouTube blog](https://blog.youtube/creator-and-artist-stories/cracking-the-code-to-youtube-shorts/)

No source gives a best use rate, a cost for a sign-off before the call to action, or a sourced signature line from a short-form tech creator. The lines below follow the channel's tone: plain, specific, never salesy.

## Goals

- Product and topic renders draw from their own pools, each with its own use rate.
- An opener runs into the first sentence with no comma and is at most five words, so the search phrase stays in the first caption.
- The lines ship in the bundled config with the feature switched off, so turning it on is a one-key change and the choice of each line is recorded here.

## Non-goals

- **Choosing a line by channel history.** Not repeating the last video's line would need the render history at selection time. Three or four lines per pool and a use rate under one already vary it, and the variety report ([0006](0006-render-choices-and-variety-report.md)) shows the spread.
- **On-screen signature text.** All three elements are spoken.
- **Changing the sting** (`audio_settings.signature_sting`), which stays a separate setting.

## Design

`script_templates.signature` in `config/ai_services.yaml`:

- `enabled` (default `false`). While false, no line is drawn and every prompt is unchanged, whatever the pools hold.
- `product` and `topic`, each with `use_rate`, `openers`, `transitions` and `signoffs`. A topic render draws from `topic` and a product render from `product`. The flat keys (`use_rate`, `openers`, `transitions`, `signoffs` directly under `signature`) are refused.
- An opener over five words is refused at load.
- `opener_templates` on each pool names the script templates that may draw an opener; empty means every one. The bundled topic pool lists `topic_from_steps` and `topic_answer_first`, the templates that open on the task. Every element is still drawn, so leaving out the opener does not move the transition's or the sign-off's draw.

The opener rule asks the model to start the first sentence with the opener and run on with no comma into the hook. The transition rule is unchanged. A step-list tutorial's sign-off follows its recap, as before. The draw is unchanged: one salted MD5 per element on the product id, so a render repeats its choice.

The bundled lines:

| Pool | Product (use rate 0.5) | Topic (use rate 0.4) |
|---|---|---|
| Openers | None: the product name is the hook | "Here's how to", "Quick fix to", "The fast way to", "Easy way to" |
| Transitions | "Here's what matters.", "Here's what it gets right.", "Here's the useful part." | "Here's the fix.", "Now the fix.", "Here's what works.", "Let's fix it." |
| Sign-offs | "That's my honest take.", "That's the honest version.", "Now you know the catch.", "That's the real picture." | "That's the whole fix.", "And that's the fix.", "That's all it takes.", "And you're done." |

Why these:

- **Topic openers** name the kind of video and lead straight into the task ("Here's how to turn off background app refresh").
- **Topic sign-offs** are factual and make no claim about time. Lines echoing the channel's promise of a fix in under a minute had the one distinctive-asset case in the evidence, but a fix with many steps can take longer than a minute to follow, so the claim would sometimes be false.
- **Product sign-offs** carry an opinion, because every product script states one drawback.
- **The topic use rate is lower** (0.4 against 0.5) because a topic render's opener sits in the first three seconds, where the cost would be.
- **Every line is short and plain**, with no question mark, ellipsis, dash or capitals for the voice to misread.

Tests:

- each arm draws only its own pools, in the prompt and in the script step's record;
- filled pools with `enabled: false` leave the prompt byte-identical;
- an opener over five words and the flat keys are refused;
- the opener rule asks for no comma;
- the bundled config ships the lines switched off;
- the reach-test holdout checks the signature stays off.

## Alternatives considered

- **Keep one pool and drop elements per video type.** An earlier fix dropped the transition from step-list tutorials. A per-type pool says the same thing in the config and lets the topic transitions be written for the turn to the steps.
- **Keep the comma join and move the opener after the first sentence.** This would contradict REQ-CNT-045, which puts the opener at the start, and still leave a sentence of its own.
- **Empty pools as the off switch.** Then the lines could not ship until the readout, and turning the feature on would mean writing them under time pressure.

## Rollout

Ships off: `script_templates.signature.enabled: false`, and `tests/test_reach_test_holdout.py` fails if it is set during the reach test. Turn it on after the readout (#540), topic arm first, when one batch with it on shows "viewed vs. swiped away" and average percentage viewed no worse than a batch without it. The topic opener sits in the first seconds, where the cost would show.

Remove the switch when: `enabled` has been on in the bundled config for 30 days with no measured loss. The key and the off path then go in a minor release with a `**Breaking**:` entry.

## Open questions

- **Use rates.** 0.4 and 0.5 are judgement calls, with no source behind them.
- **A recognisable core.** Without the tagline echo, no line carries the distinctive-asset case. A tagline line that makes no time claim could, if one reads naturally.

## As built

With a topic opener drawn, the symptom-first and mistake-first topic templates open on the task instead: live scripts for `topic_symptom_cause` and `topic_mistake_fix` began "Here's how to stop your iPhone battery draining overnight" and "Easy way to fix your iPhone battery draining overnight is to turn off Background App Refresh". They read well, but the opener overrode those templates' opening, which narrows the variety the templates exist for, so openers are limited to the task-first templates (below). A step-list tutorial can now draw a transition, from the topic pool.


Built as designed, and on since the end of the reach-test hold ([decision 0014](../decisions/0014-the-reach-test-hold-ends-when-its-posts-are-queued.md)). With the bundled lines on and both use rates at 1.0, one live script per arm placed every drawn line where its rule puts it. The topic script opened "The fast way to turn off Background App Refresh on iPhone is to go into your Settings", with no comma and the search phrase in the first sentence. The product script drew no opener, used "Here's what matters." mid-script and signed off with "That's the real picture." before the call to action.

Openers are limited to the task-first templates (`opener_templates`). With the bundled config on and the opener rate at 1.0, live scripts per template opened: `topic_answer_first` "The fast way to stop your iPhone battery draining overnight is to turn off Background App Refresh"; `topic_symptom_cause` "Your iPhone battery drains overnight because of Background App Refresh"; `topic_mistake_fix` "You're likely leaving Background App Refresh on, and that's what's killing your iPhone battery overnight".

A live sample with it on had a product script speak its sign-off before the closing line rather than before the call to action; the script step now moves a misplaced sign-off to directly before the CTA.
