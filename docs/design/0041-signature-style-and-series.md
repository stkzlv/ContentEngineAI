# 0041. Signature style and recurring series

- **Status:** Accepted
- **Issue:** #857
- **Requirements:** REQ-CNT-186 to REQ-CNT-195, REQ-VID-166 to REQ-VID-168, REQ-PUB-168, REQ-PUB-169

## Context

`narrator_identity` (design 0031) pins what every render of a content kind keeps the same: the voice, a sign-off, a sting, music and a caption template. A recurring narrator also needs a recognisable way of opening, a sonic cue, a running bit, a script shape that holds viewers, and recurring series. Today each of these lives in hand-written prompts or nowhere, and none can be switched per kind. Roadmap item 4.8 plans series with an episode counter, but numbering doesn't suit standalone episodes.

Evidence, graded per [evidence grades](README.md#evidence-grades):

- **Creators repeat a stance, a fixed opening move, a visual system and named series; an audio logo on every video is rare.** Kurzgesagt's narrator "narrates every video in his signature tone"; Daniel Mac asks the same question in every video; Will It Blend? opened every episode with the same line for 14 years (C: creator pages and case reports; survivorship bias).
- **Early sonic assets and brief repeated brand cues reduce avoidance.** In 887 short-form TikTok ads, sonic assets in the first two seconds gave the largest brand-awareness lift and logo overlays reduced it (B: System1 and TikTok, ads). A centrally placed brand raised ad avoidance; "brand pulsing", short repeated appearances at the same total exposure, reduced it (A: Teixeira, Wedel and Pieters, Marketing Science 2010).
- **Misconception first.** Explainers that state and then refute a common misconception taught more than plain exposition (B: Muller's research, secondary report). YouTube advises moving top moments earlier because "audience sizes typically decrease over the length of the video" (A: YouTube Help).
- **Wrap-up cues precede drop-off** in long-form hosted video (C: Wistia); nothing tests a sign-off on short-form.
- **Self-introductions:** no evidence either way; no short-form-native creator in the sample opens with one (C).
- **Numbering:** YouTube calls non-serial shows, "watched in any order and listed newest to oldest", the most common type; a Shorts series is a playlist of Shorts up to 180 s, shown with a "watch series" button (A: YouTube Help). Instagram linked Reels add a next button (A: Instagram via TechCrunch, no data).
- **Repetition wears out and templating is penalised.** Liking follows an inverted U with exposure (A: Montoya et al. 2017). TikTok makes unoriginal or reproduced content ineligible for For You (A); YouTube's inauthentic-content policy targets mass-produced templated video (A).

No study tests any of these levers on organic short-form, so every feature ships behind a switch and is judged on the channel's own swipe-away and retention data.

## Goals

- Per content kind: an opening rule, a self-introduction ban, a title-card switch, a motif, a running bit with a cap, and timing targets for the script's shape.
- Named series as config: a title prefix, an opening line, a verdict from a fixed set with a verdict beat, optional numbering, platform grouping, and a claim queue whose status follows each episode.
- Every key empty or off in the shipped config; an installation sets its style in its private overlay.

## Non-goals

- Finding and verifying new claims for a series queue. That is editorial research, done by a person or an agent outside the producer.
- A visual-variety rule. The variety guard (design 0006) already spreads unpinned dimensions; series episodes reuse it.
- Choosing the style itself (the words, the motif, the timings). Those are each installation's values.

## Design

### Opening and self-introduction (narrator identity)

`narrator_identity.<kind>` gains:

- `opening`: `null` (default) or `claim_first`. With `claim_first`, the script prompt asks for the take's claim, with one concrete number or name where the detail has one, as the first sentence, and no greeting or setup before it.
- `self_intro_patterns`: a list of regular expressions (default empty). A first sentence matching one is rejected like any soft check, ranked after the script lint in REQ-CNT-164's keep order. The shipped list is empty; an installation adds its greeting forms and its narrator's name.
- `title_card`: `null` (default, `video_settings.hook_overlay` decides) or `false`, which renders the kind without the hook title card whatever that setting says.

### Motif (narrator identity)

`narrator_identity.<kind>.motif`: `{path, at, volume_db, max_sec}`, default `null`.

- `at` lists placements: `first_word` (under the first spoken word) and `turn` (under the turn sentence's first word). Times come from the word timings after transcription.
- The clip is trimmed to `max_sec` (bundled 1.5 s) with a short fade, mixed under the voice at `volume_db`, and mastered with the rest. It sits beside `sting`, which keeps its start or end position.
- The turn sentence is the one the script step names (below); without one, the `turn` placement is skipped and logged.

### Running bit (narrator identity)

`narrator_identity.<kind>.running_bits`: a list of `{id, hint}` (default empty), with `running_bit_cap: {per_batch, min_gap}` (defaults 1 and 2).

- While the cap allows, the script prompt offers one bit's hint and asks the model to use it only where it fits the line; the render record stores the bit's id when the model reports using it.
- `per_batch` counts uses in the current batch run; `min_gap` is the number of renders of the kind since the last use, read from the render-choices records.

### Script shape (structure targets)

`script_structure.<kind>` (default `enabled: false`):

- `turn_by_sec`, `payoff_by_sec` and `signoff_max_sec`: timing targets for the turn sentence, the first payoff sentence and the sign-off.
- `setup`: `any` (default) or `misconception`, which asks for the setup framed as what most people believe, not as a chronology.
- `no_recap: true` asks for no summary line before the closing question.
- The script step asks the model to name the turn and first-payoff sentences by index; both are stored in state.
- **Before voicing**, the producer estimates each named sentence's start from the word count and the block's `wpm` (bundled 150, a generic speaking rate; an installation sets its voice's measured rate). A miss is logged, or, with `on_miss: retry`, treated as a soft check ranked after the self-introduction check in REQ-CNT-164's keep order.
- **After transcription**, the producer measures the same starts from the word timings and records them, with the targets, in the render record.
- `variants`: named overrides of the targets (for example a payoff at 12 s against 25 s). One is drawn per render, and its name is recorded as a render-choices dimension, so a retention read compares variants.

### Series

```
series:
  <id>:
    title_prefix: ""          # e.g. "Myth check: "
    opening_line: ""          # e.g. "Myth check: {claim}." ; {claim} is the topic's claim; empty = none
    verdicts: []              # e.g. [YES, NO, PARTLY]
    verdict_beat: {enabled: false, colours: null}
    numbering: none           # none | episode
    serial: false
    queue_file: null
    credit_commenter: false
    groups: {youtube_playlist_id: null, tiktok_playlist: null, instagram_link: false}
```

The shipped `series` map is empty.

- **Topic fields.** A topic marks `series: <id>`, `claim`, `verdict` and optionally `credit` (a commenter's handle). The producer rejects a verdict outside `verdicts`, and an id that isn't configured.
- **Script and title.** The opening line is filled in and required word for word as the script's first sentence. The title and the platform captions start with `title_prefix`, then the episode number under `numbering: episode`, then the search phrase the hook rules lead with (REQ-CNT-054), so the two rules agree. `numbering: episode` requires a `queue_file`, and config validation names the series that lacks one. With `none`, nothing is numbered.
- **Verdict beat.** With `verdict_beat.enabled`, the sentence that states the verdict gets a stamp card showing the verdict label in `verdict_beat.colours` (default: the caption template's colours). The script step names the verdict sentence by index, as it names the turn; the card lands on that sentence's first word, or on the label's first spoken mention after the opening where the script says the label itself. With neither, the card is skipped and the skip is logged.
- **Commenter credit.** With `credit_commenter` and a `credit` on the topic, a lower-third line names the commenter while the claim is spoken.
- **Claim queue.** `queue_file` is a CSV with `id, claim, verdict, sources, status, episode, output_id, post_id`. Under `numbering: episode`, the producer fills an empty `episode` when it loads the series topic, before the script and description steps, with one more than the highest in the file, and writes it at once under a file lock, so two renders in one batch never share a number. It sets the row to `rendered` with its `output_id` when the render succeeds. The publisher sets it to `scheduled` with its `post_id`, and to `live` once the post-status check sees it published. Other statuses are left alone, so editorial states (`queued`, `picked`, `dropped`) belong to whoever edits the file.
- **Grouping.** After a series post goes live, the publisher adds it to the configured YouTube playlist and TikTok playlist, and links it to the previous Instagram episode, where the publishing provider exposes that call. Otherwise it logs the manual step with the post's URL. Making the playlist a Shorts series is a one-time Studio setting, outside the pipeline.

### Config validation

Every new key is optional, and an unknown series id or verdict fails config or topic validation with the key named. Nothing changes when all keys are at their defaults.

### Tests

- Each key at its default leaves the prompt, the mix, the title and the captions unchanged (snapshot of the prompt and the ffmpeg graph).
- A first sentence matching a pattern is rejected and retried.
- The motif lands within one frame of the first word and the turn word on a fixture with known timings.
- The running-bit cap holds across a batch of three renders.
- A structure variant is drawn, recorded and reported.
- Series validation, the title prefix with and without numbering, the verdict beat's time, and queue status writes for rendered, scheduled and live.

## Alternatives considered

- **One free-text "style prompt" per kind.** Cheapest, but nothing in it can be checked, switched or measured. The opening rule and the timing targets are the parts worth checking.
- **Numbered episodes by default** (roadmap 4.8 as written). Standalone episodes read cold, and YouTube's non-serial type is the common one. Numbering stays an option for serial formats.
- **Motif as a branded intro before the voice.** It delays the claim, and the ad research found centrally placed branding raised avoidance. The motif goes under the voice instead.

## Rollout

Ships off: every key is empty, null or false in the shipped config, so a render is unchanged. An installation sets its style in its private overlay. Each feature is turned on alone and read against renders without it, using swipe-away or skip rate, 3 s retention and the curve at the changed second.

Remove the switch when: the switches stay as lasting options. They describe one installation's style, and the generic default is no style.

## Open questions

- Whether the verdict beat reuses the drawn-beat renderer or a fixed card template.
- Whether the publishing provider exposes playlist and linked-Reel calls, or the grouping stays a logged manual step.
- Whether `min_gap` should count renders or days.
