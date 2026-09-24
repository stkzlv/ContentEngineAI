# What Successful Short-Form Creators Do

Research into the techniques behind successful short-form video: published teardowns of top creators, platform statements on how ranking works, and the few measured studies that exist. It complements the existing guides rather than repeating them. Hook text patterns and cut cadence are in [promotional-video-best-practices.md](promotional-video-best-practices.md), tutorial structure is in [tutorial-video-best-practices.md](tutorial-video-best-practices.md), caption typography is in [subtitle-best-practices.md](subtitle-best-practices.md), and voice and music levels are in [audio-best-practices.md](audio-best-practices.md).

The requirements drawn from this research are in [requirements.md](requirements.md), each marked `(planned, #N)` in the section it belongs to, and the technical design for each is in [creator-techniques-spec.md](creator-techniques-spec.md).

Researched September 2026.

## How to read the evidence

Most "technique X adds N% retention" figures online come from tool vendors and publish no method. Each claim here carries a grade:

| Grade | Meaning |
|---|---|
| **A** | Peer-reviewed or registered research, or a platform's own policy or documentation |
| **B** | A large observational dataset with a stated method, a preprint or working paper, agency research, or a platform's first-party ad research. Correlation, not cause. |
| **C** | A creator's or vendor's claim, often with one data point or no method |

Two cautions apply throughout:

- **Ad research is not organic research.** TikTok's creative studies measure paid ads (recall, awareness), not how organic posts are distributed.
- **Several key sources predate 2025.** They are flagged where used.

## The rules that matter (cheat-sheet)

1. **Win the first second.** The first gate on every platform is whether the viewer stops or swipes. Frame 0 should be a finished composition: the product or the result, plus the hook text. No fade-in, no preamble.
2. **Variation between videos is a defence, not polish.** YouTube's "inauthentic content" policy, TikTok's For You feed standards and Instagram's originality rules all target templated output with minimal variation. An automated pipeline is the exact profile they look for.
3. **End on the peak.** A tail after the last line (fading music, a standalone CTA card) looks like an exit point; one creator measured it, nobody has at scale. Cutting it is cheap to test.
4. **Keep effects moderate.** Stimulation follows an inverted U: moderate intensity gets the most likes, shares and comments, heavy effects get fewer.
5. **Put motion on every still.** TikTok lists static images as low-quality content. Slow, varied motion on every image is close to mandatory.
6. **Ask genuine questions, never bait.** A choice or experience question is safe. Asking for a specific word, emoji, share or tag is demoted by Meta and TikTok.
7. **Say the search phrase three times.** In the first spoken line, in the on-screen text and at the start of the caption.
8. **Measure the right thing.** YouTube's "viewed vs swiped away", engaged views, TikTok completion and Instagram sends per reach, not raw view counts.

## 1. How the platforms rank short video in 2026

### YouTube Shorts

- **Ranking inputs (A).** Whether the viewer chose to watch or swiped away, the share of viewers who viewed, average view duration and percentage viewed, likes, and survey responses. Shorts are ranked by "performance and relevancy to that individual viewer". https://support.google.com/youtube/answer/11914225
- **The first gate is "viewed vs swiped away" (A).** YouTube Studio shows it per Short. https://support.google.com/youtube/community-video/273390203/new-youtube-shorts-metric-viewed-vs-swiped-away
- **Views were redefined (A).** Since 31 March 2025 every play and replay counts as a Shorts view. The older measure is now "engaged views", which monetisation uses. Raw Shorts view counts are inflated by loops, so judge performance on engaged views and swipe-away. https://support.sproutsocial.com/hc/en-us/articles/35874991211533-YouTube-Shorts-View-Count-Update-March-2025
- **Likes and comments matter less than retention (B, 2023).** Paddy Galloway studied 5,400 Shorts across 33 channels (3.3 billion views). Shorts below 60% "viewed vs swiped away" rarely performed well, and likes, comments and shares had no strong relationship with performance. https://threadreaderapp.com/thread/1646898356419981315.html
- **No Shorts A/B testing yet (A).** YouTube's title and thumbnail testing excludes Shorts. Testing three cuts of a Short is announced for 2027. https://support.google.com/youtube/answer/16391400 , https://techcrunch.com/2026/09/23/youtube-adds-new-creator-tools-like-video-a-b-testing-dynamic-thumbnails-and-live-dubbing/
- **Custom Shorts thumbnails (A).** Available to Partner Program creators on desktop since July 2026. They show in search, the channel page and home, not in the swipe feed. https://blog.youtube/news-and-events/youtube-studio-custom-thumbnail-updates/

### TikTok

- **Ranking inputs (A).** User interactions, video information (captions, sounds, hashtags) and device settings. Finishing a longer video carries more weight than weak signals. Follower count and past hits are "not direct factors". https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you
- **The feed avoids repetition (A).** It avoids consecutive videos from the same creator or with the same sound. https://www.tiktok.com/transparency/en/recommendation-system
- **No published weights.** The widely shared weight tables and "batches of 300-500 viewers" are folklore.
- **For You feed eligibility (A).** Unoriginal content without new edits, other platforms' watermarks, static images and very short clips, QR codes, and engagement manipulation ("like-for-like", false incentives) are ineligible. A stricter update took effect on 24 September 2026. https://www.tiktok.com/community-guidelines/en/fyf-standards
- **AI content (A).** Realistic AI content and AI-generated speech, including a TTS voiceover, must be labelled. Since November 2025 viewers have a slider to see less AI content, so a labelled video's reachable audience shrinks by however many viewers turned it down. https://newsroom.tiktok.com/more-ways-to-spot-shape-and-understand-ai-content?lang=en
- **Hashtags (A).** TikTok's own advice: "less is more", two or three relevant hashtags. https://www.tiktok.com/creator-academy/en/article/elements-of-tiktok-video?lang=en

### Instagram Reels

- **Top signals (A, via secondary quotes).** Adam Mosseri, January 2025: watch time, likes per reach and sends per reach. Sends weigh slightly more for viewers who do not follow the account. https://blog.hootsuite.com/instagram-algorithm/
- **Hashtags do not add reach (A).** They only categorise. Posts have been capped at 5 hashtags since December 2025. https://www.socialmediatoday.com/news/instagram-implements-new-limits-on-hashtag-use/808309/
- **Originality (A).** Since 30 April 2026, accounts that mainly post content they did not create are not recommended to non-followers. Watermarks and speed changes do not count as transformation. "Unique text, creative edits, and voiceover" do. https://techcrunch.com/2026/04/30/instagram-restricts-reach-of-content-aggregators-in-new-crackdown/
- **Trial Reels (A).** A Reel can be shown to non-followers first, with automatic sharing to followers if it performs in the first 72 hours. Whether third-party publishing tools can post Trial Reels is not confirmed. https://creators.instagram.com/blog/instagram-trial-reels
- **Captions are searchable on the web (A).** Since July 2025 public professional posts can be indexed by Google and Bing.

### Across all three

- **Cross-posting your own master is fine.** What is penalised is another platform's watermark or a re-upload of a platform download.
- **Engagement bait is demoted (A).** Meta demotes asking for specific words, emojis, votes, shares or tags, and demotes repeat offenders harder. Genuine requests for advice or opinions are exempt. https://transparency.meta.com/features/approach-to-ranking/content-distribution-guidelines/engagement-bait/
- **AI disclosure costs a little engagement (A).** A study of about one million TikTok posts found AI disclosure cut engagement by 7-8%, through lower perceived creator effort rather than lower quality. Disclosure is still mandatory where it applies. https://academic.oup.com/jcr/advance-article/doi/10.1093/jcr/ucag013/8672493
- **No measured study compares faceless or AI-voice content with on-camera content.** Treat claims either way as untested.

## 2. Hooks and the first frame

- **Key message in the first three seconds (B, ads, 2021).** 63% of TikTok's highest-click-through ads show the key message or product within three seconds. The first 2-2.5 seconds carry most of the recall. https://ads.tiktok.com/business/creativecenter/quicktok/online/Power_Creative_Elements/pc/en
- **Treat the opening like a thumbnail (B, 2023).** Galloway's advice from the swipe-away finding above.
- **The first frame must work on mute (C).** Jenny Hoyos, who averages about 10 million views per Short, also keeps hooks at a fifth-grade reading level and foreshadows the payoff within three seconds. https://podcast.creatorscience.com/jenny-hoyos/
- **Concrete beats mysterious (A).** A registered meta-analysis of 8,977 headline tests found concreteness follows an inverted U. Name the product, number or situation, and withhold only the answer. "You won't believe this" is past the optimum. https://www.nature.com/articles/s41598-024-81575-9
- **Negative framing lifts clicks (A).** Each negative word raised click-through by about 2.3% across 105,000 headline variants. It supports honest "mistake" and "don't buy" hooks. https://www.nature.com/articles/s41562-023-01538-4
- **Hook length (B).** Across 4,148 hooks from million-view videos, the median Shorts hook was 11 words. A third used "you" or "your", and no single feature tracked with higher view tiers. https://www.overseeros.com/blog/best-youtube-hooks
- **Specific questions (B, confounded by topic).** Across 14,424 videos from 355 accounts, spoken question hooks averaged 10.08 times account baseline views against 7.04 for statements. Only specific, audience-targeted questions won. https://thecontentlabs.app/blog/question-hooks-data-study
- **Text overlay should stand out while fitting in (A).** https://journals.sagepub.com/doi/10.1177/00222429251322773
- **Visual and spoken hooks should complement, not duplicate (C).** The on-screen text must work alone for muted viewers. This is already how the authored hook headline works.

## 3. Story structure and pacing

- **Hook, retain, reward (C).** Alex Hormozi's model: the hook earns the watch, the middle pays small curiosity loops, the end fulfils the hook's promise. His testing method is the useful part for automation: one fixed body with several hooks, then iterate on the winners. https://dickiebush.substack.com/p/i-invested-45000-in-alex-hormozis
- **Foreshadow, then progress visibly (C).** Hoyos uses "but" and "then" beats and a visible progression ("three steps") so viewers stay to the end.
- **"But" and "therefore", never "and then" (C).** A screenwriting rule that suits a 90-word script: each sentence causes or complicates the next. https://thescriptlab.com/features/screenwriting-101/13636-how-south-park-creators-plot-better-scripts/
- **Pay off the promise immediately (C).** The leaked MrBeast production memo, written for long-form: the video must confirm what the thumbnail promised at once, and re-engage with a new high point at planned intervals. For Shorts that means the hook headline and first frame must match the title and cover, and one mid-video "re-hook" around the middle. https://simonwillison.net/2024/Sep/15/how-to-succeed-in-mrbeast-production/
- **Storytelling outperforms other types on views (B, working paper, small study).** A four-month field experiment across 202 TikTok posts. https://mpra.ub.uni-muenchen.de/123280/1/MPRA_paper_123280.pdf
- **One honest trade-off is supported (A).** Two-sided messages raise credibility, best when the negative is small, real and tied to a positive. https://www.sciencedirect.com/science/article/abs/pii/S0167811606000267
- **Pace (weak evidence).** No study sets an optimal words-per-minute for Shorts. 150-170 WPM (about 80-125 words for 30-45 seconds) is the common practice. Lab work found fast speech reduced listeners' ability to judge arguments. https://journals.sagepub.com/doi/10.1177/01461672952110006
- **Length (B).** Hoyos targets 34 seconds. Galloway found Shorts that held viewers past 40 seconds were favoured. 30-45 seconds is defensible; lengthen only when the retention curve holds.
- **Machine-written phrasing has known tells (C).** Whether they cost engagement is unmeasured; the AI-disclosure finding in section 1 is about labels, not phrasing. The tells are well catalogued: "it's not X, it's Y", reflexive lists of three, "delve", "game-changer", "seamless". https://www.pangram.com/signs-of-ai-writing

## 4. Editing and visual effects

- **Moderate stimulation wins (B, preprint).** A 2026 study of 1,200 rated short videos, validated on 14,492 more, found message sensation value (cuts, motion, sound and text combined) raises engagement up to a point, then heavy effects reduce likes, shares and comments. https://arxiv.org/abs/2604.19995
- **Filters and pace (B, preprint, one study).** In 9,654 brand TikToks, original spoken audio had the largest modelled effect on likes, having no visual filter predicted better performance, and editing pace had only modest predictive value. https://arxiv.org/html/2606.16053
- **A visual change every 3-5 seconds, not hyper-cutting (C, measured on four videos).** Ali Abdaal's Shorts change visual state about every 5 seconds, with list items every 3 seconds and at most one punch-in zoom. https://www.writepanda.ai/blog/how-to-edit-shorts-like-ali-abdaal/ . "A cut every 2 seconds" has no measured support.
- **Transitions (B, ads).** In TikTok's ad coding, seamless transitions gave 14% more view time and surprising transitions 53% more brand recall.
- **Motion on stills (A policy, C timing).** TikTok's feed standards treat static images as low quality. How long a still can hold before viewers leave is unmeasured.
- **Callouts count as transformation (A for originality).** Instagram names "creative graphics or contextual overlays" among the edits that make content original. No retention data exists for arrows or circles.
- **Colour (B, preprint, weak).** Filters predicted worse performance in the brand study. Keep the grade neutral with mild contrast.
- **Progress bars (C).** Only the vendors that sell them report gains, and a bar on every video is a template tell.
- **Flashes (A, safety).** Never more than three flashes per second (WCAG 2.3.1). https://w3c.github.io/wcag21/understanding/three-flashes-or-below-threshold.html
- **Automation pitfall.** FFmpeg `zoompan` rounds positions to whole pixels, so slow moves shudder unless the image is upscaled first or animated with a sub-pixel `crop`.

## 5. Sound

- **Most TikTok viewers have sound on (B, older).** 93% of users keep sound on, which makes voice, music and effects at least as important as captions. https://ads.tiktok.com/business/en/blog/kantar-report-how-brands-are-making-noise-and-driving-impact-with-sound-on-tiktok
- **AI voice carries a measured penalty that pitch narrows (A, ads).** AI voiceovers drew lower engagement than human voices on real TikTok ads, and the gap narrowed with a lower-pitched AI voice. https://www.sciencedirect.com/science/article/abs/pii/S0268401225000945
- **Listeners often cannot tell (B, agency research).** WPP Media found listeners identified generic AI voices less than half the time. AI voices matched human ones on attention and purchase intent, but voices believed human scored higher on relatability. Prosody aligned with the information structure raised human-likeness. https://www.wppmedia.com/news/ai-voices-audio-ads
- **Tempo shifts mood, not attention (A).** Fast music (108 BPM and up) raised arousal and purchase intent in short ad studies, while an EEG study found tempo did not change attention. https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1236006/full
- **Cuts on the beat feel better (A).** Cuts on accented downbeats, even unnoticed ones, increase perceptual pleasure. https://www.sciencedirect.com/science/article/pii/S030439402200180X
- **Sound effects: sparse, not on every cut (C).** No retention study exists. Editors agree an effect on every cut is worse than none. Reserve them for the hook, the reveal and the CTA.
- **Voice polish for TTS (C).** The usual chain is a high-pass filter, a small cut at a harsh 2-4 kHz peak, gentle compression, de-essing, a slight air shelf and a limiter.

## 6. Endings and loops

- **End on the peak (C, one creator's data point).** Hoyos found a one-second retention cliff at the end of a Short. Cutting it took retention from 83% to 88%, and the Short took off. A trailing music fade or a standalone CTA card is that "goodbye second".
- **Loops count on YouTube (A).** Every replay is a view since March 2025, though engaged views exclude loops. A seamless loop, where the last line or frame flows into the first, raises the replay signal.
- **No end screens on Shorts (A).** The substitute is the "related video" link. https://support.google.com/youtube/answer/14075157

## 7. Packaging and search

- **Covers matter off-feed (A, C).** The feed autoplays, so the cover barely affects feed reach. It matters for the profile grid, search and the follow decision. Instagram's grid crops to 3:4, so cover text belongs in the centred 3:4 area.
- **The search phrase goes in three places (A mechanism, C effect sizes).** TikTok reads captions, on-screen text and transcribed speech. Put the query phrase in the first spoken line, the on-screen text and the start of the caption. The "300-500% more search views" figures are unsourced.
- **Short captions (B).** Instagram posts under 30 words had higher engagement across 9.1 million posts. https://www.socialinsider.io/blog/instagram-caption-length/
- **YouTube Shorts titles (B, descriptive).** Across 10,000 trending Shorts, the median title was about 8 words, 20-40 characters.

## 8. Series, consistency and community

- **Frequency (B, correlational).** Buffer's data across 100,000+ users: 3-5 posts a week roughly doubles follower growth over 1-2, with reach per post compressing as frequency rises. https://buffer.com/resources/social-media-frequency-guide/
- **Replying to comments (B).** A within-account comparison across about 2 million posts found 21% more engagement on Instagram posts that got replies, with causation uncertain. https://buffer.com/resources/instagram-comments-engagement/
- **A closing prompt drives comments (B, preprint, one study).** Endings that invite a response were the main comment predictor in the brand TikTok study.
- **Consistent format, varied content.** A recognisable format helps (MKBHD, Khaby Lame), but templated sameness is what the inauthentic-content rules target. Keep the format, vary everything inside it.
- **"Follow for part 2" is a risk (A).** TikTok's feed standards exclude false incentives and payoffs withheld to force a follow. An honest, self-contained series linked by YouTube's related video or a playlist is fine.

## 9. What does not transfer to a faceless pipeline

- **Personality.** Mrwhosetheboss: "you have to be a person". Faces and expressions carry a lot of what top creators do.
- **Spectacle.** MrBeast's stunts, Zach King's practical-magic reveals and cinematic sound design are production, not technique.
- **Tech-channel teardowns.** No rigorous published analysis of top tech-gadget or tech-tips Shorts channels was found. Niche advice (hidden features, "settings to turn off", comparisons) is opinion, though comparison and "mistake" framing are backed by the concreteness and negativity findings above.

## 10. Folklore to drop

- Hashtags boost Instagram reach. They do not.
- TikTok favours big accounts. Follower count is not a direct factor.
- Fixed numeric weight tables for any platform.
- You can A/B test Shorts titles. Not until 2027, and then it is cuts.
- AI labels kill reach. YouTube says they do not; TikTok's effect runs only through viewers who turn AI content down.
- "Most viewers watch on mute." True for some feeds, not TikTok.
- Any unsourced "+N% retention" figure for an effect.

## 11. Honest gaps in the evidence

- No public controlled study measures the organic retention effect of an individual edit effect (zooms, SFX, transitions) on Shorts or Reels.
- No study compares faceless or AI-voice content reach with on-camera content.
- The strongest numbers on hooks and transitions measure paid ads, not organic posts.
- Galloway's Shorts study is from 2023, before the view-count change.

The durable guidance is the platform policies (originality, bait, disclosure, hashtag caps) and the few measured findings (first seconds, moderate stimulation, concreteness, voice pitch). Everything else is a hypothesis for the pipeline's own A/B data, which the specs are designed to produce.
