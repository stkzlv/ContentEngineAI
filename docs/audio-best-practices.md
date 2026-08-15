# Audio Best Practices

The sound-on layer for short-form vertical promotional video. The
[promotional-video-best-practices.md](promotional-video-best-practices.md)
doc treats sound-off as the primary audience (85% of views); this doc covers
the other 15% and the parts of the audio track the algorithm reads even when
the viewer has the sound off.

**Audience**: 30-60 second 9:16 vertical product videos for TikTok, Instagram
Reels, and YouTube Shorts.

**Related docs**:
- [promotional-video-best-practices.md](promotional-video-best-practices.md)
  -- promo strategy; treats captions as the primary channel for the sound-off
  majority.
- [tutorial-video-best-practices.md](tutorial-video-best-practices.md) —
  the how-to counterpart: answer-first structure, search discovery,
  and why durability needs a longer measurement window.
- [subtitle-best-practices.md](subtitle-best-practices.md) -- caption design,
  including the timing smoother that leads the audio.
- [tts-voice-profiles.md](tts-voice-profiles.md) -- TTS voice selection and the
  voice-profile config.

---

## The rules that matter (cheat-sheet)

1. **Audio is a search signal even on mute.** TikTok 2026 transcribes the
   spoken track via ASR and indexes the transcript. The hook keyword must land
   in the first 5 s of spoken audio, not just on-screen.
2. **Voiceover sits at -3 to -6 dB peak (~-14 LUFS integrated).** Clear,
   foreground, never fighting the music.
3. **Music ducks 18-24 dB under the voice.** Background music sits around
   -25 to -30 dB while narration plays; it can rise to -6 to -10 dB in
   voice-free beats (intro sting, outro).
4. **Lead with an audio hook in the first 3 seconds.** A beat drop, a spoken
   punchline, or a recognizable sound, aligned with the visual hook.
5. **Original audio over borrowed trending sound for product video.** A clear
   product voiceover beats a trending song the viewer can't act on. Trending
   sound helps discovery only inside its first 3-7 day window and rarely fits a
   narration-led review.
6. **Normalize to each platform's loudness target** so the video isn't
   auto-attenuated louder or quieter than the feed around it.

---

## 1. Audio as a ranking signal

TikTok 2026 runs ASR over the spoken track and indexes the transcript
alongside captions and hashtags. This makes the voiceover a search surface, not
just an accessibility layer. Practical consequences:

- The hook keyword (product category, price band, audience cue, pain point)
  has to be **spoken** in the first 5 seconds, not only shown. The project's
  script templates already require this in line one.
- Clear diction beats stylized delivery for the indexable portion. A muddy or
  over-processed voice degrades the transcript and the search match.
- Original spoken audio can itself become discoverable. A borrowed trending
  song indexes as the song, not as your product.

## 2. Voiceover and music levels

Mix targets converge across audio-for-video guidance:

| Element | Level | Notes |
|---|---|---|
| Voiceover / narration | -3 to -6 dB peak | The foreground. Compress lightly for consistent level. |
| Integrated loudness (voice) | ~-14 LUFS | Matches streaming/social loudness norms |
| Music under voice | -25 to -30 dB | 18-24 dB below the voiceover |
| Music in voice-free beats | -6 to -10 dB | Intro sting, outro, B-roll moments |
| Sound effects (caption pops) | ~-18 dB rel. voice | Only on emphasized words, not every segment |

**Pipeline mapping.** Levels live in `audio_settings` in
`config/video_production.yaml`: `music_volume_db` (default -24.0) and
`voiceover_volume_db` (default 3.0). The -24.0 music level sits at the top of
the "music under voice" band above. Fades are `music_fade_in_duration` /
`music_fade_out_duration`.

These two numbers are not the same measure as the target levels in the table.
`voiceover_volume_db` and `music_volume_db` are per-track gain offsets applied
at the `amix` stage (the voiceover track gets +3 dB, the music track -24 dB),
not absolute peak or LUFS targets. The "-3 to -6 dB peak" figure is the goal
for the voiceover's peak level in the finished mix. A +3 dB mix offset and a
-3 to -6 dB peak target don't conflict: the offset sets the voice above the
music inside the mix, and the source voiceover level plus that offset is what
lands near the peak target. There's no separate peak-normalization or
loudness-target stage in the pipeline yet, so treat the table as the goal and
the config gains as the lever.

**Ducking** (music level keyed to the presence of voice) keeps the gap clean:
the music drops when narration plays and recovers in the gaps, instead of one
fixed level for the whole clip. The assembler's audio builder
(`src/video/assembler/audio_builder.py`) currently mixes music at the fixed
`music_volume_db` with fade in/out, not a voice-keyed sidechain duck. A real
sidechain duck (FFmpeg `sidechaincompress`) is a future improvement.

**`silence_min_duration_sec` is a trim knob, not a level knob.** Audio
trimming and music ducking are separate concerns. The field maps to the ffmpeg
`silenceremove` `start_duration`: the continuous non-silence window the filter
must detect before it stops trimming, so larger values trim MORE, not less, and
audio inside that window is discarded. Short trailing words (under ~0.4 s) get
eaten if the value exceeds the word length, so keep it at 0.1 s or below. It
lives in `config/ai_services.yaml::audio_processing`.

## 3. Trending sound vs. original audio

The trending-sound playbook fits dance/lip-sync/meme content. Product video is
narration-led, so the trade-off is different:

- **Original audio (voiceover) is the default.** The viewer needs to hear the
  product claim, the price, and the CTA. A song can't carry that.
- **Trending sound has a 3-7 day window.** Its algorithmic lift decays fast and
  only applies if you ride it early. For an evergreen product review, the song
  dates the video.
- **A low trending bed under the voice** is a middle path: it picks up some of
  the trend signal without burying the narration. Keep it ducked per section 2.
- **Licensing matters for a commercial pipeline.** Borrowed commercial music
  carries takedown and monetization risk. The pipeline's audio providers
  (Jamendo CC-licensed as primary, then Freesound, with local files as the last
  resort) exist for this reason. The provider chain is configured via the
  `audio_providers` list in `config/video_production.yaml` and tried in order,
  first successful download wins.

## 4. The audio hook

The first 3 seconds need an audio event, not just a visual one:

- A spoken punchline that states the concrete fact (matches the punchline-first
  visual opener).
- A beat drop or musical accent on the cut to motion.
- A recognizable sound effect tied to the product (a click, a pour, a snap).

Align the audio hook with the visual hook and the burned-in hook overlay so the
opening beat lands once, hard, across all three channels (audio, on-screen
text, caption). Mismatched audio and visual hooks split attention in the
1.7-second decision window.

## 5. Platform loudness normalization

Each platform normalizes loudness on playback. Master near the platform target
so the video isn't pushed up or down relative to the surrounding feed:

- Target roughly **-14 LUFS integrated** as a cross-platform compromise.
- True peak below **-1 dBTP** to avoid clipping after platform transcoding.
- Don't over-compress to chase loudness; the platforms attenuate it back down
  and the only audible result is a flatter, more fatiguing track.

## 6. Honest gaps in the evidence

- **Exact platform loudness targets drift and aren't all published.** -14 LUFS
  is a defensible cross-platform compromise, not an official spec for each app.
- **"Trending sound boosts reach by X%" numbers are vendor claims**, not
  controlled trials, and almost all come from non-narration content. Treat the
  3-7 day window as directional.
- **ASR transcript weighting is asserted by TikTok**, not independently
  measured. The defensible posture: clear spoken keywords can only help, so
  prioritize diction over stylization in the indexable opening.

---

## Sources

- [Brandwatch -- TikTok voiceovers guide (2026)](https://www.brandwatch.com/blog/tiktok-voiceovers/)
- [Gumlet -- Audio levels for video: dB guide](https://www.gumlet.com/learn/audio-levels-for-video/)
- [Metricool -- TikTok trends 2026](https://metricool.com/tiktok-trends/)
