# Audio: why the mix sounds the way it does

This page explains the sound of a render: original voiceover rather than trending sound, the voiceover and music levels, ducking and the final loudness pass. The mix settings live in `audio_settings` in `config/video_production.yaml` (levels, fades, ducking, loudness, the music provider chain); caption template sounds in `config/subtitles.yaml`; voice profiles, TTS and audio trimming in `config/ai_services.yaml`. The requirements are in [the content requirements](../requirements/content.md) and [the video requirements](../requirements/video.md). Voice selection is in [TTS voice profiles](tts-voice-profiles.md), and the defects behind the music chain are in [the audio module notes](../notes/audio.md). The `[A]`, `[B]` and `[C]` marks are defined in [the evidence grades](../design/README.md#evidence-grades).

## What goes into the mix

A render's audio is the TTS voiceover and one background music track; the source videos' own audio is dropped (`REQ-VID-026`). The mix runs for the voiceover's length (`audio_mix_duration: "first"`), so audio and captions stay in sync. Caption templates add no sound while `mute_template_sound_effects` is on, the default in `config/subtitles.yaml` (`REQ-VID-045`).

Why: the audio track reaches most TikTok viewers. 93% of US users spend time with sound on, in TikTok Marketing Science's 2020 data [B] ([TikTok Creative Center](https://ads.tiktok.com/business/creativecenter/quicktok/online/Power_Creative_Elements/pc/en)); muted viewing is more common on other platforms, which is why captions carry the same words.

Built and held off: a voice processing chain on the voiceover before the mix (`audio_settings.voice_chain`, `REQ-CNT-073`): high-pass, a small cut at the harsh 2-4 kHz peak, gentle compression, de-essing, an air shelf and a -1 dBFS limiter. On a bundled voiceover with music, the mastered mix measured -14.2 LUFS with it and -14.8 without, against the -14 target: the limited voice peaks let `loudnorm` stay nearer linear. Each render records whether it was on, beside the voice name (`REQ-CNT-074`).

Planned: a signature sting at the start or end (`audio_settings.signature_sting`, `null` by default, `REQ-VID-027`, held).

Built and held off: sparse event sound effects (`audio_settings.sound_effects`, [design 0003](../design/0003-sparse-sound-effects.md), `REQ-VID-012`). Effects mark the first frame under the hook headline, the start of the sentence after the hook and the start of the last sentence, as the research reserves them, at most two per 10 seconds by default with the call to action and the reveal kept first, and never in a spoken word's first 100 ms. Why sparse: a 2026 study of rated short videos found engagement rises with sensation value up to a point and then falls [B]. The project bundles five public-domain (CC0) effects per event under `static/sfx`, listed with their Freesound sources in `static/sfx/SOURCES.md`; they were picked by name, rating and length, not by listening, so an A/B should start with a listen.

## Original audio over trending sound

The pipeline narrates every render with its own voiceover and never uses a borrowed trending track. Music comes from Creative Commons and local sources: the `audio_providers` chain in `config/video_production.yaml` tries Jamendo, then Freesound, then the local files in `background_music_paths` (`REQ-CNT-077`, `REQ-CNT-078`). Within a provider, candidates are ranked by how many of the search query's words their title or tags carry, a candidate that matches none is dropped, and a provider with no match hands over to the next one (`REQ-CNT-084` to `REQ-CNT-086`). A candidate that matches only a generic word such as `instrumental` still passes, as a fallback after the fuller matches, so a query's mood words decide the track only when a provider returns one that carries them. The chain records each track's attribution (`REQ-CNT-081`) and stays inside `music_search_budget_sec` (default 120 seconds, `REQ-CNT-083`).

Why:

- A product video is narration-led: the viewer needs to hear the product claim and the call to action, and a song cannot carry them.
- Original spoken audio had the largest modelled effect on likes in a 9,654-video brand study [B] ([arXiv 2606.16053](https://arxiv.org/html/2606.16053)).
- TikTok's feed avoids showing consecutive videos with the same sound [A] ([TikTok transparency center](https://www.tiktok.com/transparency/en/recommendation-system)).
- Borrowed commercial music carries takedown and monetization risk for a commercial pipeline, which is why the providers are Creative Commons sources.
- Generic or mood-mismatched music is one of the tells that make an automated video read as AI slop [C]. This is why the chain matches tracks on mood words; [the audio module notes](../notes/audio.md) record the drill instrumental that a calm query once returned.

Not supported: "trending sound boosts reach by X%" figures are vendor claims, not controlled trials, and almost all come from non-narration content. The often-quoted 3-7 day window in which a trending sound gives lift ([Metricool](https://metricool.com/tiktok-trends/)) is directional at best.

## The spoken opening and the search phrase

Every product template instructs the script writer to state a concrete fact, result or observation in the first line (`REQ-CNT-018`), and every topic template to speak the search phrase within the first five seconds (`REQ-CNT-027`). Before TTS, the producer strips speaker labels, stage directions, markdown, emojis and hashtags so none of them is spoken (`REQ-CNT-052`).

Why: TikTok states that it transcribes the spoken track and indexes the transcript beside captions and hashtags ([Brandwatch](https://www.brandwatch.com/blog/tiktok-voiceovers/)), so the voiceover is a search surface as well as an accessibility layer. A borrowed song indexes as the song, not as the product. The weighting of that transcript is asserted by TikTok, not independently measured; clear spoken keywords can only help, so the opening favours diction over stylized delivery.

Planned: a script lint and a report on whether the search phrase appears in the first spoken sentence, the hook headline and each platform caption ([design 0007](../design/0007-script-lint.md), `REQ-CNT-054`, `REQ-CNT-055`).

## Voiceover and music levels

The levels are per-track gain offsets applied before the `amix` stage, in `audio_settings`:

| Key | Default | Effect |
|---|---|---|
| `voiceover_volume_db` | `3.0` | The voiceover track gets +3 dB |
| `music_volume_db` | `-24.0` | The music track gets -24 dB, 27 dB under the voice |
| `music_fade_in_duration` | `2.0` | Seconds of music fade-in |
| `music_fade_out_duration` | `3.0` | Seconds of music fade-out; a `peak` or `loop` ending fades within `peak_margin_sec` instead |

Why: audio-for-video guidance converges on these targets for the finished mix ([Gumlet](https://www.gumlet.com/learn/audio-levels-for-video/)):

| Element | Target |
|---|---|
| Voiceover | -3 to -6 dB peak, the foreground |
| Music under voice | -25 to -30 dB, 18-24 dB below the voiceover |
| Music in voice-free beats | -6 to -10 dB (intro, outro) |
| Sound effects | about -18 dB relative to the voice, sparse (the held effects default to -15 dB, [design 0003](../design/0003-sparse-sound-effects.md)) |

The pipeline's two numbers are not the same measure as these targets. A +3 dB offset and a -3 to -6 dB peak target do not conflict: the offset sets the voice above the music inside the mix, the source voiceover level plus that offset is what lands near the peak target, and the loudness pass below sets the master level. The -24 dB music gain sits just above the "music under voice" band. The pipeline holds the music at one level for the whole clip, so it does not raise the music in voice-free beats.

## Ducking

Voice-keyed ducking drops the music while narration plays and lets it recover in the gaps, using FFmpeg `sidechaincompress` keyed by the voiceover (`src/video/assembler/audio_builder.py`). It is off by default (`music_ducking_enabled: false`, `REQ-CNT-101`), with `music_ducking_threshold` 0.1, `music_ducking_ratio` 4.0, `music_ducking_attack_ms` 20 and `music_ducking_release_ms` 300. The measured duck depth for several threshold and ratio pairs is in the comment above these keys in `config/video_production.yaml`; the default pair ducks about 5 dB.

Why off, and why shallow:

- Ducking changes the sound of every render, and output changes ship off by default ([decision 0002](../decisions/0002-output-changes-ship-off-by-default.md)). The fixed-level mix works on its own.
- `music_volume_db` already puts the music 27 dB under the voice, so a deep duck on top of it makes the music inaudible rather than unobtrusive.
- A duck attenuates and never boosts: in a gap the music returns to `music_volume_db`, not above it. Music that rises in voice-free beats needs a louder `music_volume_db` paired with a deeper duck, not the duck alone.

## Loudness normalization

The assembler masters the finished mix with `loudnorm` (EBU R128), on by default (`REQ-CNT-100`):

| Key | Default |
|---|---|
| `loudness_normalization_enabled` | `true` |
| `loudness_target_lufs` | `-14.0` |
| `loudness_true_peak_db` | `-1.0` |
| `loudness_range_lu` | `7.0` |
| `output_audio_sample_rate` | `48000` |

Why:

- Each platform normalizes loudness on playback, so mastering near the target keeps a video level with the feed around it. -14 LUFS integrated is a cross-platform compromise: exact platform targets drift and are not all published.
- A true peak below -1 dBTP leaves headroom for clipping after platform transcoding.
- Over-compressing to chase loudness gains nothing: the platforms attenuate it back down, and the audible result is a flatter, more fatiguing track. `loudness_range_lu` keeps the dynamics.
- Before the pass existed, two real renders measured -17.4 and -17.6 LUFS with true peaks of -0.1 and -0.2 dBFS: quiet against the target and nearly touching full scale at once. A fixed gain change would clip, and a limiter alone would leave the mix quiet.

What a render measures: real renders land about 1 LU short of the target. The same product comes out at -14.9 LUFS on the FFmpeg subtitle engine and -15.1 on pycaps, both peaking at -0.8 dBFS against a requested -1.0.

- The cause is the true-peak ceiling, not single-pass operation. Mixed narration arrives above 0 dBTP, so the gain that would reach -14 LUFS linearly would breach the ceiling; `loudnorm` refuses linear normalization, falls back to dynamic mode and reports the shortfall as `target_offset` (measured: 1.19 LU).
- A second pass fed only the four `measured_*` values reports the same offset and produces a byte-identical file. The loudnorm author's two-pass also feeds back `offset=<target_offset>`, which closes about half the shortfall (measured -15.2 to -14.6 LUFS, true peak still -1.0). The pipeline does not use it because two-pass needs the mixed audio as a file, and that exists only inside the filtergraph.
- A constant tone lands on -14.0 exactly, because its crest factor never brings the ceiling into play, so a synthetic measurement does not predict a render.
- The 0.2 dB by which the delivered file exceeds the ceiling comes from the AAC encode: `loudnorm` reports `output_tp: -1.00` and the graph's WAV output measures -1.0 dBFS. To stay under -1 dBTP in the delivered file, lower `loudness_true_peak_db`.
- `loudnorm` outputs at 192 kHz whatever it is handed, so the chain resamples to `output_audio_sample_rate` afterwards, whether or not normalization runs.

## Voice and music choice

The default voice profile delivers calm, confident speech rather than high energy (`REQ-CNT-059`), a product gets the same voice on every run (`REQ-CNT-064`), and the voiceover enters the mix with only a volume adjustment. [TTS voice profiles](tts-voice-profiles.md) covers the profiles and selection order. The default Jamendo queries ask for calm moods (`ambient chill`, `soft background`, `calm lofi`).

Why:

- AI voiceovers drew lower engagement than human voices on real TikTok ads, and a lower-pitched AI voice narrowed the gap [A] ([design 0015](../design/0015-narrator-voice-evaluation.md) has the source).
- Fast music (108 BPM and up) raised arousal and purchase intent in short ad studies, while an EEG study found tempo did not change attention: tempo shifts mood, not attention [A] ([Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1236006/full)).
- Cuts on accented downbeats feel better [A] ([design 0005](../design/0005-beat-snapped-cuts.md) has the source).

Built and held off: an optional voice processing chain ([design 0004](../design/0004-voice-processing-chain.md), `REQ-CNT-073`) and visual cuts snapped to music beats ([design 0005](../design/0005-beat-snapped-cuts.md), `REQ-VID-013`). Planned: an evaluation of a distinctive or owned narrator voice ([design 0015](../design/0015-narrator-voice-evaluation.md)).

## Trimming is not a level

`silence_min_duration_sec` in `config/ai_services.yaml` (`audio_processing`, default 0.1) trims silence from the voiceover; it does not set a level. Larger values trim more, not less, and eat short trailing words. [The audio module notes](../notes/audio.md) explain why; keep it at 0.1 seconds or below.
