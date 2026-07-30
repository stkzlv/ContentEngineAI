# TTS Voice Profiles

Voice profiles give each product video a distinct vocal identity through deterministic voice-name selection (the same product always gets the same voice), plus optional inline markup (pauses, whispers). Voice character comes from the selected voice name, not from a text style prompt.

## How it works

1. A voice profile is selected per product using deterministic hashing (md5 of product ID, hex slice `[16:24]`)
2. If the profile uses Gemini TTS, the text is synthesized with the profile's named voice (e.g. Charon) plus optional inline markup like `[short pause]`
3. If Gemini fails, markup is stripped and the standard provider chain handles it (Google Cloud TTS preferring Chirp3-HD voices, then Coqui as final fallback)
4. The selected profile name and voice name are saved to `pipeline_state.json` under `tts_metadata`

## Configuration

Voice profiles live in `config/subtitles.yaml` under `tts_config`:

```yaml
tts_config:
  voice_profiles_enabled: true   # false to skip profiles entirely

  voice_profiles:
    warm_conversational:    # friendly, rate 1.10, pitch +1, ~165 WPM
    soft_intimate:          # soft reviewer, rate 1.05, ~155 WPM
    calm_authority:         # trusted reviewer, rate 1.10, pitch +1, ~165 WPM
    calm_confident:         # relaxed friend, rate 1.10, ~165 WPM
    gentle_storyteller:     # soft narration, rate 1.05, ~155 WPM
    chirp3_natural:         # Google Cloud Chirp3, neutral baseline
    puck:                   # Gemini TTS, pinned to "Puck" voice
    charon:                 # Gemini TTS, pinned to "Charon" voice
    fenrir:                 # Gemini TTS, pinned to "Fenrir" voice
    orus:                   # Gemini TTS, pinned to "Orus" voice

  voice_profile_pool: []         # empty = no random pool (see precedence below)
  default_voice_profile: charon  # pinned voice for unattended runs
```

Set `voice_profiles_enabled: false` to disable profiles and use the default Chirp3-HD path.

## Voice selection precedence

Selection precedence (highest first):

1. `--voice-profile <name>` CLI override (always wins)
2. Non-empty `voice_profile_pool` (random selection across the pool, deterministic per product; the testing / A-B path)
3. `default_voice_profile` (pinned voice for unattended runs)
4. Random across all `voice_profiles` (back-compat fallback when nothing above is set)

The bundled config ships `default_voice_profile: charon` and an empty `voice_profile_pool`, so an unattended `make produce-lowpri` run picks Charon every render. To opt back into random selection (e.g., to A/B test voice candidates), set `voice_profile_pool` to a non-empty list. To pin a different voice, set `default_voice_profile` to one of the profile names.

## Providers

### Google Cloud TTS (Chirp3-HD)

The current default. Uses the Cloud Text-to-Speech API with SSML input. Voices have locale-prefixed names like `en-US-Chirp3-HD-Achird`.

- No style prompt support (SSML only)
- Deterministic voice selection per product (md5 hex slice `[24:32]`)
- Requires: `GOOGLE_APPLICATION_CREDENTIALS` service account

### Gemini TTS

Uses the same `google.cloud.texttospeech` SDK but with `SynthesisInput(text=...)` instead of SSML. Speaking style comes from the selected voice name, not from a style prompt (see the note below).

- Voices use simple names: `Kore`, `Charon`, `Aoede`, `Puck`, etc.
- Requires `model_name` on `VoiceSelectionParams` (e.g. `gemini-2.5-flash-tts`)
- Requires Vertex AI API enabled on the GCP project (`aiplatform.googleapis.com`)
- Same service account auth as Cloud TTS

### Coqui TTS

Local open-source fallback. No style or markup support. Used when cloud providers are unavailable.

## Inline markup

Gemini TTS understands inline tags like `[short pause]`, `[pause]`, `[long pause]`, `[whispering]`, etc. Markup rules in profiles inject these tags into the script text using regex patterns.

When falling back to non-Gemini providers, all markup is automatically stripped so it won't be spoken literally.

## Deterministic selection

Different hash slices prevent correlation between randomized choices:

| Feature | MD5 hex slice |
|---------|---------------|
| Font family | `[0:8]` |
| Color palette | `[8:16]` |
| Voice profile | `[16:24]` |
| Voice name | `[24:32]` |

Same product ID always produces the same combination.

## Pricing (as of Feb 2026)

### Cloud TTS API (current, what we use)

| Voice tier | Free tier | Paid rate |
|------------|-----------|-----------|
| Standard | 4M chars/month | $4 / 1M chars |
| WaveNet, Neural2 | 1M chars/month | $16 / 1M chars |
| Chirp3-HD | none | $30 / 1M chars |

### Gemini TTS via Vertex AI

| Model | Input | Output (audio) |
|-------|-------|----------------|
| Flash TTS | $0.15 / 1M tokens | $0.60 / 1M tokens |
| Flash TTS (Gemini API free tier) | free | free |
| Pro TTS | $1.00 / 1M tokens | $20.00 / 1M tokens |

### Cost per video

A typical script is ~500 characters / ~125 input tokens. A 30-second voiceover produces ~750 output tokens.

| Provider | Per video | 100 videos/month | 1,000 videos/month |
|----------|-----------|-------------------|---------------------|
| Chirp3-HD (Cloud TTS) | ~$0.015 | ~$1.50 | ~$15 |
| Gemini Flash (Vertex AI) | ~$0.0005 | ~$0.05 | ~$0.50 |
| Gemini Flash (Gemini API free) | $0 | $0 | $0 |

### Why we chose Vertex AI (Cloud TTS client)

The Gemini API free tier is tempting, but it uses a different SDK (`google-genai`) with API key auth. Our code already uses the `google.cloud.texttospeech` client with service account auth, which routes through Vertex AI.

At our scale the Vertex AI cost is negligible (~$0.05/month for 100 videos). Keeping the same client avoids maintaining two auth paths and two SDK integrations.

To use Gemini TTS: enable Vertex AI API (`aiplatform.googleapis.com`) in the GCP project.

## Gemini voice catalog

Gemini 2.5 Flash TTS has 30 voices. Google doesn't publish gender labels, so test empirically.

**Best for calm/ASMR delivery:**

| Voice | Descriptor | Notes |
|-------|-----------|-------|
| Achernar | Soft | Top pick for ASMR |
| Enceladus | Breathy | Intimate feel |
| Vindemiatrix | Gentle | Soothing, low energy |
| Sulafat | Warm | Good all-rounder |
| Algieba | Smooth | Polished calm |
| Despina | Smooth | Similar to Algieba |

**Neutral/moderate:**

| Voice | Descriptor |
|-------|-----------|
| Schedar | Even |
| Gacrux | Mature |
| Sadaltager | Knowledgeable |
| Rasalgethi | Informative |
| Charon | Informative |
| Iapetus | Clear |

**Too energetic for product reviews:**

Puck, Zephyr, Fenrir, Leda, Sadachbia, Laomedeia, Autonoe (all bright/upbeat/excitable).

**Too assertive:** Kore, Orus, Alnilam, Pulcherrima (firm/forward).

The bundled `puck`, `charon`, `fenrir`, and `orus` named profiles exist as A/B candidates for picking a default voice; only `charon` ships as `default_voice_profile`. The other three are available via `--voice-profile <name>` for testing the trade-off (energy vs. trust) on your script style.

## Voice character and the `style_prompt` field

Tone and character come from the voice you pick by name (Charon, Puck, Kore, and so on), not from a text prompt. The `style_prompt` field on each profile is documentation only and is not wired to the Gemini API. Early versions passed it as `SynthesisInput.prompt`, but the model reads that text aloud as spoken content instead of treating it as a style directive, so it is no longer sent. Keep `style_prompt` in config to describe a profile's intended character for readers; to change how a profile sounds, select a different voice name.

**Parameters that adjust output:**

| Field | Range | Effect |
|-------|-------|--------|
| `speaking_rate` | 0.25-4.0 | Speed. 0.85-0.93 for calm. Gemini caveat below. |
| `pitch` | -20.0 to 20.0 | Semitones. -1.0 to -3.0 adds warmth. |
| `volume_gain_db` | -96.0 to 16.0 | Global only, not per-profile. |

**Gemini caveat on `speaking_rate`:** empirically, the Gemini TTS API appears to ignore the numeric `speaking_rate` parameter for Gemini-model voices. A 1.05 vs 1.00 A/B render produced near-identical durations (38.23s vs 38.03s). For Gemini voices, pacing is a property of the voice itself, not an API parameter. The rate field is honored on Chirp 3 HD voices via the same Cloud TTS client.

**Markup tags** Gemini understands: `[short pause]`, `[pause]`, `[long pause]`, `[whispering]`. Injected via `markup_rules` regex patterns. Stripped automatically when falling back to Google Cloud TTS.

## Setup

1. Ensure `google-cloud-texttospeech >= 2.29.0` (for Gemini voice support)
2. Enable Vertex AI API in GCP console for Gemini profiles
3. Configure profiles in `config/subtitles.yaml`
4. Run: `poetry run python -m src.video.producer outputs/<ASIN>/data.json <profile> --debug`
5. Check logs for `Selected voice profile` and `TTS metadata` entries
