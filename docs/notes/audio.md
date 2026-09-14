# Audio Module Notes

<!--
Moved out of CLAUDE.md (#454), which had grown to 224KB -- about 50k tokens
loaded at the start of every assisted session, most of it per-entry history
rather than rules that must be in front of you at all times. The text is
unchanged; CLAUDE.md points here from the section each entry left.

Each entry records a defect and what it cost, so the shape that produced it is
recognisable the next time. Add to it the same way: what broke, why it was
invisible, and what now catches it.
-->

- **Provider platform**: `src/audio/` uses `BaseAudioProvider` ABC + `AudioProviderRegistry` + `AudioManager` chain pattern (same as publisher module)
- **Adding a provider**: create `src/audio/<provider>.py` with `@register_audio_provider` decorator, add enum value to `AudioProvider`, import in `__init__.py`, enable in YAML
- **Provider chain**: configured via `audio_providers` list in `video_production.yaml`. Tried in order, first successful download wins, local files are last resort
- **Jamendo**: uses `client_id` auth only (no OAuth2), `fuzzytags` search for genre/mood matching, random query selection from configured pool
- **Jamendo CDN requires HTTP/2**: `prod-1.storage.jamendo.com` (behind Cloudflare/nginx) serves the actual MP3 only over HTTP/2; HTTP/1.1 requests get a blocking text/html page. aiohttp 3.x only supports HTTP/1.1, so `JamendoProvider.download()` uses `asyncio.create_subprocess_exec("curl", ...)` instead, which negotiates HTTP/2 by default. The search API (`api.jamendo.com`) works fine with HTTP/1.1 and still uses aiohttp. See `src/audio/jamendo_provider.py::JamendoProvider.download`.
- **Jamendo CDN is slow without proxy**: even with HTTP/2 via curl, the Jamendo CDN can be very slow in some network environments (~90 B/s, 3.3 MB tracks time out in 180s). Through a HTTP/SOCKS proxy, downloads typically complete in under 120s. The provider chain falls through to Freesound if Jamendo times out. If a run is missing background music, a proxy/VPN may be needed.
- **Freesound**: `FreesoundProvider` wraps existing `FreesoundClient` (don't modify the 728-line client directly). OAuth2 for full quality, API key for previews
- **Config patterns**: Jamendo uses `audio_providers[].settings` dict, Freesound uses legacy `freesound_*` fields on `AudioSettings`. Both work, new providers should use the `settings` dict pattern
- **CircuitBreaker**: use public `record_success()`/`record_failure()` methods, not private `_on_success()`/`_on_failure()`
- **`silence_min_duration_sec` is a trim-MORE knob, not a trim-LESS knob**: the field in `AudioProcessingSettings` maps to ffmpeg `silenceremove` `start_duration`, which is the continuous non-silence window the filter must detect before it stops trimming. Audio during that window is **discarded**, not kept. So larger values trim MORE aggressively, and short trailing words (under ~0.4s, e.g. "tips", "tech") get eaten if `start_duration` exceeds the word length. Keep at 0.1s or below. Lives in `config/ai_services.yaml::audio_processing`.
