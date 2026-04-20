# Test Coverage Follow-Ups

Deferred test-coverage gaps discovered during focused test audits. Each
item is narrow enough to pick up independently. Nothing here is a
regression — these are gaps that pre-date the audit and were called out
deliberately out of scope for whatever PR surfaced them.

Related docs: [testing.md](testing.md) (coverage targets: unit >90%,
integration >80%).

---

## 1. Gemini TTS error-path coverage

**Priority**: low — happy path is covered; error paths are defensive
code that rarely fires in practice.

**Status**: not started.

### Context

`src/video/tts.py::_generate_gemini_speech` has a retry loop with
handlers for `OSError`, `GoogleAPIError`, `DeadlineExceededError`,
`FailedPreconditionError`, `DefaultCredentialsError`, `TimeoutError`,
and a catch-all `Exception`. As of PR #78 the happy path is covered
by `test_gemini_text_padded_with_tail_token`, but the error branches
at the following lines are not exercised by any unit test:

- `src/video/tts.py:557-558` — `DefaultCredentialsError` early break
- `src/video/tts.py:578-592` — generic exception + retry-exhausted
  cleanup (`output_path.unlink`)
- `src/video/tts.py:485-486` — empty `gemini_voices` list (no Gemini
  voices in catalog) warning path
- `src/video/tts.py:492` — `_filter_and_select_voice` returns None
  (no matching voice)

Compare to `_generate_google_cloud_speech`, whose error paths are
covered in `tests/test_tts.py`. The Gemini helper has no equivalent
coverage in that file (grep: zero hits for `gemini` in test_tts.py).

### Why low priority

- These branches are defensive against GCP credential/availability
  issues, not application logic bugs.
- When they fire, the outer `generate_speech` already catches the
  exception and falls back to Google Cloud TTS — covered by
  `test_gemini_failure_falls_back_to_google_cloud`.
- The Google Cloud helper's equivalent retry/cleanup logic is already
  tested, so the shape of the error handling is validated.

### Acceptance criteria

- [ ] Add a test class `TestGenerateGeminiSpeechErrors` in
  `tests/test_tts.py` (or extend the existing Gemini test block in
  `tests/test_tts_voice_profiles.py`).
- [ ] One test per error class: `GoogleAPIError`, `TimeoutError`,
  `DefaultCredentialsError`, retry-exhausted path.
- [ ] One test for the "no Gemini voices in catalog" guard (line
  485-486).
- [ ] Each test asserts `output_path` is cleaned up on failure and
  the function returns `(None, None)`.
- [ ] Reuse the mocking pattern from
  `test_gemini_text_padded_with_tail_token` (patch
  `_global_google_cloud_client`, `_fetch_available_voices`,
  `_filter_and_select_voice`, `texttospeech`, `aiofiles.open`).

### Effort

1-2 hours. Mostly boilerplate once the first error test establishes
the mock scaffold.

---
