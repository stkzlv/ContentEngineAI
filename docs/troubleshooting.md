# Troubleshooting Guide

This guide helps you diagnose and fix common issues with ContentEngineAI. Issues are organized by category with step-by-step solutions.

## Quick Diagnostics

### System Check

Run these commands to verify your setup:

```bash
# Check Python version
poetry run python --version

# Check FFmpeg installation
ffmpeg -version

# Check configuration loading
poetry run python -c "from src.video.config_adapter import load_video_config_modular; print('✓ Config loads')"

# Check API keys
poetry run python -c "
import os
from dotenv import load_dotenv
load_dotenv()
keys = ['GEMINI_API_KEY', 'PEXELS_API_KEY', 'FREESOUND_API_KEY']
for key in keys:
    status = '✓' if os.getenv(key) else '✗'
    print(f'{status} {key}')
"
```

### Debug Mode

Always use debug mode when troubleshooting:

```bash
# Enable debug mode for producer
poetry run python -m src.video.producer products.json profile_name --debug

# Enable debug mode for scraper
poetry run python -m src.scraper.amazon.scraper --keywords "product" --debug

# Enable verbose mode (more detailed console output)
poetry run python -m src.scraper.amazon.scraper --keywords "product" --verbose
```

**Debug mode provides:**
- Detailed console logging (DEBUG level)
- Persistent log files:
  - Producer: `outputs/logs/producer.log`
  - Scraper: `outputs/logs/scraper.log`
- Intermediate file preservation in `outputs/{product_id}/temp/`
- FFmpeg command logging to `outputs/{product_id}/temp/ffmpeg_command.log`
- Step-by-step execution traces
- Debug file generation (see Debug Files section below)

**Configuration vs. CLI:**
- CLI `--debug` flag **overrides** config file settings
- Config file: Set `debug_mode: true` in `config/scraper.yaml` (scraper only)
- Debug settings: Configure in `config/performance.yaml` under `debug_settings`

## Installation Issues

### FFmpeg Not Found

**Error:** `ffmpeg: command not found` or similar

**Solutions:**

**macOS:**
```bash
# Install via Homebrew
brew install ffmpeg

# Verify installation
ffmpeg -version
```

**Ubuntu/Debian:**
```bash
# Install via apt
sudo apt update
sudo apt install ffmpeg

# Verify installation
ffmpeg -version
```

**Windows:**
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH
4. Restart terminal and verify: `ffmpeg -version`

**PATH Issues:**
```bash
# Check if FFmpeg is in PATH
which ffmpeg    # macOS/Linux
where ffmpeg    # Windows

# Temporary fix - specify full path in config
# config/video_production.yaml:
ffmpeg_settings:
  executable_path: "/usr/local/bin/ffmpeg"  # Your actual path
```

### Poetry Installation Issues

**Error:** `poetry: command not found`

**Solution:**
```bash
# Reinstall Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Restart terminal and verify
poetry --version
```

**Python Version Issues:**
```bash
# Check Python version
python --version  # Should be 3.12.x

# If wrong version, use pyenv
pyenv install 3.12.7
pyenv local 3.12.7

# Recreate Poetry environment
poetry env remove python
poetry install
```

### `openai-whisper` Fails to Build (`No module named 'pkg_resources'`)

**Error:** `poetry install` fails while building `openai-whisper (20240930)`:
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**Cause:** Whisper's legacy `setup.py` imports `pkg_resources`, which setuptools removed in `>=81`. Poetry's isolated PEP 517 build environment always pulls the latest setuptools, so the build fails on a fresh virtualenv.

**Solution:** Bootstrap a compatible setuptools into the active virtualenv, then let Poetry's build inherit it via system site-packages:
```bash
# Inside the activated ContentEngineAI venv
pip install 'setuptools>=78.1.0,<79' wheel   # 78.x still ships pkg_resources
VIRTUALENV_SYSTEM_SITE_PACKAGES=true poetry install
```
`VIRTUALENV_SYSTEM_SITE_PACKAGES=true` makes Poetry's build env see the venv's setuptools. `PIP_CONSTRAINT` does **not** propagate into Poetry's build subprocess, so that approach does not help.

**Do not** work around this by running `pip install openai-whisper` directly. pip resolves `torch` from PyPI (the multi-GB CUDA wheel) instead of the `pytorch-cpu` source pinned in `pyproject.toml`, and Poetry then treats the venv as satisfied and skips reinstalling the correct CPU build. If this has already happened, recreate the virtualenv from scratch rather than untangling the mixed state:
```bash
pyenv virtualenv-delete -f ContentEngineAI
pyenv virtualenv 3.12.13 ContentEngineAI
```

### Playwright Browser Issues

**Error:** Browser-related errors during scraping

**Solutions:**
```bash
# Reinstall browsers
poetry run playwright install

# Install system dependencies (Linux)
poetry run playwright install-deps

# For specific browser issues
poetry run playwright install chromium
```

### Playwright Chromium on Ubuntu 26.04 (`does not support chromium on ubuntu26.04-x64`)

**Error:** `playwright install chromium` fails with `ERROR: Playwright does not support chromium on ubuntu26.04-x64`. This breaks the pycaps `css` subtitle renderer, which launches Playwright's bundled Chromium. (The `pictex` renderer is browserless and isn't affected.)

**Cause:** Playwright maps the host OS to a browser build. Its platform detection treats Ubuntu 26.04 as `ubuntu26.04-x64`, but the registry only ships builds up to `ubuntu24.04`. Versions through 1.60.0 (current PyPI latest) hit this; the fix first lands in 1.61, not yet published.

**Fix:** force the binary-compatible 24.04 build with `PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64`. Only the one-time browser install needs the manual prefix:

```bash
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 poetry run playwright install chromium
```

At runtime the override is set automatically for the CSS renderer on Ubuntu-like distros at version 26+: the producer applies it in `src/video/pycaps_engine/renderer.py` before any Playwright launch, so it covers standalone `python -m src.video.producer`, the batch pipeline, `make produce-lowpri` / `batch-lowpri`, and tests with no Makefile env wiring. It matches Playwright's own distro set (`ubuntu`, `pop`, `neon`, `tuxedo`). An explicit env var always wins (the code uses `setdefault` semantics).

Once Playwright 1.61+ is on PyPI, bump it and drop the override (the renderer helper no-ops off Ubuntu 26+, so it's safe to leave until then).

### pycaps CSS renderer hangs at `Page.screenshot: Timeout 30000ms exceeded`

**Error:** the `css` renderer launches Chromium and lays out captions, then fails per word with `Page.screenshot: Timeout 30000ms exceeded`. Under the bundled `fallback_policy: fallback_ffmpeg` the run completes: the captions are burned onto the assembled video with a separate FFmpeg pass, built from the Whisper transcript the burn step already required. Under `raise` it aborts rather than ship a caption-less video, and `warn_and_skip` keeps the caption-less file. Installing a virtual display is still worth doing if you want the animated captions the profile asked for; the fallback gives static ones.

**Cause:** headless Chromium can't rasterize a frame for the screenshot when there's no usable X display, which happens on Wayland sessions (and bare headless boxes). Page navigation and layout work, so launch succeeds, but every `page.screenshot` blocks until timeout. Confirmed independent of Chrome version (the bundled 145 build and a system Chrome 149 both hang) and of the platform override above.

**Fix:** give Chromium a virtual display by wrapping the producer in `xvfb-run` (this is the same reason the scraper runs its browser under Xvfb):

```bash
xvfb-run -a make produce-lowpri ARGS="outputs/<ASIN>/data.json slideshow_images1 --pycaps-renderer css --debug"
# bare run:
xvfb-run -a poetry run python -m src.video.producer outputs/<ASIN>/data.json slideshow_images1 --pycaps-renderer css --debug
```

`xvfb-run` ships in the `xvfb` apt package. The `pictex` renderer is browserless and needs none of this, but it is preview-only and not a substitute here: it renders words with no gaps between them (issue #174). Install `xvfb` and keep the `css` renderer for anything you publish.

## API and Authentication Issues

### Gemini API Issues (Primary LLM Provider)

**Error:** `Invalid API key` or `Authentication failed`

**Diagnostics:**
```bash
# Test Gemini API key
poetry run python -c "import os; print(repr(os.getenv('GEMINI_API_KEY')))"
```

**Solutions:**
1. **Get API Key:** Go to [Google AI Studio](https://aistudio.google.com/apikey) and create a key
2. **Check `.env`:** Verify `GEMINI_API_KEY=` has no extra spaces or quotes
3. **Verify it works:** The key should not show `None` in the diagnostic above

### OpenRouter API Issues (Fallback LLM Provider)

OpenRouter activates automatically when Gemini fails. If both providers fail, check:

**Diagnostics:**
```bash
# Test OpenRouter API key
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models
```

**Solutions:**
1. **Check API Key Format:** Should start with `sk-or-`
2. **Verify API Key:** Log into [OpenRouter Dashboard](https://openrouter.ai/) and check key status
3. **Optional:** OpenRouter is a fallback. If `OPENROUTER_API_KEY` is missing, the system still works with Gemini alone

### Google Cloud Authentication

**Error:** `Google Cloud authentication failed`

**Solutions:**

1. **Service Account Setup:**
   ```bash
   # Verify service account file exists
   ls -la "$GOOGLE_APPLICATION_CREDENTIALS"
   
   # Check file contents (should be valid JSON)
   head "$GOOGLE_APPLICATION_CREDENTIALS"
   ```

2. **API Enable Check:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Text-to-Speech API
   - Enable Speech-to-Text API

3. **Permissions:**
   - Service account needs "Text-to-Speech Admin" role
   - Service account needs "Speech-to-Text Admin" role

4. **Environment Variable:**
   ```bash
   # Must be absolute path
   export GOOGLE_APPLICATION_CREDENTIALS="/full/path/to/service-account.json"
   ```

### Pexels API Issues

**Error:** `Pexels API authentication failed`

**Solutions:**
1. **Check API Key:**
   - Should be a long string without prefixes
   - Get from [Pexels API Dashboard](https://www.pexels.com/api/)

2. **Rate Limiting:**
   - Pexels has rate limits (200 requests/hour for free tier)
   - Reduce concurrent downloads in config:
   ```yaml
   stock_media_settings:
     pexels:
       concurrent_downloads: 1  # Reduce from default 3
   ```

### Freesound API Issues

**Error:** `Freesound authentication failed`

**Solutions:**
1. **Basic API Key:**
   ```bash
   # Test API key
   curl "https://freesound.org/apiv2/search/text/?query=test&token=$FREESOUND_API_KEY"
   ```

2. **OAuth2 Issues:**
   - Only needed for full-quality downloads
   - Preview downloads work with just API key
   - Check refresh token is still valid

## Pipeline Execution Issues

### Script Generation Failures

**Error:** `Script generation failed` or LLM timeouts

**Diagnostics:**
1. **Check Debug Files:**
   - `outputs/[product_id]/temp/script_prompt.txt` - Rendered prompt sent to LLM
   - `outputs/[product_id]/script.txt` - Generated script (LLM response)

2. **Test LLM Connection:**
   ```bash
   # Test Gemini API
   curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY"
   ```

**Solutions:**
1. **Model Availability:**
   ```yaml
   # config/ai_services.yaml - primary provider
   llm_settings:
     provider: "gemini"
     models:
       - "gemini-2.5-flash-lite"

     # Automatic fallback when Gemini exhausts all models
     fallback_provider:
       provider: "openrouter"
       auto_select_free_model: true
       fallback_discover_any_free: true
   ```

2. **Timeout Issues:**
   ```yaml
   llm_settings:
     timeout_seconds: 60  # Increase from default 30
     retry_attempts: 5    # Increase retries
   ```

3. **Prompt Issues:**
   - Check product data has sufficient information
   - Verify prompt template is not corrupted

### TTS (Text-to-Speech) Issues

**Error:** `TTS generation failed` or voice not found

**Diagnostics:**
```bash
# Check available Google Cloud voices
poetry run python -c "
from google.cloud import texttospeech
client = texttospeech.TextToSpeechClient()
voices = client.list_voices()
for voice in voices.voices[:5]:
    print(f'{voice.name} - {voice.language_codes[0]} - {voice.ssml_gender}')
"
```

**Solutions:**
1. **Voice Selection Issues:**
   ```yaml
   tts_config:
     google_cloud_tts:
       voice_name: "en-US-Wavenet-D"  # Use specific voice
       # OR
       voice_name_pattern: "Standard" # Use Standard instead of Wavenet
   ```

2. **Fall back to Coqui TTS.** It is not a project dependency, so it has to be
   installed, and installing the package alone is not enough. `coqui-tts` needs three things beyond the package itself:
   `transformers >=4.57,<5` (0.27.5 imports `isin_mps_friendly`, which
   `transformers` 5 removed), `torchcodec` (required on torch 2.9 and above),
   and that `torchcodec` must come from the PyTorch CPU index, because the
   default PyPI wheel is CUDA-flavoured and fails to load against this
   project's CPU-only torch:

   ```bash
   pip install coqui-tts 'transformers>=4.57,<5'
   pip install --index-url https://download.pytorch.org/whl/cpu torchcodec
   ```

   Then add it to the provider order:

   ```yaml
   tts_config:
     provider_order: ["coqui", "google_cloud"]
   ```

   Get any of that wrong and the failure is quiet: `find_spec` does not execute
   the module, so `COQUI_AVAILABLE` stays `True` and `coqui` survives config
   validation. The break surfaces as one WARN at first synthesis while every
   render falls through to the next provider.

3. **Text Sanitization Issues:**
   - Check script has proper text formatting
   - Remove special characters that break TTS

**Symptom:** Voiceover is missing the final word of the script (Whisper transcript ends mid-sentence; final MP4 duration is shorter than expected).

**Cause:** `audio_processing.silence_min_duration_sec` set too high. This YAML field maps to ffmpeg `silenceremove` `start_duration`, which is the non-silence confirmation window — audio during the window is discarded, not kept. At `0.3s`, short trailing words (~0.4s) land entirely inside the window and get stripped. This is independent of which TTS provider generated the audio; synthesizing the script in isolation will produce the complete audio, confirming the pipeline's trim step is the culprit.

**Fix:** Set `silence_min_duration_sec: 0.1` in `config/ai_services.yaml` under `audio_processing` (the code default). Thresholds from -20 to -60 dB all preserve the final word at this value.

### Subtitle Generation Issues

**Error:** Subtitles missing, poor timing, or unreadable

**Diagnostics:**
1. **Check Debug Files:**
   - `outputs/[product_id]/temp/temp_subtitles/` - Generated SRT files
   - `outputs/[product_id]/temp/voiceover_timings.json` - Timing data

2. **Test Whisper Installation:**
   ```bash
   poetry run python -c "import whisper; print('✓ Whisper available')"
   ```

**Solutions:**
1. **Provider Issues:**
   ```yaml
   whisper_settings:
     model_size: "small"  # Try larger model if "base" fails
     device: "cpu"        # Force CPU if GPU issues
   
   google_cloud_stt_settings:
     enabled: true        # Enable as fallback
   ```

2. **Timing Issues:**
   ```yaml
   subtitle_settings:
     max_duration: 2.5              # Shorter segments (seconds)
     min_duration: 0.6              # Floor so short words stay readable
     max_words_per_line: 3          # Denser line breaks
   ```

3. **Readability Issues:**
   ```yaml
   subtitle_settings:
     font_size_percent: 0.08        # Larger font (fraction of frame height)
     margin: 0.12                   # More space from anchor
     style_preset: "bold"           # High-contrast preset (colors owned by preset)
   ```

   Colors, outline, and font family are owned by the active `style_preset`.
   Edit or add an entry under the top-level `style_presets` block in
   `config/subtitles.yaml` to change them — there are no longer flat
   `font_color` / `back_color` keys on `subtitle_settings`.

### Video Assembly Issues

**Error:** `FFmpeg command failed` or invalid file index

**Diagnostics:**
1. **Check FFmpeg Command:**
   - Look in `outputs/[product_id]/temp/ffmpeg_command.log`
   - Check console logs for FFmpeg errors

2. **Verify Input Files:**
   ```bash
   # Check temp directory has all required files
   ls -la outputs/[product_id]/temp/
   ```

**Solutions:**
1. **Missing Input Files:**
   - Run previous steps individually to identify which failed
   - Use `--step` flag to run specific steps:
   ```bash
   poetry run python -m src.video.producer products.json profile --debug --step create_voiceover
   ```

2. **FFmpeg Filter Issues:**
   ```yaml
   ffmpeg_settings:
     enable_zoompan: false    # Disable complex effects
     save_command: true       # Save command for debugging
   ```

3. **Resolution/Format Issues:**
   ```yaml
   video_settings:
     output_codec: "libx264"     # Use compatible codec
     output_pixel_format: "yuv420p"  # Compatible pixel format
   ```

### File Permission Issues

**Error:** `Permission denied` or `Cannot write to directory`

**Solutions:**
```bash
# Fix directory permissions
chmod -R 755 outputs/
chmod -R 755 config/

# Check disk space
df -h

# Check if directory exists and is writable
ls -la outputs/
```

## Batch Processing Issues

### No Products Found for Batch Processing

**Error:** `No valid products found in /path/to/outputs`

**Causes:**
- No `data.json` files in product directories
- Invalid JSON structure in `data.json` files
- All directories are system directories (cache, logs, etc.)

**Solutions:**
```bash
# Check outputs directory structure
ls -la outputs/

# Find all data.json files
find outputs/ -name "data.json" -type f

# Test JSON validity
python -m json.tool outputs/PRODUCT_ID/data.json

# Run scraper to generate valid data
poetry run python -m src.scraper.amazon.scraper --keywords "B0BTYCRJSS" --debug --clean
```

### Batch Processing Fails on Some Products

**Error:** Mixed success/failure in batch processing

**Solutions:**
```bash
# Use fail-fast to identify problematic products
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --fail-fast --debug

# Process products individually for debugging
poetry run python -m src.video.producer outputs/PRODUCT_ID/data.json slideshow_images1 --debug

# Check individual product data integrity
poetry run python -c "
import json
from pathlib import Path
from src.scraper.amazon.scraper import ProductData

data = json.loads(Path('outputs/PRODUCT_ID/data.json').read_text())
if isinstance(data, list):
    product = ProductData(**data[0])  # Test first product
else:
    product = ProductData(**data)
print('✓ Product data is valid')
"
```

### Batch Processing Command Line Errors

**Error:** `--batch-profile is required when using --batch`

**Solution:**
```bash
# Correct batch processing syntax
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug

# NOT this (missing profile):
# poetry run python -m src.video.producer --batch --debug
```

**Error:** `products_file and profile arguments cannot be used with --batch`

**Solution:**
```bash
# Choose either batch mode OR single product mode:

# Batch mode (correct):
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1

# Single product mode (correct):
poetry run python -m src.video.producer outputs/PRODUCT_ID/data.json slideshow_images1

# NOT this (mixing modes):
# poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 data.json profile
```

## Performance Issues

### Slow Execution

**Diagnostics:**
```bash
# Generate performance report
make perf-report
```

**Solutions:**
1. **Reduce Concurrency:**
   ```yaml
   max_concurrent_downloads: 3      # Reduce from default 5
   stock_media_concurrent_downloads: 2  # Reduce concurrent downloads
   ```

2. **Disable Expensive Features:**
   ```yaml
   ffmpeg_settings:
     enable_zoompan: false         # Disable zoom/pan effects
   
   subtitle_settings:
     enabled: false                # Skip subtitles temporarily
   ```

3. **Use Faster Models:**
   ```yaml
   whisper_settings:
     model_size: "tiny"            # Fastest Whisper model
   
   llm_settings:
     models: ["gemini-2.5-flash-lite"]  # Fast Gemini model (primary provider)
   ```

### Memory Issues

**Error:** `Out of memory` or system freeze

**Solutions:**
1. **Reduce Memory Usage:**
   ```yaml
   memory_settings:
     max_memory_usage_mb: 1024     # Limit memory usage
     mmap_threshold_mb: 10         # Use memory mapping for larger files
   ```

2. **Process Fewer Items:**
   ```bash
   # Process one product at a time
   poetry run python -m src.video.producer products.json profile --product-index 0
   ```

3. **Clear Cache:**
   ```bash
   # Clear cache if it's too large
   rm -rf outputs/cache/
   ```

### `make *-lowpri` reports `No project interpreter found`

**Error:** `scrape-lowpri`, `produce-lowpri`, `batch-lowpri`, or `publish-lowpri` stops immediately with `No project interpreter found (tried active venv, python3, poetry env). Run 'poetry install' first.`

**Cause:** these targets deliberately do not use `poetry run`. They run the pipeline inside a `systemd-run --user --scope` cgroup to cap memory, and that scope starts the process through the user service manager, which does not carry the caller's virtualenv. `poetry run python` inside the scope resolves an interpreter without the project's dependencies and the run dies on import. The targets instead probe for an interpreter that can import a project dependency, and this error means no candidate passed.

**Fix:** install the dependencies into an environment one of the probes can find:

```bash
poetry install
# or activate the project virtualenv first, then retry
```

The probe checks the pyenv virtualenv named in `.python-version`, then the active virtualenv, the interpreter `python3` resolves to, and the environment `poetry env info -p` reports, in that order. `.python-version` comes first because it is the only one of the four that names this project; the rest read the ambient environment, so an unrelated project's virtualenv active in the shell captures all three at once and none of them can import the project's dependencies. An unusual setup that satisfies none of the four (a bare conda env, or a Poetry install the shell can't see) needs the project virtualenv activated before running `make`. The plain, non-`lowpri` targets are unaffected because they run outside the scope.

## Configuration Issues

### YAML Parsing Errors

**Error:** `YAML parsing failed` or configuration loading errors

**Diagnostics:**
```bash
# Test YAML syntax (modular config files)
poetry run python -c "
import yaml
from pathlib import Path
config_files = ['core.yaml', 'video_production.yaml', 'ai_services.yaml', 'subtitles.yaml', 'performance.yaml', 'scraper.yaml', 'pipeline.yaml', 'publisher.yaml', 'url_shortener.yaml']
for file in config_files:
    with open(f'config/{file}') as f:
        config = yaml.safe_load(f)
        print(f'✓ {file} syntax is valid')
"
```

**Solutions:**
1. **Check Indentation:**
   - YAML is sensitive to spaces vs. tabs
   - Use consistent indentation (2 or 4 spaces)

2. **Quote Special Characters:**
   ```yaml
   # Wrong
   font_color: #FFFFFF
   
   # Right  
   font_color: "#FFFFFF"
   ```

3. **Validate Lists:**
   ```yaml
   # Wrong
   models: gemini-2.5-flash-lite
   
   # Right
   models:
     - gemini-2.5-flash-lite
   ```

### Pydantic Validation Errors

**Error:** Validation errors with specific field names

**Solutions:**
1. **Check Field Types:**
   ```yaml
   # Numbers should not be quoted
   timeout_sec: 30        # Not "30"
   
   # Booleans should be lowercase
   enabled: true          # Not True
   ```

2. **Check Required Fields:**
   - See [Configuration](configuration.md) for required fields
   - Missing fields will cause validation errors

3. **Check Enum Values:**
   ```yaml
   # Check allowed values for enums
   gender: "NEUTRAL"      # Must be NEUTRAL, MALE, or FEMALE
   alignment: "bottom_center"  # Must be valid alignment option
   ```

## Network Issues

### Connection Timeouts

**Error:** `Connection timeout` or `Request failed`

**Solutions:**
1. **Increase Timeouts:**
   ```yaml
   download_timeout_sec: 120        # Increase from default 60
   api_timeout_sec: 60             # Increase from default 30
   ```

2. **Reduce Concurrency:**
   ```yaml
   max_concurrent_api_calls: 1     # Reduce concurrent requests
   ```

3. **Check Proxy Settings:**
   ```bash
   # If using proxy, ensure it's configured
   export HTTP_PROXY=http://proxy.example.com:8080
   export HTTPS_PROXY=https://proxy.example.com:8080
   ```

### SSL Certificate Issues

**Error:** `SSL certificate verification failed`

**Solutions:**
```bash
# Update certificates (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install ca-certificates

# macOS
brew install ca-certificates

# Verify Python can connect
poetry run python -c "
import ssl
import urllib.request
urllib.request.urlopen('https://google.com')
print('✓ SSL connections work')
"
```

## Test & Coverage Issues

### `pytest --cov` crashes with "_has_torch_function already has a docstring"

**Error message:**

```
RuntimeError: function '_has_torch_function' already has a docstring
```

The crash happens at conftest import time, before any test runs. Stack trace ends in `torch/overrides.py` calling `_add_docstr`.

**Cause:** coverage instrumentation re-imports modules. If a project module imports a torch-using package at module load (Coqui TTS, transformers, anything with PyTorch as a transitive dep), torch's `overrides` module rejects the second `_add_docstr` call and aborts the whole pytest-cov session.

**Fix:** defer the torch-transitive import to first use. In this repo `src/video/tts.py` uses `importlib.util.find_spec("TTS")` for the availability flag and a lazy `from TTS.api import TTS as _TTS` inside `_load_coqui_tts_class()`, which runs only when `_initialize_coqui_tts_model` is called. Coverage runs that don't exercise Coqui never trigger the torch path.

**Workaround if you can't refactor the import:** run pytest without `--cov` (the project's regular `make test` works), or add `--no-cov` to your specific run.

## Getting More Help

### Enable Maximum Debugging

```bash
# Run with maximum debugging
export PYTHONPATH=$PWD
poetry run python -m src.video.producer products.json profile \
  --debug \
  --product-index 0

# Check all log files
find outputs/logs/ -name "*.log" -exec echo "=== {} ===" \; -exec cat {} \;
```

### Collect System Information

```bash
# Create diagnostic report
echo "=== System Info ===" > debug_report.txt
uname -a >> debug_report.txt
poetry run python --version >> debug_report.txt
ffmpeg -version >> debug_report.txt 2>&1

echo -e "\n=== Environment Variables ===" >> debug_report.txt
env | grep -E "(API_KEY|GOOGLE_|FREESOUND_)" >> debug_report.txt

echo -e "\n=== Configuration Test ===" >> debug_report.txt
poetry run python -c "from src.video.config_adapter import load_video_config_modular; print('Config loads successfully')" >> debug_report.txt 2>&1

echo -e "\n=== Recent Logs ===" >> debug_report.txt
find outputs/logs/ -name "*.log" -newer $(date -d '1 hour ago' '+%Y%m%d%H%M') -exec cat {} \; >> debug_report.txt 2>/dev/null
```

## Scraper Issues

### CAPTCHA Detection

**Error:** `CAPTCHA detected on SERP/detail page`

**Solutions:**
1. **Reduce Request Frequency:**
   ```yaml
   # In config/scraper.yaml
   global_settings:
     delay_range: [3, 6]     # Increase delays between requests
     retries: 1              # Reduce retry attempts
   ```

2. **Use Different User Agents:**
   - The scraper rotates user agents automatically
   - Ensure `user_agents` list in config has variety

3. **Clear Browser Data:**
   ```bash
   # Clean run to reset browser state
   poetry run python -m src.scraper.amazon.scraper --clean --debug
   ```

### Search Parameter Issues

**Error:** Invalid search parameters or no results

**Solutions:**
1. **Check Parameter Values:**
   ```bash
   # Ensure valid price ranges
   --min-price 10.0 --max-price 100.0  # min < max
   
   # Valid rating values (1-5)
   --min-rating 4
   
   # Valid sort options
   --sort price-asc-rank  # Check available options with --help
   ```

2. **Test Basic Search First:**
   ```bash
   # Start with simple search to verify scraper works
   poetry run python -m src.scraper.amazon.scraper --keywords "test" --debug
   ```

3. **Verify Search Results:**
   - Some parameter combinations may return no results
   - Try broader search parameters
   - Check Amazon directly to verify products exist with those filters

### Selector Failures

**Error:** `All configured selectors failed for essential key`

**Solutions:**
1. **Enable Debug Mode:**
   ```bash
   poetry run python -m src.scraper.amazon.scraper --debug
   # Shows which selectors are working/failing
   ```

2. **Update Selectors:**
   - Amazon frequently changes their HTML structure
   - Check `config/scraper.yaml` for current selectors
   - Add alternative selectors as needed

3. **Test on Different Pages:**
   ```bash
   # Try different product types
   --keywords "electronics"  # vs "books" vs "clothing"
   ```

### Download Issues

**Error:** Media download failures

**Solutions:**
1. **Check Network Connectivity:**
   ```bash
   # Test download manually
   curl -I https://m.media-amazon.com/images/I/example.jpg
   ```

2. **Adjust Timeouts:**
   ```yaml
   # In config/scraper.yaml
   global_settings:
     timeouts:
       download: 120  # Increase download timeout
   ```

3. **Reduce Concurrency:**
   ```yaml
   global_settings:
     download_concurrency: 5  # Reduce from default 10
   ```

### Browser Issues

**Error:** Browser launch failures or crashes

**Solutions:**
1. **Reinstall Playwright:**
   ```bash
   poetry run playwright install --force
   ```

2. **Check System Resources:**
   - Ensure sufficient RAM (4GB+ recommended)
   - Close other browsers and applications

3. **Enable Headless Mode:**
   ```bash
   # Remove --debug to run headless (uses less resources)
   poetry run python -m src.scraper.amazon.scraper --keywords "test"
   ```

### Scraper times out / 0 products on Wayland

**Symptom:** Every scrape returns 0 products with "Document did not become ready within 60s"
in `outputs/logs/scraper.log`, and no browser window appears. Often shows up right after an OS
or session change (e.g. Ubuntu 22 to Ubuntu 26).

**Cause:** The scraper runs Chrome headful (its headless mode is detectable and crash-prone). On
Wayland, `DISPLAY` is empty, so headful Chrome has nowhere to draw and the navigation hangs. It
looks like an anti-bot block but it's a missing display.

**Check first:**
```bash
echo "$DISPLAY"          # empty on a broken Wayland session
echo "$XDG_SESSION_TYPE" # wayland vs x11
command -v Xvfb          # must be installed
```

**Fix:** Install `Xvfb` (`sudo apt install -y xvfb`). Normal runs then use a virtual display
(real headful browser, no visible window). If `Xvfb` is missing, Botasaurus silently falls back
to `--headless=new` and crashes, so a missing-Xvfb box is the real failure, not a working
degraded mode. The scraper passes Chrome `--ozone-platform=x11` so it uses the X11 backend
(Xvfb) instead of drawing a real Wayland window; unsetting `WAYLAND_DISPLAY` alone is not enough
because libwayland defaults to the `wayland-0` socket.

### Watching a debug scrape on Wayland (`make scrape-watch`)

A headful browser **cannot** be driven on a live Wayland session: once it does real work,
Chromium's CDP endpoint stops answering (`Failed to connect to Chrome URL`, then per-navigation
`Response not received` ~400s hangs). So `--debug` on a Wayland desktop runs on a virtual Xvfb
display (no visible window) rather than the live Xwayland server. To actually watch a debug run,
use `make scrape-watch`, which starts a dedicated Xvfb plus `x11vnc` and points the scraper at it:

```bash
make scrape-watch ARGS="--keywords 'wireless earbuds' --max-products 1"
# then connect a VNC viewer:
vncviewer localhost:5900
```

Needs the `xvfb` and `x11vnc` packages. `x11vnc` refuses to start on a detected Wayland session,
so the target launches it with `WAYLAND_DISPLAY`/`XDG_SESSION_TYPE` unset (an empty value is not
enough). The browser launch profile and per-navigation timing are logged at DEBUG to triage
display/CDP problems from the log alone.

### Community Support

1. **Check Existing Issues:**
   - Search [GitHub Issues](https://github.com/stkzlv/ContentEngineAI/issues)
   - Look for similar problems and solutions

2. **Create New Issue:**
   - Include system information
   - Provide error messages and logs
   - Describe steps to reproduce
   - Include configuration (remove sensitive keys)

3. **Documentation:**
   - [Installation Guide](installation.md)
   - [Configuration Guide](configuration.md)
   - [Development Guide](development.md)
   - [Architecture Documentation](architecture.md)

## Common Error Patterns

### "No module named" Errors
- Run `poetry install` to ensure all dependencies are installed
- Check you're using Poetry environment: `poetry shell`

### "File not found" Errors
- Check file paths in configuration are correct
- Ensure output directories exist and are writable
- Verify input files exist

### "Timeout" Errors
- Increase relevant timeout values in configuration
- Check network connectivity
- Reduce concurrency if system is overloaded

### "Permission denied" Errors
- Check file and directory permissions
- Ensure you have write access to output directories
- On Windows, run as administrator if needed

### "Invalid configuration" Errors
- Validate YAML syntax
- Check required fields are present
- Verify enum values are correct
- Check data types match expectations

---

## Debug Files Reference

ContentEngineAI generates various debug files to help diagnose issues. All debug files are stored in `outputs/{product_id}/temp/` unless otherwise noted.

### Debug File Locations

| File | Location | Purpose | Controlled By |
|------|----------|---------|---------------|
| **Producer Log** | `outputs/logs/producer.log` | Producer execution log | `--debug` flag |
| **Scraper Log** | `outputs/logs/scraper.log` | Scraper execution log | `--debug` or `--verbose` flag |
| **FFmpeg Commands** | `outputs/{product_id}/temp/ffmpeg_command.log` | FFmpeg commands used for video assembly | `create_ffmpeg_command_logs: true` |
| **Media Validation** | `outputs/{product_id}/temp/{product_id}_media_validation_report.json` | Media file validation results | `create_media_validation_reports: true` |
| **Pipeline Metadata** | `outputs/{product_id}/temp/metadata.json` | Pipeline state and execution tracking | `create_pipeline_metadata: true` |
| **Performance Metrics** | `outputs/{product_id}/temp/performance.json` | Operation timing and resource usage | `create_performance_metrics: true` |
| **Whisper Raw Output** | `outputs/{product_id}/temp/whisper_result_raw.json` | Raw STT transcription output | `create_whisper_debug_files: true` |
| **Whisper vs Script** | `outputs/{product_id}/temp/whisper_vs_script.txt` | Transcription vs script comparison | `create_whisper_debug_files: true` |
| **Whisper Word List** | `outputs/{product_id}/temp/whisper_word_list.json` | Word-level timing data | `create_whisper_debug_files: true` |
| **Gathered Visuals** | `outputs/{product_id}/temp/gathered_visuals.json` | Visual asset selection metadata | `create_temp_files: true` |
| **Music Choice** | `outputs/{product_id}/temp/music_choice.json` | Audio selection metadata | `create_temp_files: true` |

### Debug Settings Configuration

Edit `config/performance.yaml` to control debug file generation:

```yaml
debug_settings:
  # Logging configuration
  max_log_line_length: 200
  debug_file_retention_days: 7

  # File cleanup behavior
  intermediate_file_cleanup: true      # Master cleanup switch
  cleanup_on_success: false            # Remove files after success
  cleanup_on_failure: false            # Remove files after failure (keep for debugging)
  cleanup_whisper_files: false         # Remove Whisper temporary files

  # Debug file generation (set to false to disable specific files)
  create_media_validation_reports: true
  create_ffmpeg_command_logs: true
  create_pipeline_metadata: true
  create_performance_metrics: true
  create_whisper_debug_files: true
  create_temp_files: true
```

**Important:** CLI `--debug` flag overrides these settings and retains all debug files for troubleshooting.