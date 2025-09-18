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
poetry run python -c "from src.video.video_config import load_config; print('✓ Config loads')"

# Check API keys
poetry run python -c "
import os
from dotenv import load_dotenv
load_dotenv()
keys = ['OPENROUTER_API_KEY', 'PEXELS_API_KEY', 'FREESOUND_API_KEY']
for key in keys:
    status = '✓' if os.getenv(key) else '✗'
    print(f'{status} {key}')
"
```

### Debug Mode

Always use debug mode when troubleshooting:

```bash
poetry run python -m src.video.producer products.json profile_name --debug
```

Debug mode provides:
- Detailed console logging
- Intermediate file preservation in `outputs/temp/`
- FFmpeg command logging
- Step-by-step execution traces

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
# config/video_producer.yaml:
ffmpeg_settings:
  ffmpeg_path: "/usr/local/bin/ffmpeg"  # Your actual path
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

## API and Authentication Issues

### OpenRouter API Issues

**Error:** `Invalid API key` or `Authentication failed`

**Diagnostics:**
```bash
# Test API key manually
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models
```

**Solutions:**
1. **Check API Key Format:**
   - Should start with `sk-or-`
   - No extra spaces or quotes in `.env` file

2. **Verify API Key:**
   - Log into [OpenRouter Dashboard](https://openrouter.ai/)
   - Check key is active and has credits

3. **Environment Variable Issues:**
   ```bash
   # Check if loaded correctly
   poetry run python -c "import os; print(repr(os.getenv('OPENROUTER_API_KEY')))"
   
   # Should not show 'None' or have extra characters
   ```

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
   - `outputs/temp/[product_id]/steps/formatted_prompt.txt` - Sent to LLM
   - `outputs/temp/[product_id]/steps/script.txt` - LLM response

2. **Test LLM Connection:**
   ```bash
   # Test with simple request
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"model":"anthropic/claude-3-haiku","messages":[{"role":"user","content":"Hello"}]}' \
        https://openrouter.ai/api/v1/chat/completions
   ```

**Solutions:**
1. **Model Availability:**
   ```yaml
   llm_settings:
     models:
       - "anthropic/claude-3-haiku"    # Try different models
       - "openai/gpt-3.5-turbo"
       - "meta-llama/llama-3-8b"
   ```

2. **Timeout Issues:**
   ```yaml
   llm_settings:
     timeout_sec: 60  # Increase from default 30
     max_retries: 5   # Increase retries
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

2. **Fallback to Coqui TTS:**
   ```yaml
   tts_config:
     providers:
       - coqui_tts          # Try local TTS first
       - google_cloud_tts   # Then cloud TTS
   ```

3. **Text Sanitization Issues:**
   - Check script has proper text formatting
   - Remove special characters that break TTS

### Subtitle Generation Issues

**Error:** Subtitles missing, poor timing, or unreadable

**Diagnostics:**
1. **Check Debug Files:**
   - `outputs/temp/[product_id]/temp_subtitles/` - Generated SRT files
   - `outputs/temp/[product_id]/steps/voiceover_timings.json` - Timing data

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
     max_subtitle_duration_sec: 5    # Shorter segments
     pause_threshold_sec: 0.5        # More sensitive pause detection
     split_on_punctuation: true      # Natural breaks
   ```

3. **Readability Issues:**
   ```yaml
   subtitle_settings:
     font_size_percent: 5            # Larger font
     font_color: "#FFFFFF"           # High contrast
     back_color: "#000000AA"         # Semi-transparent background
     margin_v_percent: 15            # More space from bottom
   ```

### Video Assembly Issues

**Error:** `FFmpeg command failed` or invalid file index

**Diagnostics:**
1. **Check FFmpeg Command:**
   - Look in `outputs/temp/[product_id]/ffmpeg_command.log`
   - Check console logs for FFmpeg errors

2. **Verify Input Files:**
   ```bash
   # Check temp directory has all required files
   ls -la outputs/temp/[product_id]/
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
     models: ["anthropic/claude-3-haiku"]  # Fastest LLM
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

## Configuration Issues

### YAML Parsing Errors

**Error:** `YAML parsing failed` or configuration loading errors

**Diagnostics:**
```bash
# Test YAML syntax
poetry run python -c "
import yaml
with open('config/video_producer.yaml') as f:
    config = yaml.safe_load(f)
    print('✓ YAML syntax is valid')
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
   models: anthropic/claude-3-haiku
   
   # Right
   models:
     - anthropic/claude-3-haiku
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
   - See [CONFIGURATION.md](CONFIGURATION.md) for required fields
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
poetry run python -c "from src.video.video_config import load_config; print('Config loads successfully')" >> debug_report.txt 2>&1

echo -e "\n=== Recent Logs ===" >> debug_report.txt
find outputs/logs/ -name "*.log" -newer $(date -d '1 hour ago' '+%Y%m%d%H%M') -exec cat {} \; >> debug_report.txt 2>/dev/null
```

## Scraper Issues

### CAPTCHA Detection

**Error:** `CAPTCHA detected on SERP/detail page`

**Solutions:**
1. **Reduce Request Frequency:**
   ```yaml
   # In config/scrapers.yaml
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
   - Check `config/scrapers.yaml` for current selectors
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
   # In config/scrapers.yaml
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

### Community Support

1. **Check Existing Issues:**
   - Search [GitHub Issues](https://github.com/ContentEngineAI/ContentEngineAI/issues)
   - Look for similar problems and solutions

2. **Create New Issue:**
   - Include system information
   - Provide error messages and logs
   - Describe steps to reproduce
   - Include configuration (remove sensitive keys)

3. **Documentation:**
   - [Installation Guide](INSTALL.md)
   - [Configuration Guide](CONFIGURATION.md)
   - [Development Guide](DEVELOPMENT.md)
   - [Architecture Documentation](ARCHITECTURE.md)

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