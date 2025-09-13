# Installation Guide

This guide will walk you through installing ContentEngineAI and setting up all required dependencies.

## Prerequisites

Before installing ContentEngineAI, ensure you have the following installed:

### System Requirements

- **Python**: Version 3.12.x (Check with `python --version`)
  - We recommend using `pyenv` for Python version management
- **FFmpeg**: Required for video processing (Check with `ffmpeg -version`)
- **Git**: For cloning the repository

### Python Version Management (Recommended)

We strongly recommend using `pyenv` to manage Python versions:

```bash
# Install pyenv (macOS)
brew install pyenv

# Install pyenv (Linux)
curl https://pyenv.run | bash

# Install Python 3.12
pyenv install 3.12.7
pyenv local 3.12.7
```

### FFmpeg Installation

ContentEngineAI requires FFmpeg for video processing. Install it based on your operating system:

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add the `bin` directory to your system PATH

**Verify Installation:**
```bash
ffmpeg -version
```

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/ContentEngineAI/ContentEngineAI.git
cd ContentEngineAI
```

### 2. Install Poetry

Poetry manages dependencies and virtual environments:

```bash
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Add Poetry to your PATH:
```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"
```

### 3. Install Dependencies

```bash
# Install Python dependencies
poetry install

# Install Playwright browsers for web scraping
poetry run playwright install
```

### 4. Verify Installation

Check that all components are properly installed:

```bash
# Verify Python version
poetry run python --version

# Verify FFmpeg access
ffmpeg -version

# Verify Playwright installation
poetry run playwright --help
```

## API Keys and Credentials

ContentEngineAI uses a **dual configuration system**: YAML files for application settings and environment variables for sensitive data like API keys.

Copy the example environment file:

```bash
cp .env.example .env
```

The `.env.example` file contains all required and optional environment variables with explanations. For details on how environment variables integrate with YAML configuration, see [CONFIGURATION.md](CONFIGURATION.md).

### Required API Keys

Edit your `.env` file and replace the placeholder values with your actual API keys:

#### 1. OpenRouter API Key (Required)
For LLM script generation:
- Register at [OpenRouter](https://openrouter.ai/)
- Get your API key from the dashboard
- Replace `your_openrouter_api_key_here` in `.env`

#### 2. Pexels API Key (Required)
For stock images and videos:
- Register at [Pexels API](https://www.pexels.com/api/)
- Get your API key
- Replace `your_pexels_api_key_here` in `.env`

#### 3. Freesound API Key (Required)
For background music:
- Create account at [Freesound](https://freesound.org/)
- Register new app at [API Apps](https://freesound.org/home/app_new/)
- Replace `your_freesound_api_key_here` in `.env`

#### 4. Google Cloud Credentials (Optional)
For Google Cloud TTS and Speech-to-Text:

1. **Create Google Cloud Project:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one

2. **Enable APIs:**
   - Enable Text-to-Speech API
   - Enable Speech-to-Text API

3. **Create Service Account:**
   - Go to IAM & Admin > Service Accounts
   - Create new service account
   - Assign roles: Text-to-Speech Admin, Speech-to-Text Admin
   - Download JSON key file

4. **Update .env file:**
   - Replace `/path/to/your/google-credentials.json` with the absolute path to your downloaded JSON key file

### Optional: Freesound OAuth2 Setup

For downloading full-quality audio files from Freesound:

1. **Get OAuth2 Credentials:**
   - From your Freesound app dashboard, note Client ID and Client Secret

2. **Authorization Flow:**
   ```bash
   # Replace YOUR_CLIENT_ID with your actual Client ID
   # Open this URL in browser:
   https://freesound.org/apiv2/oauth2/authorize/?client_id=YOUR_CLIENT_ID&response_type=code
   ```

3. **Get Authorization Code:**
   - Authorize the app in browser
   - Copy the `code` parameter from redirect URL

4. **Exchange for Tokens:**
   ```bash
   curl -X POST https://freesound.org/apiv2/oauth2/access_token/ \
   -d "client_id=YOUR_CLIENT_ID" \
   -d "client_secret=YOUR_CLIENT_SECRET" \
   -d "grant_type=authorization_code" \
   -d "code=AUTHORIZATION_CODE_FROM_STEP_3" \
   -d "redirect_uri=YOUR_REDIRECT_URI"
   ```

5. **Update .env file:**
   - Replace the Freesound OAuth2 placeholder values with your actual credentials from the curl response

## Configuration

### Initial Configuration

Copy the example configuration file:

```bash
cp config/video_producer.yaml.example config/video_producer.yaml  # If example exists
```

### Basic Configuration

The default configuration in `config/video_producer.yaml` works for most use cases. Key settings to review:

- **Output directories**: Where videos and data are stored
- **API settings**: Environment variable names for your API keys
- **Video settings**: Resolution (default: 1080x1920), frame rate, duration
- **Provider preferences**: Order for TTS and STT providers

For detailed configuration options, see [CONFIGURATION.md](CONFIGURATION.md).

## Verification

### Test Installation

Run a quick verification to ensure everything works:

```bash
# Check configuration loading
poetry run python -c "from src.video.video_config import load_config; print('✓ Configuration loads successfully')"

# Test API connections (optional, requires API keys)
poetry run python -c "
import os
from dotenv import load_dotenv
load_dotenv()
if os.getenv('OPENROUTER_API_KEY'): print('✓ OpenRouter API key found')
if os.getenv('PEXELS_API_KEY'): print('✓ Pexels API key found')
if os.getenv('FREESOUND_API_KEY'): print('✓ Freesound API key found')
"
```

### Test Basic Functionality

If you have API keys configured, test the scraper:

```bash
# Test Amazon scraper (requires no API keys)
poetry run python -m src.scraper.amazon.scraper --keywords "test" --debug

# Test advanced search filtering
poetry run python -m src.scraper.amazon.scraper \
  --keywords "wireless earbuds" --min-price 20 --max-price 100 \
  --min-rating 4 --debug --clean
```

## Troubleshooting

### Common Issues

**FFmpeg Not Found:**
- Ensure FFmpeg is installed and in your system PATH
- On Windows, verify the `bin` directory is added to PATH
- Restart your terminal after PATH changes

**Poetry Installation Issues:**
- Ensure Python 3.12 is installed and accessible
- Try reinstalling Poetry: `curl -sSL https://install.python-poetry.org | python3 - --uninstall`

**Playwright Browser Issues:**
- Re-run browser installation: `poetry run playwright install`
- On Linux, install system dependencies: `poetry run playwright install-deps`

**Permission Errors:**
- Ensure you have write permissions to the project directory
- On macOS/Linux: `chmod +x scripts/*` if using shell scripts

**Import Errors:**
- Verify virtual environment: `poetry env info`
- Reinstall dependencies: `poetry install --no-cache`

For more troubleshooting help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Development Installation

If you plan to contribute to ContentEngineAI:

```bash
# Install development dependencies
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install

# Run tests to verify setup
poetry run pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development setup and [TESTING.md](TESTING.md) for comprehensive testing documentation.

## Next Steps

Once installation is complete:

1. **Review Configuration**: Check `config/video_producer.yaml` settings
2. **Set Up API Keys**: Configure your `.env` file with required API keys
3. **Run First Test**: Try scraping a product or generating a test video
4. **Read Documentation**: Check out the [main README](README.md) for usage examples

## Getting Help

If you encounter issues during installation:

- Check [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review [GitHub Issues](https://github.com/ContentEngineAI/ContentEngineAI/issues)
- Create a new issue with your system details and error messages