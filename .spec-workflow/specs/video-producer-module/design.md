# Design Document: Video Producer Module

## Overview

This design documents the Video Producer Module architecture, which orchestrates the complete video production pipeline from product data to polished video output. The module integrates video assembly strategies, subtitle systems, background music acquisition, and batch processing capabilities.

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                           cli.py                                         ││
│  │  • Argument parsing (60+ options)                                        ││
│  │  • Batch discovery and orchestration                                     ││
│  │  • Profile selection (fixed/random)                                      ││
│  │  • Progress reporting                                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Pipeline Orchestration                               │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────┐ │
│  │   orchestration.py │  │      state.py      │  │      context.py        │ │
│  │  • Step sequencing │  │  • State tracking  │  │  • Pipeline context    │ │
│  │  • Error handling  │  │  • Artifact load   │  │  • Exception classes   │ │
│  │  • Resume support  │  │  • Persistence     │  │  • Run paths           │ │
│  └────────────────────┘  └────────────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Pipeline Steps                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                            steps.py                                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │   Gather    │  │  Generate   │  │   Create    │  │  Download   │   │  │
│  │  │   Visuals   │  │   Script    │  │  Voiceover  │  │    Music    │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │  Generate   │  │   Fetch     │  │  Assemble   │  │   Apply     │   │  │
│  │  │  Subtitles  │  │   Stock     │  │    Video    │  │  Subtitles  │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│  Video Assembler │    │   Subtitle System    │    │  Audio Services  │
│  ┌────────────┐  │    │  ┌────────────────┐  │    │  ┌────────────┐  │
│  │ Strategies │  │    │  │   Positioning  │  │    │  │ Freesound  │  │
│  │ ─────────  │  │    │  │   ──────────   │  │    │  │  Client    │  │
│  │ Sequential │  │    │  │ Anchor-based   │  │    │  │ ────────── │  │
│  │ SingleBest │  │    │  │ Content-aware  │  │    │  │ OAuth2     │  │
│  │ MixedMedia │  │    │  │ Two-part       │  │    │  │ Circuit    │  │
│  │ VideoFirst │  │    │  └────────────────┘  │    │  │ Breaker    │  │
│  └────────────┘  │    │  ┌────────────────┐  │    │  │ Fallback   │  │
│  ┌────────────┐  │    │  │  ASS Effects   │  │    │  └────────────┘  │
│  │ Processors │  │    │  │  ──────────    │  │    │  ┌────────────┐  │
│  │ ─────────  │  │    │  │ Karaoke, Fade  │  │    │  │    TTS     │  │
│  │ AspectRatio│  │    │  │ Typewriter     │  │    │  │  Provider  │  │
│  │ AudioNorm  │  │    │  │ Glow, Bounce   │  │    │  └────────────┘  │
│  │ FormatNorm │  │    │  └────────────────┘  │    └──────────────────┘
│  └────────────┘  │    │  ┌────────────────┐  │
└──────────────────┘    │  │ Style Presets  │  │
                        │  │  ──────────    │  │
                        │  │ Minimal, Bold  │  │
                        │  │ Modern, Random │  │
                        │  └────────────────┘  │
                        └──────────────────────┘
```

### Data Flow

```
Product Data Flow:
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  data.json │───▶│   Gather   │───▶│  Generate  │───▶│   Create   │
│  (product) │    │  Visuals   │    │   Script   │    │  Voiceover │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                                                             │
                                                             ▼
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│   Final    │◀───│   Apply    │◀───│  Assemble  │◀───│  Download  │
│   Video    │    │  Subtitles │    │   Video    │    │   Music    │
└────────────┘    └────────────┘    └────────────┘    └────────────┘

Batch Processing Flow:
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│   Scan     │───▶│  Select    │───▶│  Process   │───▶│  Report    │
│  Products  │    │  Profile   │    │  Product   │    │  Summary   │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                        │                 │
                        ▼                 ▼
                 ┌────────────┐    ┌────────────┐
                 │ Fixed Mode │    │ [N/total]  │
                 │ Random Mode│    │  Progress  │
                 └────────────┘    └────────────┘
```

## Detailed Design

### 1. Video Assembly Strategies

**Location**: `src/video/assembler/video_strategies.py`

**Purpose**: Implement interchangeable video assembly strategies using Strategy Pattern.

**Class Hierarchy**:
```python
class VideoAssemblyStrategy(ABC):
    """Abstract base for video assembly strategies."""

    @abstractmethod
    def assemble(
        self,
        videos: list[Path],
        images: list[Path],
        target_duration: float,
    ) -> list[tuple[Path, float, bool]]:
        """Return list of (file, duration, is_video) for assembly."""
        ...

class SequentialStrategy(VideoAssemblyStrategy):
    """Concatenate all videos in sequence with crossfade transitions."""

class SingleBestStrategy(VideoAssemblyStrategy):
    """Use only the longest video, loop/trim to target duration."""

class MixedMediaStrategy(VideoAssemblyStrategy):
    """Interleave videos and images for variety."""

class VideoFirstFallbackStrategy(VideoAssemblyStrategy):
    """Prioritize videos, fall back to images when needed."""

class VideoStrategyFactory:
    """Factory for creating strategy instances."""

    @staticmethod
    def create(mode: str, config: VideoConfig) -> VideoAssemblyStrategy:
        strategies = {
            "sequential": SequentialStrategy,
            "single_best": SingleBestStrategy,
            "mixed_media": MixedMediaStrategy,
            "video_first_fallback": VideoFirstFallbackStrategy,
        }
        return strategies[mode](config)
```

**Strategy Selection**:
| Mode | Behavior | Best For |
|------|----------|----------|
| sequential | Concat all videos | Product demos with multiple clips |
| single_best | Use longest video | Products with one good video |
| mixed_media | Interleave videos/images | Visual variety |
| video_first_fallback | Videos preferred, images backup | Uncertain media availability |

### 2. Aspect Ratio Handling

**Location**: `src/video/assembler/video_strategies.py`

**Modes**:
```python
class AspectRatioMode(Enum):
    LETTERBOX = "letterbox"        # Add black bars
    CROP_TO_FIT = "crop_to_fit"    # Center-crop
    SMART_SCALE = "smart_scale"    # Intelligent scaling
```

**FFmpeg Filter Construction**:
```python
def build_aspect_filter(mode: AspectRatioMode, target: tuple[int, int]) -> str:
    width, height = target
    if mode == AspectRatioMode.LETTERBOX:
        return f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    elif mode == AspectRatioMode.CROP_TO_FIT:
        return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    else:  # SMART_SCALE
        return f"scale={width}:{height}:force_original_aspect_ratio=decrease"
```

### 3. Freesound Client

**Location**: `src/audio/freesound_client.py`

**Purpose**: Async music search and download with three-tier reliability.

**Class Structure**:
```python
class FreesoundClient:
    """Async Freesound API client with OAuth2 and resilience."""

    def __init__(
        self,
        api_key: str,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.oauth_token: str | None = None
        self.oauth_token_expiry: datetime | None = None
        self._session: aiohttp.ClientSession | None = None

    @freesound_circuit_breaker
    async def search_music(
        self,
        query: str,
        target_duration: float | None = None,
        min_duration: float | None = None,
        max_duration: float | None = None,
    ) -> list[FreesoundTrack]:
        """Search for music with duration matching."""
        ...

    async def download_track(
        self,
        track: FreesoundTrack,
        output_path: Path,
    ) -> bool:
        """Download track with OAuth2 → API key fallback."""
        ...

    async def _refresh_oauth_token(self) -> bool:
        """Refresh OAuth2 token using refresh token."""
        ...
```

**Reliability Tiers**:
```
Tier 1: OAuth2 HQ Download
    │
    ▼ (on failure)
Tier 2: API Key Preview Download
    │
    ▼ (on failure)
Tier 3: Local Fallback Library
```

**Circuit Breaker Integration**:
| Operation | Threshold | Timeout | Recovery |
|-----------|-----------|---------|----------|
| search_music | 3 failures | 30s | Half-open probe |
| download | 3 failures | 30s | Half-open probe |
| token_refresh | 3 failures | 60s | Half-open probe |

### 4. Subtitle System

**Location**: `src/video/config/subtitle_models.py`, `src/video/subtitle/`

**Positioning Model**:
```python
class SubtitleAnchor(Enum):
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"
    ABOVE_CONTENT = "above_content"
    BELOW_CONTENT = "below_content"

class UnifiedSubtitleConfig(BaseModel):
    anchor: SubtitleAnchor = SubtitleAnchor.BOTTOM
    margin: float = 0.05  # Fraction of frame height
    content_aware: bool = True
    horizontal_alignment: str = "center"
```

**ASS Effects Configuration**:
```python
class SubtitleEffectsSettings(BaseModel):
    # Karaoke
    karaoke_enabled: bool = False
    karaoke_timing_min_ms: int = 20
    karaoke_timing_max_ms: int = 200
    karaoke_use_fill: bool = True

    # Fade
    fade_in_duration_ms: int = 300
    fade_out_duration_ms: int = 300

    # Typewriter
    typewriter_enabled: bool = False
    typewriter_char_delay_ms: int = 50

    # Glow
    glow_enabled: bool = False
    glow_color: str = "&HFFFFFF"

    # Bounce
    bounce_enabled: bool = False
    bounce_rotation_min: float = -5.0
    bounce_rotation_max: float = 5.0

    # Pulse
    pulse_enabled: bool = False
    pulse_scale_min: float = 100.0
    pulse_scale_max: float = 110.0
```

**Style Presets**:
```python
class StylePreset(Enum):
    MINIMAL = "minimal"      # Clean, no effects
    MODERN = "modern"        # Subtle fade only
    BOLD = "bold"            # High contrast, glow
    ANIMATED = "animated"    # Full effects suite
    RANDOM = "random"        # Random effect selection
```

### 5. Batch Processing

**Location**: `src/video/producer/cli.py`, `src/video/producer/utils.py`

**Product Discovery**:
```python
def discover_products_for_batch(
    outputs_dir: Path,
) -> list[tuple[Path, ProductData]]:
    """Scan outputs directory for valid products."""
    products = []
    skip_dirs = {"cache", "logs", "reports", ".git"}

    for subdir in outputs_dir.iterdir():
        if subdir.name in skip_dirs or not subdir.is_dir():
            continue
        data_json = subdir / "data.json"
        if data_json.exists():
            try:
                product = ProductData.from_json(data_json)
                products.append((subdir, product))
            except Exception as e:
                logger.warning(f"Skipping {subdir.name}: {e}")

    return products
```

**Profile Randomization**:
```python
def select_profile_for_product(
    product_id: str,
    profile_pool: list[str],
) -> str:
    """Deterministically select profile using product ID hash."""
    # Use product ID as seed for reproducibility
    hash_value = int(hashlib.md5(product_id.encode()).hexdigest(), 16)
    index = hash_value % len(profile_pool)
    return profile_pool[index]

class ProfileUsageTracker:
    """Track profile usage across batch run."""

    def __init__(self) -> None:
        self.usage: dict[str, int] = defaultdict(int)

    def record(self, profile: str) -> None:
        self.usage[profile] += 1

    def format_summary(self) -> str:
        total = sum(self.usage.values())
        lines = [f"Profile Distribution ({total} products):"]
        for profile, count in sorted(self.usage.items()):
            pct = count / total * 100
            lines.append(f"  {profile}: {count} ({pct:.1f}%)")
        return "\n".join(lines)
```

**Batch Execution Flow**:
```python
async def run_batch(
    outputs_dir: Path,
    profile: str | None,
    random_profile: bool,
    profile_pool: list[str] | None,
    fail_fast: bool,
) -> BatchSummary:
    products = discover_products_for_batch(outputs_dir)
    total = len(products)
    tracker = ProfileUsageTracker()

    for i, (path, product) in enumerate(products, 1):
        selected_profile = (
            profile if profile
            else select_profile_for_product(product.id, profile_pool)
        )
        tracker.record(selected_profile)

        try:
            await process_product(path, product, selected_profile)
            logger.info(f"[{i}/{total}] SUCCESS: {product.id}")
            summary.succeeded.append(product.id)
        except Exception as e:
            logger.error(f"[{i}/{total}] FAILED: {product.id} - {e}")
            summary.failed.append((product.id, str(e)))
            if fail_fast:
                logger.error(f"Fail-fast: {total - i} pending")
                break

    return summary
```

### 6. Profile System

**Location**: `src/video/config/visual_models.py`, `src/video/config/core_models.py`

**VideoProfile Structure**:
```python
class VideoProfile(BaseModel):
    """Per-profile video production settings."""

    # Media selection
    use_scraped_images: bool = True
    use_scraped_videos: bool = False
    use_stock_images: bool = False
    use_stock_videos: bool = False
    stock_image_count: int = 0
    stock_video_count: int = 0

    # Video assembly overrides
    video_assembly_mode: str | None = None
    video_aspect_mode: str | None = None
    video_audio_handling: str | None = None
    video_original_volume: float = -30.0
    video_transition_duration: float = 0.5
    enable_format_normalization: bool = True

    # Subtitle overrides
    subtitle_anchor: str | None = None
    subtitle_margin: float | None = None
    subtitle_content_aware: bool | None = None
    subtitle_style_preset: str | None = None
    subtitle_font_size_scale: float | None = None

    # Two-part subtitles
    two_part_subtitles_enabled: bool = False
    upper_line_source_field: str | None = None
    upper_line_anchor: str | None = None
    lower_line_anchor: str | None = None
```

**Profile Resolution**:
```
CLI Override > Profile Setting > Global Config > Default Value
```

### 7. Pipeline Steps

**Location**: `src/video/producer/steps.py`

**Step Sequence**:
| Step | Purpose | Input | Output |
|------|---------|-------|--------|
| gather_visuals | Collect images/videos | data.json | visuals_info.json |
| generate_script | LLM voiceover script | product data | script.txt |
| create_voiceover | TTS audio generation | script.txt | voiceover.mp3 |
| generate_subtitles | Word timing | voiceover.mp3 | subtitles.srt |
| download_music | Background music | duration | music.mp3 |
| fetch_stock | Stock media | profile config | stock files |
| assemble_video | Video composition | all media | video.mp4 |
| apply_subtitles | Burn-in subtitles | video + srt/ass | final.mp4 |

**Step Resumability**:
```python
VALID_STEPS = [
    "gather_visuals",
    "generate_script",
    "generate_description",
    "create_voiceover",
    "generate_subtitles",
    "download_music",
    "fetch_stock",
    "assemble_video",
    "apply_subtitles",
]

def load_artifacts_for_step(step: str, run_paths: RunPaths) -> dict:
    """Load artifacts needed to resume from specific step."""
    loaders = {
        "gather_visuals": _load_artifacts_gather_visuals,
        "generate_script": _load_artifacts_generate_script,
        # ...
    }
    return loaders[step](run_paths)
```

## File Structure

```
src/video/
├── producer/
│   ├── __init__.py
│   ├── __main__.py           # Entry point
│   ├── cli.py                # CLI and batch processing
│   ├── orchestration.py      # Pipeline orchestration
│   ├── steps.py              # Pipeline step implementations
│   ├── state.py              # State management
│   ├── context.py            # Pipeline context
│   └── utils.py              # Profile selection utilities
├── assembler/
│   ├── video_strategies.py   # Assembly strategy classes
│   └── ...
├── subtitle/
│   ├── subtitle_positioning.py
│   ├── subtitle_builder.py
│   └── ...
├── config/
│   ├── core_models.py        # VideoConfig
│   ├── visual_models.py      # VideoProfile
│   └── subtitle_models.py    # SubtitleEffectsSettings
└── video_config.py           # Config loading

src/audio/
├── freesound_client.py       # Freesound API client
└── ...

src/utils/
├── circuit_breaker.py        # Circuit breaker implementation
└── logging_setup.py          # Logging configuration
```

## Integration Points

### Component Integration

| Component | Config Source | Logging | Circuit Breaker |
|-----------|--------------|---------|-----------------|
| Producer CLI | VideoConfig | ✓ debug_mode | - |
| Steps | VideoProfile | ✓ per-step | ✓ TTS, LLM |
| Freesound | .env secrets | ✓ API calls | ✓ search, download |
| Video Assembly | Profile overrides | ✓ strategy selection | - |
| Subtitles | Profile + effects | ✓ ASS generation | - |

### CLI Integration

| Flag | Purpose | Mutual Exclusivity |
|------|---------|-------------------|
| --batch | Enable batch mode | - |
| --batch-profile | Fixed profile for batch | vs --random-profile |
| --random-profile | Random profile selection | vs --batch-profile |
| --profile-pool | Limit random selection | requires --random-profile |
| --fail-fast | Stop on first error | - |
| --step | Resume from specific step | single product mode |

## Error Handling

### Error Categories

1. **Recoverable** (retry with backoff):
   - Network timeouts
   - Rate limiting
   - Temporary API failures

2. **Skippable** (log and continue):
   - Missing optional media
   - Invalid individual product
   - Music download failure

3. **Fatal** (stop pipeline):
   - Invalid profile configuration
   - Missing required secrets
   - Filesystem errors

### Circuit Breaker States

```
CLOSED ──(3 failures)──▶ OPEN ──(30s timeout)──▶ HALF_OPEN
   ▲                                                 │
   │                    success                      │
   └─────────────────────────────────────────────────┘
                          │
                       failure
                          ▼
                        OPEN
```

## Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| aiohttp | Async HTTP for Freesound | ^3.9 |
| pydantic | Configuration models | ^2.0 |
| ffmpeg-python | Video processing | ^0.2 |
| pyyaml | YAML config loading | ^6.0 |
| python-dotenv | .env file loading | ^1.0 |

## Alternatives Considered

### Video Assembly

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Single mode | Simple | No flexibility | Not chosen |
| Strategy Pattern | Flexible, testable | More code | **Chosen** |
| Plugin system | Extensible | Over-engineered | Not chosen |

### Profile Randomization

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| True random | Varied | Non-reproducible | Not chosen |
| Hash-based seed | Reproducible | Deterministic | **Chosen** |
| Round-robin | Even distribution | Predictable | Not chosen |

### Music Service

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Freesound only | Free, large library | Requires attribution | **Chosen** |
| Premium services | Higher quality | Cost, licensing | Not chosen |
| Local only | No network | Limited variety | Fallback only |
