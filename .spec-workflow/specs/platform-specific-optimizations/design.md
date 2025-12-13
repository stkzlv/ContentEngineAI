# Design Document

## Overview

This feature introduces platform-specific metadata generation for YouTube, TikTok, and Instagram, replacing the current unified description generator. The design follows a modular, extensible architecture where each platform has its own metadata generator implementing a common interface. The system integrates seamlessly into the existing video production pipeline as an enhancement to the `generate_description` step.

**Architecture Pattern:** Factory + Strategy pattern with async LLM integration
**Integration Point:** Video production pipeline (`src/video/producer/steps.py`)
**Configuration:** Pydantic models in `src/video/config/core_models.py` + YAML config files

## Steering Document Alignment

### Technical Standards (tech.md)

**Async-First Architecture:**
- All metadata generators use `async def` methods with `aiohttp` for LLM API calls
- Parallel generation for multi-platform mode using `asyncio.gather()`
- Reuse existing `openrouter_circuit_breaker` and retry logic patterns

**Pydantic Configuration:**
- Extend existing `DescriptionSettings` model with platform-specific fields
- Add `PlatformMetadataSettings` model with validation rules per platform
- Follow existing validation patterns (min/max constraints, required fields)

**LLM Integration:**
- Reuse `_fetch_and_select_model()`, `_call_llm_api_with_retry()` patterns from `script_generator.py`
- Share HTTP session from connection pool across all generators
- Apply circuit breaker pattern for API reliability

### Project Structure (structure.md)

**File Organization:**
- `src/ai/platform_metadata/` - New package for platform-specific generators
  - `__init__.py` - Factory and base interface
  - `base.py` - Abstract base class
  - `youtube.py` - YouTube metadata generator
  - `tiktok.py` - TikTok metadata generator
  - `instagram.py` - Instagram metadata generator
  - `models.py` - Platform metadata data models

**Prompt Templates:**
- `src/ai/prompts/youtube_metadata.md` - YouTube-specific prompt
- `src/ai/prompts/tiktok_caption.md` - TikTok-specific prompt
- `src/ai/prompts/instagram_caption.md` - Instagram-specific prompt

**Configuration:**
- `config/ai_services.yaml` - Add `platform_metadata` section
- `src/video/config/core_models.py` - Extend `DescriptionSettings` model

**Naming Conventions:**
- Classes: `YouTubeMetadataGenerator`, `PlatformMetadata` (PascalCase)
- Functions: `generate_platform_metadata()`, `validate_metadata()` (snake_case)
- Files: `youtube.py`, `platform_metadata_generator.py` (snake_case)

## Code Reuse Analysis

### Existing Components to Leverage

**LLM Infrastructure (`src/ai/description_generator.py`):**
- `_fetch_and_select_model()` - Auto-select free models from OpenRouter API
- `_call_llm_api_with_retry()` - Retry logic with exponential backoff
- `load_prompt_template()` - Template file loading
- `format_prompt()` - Product data injection pattern
- `validate_description_completeness()` - Validation framework

**Configuration System (`src/video/config/`):**
- `LLMSettings` - Shared LLM configuration (API keys, models, timeouts)
- `DescriptionSettings` - Extend with platform-specific fields
- `VideoConfig` - Pipeline integration point

**Connection Management (`src/utils/connection_pool.py`):**
- `get_http_session()` - Shared async HTTP session with connection pooling

**Circuit Breaker (`src/utils/circuit_breaker.py`):**
- `openrouter_circuit_breaker` - Decorator for API reliability

### Integration Points

**Video Production Pipeline (`src/video/producer/steps.py`):**
- `step_generate_description()` - Replace with platform-aware version
- `_load_artifacts_generate_description()` - Load multiple platform metadata files
- Pipeline graph dependencies remain unchanged

**Product Data (`src/scraper/amazon/scraper.py`):**
- `ProductData.shortened_affiliate_link` - URL source for metadata
- `ProductData.title`, `ProductData.description` - Content sources

**File System (`outputs/{product_id}/`):**
- Save multiple metadata files: `metadata_youtube.json`, `metadata_tiktok.json`, `metadata_instagram.json`
- Maintain backward compatibility with `description.txt` fallback

## Architecture

```mermaid
graph TD
    A[Pipeline: step_generate_description] --> B[PlatformMetadataFactory]
    B --> C{Target Platform}
    C -->|YouTube| D[YouTubeMetadataGenerator]
    C -->|TikTok| E[TikTokMetadataGenerator]
    C -->|Instagram| F[InstagramMetadataGenerator]
    C -->|Multi| G[All Generators in Parallel]

    D --> H[Load youtube_metadata.md template]
    E --> I[Load tiktok_caption.md template]
    F --> J[Load instagram_caption.md template]

    H --> K[LLM API Call with Retry]
    I --> K
    J --> K

    K --> L[Validate Character Limits & Required Elements]
    L --> M{Valid?}
    M -->|Yes| N[Save metadata_{platform}.json]
    M -->|No| O[Retry with Fallback Model]
    O --> K

    N --> P[Pipeline: step_create_voiceover]

    style A fill:#e1f5e1
    style P fill:#e1f5e1
    style B fill:#fff4e6
    style M fill:#ffe6e6
```

### Modular Design Principles

**Single File Responsibility:**
- `base.py` - Abstract interface only, no implementation
- `youtube.py` - YouTube-specific generation logic only
- `tiktok.py` - TikTok-specific generation logic only
- `instagram.py` - Instagram-specific generation logic only
- `models.py` - Data models only, no business logic

**Component Isolation:**
- Each platform generator is independent, no cross-platform dependencies
- Factory pattern enables runtime platform selection without code changes
- Validation logic encapsulated per platform generator

**Service Layer Separation:**
- LLM API calls abstracted via shared utility functions
- Prompt template loading separate from generation logic
- Metadata persistence separate from generation logic

## Components and Interfaces

### Component 1: BasePlatformMetadataGenerator (Abstract Base Class)

**Purpose:** Define common interface for all platform-specific generators

**Interfaces:**
```python
class BasePlatformMetadataGenerator(ABC):
    @abstractmethod
    async def generate(
        self,
        product: ProductData,
        settings: LLMSettings,
        secrets: dict[str, str],
        session: aiohttp.ClientSession,
        intermediate_paths: dict[str, Path],
        debug_mode: bool,
        api_settings=None,
    ) -> PlatformMetadata | None:
        """Generate platform-specific metadata."""

    @abstractmethod
    def validate(self, metadata: PlatformMetadata) -> tuple[bool, str]:
        """Validate generated metadata against platform rules."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return platform identifier (youtube, tiktok, instagram)."""
```

**Dependencies:** `aiohttp`, `ProductData`, `LLMSettings`, `PlatformMetadata`

**Reuses:** None (abstract base class)

### Component 2: YouTubeMetadataGenerator

**Purpose:** Generate YouTube-optimized titles, descriptions, and hashtags

**Interfaces:**
```python
class YouTubeMetadataGenerator(BasePlatformMetadataGenerator):
    async def generate(...) -> PlatformMetadata | None:
        """Generate YouTube metadata with SEO optimization."""

    def validate(self, metadata: PlatformMetadata) -> tuple[bool, str]:
        """Validate YouTube-specific rules (title 50-60 chars, 3-5 hashtags, #Shorts)."""

    @property
    def platform_name(self) -> str:
        return "youtube"
```

**Dependencies:**
- `load_prompt_template()` - Load `youtube_metadata.md`
- `_call_llm_api_with_retry()` - LLM API integration
- `openrouter_circuit_breaker` - Reliability

**Reuses:**
- `_fetch_and_select_model()` from `description_generator.py`
- `format_prompt()` pattern for product data injection
- HTTP session from connection pool

### Component 3: TikTokMetadataGenerator

**Purpose:** Generate TikTok-optimized captions and hashtags with search focus

**Interfaces:**
```python
class TikTokMetadataGenerator(BasePlatformMetadataGenerator):
    async def generate(...) -> PlatformMetadata | None:
        """Generate TikTok caption with SEO keywords and niche hashtags."""

    def validate(self, metadata: PlatformMetadata) -> tuple[bool, str]:
        """Validate TikTok rules (100-300 chars optimal, 3-5 hashtags, no #fyp)."""

    @property
    def platform_name(self) -> str:
        return "tiktok"
```

**Dependencies:** Same as YouTubeMetadataGenerator

**Reuses:** Same as YouTubeMetadataGenerator

### Component 4: InstagramMetadataGenerator

**Purpose:** Generate Instagram Reels captions and 15-30 hashtags

**Interfaces:**
```python
class InstagramMetadataGenerator(BasePlatformMetadataGenerator):
    async def generate(...) -> PlatformMetadata | None:
        """Generate Instagram caption (short or SEO style) with 15-30 hashtags."""

    def validate(self, metadata: PlatformMetadata) -> tuple[bool, str]:
        """Validate Instagram rules (15-30 hashtags in caption, proper format)."""

    @property
    def platform_name(self) -> str:
        return "instagram"

    def _determine_caption_style(self, product: ProductData) -> str:
        """Choose between 'short' (3-5 words) or 'seo' (100-200 chars) style."""
```

**Dependencies:** Same as YouTubeMetadataGenerator

**Reuses:** Same as YouTubeMetadataGenerator

### Component 5: PlatformMetadataFactory

**Purpose:** Instantiate appropriate metadata generator based on target platform

**Interfaces:**
```python
class PlatformMetadataFactory:
    @staticmethod
    def create(platform: str) -> BasePlatformMetadataGenerator:
        """Create generator for specified platform."""

    @staticmethod
    async def generate_multi_platform(
        platforms: list[str],
        product: ProductData,
        settings: LLMSettings,
        ...
    ) -> dict[str, PlatformMetadata]:
        """Generate metadata for multiple platforms in parallel."""
```

**Dependencies:** All platform generator classes

**Reuses:** `asyncio.gather()` for parallel execution

### Component 6: Pipeline Integration (Modified `step_generate_description`)

**Purpose:** Integrate platform metadata generation into video production pipeline

**Interfaces:**
```python
async def step_generate_description(ctx: PipelineContext):
    """Generate platform-specific metadata based on configuration."""
```

**Dependencies:**
- `PlatformMetadataFactory`
- `ctx.config.description_settings.target_platform`
- `ctx.product`

**Reuses:**
- Existing `performance_monitor.measure_step()` pattern
- Existing artifact loading/saving patterns
- Existing error handling framework

## Data Models

### PlatformMetadata
```python
@dataclass
class PlatformMetadata:
    platform: str  # "youtube", "tiktok", "instagram"
    title: str | None  # YouTube only, None for others
    description: str  # Platform-specific description/caption
    hashtags: list[str]  # Platform-optimized hashtag list
    keywords: list[str]  # SEO keywords (YouTube primary, TikTok secondary)
    character_counts: dict[str, int]  # {"title": 58, "description": 487}
    generated_at: str  # ISO 8601 timestamp
    product_id: str  # ASIN or product identifier
    validation_status: str  # "valid", "warning", "error"
    validation_messages: list[str]  # Specific validation details
```

### PlatformMetadataSettings (New Pydantic Model)
```python
class PlatformMetadataSettings(BaseModel):
    enabled: bool = True
    target_platform: str = "multi"  # "youtube", "tiktok", "instagram", "multi"

    youtube: YouTubePlatformSettings
    tiktok: TikTokPlatformSettings
    instagram: InstagramPlatformSettings

class YouTubePlatformSettings(BaseModel):
    enabled: bool = True
    title_length_max: int = 60
    description_length_max: int = 5000
    hashtag_count_min: int = 3
    hashtag_count_max: int = 5
    include_shorts_tag: bool = True
    seo_keywords: bool = True

class TikTokPlatformSettings(BaseModel):
    enabled: bool = True
    caption_length_optimal: int = 150
    caption_length_max: int = 2200
    hashtag_count_min: int = 3
    hashtag_count_max: int = 5
    seo_focused: bool = True
    avoid_generic_tags: list[str] = ["foryoupage", "fyp", "viral"]

class InstagramPlatformSettings(BaseModel):
    enabled: bool = True
    caption_style: str = "seo"  # "short" or "seo"
    caption_length_short: int = 15
    caption_length_seo: int = 200
    hashtag_count_min: int = 15
    hashtag_count_max: int = 30
    emoji_enabled: bool = True
```

### Extended DescriptionSettings (Modified)
```python
class DescriptionSettings(BaseModel):
    enabled: bool = True

    # Legacy unified mode (backward compatibility)
    prompt_template_path: str = "src/ai/prompts/video_description.md"

    # New platform-specific mode
    platform_metadata: PlatformMetadataSettings | None = None
```

## Error Handling

### Error Scenarios

**1. LLM API Failures**
- **Handling:** Retry up to 2 times per model, fallback to alternative models, circuit breaker prevents cascading failures
- **User Impact:** Graceful degradation to unified description generator if all platform-specific attempts fail
- **Logging:** Detailed error logs with model name, attempt number, error message

**2. Invalid Platform Specification**
- **Handling:** Validate platform name at configuration load time, raise `ValidationError` with allowed values
- **User Impact:** Clear error message: "Invalid platform 'tiktak', allowed: youtube, tiktok, instagram, multi"
- **Logging:** Configuration validation error logged at startup

**3. Character Limit Violations**
- **Handling:** Truncate gracefully with ellipsis, log warning, retry with fallback model if critical
- **User Impact:** Warning logged but metadata still saved, user informed of truncation
- **Logging:** "YouTube title truncated from 75 to 60 chars: 'Best Wireless Earbuds...'"

**4. Missing Required Elements (#ad hashtag)**
- **Handling:** Automatically append required hashtags, log info message
- **User Impact:** Transparent addition of mandatory elements
- **Logging:** "Added required #ad hashtag to YouTube description"

**5. Prompt Template Not Found**
- **Handling:** Raise `FileNotFoundError` with full path, suggest template installation
- **User Impact:** Pipeline fails fast with actionable error message
- **Logging:** "Prompt template not found: src/ai/prompts/youtube_metadata.md - ensure templates are installed"

**6. Multi-Platform Partial Failures**
- **Handling:** Continue processing remaining platforms, collect all errors, return partial results
- **User Impact:** Some platforms succeed, others fall back to unified description
- **Logging:** "YouTube metadata generated successfully, TikTok failed (retrying), Instagram succeeded"

## Testing Strategy

### Unit Testing

**Test Files:**
- `tests/ai/test_platform_metadata_generators.py` - Generator logic
- `tests/ai/test_platform_metadata_validation.py` - Validation rules
- `tests/ai/test_platform_metadata_factory.py` - Factory pattern

**Key Test Cases:**
- Mock LLM API responses for each platform
- Validate character limit enforcement
- Test hashtag count constraints
- Verify required element addition (#ad, #Shorts)
- Test validation logic per platform
- Test factory platform selection
- Test multi-platform parallel generation

**Coverage Target:** >90% code coverage for all generator modules

### Integration Testing

**Test Files:**
- `tests/test_platform_metadata_integration.py` - End-to-end pipeline integration

**Key Flows:**
- Single platform generation (YouTube)
- Multi-platform generation (all three)
- Fallback to unified description on failures
- CLI argument override (--target-platform)
- YAML configuration loading
- Batch processing with platform targeting

**Coverage Target:** >80% integration coverage

### End-to-End Testing

**Test Scenarios:**
- Generate YouTube video with platform-specific metadata
- Generate multi-platform batch (10 products, random profiles)
- Test metadata file persistence and loading
- Verify backward compatibility with existing unified mode
- Test graceful degradation on API failures

**User Scenarios:**
- "As a YouTuber, I want optimized titles and descriptions for my product videos"
- "As a multi-platform marketer, I want metadata for all platforms in one command"
- "As a batch processor, I want platform targeting applied consistently across 100 products"
