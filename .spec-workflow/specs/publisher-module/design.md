# Design Document: Publisher Module

## Overview

The Publisher Module provides multi-platform video publishing through the Late.dev service, with scheduling capabilities and post-publication cleanup. The architecture follows a provider pattern with clear separation between publishing, scheduling, and cleanup concerns.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Publisher Module                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   CLI Layer  │    │  Batch Layer │    │ Schedule CLI │              │
│  │  late/cli.py │    │   batch.py   │    │  schedule.py │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │                    Core Services                          │          │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │          │
│  │  │LatePublisher│  │ScheduleMgr │  │ CleanupMgr  │       │          │
│  │  │  client.py  │  │ schedule.py │  │ cleanup.py  │       │          │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │          │
│  └─────────┼────────────────┼────────────────┼──────────────┘          │
│            │                │                │                          │
│            ▼                ▼                ▼                          │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │                   Support Services                        │          │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌─────────┐ │          │
│  │  │ Metadata │  │ Validator │  │ Tracking │  │ Config  │ │          │
│  │  │metadata.py│ │schedule_  │  │tracking.py│ │config.py│ │          │
│  │  │          │  │validator  │  │          │  │         │ │          │
│  │  └──────────┘  └───────────┘  └──────────┘  └─────────┘ │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │                   Data Models (models.py)                 │          │
│  │  PublishResult, ScheduleSlot, CleanupRecord, BatchSummary │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────┐
                         │   Late.dev API   │
                         │  (via late-sdk)  │
                         └──────────────────┘
```

## Component Design

### Component 1: BasePublisher Abstract Interface

**File:** `src/publisher/base.py`

**Purpose:** Define abstract interface for publishing providers.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PublishResult:
    success: bool
    post_id: str | None
    platform: str
    video_path: Path
    scheduled_time: datetime | None
    error: str | None
    url: str | None

class BasePublisher(ABC):
    @abstractmethod
    async def publish(
        self,
        video_path: Path,
        platforms: list[str],
        metadata: dict[str, Any],
        schedule_time: datetime | None = None,
    ) -> list[PublishResult]:
        """Publish video to specified platforms."""
        pass

    @abstractmethod
    async def get_accounts(self) -> list[dict[str, Any]]:
        """Get connected social media accounts."""
        pass

    @abstractmethod
    async def get_status(self, post_id: str) -> dict[str, Any]:
        """Get status of a published/scheduled post."""
        pass
```

### Component 2: LatePublisher Implementation

**File:** `src/publisher/late/client.py`

**Purpose:** Late.dev API integration using official SDK.

```python
from late import Client as LateClient
from src.publisher.base import BasePublisher, PublishResult

class LatePublisher(BasePublisher):
    def __init__(self, api_key: str, user_id: str):
        self.client = LateClient(api_key=api_key)
        self.user_id = user_id
        self._accounts_cache: list[dict] | None = None

    async def publish(
        self,
        video_path: Path,
        platforms: list[str],
        metadata: dict[str, Any],
        schedule_time: datetime | None = None,
    ) -> list[PublishResult]:
        # Upload media (handles small/large files automatically)
        media_url = await self._upload_media(video_path)

        results = []
        for platform in platforms:
            platform_meta = self._prepare_metadata(platform, metadata)
            result = await self._publish_to_platform(
                media_url, platform, platform_meta, schedule_time
            )
            results.append(result)
        return results

    async def _upload_media(self, video_path: Path) -> str:
        file_size = video_path.stat().st_size
        if file_size <= 4 * 1024 * 1024:  # 4MB
            return await self._upload_direct(video_path)
        else:
            return await self._upload_with_token(video_path)
```

### Component 3: Metadata Loader

**File:** `src/publisher/metadata.py`

**Purpose:** Generate platform-specific metadata from product data.

```python
@dataclass
class PlatformMetadata:
    title: str
    description: str
    tags: list[str]
    platform_specific: dict[str, Any]

class MetadataLoader:
    PLATFORM_LIMITS = {
        "youtube": {"title": 100, "description": 5000, "tags": 500},
        "tiktok": {"title": 150, "description": 2200},
        "instagram": {"caption": 2200},
    }

    def load_from_product(
        self, product_data: dict[str, Any], platform: str
    ) -> PlatformMetadata:
        """Generate metadata from product data with platform limits."""
        title = self._truncate(
            product_data.get("title", ""),
            self.PLATFORM_LIMITS[platform]["title"]
        )
        # Generate description, tags, platform-specific fields
        return PlatformMetadata(...)

    def load_from_file(self, metadata_path: Path) -> PlatformMetadata:
        """Load explicit metadata from JSON/YAML file."""
        pass
```

### Component 4: ScheduleManager

**File:** `src/publisher/schedule.py`

**Purpose:** Manage publishing schedule with recurring slots.

```python
@dataclass
class ScheduleSlot:
    day_of_week: int  # 0=Monday, 6=Sunday
    time: time
    platforms: list[str]
    timezone: str = "UTC"

@dataclass
class ScheduledPost:
    post_id: str
    product_id: str
    scheduled_time: datetime
    platforms: list[str]
    status: str  # pending, published, failed

class ScheduleManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.slots: list[ScheduleSlot] = []
        self.scheduled_posts: list[ScheduledPost] = []

    def get_calendar_view(
        self, start_date: date, days: int = 7
    ) -> dict[date, list[ScheduledPost]]:
        """Get calendar view of scheduled posts."""
        pass

    def find_next_available_slots(
        self, count: int, start_date: date | None = None
    ) -> list[datetime]:
        """Find next N available slots from recurring schedule."""
        pass

    def auto_schedule(
        self, product_ids: list[str], start_date: date | None = None
    ) -> list[ScheduledPost]:
        """Auto-schedule products to next available slots."""
        pass
```

### Component 5: ScheduleValidator

**File:** `src/publisher/schedule_validator.py`

**Purpose:** Validate scheduling constraints.

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]

class ScheduleValidator:
    def __init__(
        self,
        min_spacing_hours: int = 2,
        max_daily_posts: int = 5,
        require_unique_slots: bool = True,
    ):
        self.min_spacing_hours = min_spacing_hours
        self.max_daily_posts = max_daily_posts
        self.require_unique_slots = require_unique_slots

    def validate_slot(
        self, slot: datetime, platform: str, existing: list[ScheduledPost]
    ) -> ValidationResult:
        """Validate a single slot against constraints."""
        errors, warnings = [], []

        # Check duplicate slots
        if self.require_unique_slots:
            for post in existing:
                if post.scheduled_time == slot and platform in post.platforms:
                    errors.append(f"Duplicate slot: {slot} for {platform}")

        # Check minimum spacing
        for post in existing:
            if platform in post.platforms:
                gap = abs((post.scheduled_time - slot).total_seconds() / 3600)
                if gap < self.min_spacing_hours:
                    errors.append(f"Too close to existing post (gap: {gap}h)")

        # Check daily limit
        same_day = [p for p in existing if p.scheduled_time.date() == slot.date()]
        if len(same_day) >= self.max_daily_posts:
            errors.append(f"Daily limit exceeded ({self.max_daily_posts})")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
```

### Component 6: CleanupManager

**File:** `src/publisher/cleanup.py`

**Purpose:** Handle post-publication directory cleanup.

```python
@dataclass
class CleanupRecord:
    product_id: str
    cleanup_time: datetime
    platforms_verified: list[str]
    space_freed_mb: float
    archived: bool
    archive_path: Path | None

class CleanupManager:
    def __init__(
        self,
        outputs_dir: Path,
        archive_dir: Path | None = None,
        delay_hours: int = 24,
    ):
        self.outputs_dir = outputs_dir
        self.archive_dir = archive_dir
        self.delay_hours = delay_hours
        self.audit_log: list[CleanupRecord] = []

    async def verify_publication(
        self, product_id: str, platforms: list[str], publisher: BasePublisher
    ) -> dict[str, bool]:
        """Verify video is published on all specified platforms."""
        pass

    async def cleanup_product(
        self,
        product_id: str,
        platforms: list[str],
        publisher: BasePublisher,
        force: bool = False,
        archive: bool = True,
    ) -> CleanupRecord:
        """Cleanup product directory after verification."""
        if not force:
            verified = await self.verify_publication(product_id, platforms, publisher)
            if not all(verified.values()):
                unverified = [p for p, v in verified.items() if not v]
                raise CleanupError(f"Unverified platforms: {unverified}")

        product_dir = self.outputs_dir / product_id
        space_freed = self._get_directory_size(product_dir)

        if archive and self.archive_dir:
            archive_path = self._archive_directory(product_dir)
        else:
            archive_path = None

        shutil.rmtree(product_dir)

        record = CleanupRecord(
            product_id=product_id,
            cleanup_time=datetime.now(),
            platforms_verified=platforms,
            space_freed_mb=space_freed / (1024 * 1024),
            archived=archive,
            archive_path=archive_path,
        )
        self.audit_log.append(record)
        return record
```

### Component 7: BatchPublisher

**File:** `src/publisher/batch.py`

**Purpose:** Orchestrate batch publishing operations.

```python
@dataclass
class BatchConfig:
    concurrency: int = 3
    fail_fast: bool = False
    auto_cleanup: bool = False
    cleanup_delay_hours: int = 24

@dataclass
class BatchSummary:
    total: int
    succeeded: int
    failed: int
    skipped: int
    results: list[PublishResult]
    cleanup_records: list[CleanupRecord]
    duration_seconds: float

class BatchPublisher:
    def __init__(
        self,
        publisher: BasePublisher,
        schedule_manager: ScheduleManager,
        cleanup_manager: CleanupManager,
        config: BatchConfig,
    ):
        self.publisher = publisher
        self.schedule_manager = schedule_manager
        self.cleanup_manager = cleanup_manager
        self.config = config

    async def publish_batch(
        self,
        product_dirs: list[Path],
        platforms: list[str],
        use_auto_schedule: bool = False,
    ) -> BatchSummary:
        """Publish multiple products with progress tracking."""
        total = len(product_dirs)
        results = []

        for i, product_dir in enumerate(product_dirs, 1):
            logger.info(f"[{i}/{total}] Publishing {product_dir.name}")
            try:
                # Load metadata, get video, publish
                result = await self._publish_product(product_dir, platforms)
                results.append(result)
                logger.info(f"[{i}/{total}] SUCCESS: {product_dir.name}")

                if self.config.auto_cleanup:
                    await self._schedule_cleanup(product_dir.name, platforms)

            except Exception as e:
                logger.error(f"[{i}/{total}] FAILED: {product_dir.name} - {e}")
                if self.config.fail_fast:
                    raise

        return BatchSummary(...)
```

### Component 8: Publisher Configuration

**File:** `src/publisher/config.py`

**Purpose:** Configuration management with three-tier precedence.

```python
@dataclass
class PublisherConfig:
    # Late.dev credentials
    api_key: str
    user_id: str

    # Publishing defaults
    default_platforms: list[str] = field(default_factory=lambda: ["youtube"])

    # Scheduling settings
    min_spacing_hours: int = 2
    max_daily_posts: int = 5
    timezone: str = "UTC"

    # Cleanup settings
    auto_cleanup: bool = False
    cleanup_delay_hours: int = 24
    archive_before_delete: bool = True

    # Batch settings
    batch_concurrency: int = 3
    fail_fast: bool = False

    @classmethod
    def load(cls, cli_args: dict | None = None) -> "PublisherConfig":
        """Load config with CLI > ENV > YAML > defaults precedence."""
        # Load YAML defaults
        yaml_config = cls._load_yaml("config/publisher.yaml")

        # Override with environment
        env_config = cls._load_env()

        # Override with CLI args
        merged = {**yaml_config, **env_config, **(cli_args or {})}

        return cls(**merged)
```

### Component 9: Status Tracking

**File:** `src/publisher/tracking.py`

**Purpose:** Track publishing status and history.

```python
@dataclass
class PublishingStatus:
    product_id: str
    post_id: str
    platform: str
    status: str  # pending, uploading, scheduled, published, failed
    scheduled_time: datetime | None
    published_time: datetime | None
    url: str | None
    error: str | None

class StatusTracker:
    def __init__(self, tracking_file: Path):
        self.tracking_file = tracking_file
        self.statuses: dict[str, PublishingStatus] = {}

    def record(self, status: PublishingStatus) -> None:
        """Record or update status."""
        key = f"{status.product_id}:{status.platform}"
        self.statuses[key] = status
        self._save()

    def get_pending_cleanup(self) -> list[str]:
        """Get product IDs that are published and ready for cleanup."""
        published = [
            s.product_id for s in self.statuses.values()
            if s.status == "published"
        ]
        # Group by product_id and check all platforms published
        return self._filter_fully_published(published)
```

## Data Flow

### Single Video Publishing

```
1. CLI receives: video_path, --platforms, --schedule
2. MetadataLoader generates platform-specific metadata
3. LatePublisher.publish() called:
   a. Upload media (direct or token-based)
   b. For each platform: create post with metadata
   c. Return list of PublishResult
4. StatusTracker records results
5. If auto_cleanup enabled: CleanupManager schedules cleanup
```

### Batch Publishing with Auto-Schedule

```
1. CLI receives: --batch, --auto-schedule
2. BatchPublisher scans outputs/ for product directories
3. ScheduleManager.find_next_available_slots() gets slots
4. ScheduleValidator validates each slot
5. For each product:
   a. Assign next available slot
   b. Publish with scheduled_time
   c. Track status
   d. Schedule cleanup if enabled
6. Output BatchSummary
```

### Post-Publication Cleanup

```
1. CLI receives: --cleanup or auto-cleanup triggers
2. CleanupManager.verify_publication() checks Late API
3. If verified on all platforms:
   a. Archive directory (if enabled)
   b. Delete product directory
   c. Record in cleanup_audit.json
4. If verification fails: skip and log warning
```

## File Structure

```
src/publisher/
├── __init__.py
├── base.py              # BasePublisher ABC
├── batch.py             # BatchPublisher orchestrator
├── cleanup.py           # CleanupManager
├── config.py            # PublisherConfig
├── metadata.py          # MetadataLoader
├── models.py            # Data models
├── registry.py          # Provider registry
├── schedule.py          # ScheduleManager
├── schedule_validator.py # ScheduleValidator
├── tracking.py          # StatusTracker
└── late/
    ├── __init__.py
    ├── __main__.py      # Entry point
    ├── cli.py           # CLI implementation
    └── client.py        # LatePublisher implementation

config/
└── publisher.yaml       # Default configuration

tests/publisher/
├── __init__.py
├── test_batch.py
├── test_cleanup.py
├── test_metadata.py
├── test_schedule_manager.py
├── test_schedule_validator.py
├── test_tracking.py
└── late/
    ├── __init__.py
    └── test_client.py
```

## Configuration File

**File:** `config/publisher.yaml`

```yaml
# Late.dev settings (credentials in environment)
late:
  timeout_seconds: 30
  max_retries: 3

# Platform defaults
platforms:
  youtube:
    privacy: "public"
    category: "22"  # People & Blogs
    made_for_kids: false
  tiktok:
    privacy: "public"
    allow_comments: true
    allow_duet: true
    allow_stitch: true
  instagram:
    share_to_feed: true

# Scheduling
schedule:
  min_spacing_hours: 2
  max_daily_posts: 5
  timezone: "America/New_York"
  recurring_slots:
    - day: "monday"
      time: "10:00"
      platforms: ["youtube", "tiktok"]
    - day: "wednesday"
      time: "14:00"
      platforms: ["youtube", "instagram"]
    - day: "friday"
      time: "18:00"
      platforms: ["youtube", "tiktok", "instagram"]

# Cleanup
cleanup:
  auto_cleanup: false
  delay_hours: 24
  archive_before_delete: true
  archive_dir: "archive/"
  require_all_platforms: true

# Batch processing
batch:
  concurrency: 3
  fail_fast: false
```

## Error Handling Strategy

```python
class PublisherError(Exception):
    """Base exception for publisher errors."""
    pass

class UploadError(PublisherError):
    """Media upload failed."""
    pass

class ScheduleError(PublisherError):
    """Scheduling validation failed."""
    pass

class CleanupError(PublisherError):
    """Cleanup verification failed."""
    pass

class RateLimitError(PublisherError):
    """API rate limit exceeded."""
    retryable = True

class AuthenticationError(PublisherError):
    """API authentication failed."""
    retryable = False
```

## CLI Commands

```bash
# Publishing
python -m src.publisher.late publish <video> --platforms youtube tiktok
python -m src.publisher.late publish <video> --schedule "2024-01-15T10:00:00"
python -m src.publisher.late publish --batch --auto-schedule

# Status
python -m src.publisher.late status <post_id>
python -m src.publisher.late accounts

# Schedule
python -m src.publisher.late calendar --week
python -m src.publisher.late calendar --month
python -m src.publisher.late schedule auto --products B0ASIN1 B0ASIN2

# Cleanup
python -m src.publisher.late cleanup --product B0ASIN1
python -m src.publisher.late cleanup --date-range 2024-01-01 2024-01-15
python -m src.publisher.late cleanup --dry-run
```
