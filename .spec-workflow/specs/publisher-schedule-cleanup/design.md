# Design Document

## Introduction

This design document provides the technical architecture and implementation strategy for adding **Publishing Schedule & Calendar** and **Post-Publication Cleanup** capabilities to the ContentEngineAI Publisher module.

**Design Goals:**
1. **Modularity**: Separate schedule management, validation, and cleanup logic into distinct, testable components
2. **Consistency**: Follow existing publisher architecture patterns (BasePublisher interface, data models, CLI structure)
3. **Extensibility**: Design for easy addition of new platforms and schedule rules
4. **Reliability**: Ensure data integrity with validation, atomic operations, and audit logging
5. **Usability**: Provide clear CLI commands with dry-run support and comprehensive feedback

## Architecture Overview

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Layer (cli.py)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  calendar    │  │ schedule auto│  │   cleanup    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ ScheduleManager │  │ScheduleValidator│  │ CleanupManager  │
│  - list()       │  │  - validate()   │  │  - cleanup()    │
│  - auto_sched() │  │  - check_dupes()│  │  - verify()     │
│  - add_entry()  │  │  - check_space()│  │  - archive()    │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                     │
         └────────────────────┴─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  LatePublisher   │
                    │  - posts.list()  │
                    │  - get_status()  │
                    └──────────────────┘
```

### Data Flow

**Schedule Auto-Assignment Flow:**
```
User Command → CLI Parser → Load Config
                                 ↓
                         Scan outputs dir
                                 ↓
                         Filter unpublished videos
                                 ↓
                         Get recurring slots
                                 ↓
                         For each video:
                           - Find next available slot
                           - Validate schedule (ScheduleValidator)
                           - Create schedule entry (ScheduleManager)
                           - Call LatePublisher.publish() with scheduled_time
                           - Record in schedule.json
                                 ↓
                         Display summary
```

**Post-Publication Cleanup Flow:**
```
Successful Publish → Check cleanup.enabled config
                                 ↓
                         Verify all platforms published (CleanupManager)
                                 ↓
                         Query Late API for status (LatePublisher.get_status())
                                 ↓
                         Confirm all platforms = "published"
                                 ↓
                         Archive if cleanup.archive_before_delete=true
                                 ↓
                         Remove product directory (outputs/<product_id>/)
                                 ↓
                         Log to cleanup_audit.json
```

## Detailed Component Design

### 1. ScheduleManager Class

**Location:** `src/publisher/schedule.py`

**Purpose:** Manage calendar operations and recurring schedule slots

**Class Definition:**
```python
@dataclass
class RecurringSlot:
    """Single recurring time slot configuration."""
    day_of_week: str  # "monday", "tuesday", etc.
    time: str         # "14:00:00" (HH:MM:SS)
    timezone: str     # "UTC", "America/New_York", etc.

    def next_occurrence(self, after: datetime) -> datetime:
        """Calculate next occurrence of this slot after given datetime."""
        pass

@dataclass
class ScheduleEntry:
    """Single scheduled post entry."""
    product_id: str
    scheduled_time: datetime
    platforms: list[str]
    post_id: str | None
    status: str  # "pending", "scheduled", "published", "failed"
    created_at: datetime
    slot_index: int | None  # Which recurring slot was used

class ScheduleManager:
    """Manages calendar view and recurring schedule operations."""

    def __init__(self,
                 schedule_path: Path = Path("outputs/schedule.json"),
                 config: ScheduleConfig | None = None):
        """Initialize with schedule file path and configuration."""
        self.schedule_path = schedule_path
        self.config = config or ScheduleConfig()
        self.entries: list[ScheduleEntry] = []
        self._load_schedule()

    def list_scheduled(self,
                      platform: str | None = None,
                      status: str | None = None,
                      date_from: datetime | None = None,
                      date_to: datetime | None = None) -> list[ScheduleEntry]:
        """List scheduled posts with optional filtering."""
        pass

    def auto_schedule(self,
                     videos: list[Path],
                     platforms: list[Platform],
                     start_slot: int = 0,
                     dry_run: bool = False) -> dict[str, int]:
        """Auto-assign videos to recurring slots."""
        pass

    def add_entry(self, entry: ScheduleEntry) -> None:
        """Add schedule entry (validates before adding)."""
        pass

    def get_next_slot(self, after: datetime, slot_index: int = 0) -> tuple[datetime, int]:
        """Get next available recurring slot."""
        pass
```

**Key Methods:**

1. **`list_scheduled()`**: Query schedule.json with filters
   - Filter by platform, status, date range
   - Return sorted by scheduled_time
   - Display both UTC and local timezone

2. **`auto_schedule()`**: Batch assignment to slots
   - Load recurring_schedule from config
   - For each video: find next available slot (respecting min_post_spacing)
   - Validate with ScheduleValidator
   - Call LatePublisher.publish() with scheduled_time
   - Record in schedule.json
   - Return summary: {scheduled: N, skipped: N, failed: N}

3. **`add_entry()`**: Atomic write to schedule.json
   - Validate entry with ScheduleValidator
   - Append to entries list
   - Save atomically (write temp file → rename)

### 2. ScheduleValidator Class

**Location:** `src/publisher/schedule_validator.py`

**Purpose:** Enforce scheduling rules and prevent conflicts

**Class Definition:**
```python
@dataclass
class ScheduleConfig:
    """Schedule validation configuration."""
    min_post_spacing_hours: int = 2
    prevent_duplicates: bool = True
    allow_past_schedules: bool = False
    max_posts_per_day: int = 10
    timezone: str = "UTC"

class ScheduleValidator:
    """Validates schedule entries against rules."""

    def __init__(self, config: ScheduleConfig, existing_entries: list[ScheduleEntry]):
        """Initialize with config and existing schedule entries."""
        self.config = config
        self.existing_entries = existing_entries

    def validate(self, entry: ScheduleEntry) -> tuple[bool, str]:
        """
        Validate schedule entry against all rules.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check 1: Duplicate prevention
        if self.config.prevent_duplicates:
            if self._is_duplicate(entry):
                return False, f"Duplicate schedule: {entry.product_id} already scheduled to {entry.platforms} at {entry.scheduled_time}"

        # Check 2: Past schedule validation
        if not self.config.allow_past_schedules:
            if entry.scheduled_time < datetime.now(UTC):
                return False, f"Cannot schedule in past: {entry.scheduled_time}"

        # Check 3: Timezone validation
        if entry.scheduled_time.tzinfo is None:
            return False, "scheduled_time must include timezone info"

        # Check 4: Minimum spacing enforcement
        if not self._check_spacing(entry):
            return False, f"Minimum spacing violation: posts must be {self.config.min_post_spacing_hours}h apart on same platform"

        # Check 5: Daily limit enforcement
        if not self._check_daily_limit(entry):
            return False, f"Daily limit reached: max {self.config.max_posts_per_day} posts per day"

        return True, "Valid"

    def _is_duplicate(self, entry: ScheduleEntry) -> bool:
        """Check if same product+platform+time already scheduled."""
        pass

    def _check_spacing(self, entry: ScheduleEntry) -> bool:
        """Check minimum spacing between posts on same platform."""
        pass

    def _check_daily_limit(self, entry: ScheduleEntry) -> bool:
        """Check daily post limit."""
        pass
```

**Validation Rules Implementation:**

1. **Duplicate Prevention**: Check if `(product_id, platform, scheduled_time)` exists in `existing_entries`
2. **Past Schedule Check**: Compare `entry.scheduled_time` with `datetime.now(UTC)`
3. **Timezone Validation**: Ensure `entry.scheduled_time.tzinfo is not None`
4. **Spacing Enforcement**: For each platform in entry, find all entries within ±N hours on same platform
5. **Daily Limit**: Count entries on same day (date component), reject if >= `max_posts_per_day`

### 3. CleanupManager Class

**Location:** `src/publisher/cleanup.py`

**Purpose:** Handle post-publication cleanup with safety checks

**Class Definition:**
```python
@dataclass
class CleanupConfig:
    """Cleanup behavior configuration."""
    enabled: bool = False
    verify_before_delete: bool = True
    require_all_platforms: bool = True
    archive_before_delete: bool = False
    archive_dir: Path = Path("outputs/archive")
    keep_published_days: int = 0
    preserve_metadata: bool = False
    preserve_logs: bool = True

class CleanupManager:
    """Manages post-publication cleanup operations."""

    def __init__(self,
                 outputs_dir: Path = Path("outputs"),
                 config: CleanupConfig | None = None,
                 publisher: BasePublisher | None = None):
        """Initialize with outputs directory and configuration."""
        self.outputs_dir = outputs_dir
        self.config = config or CleanupConfig()
        self.publisher = publisher
        self.audit_log_path = outputs_dir / "cleanup_audit.json"

    async def cleanup(self,
                      product_id: str,
                      platforms: list[str],
                      dry_run: bool = False) -> dict[str, bool | str]:
        """
        Cleanup published product directory.

        Returns:
            Dict with: success (bool), message (str), disk_freed (int)
        """
        pass

    async def cleanup_all(self,
                         platforms: list[str],
                         dry_run: bool = False) -> dict[str, int]:
        """
        Cleanup all successfully published products.

        Returns:
            Dict with: cleaned (int), skipped (int), disk_freed (int)
        """
        pass

    async def verify_publication(self,
                                product_id: str,
                                platforms: list[str]) -> tuple[bool, dict[str, str]]:
        """
        Verify all platforms successfully published via API.

        Returns:
            Tuple of (all_published, platform_statuses)
        """
        pass

    def archive_directory(self, product_dir: Path) -> Path:
        """Create ZIP archive of directory before deletion."""
        pass

    def _should_cleanup(self, product_id: str, platforms: list[str]) -> bool:
        """Check if product should be cleaned up (age, status, etc.)."""
        pass

    def _log_cleanup(self, product_id: str, platforms: list[str], post_urls: list[str]):
        """Log cleanup operation to audit log."""
        pass
```

**Key Methods:**

1. **`cleanup()`**: Single product cleanup
   - Check `config.enabled`
   - Call `verify_publication()` if `verify_before_delete=true`
   - Archive if `archive_before_delete=true`
   - Remove directory: `shutil.rmtree(product_dir)`
   - Log to `cleanup_audit.json`
   - Return disk space freed

2. **`verify_publication()`**: Multi-platform status check
   - Load `publish_history.json` to get post_ids
   - For each platform: call `publisher.get_status(post_id)`
   - Check status == "published" for ALL platforms if `require_all_platforms=true`
   - Return boolean + dict of platform statuses

3. **`archive_directory()`**: ZIP creation
   - Use `shutil.make_archive()` to create ZIP
   - Save to `archive_dir/<product_id>_<timestamp>.zip`
   - Return archive path

4. **`_should_cleanup()`**: Age/status checks
   - If `keep_published_days > 0`: check publish timestamp age
   - If `preserve_metadata=true`: only delete video files, keep JSON
   - Return boolean decision

### 4. CLI Integration

**Location:** `src/publisher/late/cli.py`

**New Subcommands:**

```python
# calendar command
calendar_parser = subparsers.add_parser("calendar", help="View scheduled posts calendar")
calendar_parser.add_argument("action", choices=["list"], help="Calendar action")
calendar_parser.add_argument("--platform", help="Filter by platform")
calendar_parser.add_argument("--status", choices=["scheduled", "published", "failed", "partial"], help="Filter by status")
calendar_parser.add_argument("--date-from", help="Start date (YYYY-MM-DD)")
calendar_parser.add_argument("--date-to", help="End date (YYYY-MM-DD)")
calendar_parser.add_argument("--debug", action="store_true")

# schedule command
schedule_parser = subparsers.add_parser("schedule", help="Auto-schedule videos")
schedule_parser.add_argument("action", choices=["auto"], help="Schedule action")
schedule_parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
schedule_parser.add_argument("--platform", action="append", dest="platforms", required=True)
schedule_parser.add_argument("--start-slot", type=int, default=0, help="Start from slot N")
schedule_parser.add_argument("--dry-run", action="store_true")
schedule_parser.add_argument("--debug", action="store_true")

# cleanup command
cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup published products")
cleanup_parser.add_argument("--product-id", help="Cleanup specific product")
cleanup_parser.add_argument("--all", action="store_true", help="Cleanup all published")
cleanup_parser.add_argument("--dry-run", action="store_true")
cleanup_parser.add_argument("--debug", action="store_true")

# Modify single/batch commands
single_parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup")
batch_parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup")
```

**Command Handler Functions:**

```python
async def cmd_calendar(args: argparse.Namespace, config, session):
    """Execute calendar list command."""
    schedule_mgr = ScheduleManager()

    # Parse date filters
    date_from = datetime.fromisoformat(args.date_from) if args.date_from else None
    date_to = datetime.fromisoformat(args.date_to) if args.date_to else None

    # List with filters
    entries = schedule_mgr.list_scheduled(
        platform=args.platform,
        status=args.status,
        date_from=date_from,
        date_to=date_to
    )

    # Display results
    logger.info(f"Found {len(entries)} scheduled post(s)")
    for entry in entries:
        logger.info(f"Product: {entry.product_id}")
        logger.info(f"Scheduled: {entry.scheduled_time} (UTC)")
        logger.info(f"Platforms: {', '.join(entry.platforms)}")
        logger.info(f"Status: {entry.status}")
        logger.info("-" * 80)

async def cmd_schedule_auto(args: argparse.Namespace, config, session):
    """Execute schedule auto command."""
    schedule_mgr = ScheduleManager(config=config.schedule_config)
    validator = ScheduleValidator(config.schedule_config, schedule_mgr.entries)

    # Scan outputs dir for videos
    videos = list(args.outputs_dir.glob("*/video_*.mp4"))
    logger.info(f"Found {len(videos)} video(s) in {args.outputs_dir}")

    # Filter unpublished
    unpublished = [v for v in videos if not is_already_published(...)]
    logger.info(f"Unpublished: {len(unpublished)}")

    # Auto-schedule
    summary = await schedule_mgr.auto_schedule(
        videos=unpublished,
        platforms=args.platforms,
        start_slot=args.start_slot,
        dry_run=args.dry_run
    )

    logger.info(f"Scheduled: {summary['scheduled']}")
    logger.info(f"Skipped: {summary['skipped']}")
    logger.info(f"Failed: {summary['failed']}")

async def cmd_cleanup(args: argparse.Namespace, config, session):
    """Execute cleanup command."""
    publisher = create_publisher(...)
    cleanup_mgr = CleanupManager(config=config.cleanup_config, publisher=publisher)

    if args.product_id:
        # Cleanup single product
        result = await cleanup_mgr.cleanup(
            product_id=args.product_id,
            platforms=config.default_platforms,
            dry_run=args.dry_run
        )
        logger.info(f"Cleanup result: {result['message']}")
        logger.info(f"Disk freed: {result['disk_freed']} bytes")

    elif args.all:
        # Cleanup all published
        summary = await cleanup_mgr.cleanup_all(
            platforms=config.default_platforms,
            dry_run=args.dry_run
        )
        logger.info(f"Cleaned: {summary['cleaned']}")
        logger.info(f"Skipped: {summary['skipped']}")
        logger.info(f"Disk freed: {summary['disk_freed']} bytes")
```

**Integration with Existing Commands:**

```python
async def cmd_single(args: argparse.Namespace, config, session):
    """Existing single command with cleanup integration."""
    # ... existing upload and publish logic ...

    # After successful publish
    result = await publisher.publish(...)

    # Check if cleanup should run
    if config.cleanup.enabled and not args.no_cleanup:
        logger.info("Cleanup enabled, verifying publication...")
        cleanup_mgr = CleanupManager(config=config.cleanup, publisher=publisher)
        cleanup_result = await cleanup_mgr.cleanup(
            product_id=product_id,
            platforms=[p.value for p in args.platforms],
            dry_run=False
        )
        if cleanup_result['success']:
            logger.info(f"Cleanup successful: {cleanup_result['message']}")
    elif args.no_cleanup:
        logger.info("Cleanup disabled via --no-cleanup flag")
```

### 5. Configuration Updates

**Location:** `config/publisher.yaml`

**New Configuration Sections:**

```yaml
# Recurring Schedule (optional)
recurring_schedule:
  enabled: false
  timezone: "UTC"
  slots:
    - day: monday
      time: "09:00:00"
    - day: monday
      time: "14:00:00"
    - day: wednesday
      time: "10:00:00"
    - day: friday
      time: "15:00:00"

# Schedule Validation
schedule_validation:
  min_post_spacing_hours: 2
  prevent_duplicates: true
  allow_past_schedules: false
  max_posts_per_day: 10

# Post-Publication Cleanup
cleanup:
  enabled: false
  verify_before_delete: true
  require_all_platforms: true
  archive_before_delete: false
  archive_dir: outputs/archive
  keep_published_days: 0
  preserve_metadata: false
  preserve_logs: true
```

**Configuration Loading:**

```python
@dataclass
class PublisherConfig:
    """Existing config with new fields."""
    # ... existing fields ...

    # New fields
    schedule_config: ScheduleConfig = field(default_factory=ScheduleConfig)
    cleanup_config: CleanupConfig = field(default_factory=CleanupConfig)

def load_publisher_config(config_path: Path, cli_overrides: dict | None = None):
    """Load configuration with three-tier precedence."""
    # Load YAML
    config_data = yaml.safe_load(config_path.read_text())

    # Parse schedule config
    schedule_data = config_data.get("recurring_schedule", {})
    schedule_config = ScheduleConfig(
        timezone=schedule_data.get("timezone", "UTC"),
        enabled=schedule_data.get("enabled", False),
        slots=[RecurringSlot(**slot) for slot in schedule_data.get("slots", [])]
    )

    # Parse validation config
    validation_data = config_data.get("schedule_validation", {})
    schedule_config.update_validation(validation_data)

    # Parse cleanup config
    cleanup_data = config_data.get("cleanup", {})
    cleanup_config = CleanupConfig(**cleanup_data)

    # ... rest of config loading ...
```

### 6. Data Persistence

**Files:**

1. **`outputs/schedule.json`** - Schedule entries
```json
{
  "entries": [
    {
      "product_id": "B0ABC123",
      "scheduled_time": "2025-01-20T14:00:00+00:00",
      "platforms": ["youtube", "tiktok"],
      "post_id": "post_xyz789",
      "status": "scheduled",
      "created_at": "2025-01-15T10:30:00+00:00",
      "slot_index": 0
    }
  ]
}
```

2. **`outputs/cleanup_audit.json`** - Cleanup audit log
```json
{
  "cleanups": [
    {
      "product_id": "B0ABC123",
      "platforms": ["youtube", "tiktok"],
      "post_urls": [
        "https://youtube.com/watch?v=abc",
        "https://tiktok.com/@user/video/123"
      ],
      "cleaned_at": "2025-01-20T16:00:00+00:00",
      "disk_freed_bytes": 150000000,
      "archive_path": "outputs/archive/B0ABC123_20250120160000.zip"
    }
  ]
}
```

3. **`outputs/publish_history.json`** - Existing tracking file (enhanced)
```json
{
  "posts": {
    "B0ABC123:youtube": {
      "product_id": "B0ABC123",
      "platform": "youtube",
      "post_id": "post_xyz789",
      "published_at": "2025-01-20T14:00:00+00:00",
      "status": "published",
      "post_url": "https://youtube.com/watch?v=abc"
    }
  }
}
```

## Technology Choices

### Late SDK Integration

**Calendar View:**
```python
# Use Late SDK posts.list() method
async def fetch_scheduled_posts(platform: str | None = None,
                                date_from: datetime | None = None,
                                date_to: datetime | None = None):
    """Fetch scheduled posts from Late API."""
    posts = await client.posts.list(
        status="SCHEDULED",  # Filter by PostStatus enum
        platform=platform,
        date_from=date_from.isoformat() if date_from else None,
        date_to=date_to.isoformat() if date_to else None,
        page=1,
        limit=100
    )
    return posts.posts  # PostsListResponse.posts
```

**Status Verification:**
```python
# Use existing LatePublisher.get_status() method
async def verify_publication_status(post_id: str) -> str:
    """Check if post is published via API."""
    status_info = await publisher.get_status(post_id)
    return status_info['status']  # "scheduled", "published", "failed", etc.
```

### File Operations

- **Atomic writes**: Use `tempfile.NamedTemporaryFile()` + `os.rename()` for schedule.json writes
- **Directory removal**: Use `shutil.rmtree()` with error handling
- **Archive creation**: Use `shutil.make_archive(format='zip')`
- **Disk usage**: Use `pathlib.Path.stat().st_size` for size calculation

### Date/Time Handling

- **Timezone support**: Use `zoneinfo.ZoneInfo` (Python 3.9+) for timezone conversions
- **Parsing**: Use `datetime.fromisoformat()` for ISO 8601 strings
- **Storage**: Store all datetimes in UTC, convert to local for display
- **Recurring slots**: Use `dateutil.rrule` for calculating next occurrences

## Error Handling

### Schedule Validation Errors

```python
class ScheduleValidationError(ValidationError):
    """Raised when schedule validation fails."""
    def __init__(self, message: str, violations: list[str]):
        super().__init__(message)
        self.violations = violations
```

**Error Messages:**
- "Duplicate schedule: B0ABC123 already scheduled to youtube at 2025-01-20 14:00"
- "Minimum spacing violation: 2h required between posts on youtube"
- "Daily limit reached: max 10 posts per day (currently 10)"
- "Cannot schedule in past: 2025-01-15 14:00 < 2025-01-16 10:00 (now)"

### Cleanup Errors

```python
class CleanupError(PublisherError):
    """Raised when cleanup operation fails."""
    pass
```

**Error Messages:**
- "Cannot cleanup B0ABC123: not published to all platforms (youtube=published, tiktok=failed)"
- "API status check failed: unable to verify publication status"
- "Archive creation failed: insufficient disk space"
- "Directory removal failed: permission denied"

### Graceful Degradation

1. **API Unavailable**: Skip cleanup rather than fail publish operation
2. **Archive Failure**: Log warning but proceed with cleanup (if `archive_before_delete=false` fallback)
3. **Status Check Timeout**: Treat as unpublished and skip cleanup

## Testing Strategy

### Unit Tests

**Schedule Tests:**
```python
# test_schedule_manager.py
def test_recurring_slot_next_occurrence()
def test_schedule_manager_add_entry()
def test_schedule_manager_list_filtered()
def test_auto_schedule_batch()

# test_schedule_validator.py
def test_duplicate_detection()
def test_minimum_spacing_enforcement()
def test_daily_limit_enforcement()
def test_past_schedule_rejection()
def test_timezone_validation()
```

**Cleanup Tests:**
```python
# test_cleanup_manager.py
def test_cleanup_single_product()
def test_cleanup_all_published()
def test_verify_publication_all_platforms()
def test_archive_before_delete()
def test_preserve_metadata_option()
def test_skip_cleanup_if_not_all_published()
```

### Integration Tests

```python
# test_publisher_e2e.py
async def test_auto_schedule_and_cleanup_workflow():
    """Test full workflow: auto-schedule → publish → cleanup."""
    # 1. Configure recurring schedule
    # 2. Run schedule auto command
    # 3. Verify schedule.json entries
    # 4. Mock publish success
    # 5. Run cleanup command
    # 6. Verify directory removed
    # 7. Verify cleanup_audit.json entry
```

### Edge Cases

1. **Empty recurring slots**: Error message with helpful guidance
2. **All slots occupied**: Report next available slot time
3. **Cleanup during publish**: Use file locking to prevent race conditions
4. **Timezone conversion errors**: Fallback to UTC with warning
5. **Partial platform success**: Only cleanup if ALL platforms succeeded (when `require_all_platforms=true`)

## Performance Considerations

### Schedule Operations

- **Calendar list**: O(n) filtering on schedule entries (expected <1000 entries)
- **Auto-schedule**: O(n*m) where n=videos, m=slots (typical: 50 videos × 10 slots = 500 ops)
- **Validation**: O(n) where n=existing entries per validation check

**Optimization:**
- Index schedule entries by (product_id, platform) for O(1) duplicate checks
- Cache recurring slot calculations for batch operations
- Lazy-load schedule.json only when needed

### Cleanup Operations

- **API status checks**: Rate-limited by Late API (typical: 100 req/min)
- **Directory removal**: O(n) where n=number of files (typical: 10-50 files per product)
- **Archive creation**: O(n) I/O operations

**Optimization:**
- Batch status checks with `asyncio.gather()` (max 5 concurrent)
- Use `--dry-run` for preview without I/O
- Parallelize cleanup operations for `--all` mode (max 3 concurrent)

## Security Considerations

### Validation

1. **Path Traversal**: Validate product_id doesn't contain `..` or `/`
2. **File Permissions**: Check write permissions before archive/delete
3. **API Credentials**: Never log full API keys (show first 4 chars only)
4. **Timezone Injection**: Validate timezone strings against `zoneinfo.available_timezones()`

### Audit Logging

1. **Cleanup Audit**: Log all deletions with timestamps, product IDs, platforms, post URLs
2. **Archive References**: Store archive paths for recovery
3. **Tamper-Proof**: Use append-only mode for `cleanup_audit.json`

## Migration and Deployment

### Backward Compatibility

- **Existing configs**: Continue working without new sections (defaults to disabled)
- **Existing CLI**: No breaking changes to `single` and `batch` commands
- **Data files**: New files created on first use (schedule.json, cleanup_audit.json)

### Migration Steps

1. Add new config sections to `config/publisher.yaml` (disabled by default)
2. Deploy new CLI commands (`calendar`, `schedule`, `cleanup`)
3. Users opt-in by enabling features in config
4. No database migrations required (file-based storage)

### Rollback Plan

1. Set `cleanup.enabled: false` in config
2. Restore archived products from `outputs/archive/` if needed
3. Remove schedule.json and cleanup_audit.json files
4. Revert to previous publisher module version

## Open Questions and Future Enhancements

### Open Questions

1. **Late API Rate Limits**: What are the exact rate limits for `posts.list()` and `posts.get()`?
   - **Resolution**: Test with burst of 100 requests, implement rate limiting if needed

2. **Timezone Handling**: Should we support per-user timezone preferences?
   - **Resolution**: Use system timezone by default, allow override in config

3. **Cleanup Safety**: Should we require explicit confirmation for `cleanup --all`?
   - **Resolution**: Add `--confirm` flag requirement for `--all` mode

### Future Enhancements

1. **Calendar UI**: Web-based calendar view with drag-and-drop rescheduling
2. **Smart Scheduling**: ML-based optimal posting time recommendations
3. **Multi-Provider**: Extend schedule system to Buffer, Hootsuite, etc.
4. **Webhook Integration**: Real-time status updates via Late API webhooks
5. **Cloud Backup**: Automatic backup of cleanup archives to S3/GCS before deletion
6. **Cleanup Statistics**: Dashboard showing disk space trends over time

## Appendix

### File Structure

```
src/publisher/
├── late/
│   ├── cli.py              # Extended with calendar/schedule/cleanup commands
│   └── client.py           # Existing LatePublisher (no changes)
├── base.py                 # Existing BasePublisher (no changes)
├── config.py               # Extended with ScheduleConfig, CleanupConfig
├── models.py               # Extended with RecurringSlot, ScheduleEntry
├── schedule.py             # NEW: ScheduleManager class
├── schedule_validator.py   # NEW: ScheduleValidator class
├── cleanup.py              # NEW: CleanupManager class
├── batch.py                # Updated with cleanup integration
├── tracking.py             # Enhanced with status tracking
└── metadata.py             # Existing (no changes)

config/
└── publisher.yaml          # Extended with schedule/cleanup sections

outputs/
├── schedule.json           # NEW: Schedule entries
├── cleanup_audit.json      # NEW: Cleanup audit log
├── publish_history.json    # Enhanced with status tracking
└── archive/                # NEW: Archived product directories
```

### Dependencies

**Existing:**
- `late` - Late.dev Python SDK
- `aiohttp` - Async HTTP client
- `pyyaml` - YAML configuration parsing
- `python-dotenv` - Environment variable loading

**New:**
- None (all features use stdlib: `datetime`, `zoneinfo`, `shutil`, `pathlib`, `json`)

### Configuration Schema

```yaml
# Complete publisher.yaml schema with new sections
provider: late
api_key: ${LATE_API_KEY}
vercel_token: ${LATE_VERCEL_TOKEN}

# Existing sections
immediate_publish: true
default_platforms: [youtube, tiktok, instagram]
max_retries: 3
timeout: 120.0
backoff_multiplier: 2.0
stagger_delay_min: 30
stagger_delay_max: 60
privacy_settings:
  youtube: public
  tiktok: public
  instagram: everyone

# NEW: Recurring schedule
recurring_schedule:
  enabled: false
  timezone: "UTC"
  slots:
    - day: monday
      time: "09:00:00"
    - day: monday
      time: "14:00:00"
    - day: wednesday
      time: "10:00:00"
    - day: friday
      time: "15:00:00"

# NEW: Schedule validation
schedule_validation:
  min_post_spacing_hours: 2
  prevent_duplicates: true
  allow_past_schedules: false
  max_posts_per_day: 10

# NEW: Post-publication cleanup
cleanup:
  enabled: false
  verify_before_delete: true
  require_all_platforms: true
  archive_before_delete: false
  archive_dir: outputs/archive
  keep_published_days: 0
  preserve_metadata: false
  preserve_logs: true
```
