# Assembler.py Refactoring Research Report

**Date**: 2025-11-24
**Status**: Research Complete - Implementation Deferred

## Executive Summary

VideoAssembler is a 3,311-line GOD CLASS requiring **Strategy Pattern + Composition** refactoring. Research identified 9 distinct responsibility groups with 2 standalone utilities (zero dependencies) as low-risk extraction targets.

## Current State Analysis

**VideoAssembler** (`src/video/assembler.py`)
- **Size**: 3,311 lines
- **Classes**: 3 (VisualGeometry, SubtitleEntry, VideoAssembler)
- **Methods**: 32 methods in VideoAssembler
- **Usage**: Single point in `src/video/producer/steps.py:step_assemble_video()`
- **Pattern**: GOD CLASS antipattern with tight coupling

### Class Structure
```python
@dataclass
class VisualGeometry:
    """Position/dimensions of visual elements."""
    rendered_x, rendered_y, rendered_w, rendered_h

@dataclass
class SubtitleEntry:
    """Subtitle timing and text."""
    start, end, text

class VideoAssembler:
    """Orchestrates FFmpeg video assembly."""
    # 32 methods handling all video assembly logic
```

## Research Findings

### GOD Class Refactoring Best Practices (2024)

**Core Principles**:
1. **Extract Class Pattern**: Identify distinct responsibilities → separate classes
2. **SOLID Principles**: Single Responsibility Principle guides decomposition
3. **Composition Over Inheritance**: Build complex functionality from simpler objects

**5-Step Process**:
1. Build comprehensive unit test suite
2. Identify most relied-upon components
3. Move static/utility methods first (lowest risk)
4. Group common methods with loose coupling
5. Gradual migration with backward compatibility

### Python Video Processing Architecture Patterns

**Modular FFmpeg Wrapper Approaches**:
- **Separation of Concerns**: Stores (config) vs Options (arguments) vs Execution
- **Builder Pattern**: Programmatically construct complex FFmpeg commands
- **Hardware Acceleration**: VPF framework separates decode/encode/transcode/color-conversion
- **Generator Pattern**: imageio-ffmpeg uses generators for frame processing

**Key Libraries**:
- **imageio-ffmpeg**: Simple subprocess wrapper with pipe communication
- **VidGear**: Extensible APIs for WriteGear, StreamGear (DASH/HLS)
- **ffmpeg-python**: Programmatic FFmpeg command construction

## Detailed Dependency Analysis

### Method Groupings by Responsibility

**9 Logical Groups Identified**:

#### 1. Configuration Management (5 methods)
- `__init__`, `set_profile_settings`, `set_product_id`
- `_get_effective_video_settings`, `_get_effective_subtitle_settings`
- **Risk**: Shared state across entire class
- **Strategy**: Keep in core VideoAssembler

#### 2. Media Inspection (4 methods) - ⭐ **STANDALONE**
- `_is_video`, `_get_media_duration`, `_get_media_dimensions`, `_detect_video_rotation`
- **Dependencies**: ZERO - only calls external ffprobe/ffmpeg
- **Extract Target**: `MediaInspector` utility class
- **Risk**: 🟢 Very Low

#### 3. Subtitle Processing (5 methods) - ⭐ **STANDALONE**
- `_parse_srt`, `_parse_ass_time`, `_resolve_font_path`
- `_convert_ass_color_to_ffmpeg`, `_normalize_text_for_verification`
- **Dependencies**: ZERO - pure utility functions
- **Extract Target**: `SubtitleUtils` utility class
- **Risk**: 🟢 Very Low

#### 4. Audio Processing (3 methods)
- `_prepare_audio_inputs`, `_build_audio_filters`, `_build_audio_filters_with_video_audio`
- **Dependencies**: Moderate - calls media inspection
- **Extract Target**: `AudioFilterBuilder` class
- **Risk**: 🟡 Medium

#### 5. Video Mode Strategies (5 methods)
- `_build_video_mode_sequential`, `_build_video_mode_single_best`
- `_build_video_mode_mixed`, `_handle_fallback_image`
- **Dependencies**: High - calls visual processing, media inspection
- **Extract Target**: Strategy pattern classes
- **Risk**: 🟠 Medium-High

#### 6. Visual Processing (2 methods)
- `_apply_aspect_ratio_mode`, `_build_visual_chain`
- **Dependencies**: Very High - core to video assembly
- **Extract Target**: `VisualFilterBuilder` class
- **Risk**: 🔴 High

#### 7. Subtitle Graph Building (3 methods)
- `_build_subtitle_graph_single`, `_build_subtitle_graph_dual`
- `_build_subtitle_graph_content_aware`
- **Dependencies**: Very High - depends on visual processing
- **Extract Target**: `SubtitleGraphBuilder` class
- **Risk**: 🔴 High

#### 8. FFmpeg Orchestration (3 methods)
- `_build_ffmpeg_command`, `_should_create_ffmpeg_logs`, `assemble_video`
- **Dependencies**: Calls everything - orchestration layer
- **Strategy**: Keep as core responsibility of VideoAssembler

#### 9. Video Verification (1 method)
- `verify_video`
- **Dependencies**: None - standalone validator
- **Extract Target**: Could move to `MediaInspector`

### Call Frequency Analysis

Most-called helper methods:
- `_get_effective_subtitle_settings()` - 8 calls
- `_convert_ass_color_to_ffmpeg()` - 8 calls
- `_resolve_font_path()` - 5 calls
- `_get_media_duration()` - 5 calls

## Recommended Refactoring Strategy

### Phase 1: Extract Standalone Utilities (Low Risk) 🟢

**Create**: `src/video/assembler/media_inspector.py`
```python
class MediaInspector:
    """Standalone media file inspection utilities."""

    @staticmethod
    def is_video(path: Path) -> bool:
        """Check if file is a video."""

    @staticmethod
    async def get_media_duration(path: Path) -> float:
        """Get media duration in seconds."""

    @staticmethod
    async def get_media_dimensions(path: Path) -> tuple[int, int]:
        """Get media width and height."""

    @staticmethod
    async def detect_video_rotation(path: Path) -> int:
        """Detect video rotation metadata."""
```

**Create**: `src/video/assembler/subtitle_utils.py`
```python
class SubtitleParser:
    """SRT/ASS subtitle parsing."""

    @staticmethod
    def parse_srt(subtitle_path: Path) -> list[SubtitleEntry]:
        """Parse SRT file into SubtitleEntry list."""

    @staticmethod
    def parse_ass_time(time_str: str) -> float:
        """Convert ASS timestamp to seconds."""

class SubtitleStyler:
    """Font and color utilities."""

    @staticmethod
    def resolve_font_path(font_name: str) -> Path | None:
        """Resolve font name to system font path."""

    @staticmethod
    def convert_ass_color_to_ffmpeg(ass_color: str) -> str:
        """Convert ASS color format to FFmpeg format."""

    @staticmethod
    def normalize_text_for_verification(text: str) -> str:
        """Normalize subtitle text for comparison."""
```

**Risk**: 🟢 Very Low - zero dependencies, pure functions
**Validation**: Run full test suite, verify subtitle rendering unchanged
**Lines**: Extract ~400 lines from assembler.py

### Phase 2: Extract Audio Processing (Medium Risk) 🟡

**Create**: `src/video/assembler/audio_builder.py`
```python
class AudioFilterBuilder:
    """Audio filter chain construction."""

    def __init__(self, media_inspector: MediaInspector):
        self.inspector = media_inspector

    async def prepare_audio_inputs(
        self, voiceover_path: Path, music_path: Path | None
    ) -> list[str]:
        """Prepare FFmpeg audio input arguments."""

    async def build_audio_filters(
        self, voiceover_path: Path, music_path: Path | None,
        music_volume: float, ducking_amount: float
    ) -> str:
        """Build audio filter graph with ducking."""

    async def build_audio_filters_with_video_audio(
        self, voiceover_path: Path, music_path: Path | None,
        video_audio_paths: list[Path], ...
    ) -> str:
        """Build audio filter graph including video audio tracks."""
```

**Risk**: 🟡 Medium - depends on MediaInspector (Phase 1)
**Validation**: Test audio ducking, multiple audio inputs
**Lines**: Extract ~300 lines from assembler.py

### Phase 3: Extract Video Strategies (Medium-High Risk) 🟠

**Create**: `src/video/assembler/video_strategies.py`
```python
from typing import Protocol

class VideoModeStrategy(Protocol):
    """Protocol for video mode strategies."""
    async def build_filter_graph(
        self, visuals: list[Path], durations: list[float], ...
    ) -> str:
        """Build FFmpeg filter graph for video mode."""

class SequentialVideoMode(VideoModeStrategy):
    """Sequential video mode: play all visuals in order."""
    async def build_filter_graph(...) -> str:
        ...

class SingleBestVideoMode(VideoModeStrategy):
    """Single best video mode: use highest quality visual."""
    async def build_filter_graph(...) -> str:
        ...

class MixedVideoMode(VideoModeStrategy):
    """Mixed video mode: combine multiple visuals."""
    async def build_filter_graph(...) -> str:
        ...

class FallbackImageHandler:
    """Handle fallback image for insufficient media."""
    async def handle_fallback_image(
        self, fallback_path: Path, duration: float, ...
    ) -> str:
        ...
```

**Risk**: 🟠 Medium-High - complex logic, many dependencies
**Validation**: Test all video modes (sequential, single_best, mixed, fallback)
**Lines**: Extract ~600 lines from assembler.py

### Phase 4: Extract Visual & Subtitle Builders (High Risk) 🔴

**Create**: `src/video/assembler/visual_builder.py`
```python
class VisualFilterBuilder:
    """Visual filter chain construction."""

    async def apply_aspect_ratio_mode(
        self, input_stream: str, mode: str, target_width: int, target_height: int
    ) -> str:
        """Apply aspect ratio mode (fit/fill/stretch) to visual."""

    async def build_visual_chain(
        self, visuals: list[Path], mode: str, ...
    ) -> tuple[str, list[VisualGeometry]]:
        """Build visual filter chain and return geometry."""
```

**Create**: `src/video/assembler/subtitle_builder.py`
```python
class SubtitleGraphBuilder:
    """Subtitle filter graph construction."""

    def __init__(self, subtitle_utils: SubtitleStyler):
        self.styler = subtitle_utils

    async def build_subtitle_graph_single(
        self, subtitle_path: Path, visual_geometries: list[VisualGeometry], ...
    ) -> str:
        """Build subtitle graph for single-line subtitles."""

    async def build_subtitle_graph_dual(
        self, subtitle_path: Path, visual_geometries: list[VisualGeometry], ...
    ) -> str:
        """Build subtitle graph for dual-line subtitles."""

    async def build_subtitle_graph_content_aware(
        self, subtitle_path: Path, visual_geometries: list[VisualGeometry], ...
    ) -> str:
        """Build subtitle graph with content-aware positioning."""
```

**Risk**: 🔴 High - tightly coupled, shared state, complex logic
**Validation**: Test all subtitle positioning modes, content-aware detection
**Lines**: Extract ~800 lines from assembler.py

### Phase 5: Simplify Core Orchestrator (Final)

**Refactored**: `src/video/assembler/core.py`
```python
class VideoAssembler:
    """Slim orchestrator coordinating specialized builders.

    Reduced from 3,311 lines to ~500 lines by delegating to:
    - MediaInspector: media file inspection
    - SubtitleUtils: subtitle parsing and styling
    - AudioFilterBuilder: audio filter chains
    - VideoModeStrategy: video mode implementations
    - VisualFilterBuilder: visual filter chains
    - SubtitleGraphBuilder: subtitle positioning
    """

    def __init__(self, config: VideoConfig, debug_mode: bool = False):
        # Configuration state
        self.config = config
        self.debug_mode = debug_mode
        self.profile_name = None
        self.product_id = None
        self.cli_overrides = {}

        # Initialize specialized builders
        self.media_inspector = MediaInspector()
        self.subtitle_parser = SubtitleParser()
        self.subtitle_styler = SubtitleStyler()
        self.audio_builder = AudioFilterBuilder(self.media_inspector)
        self.visual_builder = VisualFilterBuilder()
        self.subtitle_builder = SubtitleGraphBuilder(self.subtitle_styler)

        # Video mode strategies
        self.video_strategies = {
            "sequential": SequentialVideoMode(),
            "single_best": SingleBestVideoMode(),
            "mixed": MixedVideoMode(),
        }
        self.fallback_handler = FallbackImageHandler()

    def set_profile_settings(self, profile_name: str, cli_overrides: dict | None):
        """Configure profile and overrides."""

    def set_product_id(self, product_id: str):
        """Set product ID for logging context."""

    async def assemble_video(
        self, visuals: list[Path], voiceover_path: Path,
        subtitle_path: Path, music_path: Path | None, output_path: Path
    ) -> Path:
        """Orchestrate video assembly using specialized builders."""
        # 1. Prepare audio filters
        audio_inputs = await self.audio_builder.prepare_audio_inputs(...)
        audio_filters = await self.audio_builder.build_audio_filters(...)

        # 2. Build visual chain
        visual_filters, geometries = await self.visual_builder.build_visual_chain(...)

        # 3. Build subtitle graph
        subtitle_filters = await self.subtitle_builder.build_subtitle_graph_content_aware(...)

        # 4. Assemble FFmpeg command
        ffmpeg_cmd = self._build_ffmpeg_command(
            audio_inputs, audio_filters, visual_filters, subtitle_filters, output_path
        )

        # 5. Execute FFmpeg
        await async_run_ffmpeg(ffmpeg_cmd, ...)

        return output_path

    def verify_video(self, video_path: Path, expected_subtitles: list[str]) -> bool:
        """Verify video output quality."""
```

**Create**: `src/video/assembler/__init__.py`
```python
"""Video assembler module - Modular structure with backward compatibility.

This module has been refactored into smaller, focused modules:
- media_inspector.py: Media file inspection utilities
- subtitle_utils.py: Subtitle parsing and styling
- audio_builder.py: Audio filter chain construction
- video_strategies.py: Video mode implementations
- visual_builder.py: Visual filter chain construction
- subtitle_builder.py: Subtitle graph construction
- core.py: Slim VideoAssembler orchestrator

All classes are re-exported from this module to maintain backward compatibility
with existing imports like:
    from src.video.assembler import VideoAssembler, VisualGeometry, SubtitleEntry
"""

# Re-export core classes for backward compatibility
from src.video.assembler.core import (
    VideoAssembler,
    VisualGeometry,
    SubtitleEntry,
)

__all__ = ["VideoAssembler", "VisualGeometry", "SubtitleEntry"]
```

**Risk**: 🟢 Low - composition pattern, backward compatible via re-exports
**Lines**: VideoAssembler reduced from 3,311 → ~500 lines
**No changes required**: `src/video/producer/steps.py` continues using same imports

## Implementation Approach

### Testing Strategy

1. **Before Each Phase**: Capture baseline test results
   ```bash
   poetry run pytest tests/ --cov=src.video.assembler -v
   ```

2. **Extract Pure Functions First**: Phases 1-2 are safest (zero/low dependencies)

3. **Incremental Testing**: Run test suite after each extraction
   - Unit tests for extracted classes
   - Integration tests for video assembly pipeline

4. **Regression Testing**: Verify end-to-end video assembly unchanged
   - Same FFmpeg commands generated
   - Identical video output (frame-by-frame comparison)
   - Subtitle positioning unchanged

### Migration Path

**Backward Compatibility**:
- Keep original imports working via `__init__.py` re-exports
- No changes required to `src/video/producer/steps.py`
- Gradual refactoring over multiple PRs

**PR Strategy**:
- PR 1: Phase 1 - Extract MediaInspector + SubtitleUtils (low risk)
- PR 2: Phase 2 - Extract AudioFilterBuilder (medium risk)
- PR 3: Phase 3 - Extract VideoModeStrategy classes (medium-high risk)
- PR 4: Phase 4 - Extract VisualFilterBuilder + SubtitleGraphBuilder (high risk)
- PR 5: Phase 5 - Simplify core VideoAssembler (cleanup)

### Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| Largest file | 3,311 lines | ~500 lines |
| VideoAssembler methods | 32 | ~8-10 |
| Number of classes | 3 | ~12 |
| Test coverage | Maintain | ≥40% |
| Import compatibility | N/A | 100% backward compatible |
| Video output | Baseline | Pixel-perfect match |

## Comparison to Previous Refactorings

| Aspect | video_config.py | producer.py | assembler.py |
|--------|----------------|-------------|--------------|
| **Approach** | Domain split | Function split | **Responsibility split** |
| **Structure** | Models | Procedural | **OOP GOD CLASS** |
| **Risk** | Low | Medium | **High** |
| **Dependencies** | Clean | Moderate | **Tightly coupled** |
| **Strategy** | Extract by domain | Extract by step | **Extract by responsibility** |
| **Pattern** | Pydantic models | Module organization | **Strategy + Composition** |
| **Backward Compat** | Re-exports | Re-exports | **Re-exports** |
| **Lines Reduced** | 2,301 → 6 files | 2,514 → 8 files | **3,311 → ~500 + 7 files** |

### Key Differences

**assembler.py requires**:
1. **Strategy Pattern**: Video mode implementations (sequential/single_best/mixed)
2. **Composition**: VideoAssembler coordinates multiple builders
3. **Protocol-Oriented Design**: Define interfaces for strategies
4. **Careful State Management**: Shared config state must be threaded correctly

**Not just file splitting**: This is true OOP refactoring applying SOLID principles.

## Lessons from video_config.py & producer.py

### What Worked Well

1. **Incremental Approach**: Small, focused PRs with full test validation
2. **Backward Compatibility**: `__init__.py` re-exports preserve existing imports
3. **Zero Functional Changes**: Refactor structure, not behavior
4. **Comprehensive Testing**: Run full test suite after each change

### What to Apply Here

1. **Start with Standalone Utilities**: Phase 1 has zero dependencies (safest)
2. **Validate Each Phase**: Don't proceed until tests pass
3. **Maintain Re-exports**: Ensure `from src.video.assembler import VideoAssembler` continues working
4. **Document Changes**: Update REFACTORING_PLAN.md after each phase

## Risk Mitigation

### High-Risk Areas

1. **Shared State**: VideoAssembler config state accessed by multiple methods
   - **Mitigation**: Pass config explicitly to builders via constructor

2. **Tight Coupling**: Subtitle positioning depends on visual geometry
   - **Mitigation**: Pass visual_geometries as parameters, not shared state

3. **Complex FFmpeg Logic**: Filter graphs have intricate dependencies
   - **Mitigation**: Extensive integration testing, frame comparison validation

### Rollback Strategy

- Keep original `assembler.py` in git history
- Tag stable commits before each phase
- If regression detected, revert PR and investigate

## Next Steps (Deferred)

1. ✅ Research complete - saved to `ASSEMBLER_REFACTORING_RESEARCH.md`
2. ✅ Update `REFACTORING_PLAN.md` with deferred status
3. ⏳ **Deferred**: Implement Phase 1 when ready
4. ⏳ **Deferred**: Subsequent phases based on Phase 1 learnings

## References

### Research Sources

**GOD Class Refactoring**:
- [Software Patterns Lexicon: God Object Anti-Pattern in Python](https://softwarepatternslexicon.com/patterns-python/11/2/4/)
- [Stack Overflow: How do you refactor a God class?](https://stackoverflow.com/questions/14870377/how-do-you-refactor-a-god-class)
- [Medium: Refactoring the God Class in Python](https://medium.com/better-programming/refactoring-the-god-class-in-python-5c13942d0e75)

**Extract Class Pattern**:
- [Refactoring Guru: Extract Class](https://refactoring.guru/extract-class)
- [Martin Fowler: This class is too large](https://martinfowler.com/articles/class-too-large.html)
- [Real Python: Refactoring Python Applications for Simplicity](https://realpython.com/python-refactoring/)

**Python Video Processing**:
- [VidGear: Hardware-Accelerated Video Processing Framework](https://pypi.org/project/vidgear/)
- [NVIDIA: VPF Hardware-Accelerated Video Processing](https://developer.nvidia.com/blog/vpf-hardware-accelerated-video-processing-framework-in-python/)
- [Gumlet: How to Use FFmpeg with Python in 2025](https://www.gumlet.com/learn/ffmpeg-python/)

---

**Report Status**: Complete
**Implementation Status**: Deferred
**Next Review**: To be determined
