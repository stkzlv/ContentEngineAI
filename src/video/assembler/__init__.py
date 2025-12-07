"""Video assembler module - Modular structure with backward compatibility.

This module has been refactored into smaller, focused modules:
- media_inspector.py: Media file inspection utilities
- subtitle_utils.py: Subtitle parsing and styling
- audio_builder.py: Audio filter chain construction
- video_strategies.py: Video mode strategy implementations
- visual_builder.py: Visual filter chain construction
- subtitle_builder.py: Subtitle graph construction
- core.py: Slim VideoAssembler orchestrator

All classes are re-exported from this module to maintain backward compatibility
with existing imports:
    from src.video.assembler import VideoAssembler, VisualGeometry, SubtitleEntry
"""

from src.video.assembler.core import VideoAssembler
from src.video.assembler.subtitle_utils import SubtitleEntry
from src.video.assembler.visual_builder import VisualGeometry

__all__ = ["VideoAssembler", "VisualGeometry", "SubtitleEntry"]
