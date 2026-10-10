"""Caption entrances are short and exits are a cut or a short fade (REQ-VID-155).

The pycaps half drives `_build_pipeline` with a fake builder, for the reason
`test_pycaps_sound_effects.py` gives: where the caller puts the override is
the part that can be wrong. The real `explosive` template is checked too,
since the fake supplies whatever attribute name the code asks for.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.video.config.subtitle_models import (
    PycapsSettings,
    SubtitleEffectsSettings,
    SubtitleSettings,
)
from src.video.subtitle_positioning import Position, StylePreset

REPO = Path(__file__).resolve().parents[2]


class FadeOut:
    """Named like pycaps' fade-out preset, which the limit keeps."""

    def __init__(self, duration: float) -> None:
        self._duration = duration


class SlideOut(FadeOut):
    """Any other exit, which the limit drops."""


class SlideIn(FadeOut):
    """An entrance."""


def _animator(animation: FadeOut, when: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        _animation=animation, _when=types.SimpleNamespace(value=when)
    )


class _Builder:
    def __init__(self, animators: list) -> None:
        self._caps_pipeline = types.SimpleNamespace(
            _layout_options=MagicMock(),
            _text_effects=[],
            _semantic_tagger=types.SimpleNamespace(_ai_rules={}),
            _sound_effects=[],
            _animators=animators,
        )

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            return self

        return _record


def _fake_pycaps(monkeypatch: pytest.MonkeyPatch, builder: _Builder) -> None:
    template = types.ModuleType("pycaps.template")
    template.TemplateFactory = lambda: types.SimpleNamespace(create=lambda n: n)

    class _Loader:
        def __init__(self, template):
            pass

        def with_input_video(self, _):
            return self

        def load(self, _):
            return builder

    template.TemplateLoader = _Loader
    transcriber = types.ModuleType("pycaps.transcriber")
    transcriber.TranscriptFormat = types.SimpleNamespace(WHISPER_JSON="whisper")
    renderer = types.ModuleType("pycaps.renderer")
    renderer.PictexSubtitleRenderer = lambda: "pictex-renderer"
    root = types.ModuleType("pycaps")
    root.template, root.transcriber, root.renderer = template, transcriber, renderer
    for name, mod in {
        "pycaps": root,
        "pycaps.template": template,
        "pycaps.transcriber": transcriber,
        "pycaps.renderer": renderer,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


def _build(monkeypatch, tmp_path: Path, animators: list, **settings) -> list:
    from src.video.pycaps_engine.renderer import PycapsRenderer

    builder = _Builder(animators)
    _fake_pycaps(monkeypatch, builder)
    monkeypatch.setattr(
        "src.video.pycaps_engine.renderer.merge_layout_with_template",
        lambda *a, **k: MagicMock(),
    )
    PycapsRenderer._build_pipeline(
        PycapsRenderer.__new__(PycapsRenderer),
        input_video=tmp_path / "in.mp4",
        transcript_path=tmp_path / "t.json",
        output_video=tmp_path / "out.mp4",
        template_name="explosive",
        visual_bounds=None,
        settings=PycapsSettings(outline_px=0, **settings),
    )
    result: list = builder._caps_pipeline._animators
    return result


def _explosive_like() -> list:
    return [
        _animator(SlideIn(0.4), "narration-starts"),
        _animator(SlideIn(0.2), "narration-starts"),
        _animator(SlideOut(0.3), "narration-ends"),
        _animator(FadeOut(0.3), "narration-ends"),
    ]


@pytest.mark.unit
@pytest.mark.req("REQ-VID-155")
class TestPycapsMotion:
    def test_entrances_are_shortened_and_short_ones_kept(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        animators = _build(
            monkeypatch, tmp_path, _explosive_like(), max_entrance_sec=0.25
        )

        entrances = [a for a in animators if a._when.value == "narration-starts"]
        assert [a._animation._duration for a in entrances] == [0.25, 0.2]

    def test_a_slide_exit_goes_and_a_fade_is_shortened(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        animators = _build(monkeypatch, tmp_path, _explosive_like(), max_exit_sec=0.08)

        exits = [a._animation for a in animators if a._when.value == "narration-ends"]
        assert [(type(e).__name__, e._duration) for e in exits] == [("FadeOut", 0.08)]

    def test_zero_drops_every_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        animators = _build(monkeypatch, tmp_path, _explosive_like(), max_exit_sec=0)

        assert all(a._when.value == "narration-starts" for a in animators)

    def test_zero_drops_every_entrance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A zero-length animation raises in pycaps, so 0 can't shorten."""
        animators = _build(monkeypatch, tmp_path, _explosive_like(), max_entrance_sec=0)

        assert all(a._when.value == "narration-ends" for a in animators)
        assert all(a._animation._duration > 0 for a in animators)

    def test_unset_leaves_the_template_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        animators = _build(monkeypatch, tmp_path, _explosive_like())

        assert [a._animation._duration for a in animators] == [0.4, 0.2, 0.3, 0.3]

    def test_a_pipeline_without_the_list_is_left_alone(self) -> None:
        from src.video.pycaps_engine.renderer import _limit_template_motion

        _limit_template_motion(types.SimpleNamespace(), 0.25, 0.08)

    def test_the_shipped_config_limits_both(self) -> None:
        import yaml

        raw = yaml.safe_load((REPO / "config" / "subtitles.yaml").read_text())
        block = PycapsSettings(**raw["subtitle_settings"]["pycaps"])

        assert (block.max_entrance_sec, block.max_exit_sec) == (0.25, 0.08)


@pytest.mark.unit
@pytest.mark.req("REQ-VID-155")
class TestAgainstTheRealTemplate:
    @pytest.fixture
    def pycaps(self):
        return pytest.importorskip(
            "pycaps",
            reason="optional group not installed (poetry install --with pycaps)",
        )

    def test_explosive_moves_within_the_limits(self, pycaps):
        from pycaps.template import TemplateFactory, TemplateLoader

        from src.video.pycaps_engine.renderer import _limit_template_motion

        builder = TemplateLoader(TemplateFactory().create("explosive")).load(False)
        before = builder._caps_pipeline._animators
        assert any(
            a._animation._duration > 0.25 for a in before
        ), "explosive no longer ships a long animation; the premise is gone"

        _limit_template_motion(builder, 0.25, 0.08)

        after = builder._caps_pipeline._animators
        assert after
        for animator in after:
            if animator._when.value == "narration-starts":
                assert animator._animation._duration <= 0.25
            else:
                assert type(animator._animation).__name__ == "FadeOut"
                assert animator._animation._duration <= 0.08


def _dialogue(monkeypatch, effect: str, effects: SubtitleEffectsSettings, last=False):
    from src.video import unified_subtitle_generator as module

    monkeypatch.setattr(module.config, "subtitle_effects", effects)
    generator = module.UnifiedSubtitleGenerator(
        config=SubtitleSettings(style_preset=StylePreset.MINIMAL),
        frame_size=(1080, 1920),
        product_id="TEST001",
        video_config=module.config,
    )
    generator._selected_effects = {effect: True}
    line = generator._create_dialogue_line(
        {"start": 0.0, "end": 2.0, "text": "a caption that runs long enough"},
        Position(x=0.5, y=0.7),
        {"primary": "&H00FFFFFF", "outline": "&H00000000"},
        is_last_segment=last,
    )
    assert line
    return line


@pytest.mark.unit
@pytest.mark.req("REQ-VID-155")
class TestFfmpegMotion:
    def test_the_fade_out_is_its_own_length(self, monkeypatch) -> None:
        effects = SubtitleEffectsSettings(fade_duration_ms=250, fade_out_duration_ms=80)

        assert "\\fad(250,80)" in _dialogue(monkeypatch, "fade", effects)

    def test_unset_fade_out_follows_the_fade_in(self, monkeypatch) -> None:
        effects = SubtitleEffectsSettings(fade_duration_ms=300)

        assert "\\fad(300,300)" in _dialogue(monkeypatch, "fade", effects)

    def test_the_capped_typewriter_is_one_short_fade_in(self, monkeypatch) -> None:
        r"""Invisible at the start, fully shown by 250 ms, nothing after.

        The uncapped pair fades the line out and then back in up to the
        event's end (`\t(R,0,...)` runs to the end in libass), so capping
        its first half lengthened the motion.
        """
        capped = SubtitleEffectsSettings(typewriter_max_reveal_ms=250)
        line = _dialogue(monkeypatch, "typewriter", capped)

        assert "\\fad(250,0)" in line
        assert "\\t(" not in line
        assert "\\alpha" not in line, "replaces the style's shadow alpha"

    def test_zero_shows_the_line_at_once(self, monkeypatch) -> None:
        line = _dialogue(
            monkeypatch,
            "typewriter",
            SubtitleEffectsSettings(typewriter_max_reveal_ms=0),
        )

        assert "\\t(" not in line
        assert "\\fad(" not in line
        assert "\\alpha" not in line

    def test_uncapped_keeps_the_old_effect(self, monkeypatch) -> None:
        line = _dialogue(monkeypatch, "typewriter", SubtitleEffectsSettings())

        assert ",0,\\alpha&H00&)" in line

    def test_the_shipped_config_limits_both(self) -> None:
        import yaml

        raw = yaml.safe_load((REPO / "config" / "subtitles.yaml").read_text())
        effects = SubtitleEffectsSettings(**raw["subtitle_effects"])

        assert effects.fade_duration_ms <= 250
        assert effects.fade_out_duration_ms is not None
        assert effects.fade_out_duration_ms <= 80
        assert effects.typewriter_max_reveal_ms is not None
        assert effects.typewriter_max_reveal_ms <= 250
