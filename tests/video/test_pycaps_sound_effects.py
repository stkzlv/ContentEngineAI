"""A template's sound effects are dropped unless the config asks for them.

`explosive` plays a `ding` on every word tagged `highlighted`, and
`ai_tag_prompt_override` widens that tag to around 15% of the words -- 10-17
dings in a 30-second render. That 15% is a recommendation about *visual*
highlighting from `docs/subtitle-best-practices.md`; nothing chose it as a
rate for a sound effect. The two decisions share one tag by accident, so
muting the audio is what lets the highlighting stay where the doc puts it.

`_sound_effects` is a plain list the template loader appends to, reached the
same way the CSS append and the AI-tag rewrite reach their targets. The caller
is driven here rather than the helper, for the reason the sentence-case test
gives: where `_build_pipeline` puts the call is the part that can be wrong.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.video.config.subtitle_models import PycapsSettings


class _RecordingBuilder:
    """Stands in for pycaps' builder, carrying a template's effect list."""

    def __init__(self, effects: list | None = None) -> None:
        self._caps_pipeline = types.SimpleNamespace(
            _layout_options=MagicMock(),
            _text_effects=[],
            _semantic_tagger=types.SimpleNamespace(_ai_rules={}),
            _sound_effects=[] if effects is None else effects,
        )

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            return self

        return _record

    @property
    def sound_effects(self) -> list:
        effects: list = self._caps_pipeline._sound_effects
        return effects


def _fake_pycaps(monkeypatch, builder: _RecordingBuilder) -> None:
    """Put a fake `pycaps` package where `_build_pipeline`'s imports look.

    Same shape as `test_pycaps_sentence_case.py`'s. Not shared between the
    two files: a test importing another test's helpers couples them, and the
    duplication is a dozen lines that change only when pycaps' own import
    surface does.
    """
    import sys

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


def _build(monkeypatch, tmp_path: Path, effects: list, **settings):
    from src.video.pycaps_engine.renderer import PycapsRenderer

    # `_build_pipeline` checks the input exists before it does anything else.
    (tmp_path / "in.mp4").write_bytes(b"")
    (tmp_path / "t.json").write_text("{}")

    builder = _RecordingBuilder(effects)
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
        settings=PycapsSettings(**settings),
    )
    return builder


@pytest.mark.unit
class TestTheEffectsAreMuted:
    def test_a_ding_per_highlighted_word_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        ding = {"name": "ding", "tag_condition": "highlighted"}
        builder = _build(
            monkeypatch, tmp_path, [ding], mute_template_sound_effects=True
        )

        assert builder.sound_effects == []

    def test_the_flag_off_keeps_them(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The template's own choice, when the operator asks for it."""
        ding = {"name": "ding", "tag_condition": "highlighted"}
        builder = _build(
            monkeypatch, tmp_path, [ding], mute_template_sound_effects=False
        )

        assert builder.sound_effects == [ding]

    def test_a_template_with_no_effects_is_fine(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """`word-focus` ships none; muting must not raise on it."""
        builder = _build(monkeypatch, tmp_path, [], mute_template_sound_effects=True)

        assert builder.sound_effects == []

    def test_it_is_on_by_default(self) -> None:
        assert PycapsSettings().mute_template_sound_effects is True


@pytest.mark.unit
class TestTheShippedConfigMutesThem:
    def test_the_yaml_sets_it(self) -> None:
        import yaml

        repo = Path(__file__).resolve().parents[2]
        raw = yaml.safe_load((repo / "config" / "subtitles.yaml").read_text())
        block = raw["subtitle_settings"]["pycaps"]

        assert block["mute_template_sound_effects"] is True
        assert PycapsSettings(**block).mute_template_sound_effects is True

    def test_explosive_is_the_template_this_is_for(self) -> None:
        """If it leaves the pool, this setting stops mattering."""
        import yaml

        repo = Path(__file__).resolve().parents[2]
        raw = yaml.safe_load((repo / "config" / "subtitles.yaml").read_text())

        assert "explosive" in raw["subtitle_settings"]["pycaps"]["template_pool"]
