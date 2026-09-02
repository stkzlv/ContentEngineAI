"""Captions keep the transcript's casing when the config says so.

`word-focus` ships `.word { text-transform: uppercase; }`. Rule 6 of
`docs/subtitle-best-practices.md` wants sentence case, because ascenders and
descenders carry word shape and mixed case reads faster (#100). Rather than
forking the template into `pycaps-templates/`, the renderer appends a later
rule at the same specificity, which wins the cascade, when
`subtitle_settings.pycaps.force_sentence_case` is set.

The caller is what is driven here, not the helper. `_build_pipeline` wires the
renderer *after* the layout merge, and `with_custom_subtitle_renderer` replaces
the renderer object that appended CSS lives on -- so a call placed before the
pictex swap is discarded with the CSS renderer, and a test of the helper alone
would not notice.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.video.config.subtitle_models import PycapsSettings

_CSS = "text-transform: none"


class _RecordingBuilder:
    """Records every builder call in order, in place of pycaps' own."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._caps_pipeline = types.SimpleNamespace(
            _layout_options=MagicMock(), _text_effects=[]
        )

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def _record(*args, **kwargs):
            self.calls.append((name, args))
            return self

        return _record

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]


def _fake_pycaps(monkeypatch, builder: _RecordingBuilder) -> None:
    """Put a fake `pycaps` package where `_build_pipeline`'s imports look."""
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


def _build(monkeypatch, tmp_path: Path, **settings) -> _RecordingBuilder:
    from src.video.pycaps_engine.renderer import PycapsRenderer

    builder = _RecordingBuilder()
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
        template_name="word-focus",
        visual_bounds=None,
        settings=PycapsSettings(**settings),
    )
    return builder


@pytest.mark.unit
class TestTheBuildAppendsTheOverride:
    def test_it_is_appended_when_the_setting_is_on(self, monkeypatch, tmp_path):
        builder = _build(monkeypatch, tmp_path, force_sentence_case=True)

        css = [a[0] for n, a in builder.calls if n == "add_css_content"]
        assert len(css) == 1
        assert _CSS in css[0]
        assert ".word" in css[0]

    def test_it_is_not_appended_when_the_setting_is_off(self, monkeypatch, tmp_path):
        builder = _build(monkeypatch, tmp_path, force_sentence_case=False)

        assert "add_css_content" not in builder.names()

    def test_the_model_default_renders_the_template_as_shipped(self):
        """The bundled YAML turns it on; a bare model must not."""
        assert PycapsSettings().force_sentence_case is False

    def test_it_runs_after_the_renderer_swap(self, monkeypatch, tmp_path):
        """Appended CSS lives on the renderer object the swap replaces."""
        builder = _build(
            monkeypatch, tmp_path, force_sentence_case=True, renderer="pictex"
        )

        names = builder.names()
        assert "with_custom_subtitle_renderer" in names
        assert names.index("add_css_content") > names.index(
            "with_custom_subtitle_renderer"
        ), "CSS appended before the pictex swap is discarded with the CSS renderer"


@pytest.mark.unit
class TestTheBundledConfigTurnsItOn:
    def test_the_yaml_sets_it_and_the_merge_carries_it(self):
        from src.video.config import load_video_config_modular

        config = load_video_config_modular()
        merged = config.get_profile_merged_settings("slideshow_images1")
        pycaps = merged.subtitle_settings.pycaps
        if not isinstance(pycaps, PycapsSettings):
            pycaps = PycapsSettings(**pycaps)

        assert pycaps.force_sentence_case is True


class TestAgainstTheRealTemplate:
    """The override has to reach the renderer pycaps actually uses."""

    @pytest.fixture
    def pycaps(self):
        return pytest.importorskip(
            "pycaps",
            reason="optional group not installed (poetry install --with pycaps)",
        )

    def test_word_focus_still_ships_uppercase(self, pycaps):
        """If upstream drops it, the override and this guard can go."""
        css = (
            (Path(pycaps.__file__).parent / "template" / "preset" / "word-focus")
            .joinpath("styles.css")
            .read_text()
        )

        assert "text-transform: uppercase" in css

    @pytest.mark.parametrize("template_name", ["word-focus", "explosive"])
    def test_the_override_lands_on_the_css_renderer(self, pycaps, template_name):
        from pycaps.template import TemplateFactory, TemplateLoader

        from src.video.pycaps_engine.renderer import _force_sentence_case

        builder = TemplateLoader(TemplateFactory().create(template_name)).load(False)
        _force_sentence_case(builder)

        assert _CSS in builder._caps_pipeline._renderer._custom_css

    def test_a_builder_without_the_method_is_left_alone(self):
        from src.video.pycaps_engine.renderer import _force_sentence_case

        class _Old:
            pass

        _force_sentence_case(_Old())  # must not raise
