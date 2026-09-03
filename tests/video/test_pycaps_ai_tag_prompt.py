"""The AI tagger highlights what the config asks for, not filler words.

`enable_ai_tagging` is on in the bundled config, and the pool's AI template,
`explosive`, ships the instruction "the most important phrase or word in all
the script". Gemini answers that with `also`, `can`, `all` and `from` -- the
auxiliary and preposition bucket that `docs/subtitle-best-practices.md`,
"AI-driven highlighting", says to skip. Every published video using that
template has been emphasising filler (#101).

The issue was unsure whether the prompt could be injected through
`LlmProvider.set()` or needed an `Llm` subclass. Neither: a template's AI
rules arrive as `SemanticTagger._ai_rules`, a `{Tag: instruction}` map, and
the tagger renders each entry into one line of the prompt it builds. So the
instruction is the whole lever, and overriding it needs no fork and no
subclass.

These drive `_build_pipeline`, not the helper. A helper test passes whether or
not the caller invokes it, and the call has to land after the template is
loaded, which is what puts the rules on the tagger in the first place.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.video.config.subtitle_models import PycapsSettings

RECIPE = "prices, numbers and product nouns; never articles or auxiliaries"


class _Tag:
    """Stands in for pycaps' `Tag`, which is only used as a dict key here."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Tag) and other.name == self.name


class _RecordingBuilder:
    """A builder carrying a semantic tagger with the rules a template loads."""

    def __init__(self, ai_rules: dict | None = None) -> None:
        self.calls: list[str] = []
        tagger = types.SimpleNamespace(
            _ai_rules={} if ai_rules is None else dict(ai_rules)
        )
        self._caps_pipeline = types.SimpleNamespace(
            _layout_options=MagicMock(),
            _text_effects=[],
            _semantic_tagger=tagger,
        )

    @property
    def rules(self) -> dict:
        rules: dict = self._caps_pipeline._semantic_tagger._ai_rules
        return rules

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def _record(*args, **kwargs):
            self.calls.append(name)
            return self

        return _record


def _fake_pycaps(monkeypatch, builder: _RecordingBuilder) -> None:
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


def _build(monkeypatch, tmp_path: Path, builder: _RecordingBuilder, **settings):
    from src.video.pycaps_engine.renderer import PycapsRenderer

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
class TestTheBuildOverridesTheInstruction:
    def test_every_rule_gets_the_configured_recipe(self, monkeypatch, tmp_path):
        """A template may define several rules; leaving one stock tags filler."""
        builder = _RecordingBuilder(
            {
                _Tag("highlighted"): "the most important phrase or word",
                _Tag("emphasis"): "anything striking",
            }
        )

        _build(monkeypatch, tmp_path, builder, ai_tag_prompt_override=RECIPE)

        assert list(builder.rules.values()) == [RECIPE, RECIPE]

    def test_no_override_leaves_the_template_alone(self, monkeypatch, tmp_path):
        stock = "the most important phrase or word"
        builder = _RecordingBuilder({_Tag("highlighted"): stock})

        _build(monkeypatch, tmp_path, builder)

        assert list(builder.rules.values()) == [stock]

    def test_a_template_with_no_ai_rule_is_untouched(self, monkeypatch, tmp_path):
        """`word-focus`, the other pool entry, defines none."""
        builder = _RecordingBuilder({})

        _build(monkeypatch, tmp_path, builder, ai_tag_prompt_override=RECIPE)

        assert builder.rules == {}

    def test_the_model_default_is_no_override(self):
        assert PycapsSettings().ai_tag_prompt_override is None


@pytest.mark.unit
class TestTheBundledConfigCarriesARecipe:
    def test_the_recipe_reaches_the_merged_settings(self):
        from src.video.config import load_video_config_modular

        config = load_video_config_modular()
        merged = config.get_profile_merged_settings("slideshow_images1")
        raw = merged.subtitle_settings.pycaps
        pycaps = raw if isinstance(raw, PycapsSettings) else PycapsSettings(**raw)

        override = pycaps.ai_tag_prompt_override
        assert override, "the bundled config ships no recipe, so the stock prompt runs"
        # The point of the recipe is what it excludes.
        assert "never" in override.lower()


class TestAgainstTheRealTemplate:
    """The override has to land on the object pycaps actually reads."""

    @pytest.fixture
    def pycaps(self):
        return pytest.importorskip(
            "pycaps",
            reason="optional group not installed (poetry install --with pycaps)",
        )

    def test_explosive_still_ships_an_ai_rule(self, pycaps):
        """If upstream drops it, this override has nothing to act on."""
        from pycaps.template import TemplateFactory, TemplateLoader

        builder = TemplateLoader(TemplateFactory().create("explosive")).load(False)
        rules = builder._caps_pipeline._semantic_tagger._ai_rules

        assert rules, "explosive no longer defines an AI tagging rule"

    def test_the_override_replaces_the_real_rule(self, pycaps):
        from pycaps.template import TemplateFactory, TemplateLoader

        from src.video.pycaps_engine.renderer import _override_ai_tag_prompt

        builder = TemplateLoader(TemplateFactory().create("explosive")).load(False)
        rules = builder._caps_pipeline._semantic_tagger._ai_rules
        before = list(rules.values())

        _override_ai_tag_prompt(builder, RECIPE)

        assert before != [RECIPE]
        assert list(rules.values()) == [RECIPE] * len(before)

    def test_a_builder_without_a_tagger_is_left_alone(self):
        from src.video.pycaps_engine.renderer import _override_ai_tag_prompt

        class _Old:
            pass

        _override_ai_tag_prompt(_Old(), RECIPE)  # must not raise
