"""A screen per tutorial step, timed to its narration (REQ-VID-123)."""

from __future__ import annotations

import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest
from PIL import Image

from src.video.assembler.overlay_builder import apply_step_visuals
from src.video.config.visual_models import StepVisualsSettings
from src.video.step_visuals import (
    Screen,
    draw_screen,
    load_step_visuals,
    plan_step_visuals,
)

STEPS = [
    {"ui_path": "Settings", "action": "Open Settings", "expected": "It opens"},
    {"ui_path": "General > About", "action": "Tap About", "expected": "About opens"},
    {"ui_path": "Name", "action": "Tap Name", "expected": "You can rename it"},
]
# Words 0.5 s apart. The step sentences start at 3.0 s (First), 7.0 s
# (Then) and 10.0 s (Last); the keys are Settings, About and Name.
TEXT = (
    "Rename your phone in a minute. First open Settings on the "
    "home screen now. Then tap About under General here. Last tap Name "
    "and type the new one in the box."
)


def _spoken(text: str = TEXT, gap: float = 0.5) -> list[dict]:
    return [
        {"word": w, "start_time": i * gap, "end_time": i * gap + 0.4}
        for i, w in enumerate(text.split())
    ]


def _plan(steps: list[dict] = STEPS, **kwargs) -> list[Screen]:
    options = {
        "start_screen": "Home screen",
        "video_end": 15.0,
        "opening_max_sec": 2.0,
        "recap_sec": 2.5,
    }
    options.update(kwargs)
    return plan_step_visuals(steps, _spoken(), **options)


@pytest.mark.unit
@pytest.mark.req("REQ-VID-123")
class TestThePlan:
    def test_each_step_starts_with_the_sentence_that_names_it(self) -> None:
        steps = [s for s in _plan() if s.kind == "step"]

        assert [s.start for s in steps] == [3.0, 7.0, 10.0]
        assert [s.title for s in steps] == [
            "Step 1 of 3",
            "Step 2 of 3",
            "Step 3 of 3",
        ]
        assert [s.target for s in steps] == ["Settings", "About", "Name"]

    def test_stock_keeps_only_the_opening(self) -> None:
        screens = _plan()

        # From the end of the opening to the end of the video, one screen
        # always covers the footage, with no gap between screens.
        assert screens[0].kind == "start"
        assert screens[0].start == 2.0
        assert screens[0].header == "Home screen"
        for before, after in zip(screens, screens[1:], strict=False):
            assert before.end == after.start
        assert screens[-1].end == 15.0

    def test_no_start_screen_when_the_first_step_is_early(self) -> None:
        screens = _plan(opening_max_sec=5.0)

        assert screens[0].kind == "step"

    def test_the_header_is_the_screen_the_step_happens_on(self) -> None:
        steps = [s for s in _plan() if s.kind == "step"]

        # The start screen, the path's own parent, then blank: the step list
        # doesn't name the screen Name is on, so none is guessed.
        assert [s.header for s in steps] == ["Home screen", "General", ""]

    def test_the_video_ends_on_the_recap_and_the_result(self) -> None:
        recap = _plan()[-1]

        assert recap.kind == "recap"
        assert (recap.start, recap.end) == (12.5, 15.0)
        assert recap.rows == ["Settings", "General > About", "Name"]
        assert recap.result == "You can rename it"

    def test_the_recap_leaves_the_last_step_a_second(self) -> None:
        screens = _plan(video_end=12.5, recap_sec=2.5)

        assert (screens[-2].start, screens[-2].end) == (10.0, 11.0)
        assert (screens[-1].start, screens[-1].end) == (11.0, 12.5)

    def test_the_topic_sentence_never_opens_a_step(self) -> None:
        steps = [{"ui_path": "Settings > Accessibility"}, {"ui_path": "Touch"}]
        text = (
            "Your iPhone hides a button in Accessibility settings. First open "
            "Settings and tap Accessibility. Then tap Touch now. It opens. "
            "So that is it done."
        )
        screens = plan_step_visuals(
            steps,
            _spoken(text),
            start_screen="Home",
            video_end=14.0,
            opening_max_sec=3.0,
            recap_sec=2.0,
        )

        # The stock opening, then the start screen, then step 1 at "First".
        assert [(s.kind, s.start) for s in screens[:2]] == [
            ("start", 3.0),
            ("step", 4.0),
        ]

    def test_an_earlier_segment_opens_the_step(self) -> None:
        steps = [{"ui_path": "Auto-Lock > Period > 30", "action": "Select 30"}]
        text = (
            "Here is how. Now find Auto-Lock on the page. Open the Period list. "
            "Select 30 there. It is saved. So that is Auto-Lock done."
        )
        screens = plan_step_visuals(
            steps,
            _spoken(text),
            start_screen="Inbox",
            video_end=20.0,
            opening_max_sec=1.0,
            recap_sec=2.0,
        )
        step = next(s for s in screens if s.kind == "step")

        # "Now" opens the sentence that first names a segment, not "Select 30".
        assert step.start == 1.5
        # The recap starts two sentences after the step's: "So that is ...".
        assert screens[-1].kind == "recap"
        assert screens[-1].start == 9.5

    def test_a_step_not_found_keeps_the_stock_visuals(self, caplog) -> None:
        steps = [*STEPS[:2], {"ui_path": "Bluetooth", "action": "Tap Bluetooth"}]

        with caplog.at_level("INFO", logger="src.video.step_visuals"):
            assert _plan(steps) == []
        assert "step(s) 3 (bluetooth) not found" in caplog.text

    def test_a_step_named_by_its_parent_label_still_gets_screens(self) -> None:
        """The script says "tap Name", never the path's last label."""
        steps = [*STEPS[:2], {"ui_path": "Name > Device name", "action": "Rename"}]

        screens = [s for s in _plan(steps) if s.kind == "step"]

        assert len(screens) == 3
        assert screens[2].start == 10.0


@pytest.mark.unit
@pytest.mark.req("REQ-VID-123")
class TestTheDrawing:
    def test_a_screen_is_an_opaque_frame(self) -> None:
        image = draw_screen(
            Screen("step", 0, 1, title="Step 1 of 2", header="Home", target="Wi-Fi"),
            (540, 960),
        )

        assert image.size == (540, 960)
        assert image.mode == "RGB"

    def test_the_target_is_drawn_and_marked(self) -> None:
        size = (540, 960)
        marked = draw_screen(Screen("step", 0, 1, header="Home", target="Wi-Fi"), size)
        start = draw_screen(Screen("start", 0, 1, header="Home"), size)
        other = draw_screen(
            Screen("step", 0, 1, header="Home", target="Bluetooth"), size
        )

        # The accent outline marks the target row; the start screen has none.
        accent = (47, 111, 236)
        marked_colours = marked.getcolors(1 << 20) or []
        start_colours = start.getcolors(1 << 20) or []
        assert accent in {c for _, c in marked_colours}
        assert accent not in {c for _, c in start_colours}
        assert marked.tobytes() != other.tobytes()

    def test_a_long_label_is_shortened_not_spilled(self) -> None:
        size = (540, 960)
        image = draw_screen(
            Screen("step", 0, 1, header="Home", target="Very long label " * 12), size
        )
        # Outside the panel, the background is untouched at the row's height.
        row_y = int(0.21 * 960 + (120 + 24 + 128 + 55) * 960 / 1920)
        assert image.getpixel((int(0.95 * 540), row_y)) == (27, 31, 39)


@pytest.mark.unit
@pytest.mark.req("REQ-VID-123")
class TestLoading:
    def test_draws_a_png_per_screen(self, tmp_path: Path) -> None:
        step_list = tmp_path / "step_list.json"
        step_list.write_text(
            json.dumps({"start_screen": "Home screen", "steps": STEPS}),
            encoding="utf-8",
        )

        drawn = load_step_visuals(
            step_list,
            _spoken(),
            tmp_path,
            size=(540, 960),
            video_end=15.0,
            opening_max_sec=2.0,
            recap_sec=2.5,
        )

        assert len(drawn) == 5
        assert all(p.exists() and Image.open(p).size == (540, 960) for p, _, _ in drawn)
        assert drawn[0][1:] == (2.0, 3.0)

    def test_nothing_without_a_list_or_timings(self, tmp_path: Path) -> None:
        options = {
            "size": (540, 960),
            "video_end": 15.0,
            "opening_max_sec": 2.0,
            "recap_sec": 2.5,
        }
        missing = tmp_path / "step_list.json"
        assert load_step_visuals(missing, _spoken(), tmp_path, **options) == []
        missing.write_text(json.dumps({"steps": STEPS}), encoding="utf-8")
        assert load_step_visuals(missing, None, tmp_path, **options) == []


def _chain() -> tuple:
    return (["[0:v]null[v_base]"], ["-i", "base.mp4"], [], "[v_base]", [], False)


@pytest.mark.unit
@pytest.mark.req("REQ-VID-123")
class TestTheOverlay:
    def test_none_leaves_the_chain(self) -> None:
        assert apply_step_visuals(_chain(), []) == _chain()

    def test_each_screen_is_an_input_overlaid_in_its_window(self) -> None:
        filters, inputs, _, stream, _, _ = apply_step_visuals(
            _chain(), [(Path("a.png"), 1.0, 2.5), (Path("b.png"), 2.5, 4.0)]
        )

        assert inputs == ["-i", "base.mp4", "-i", "a.png", "-i", "b.png"]
        assert filters[1] == (
            "[v_base][1:v]overlay=0:0:enable='between(t,1.000,2.500)'"
            "[v_step_visual_0]"
        )
        assert filters[2].startswith("[v_step_visual_0][2:v]overlay")
        assert stream == "[v_step_visual_1]"

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_ffmpeg_shows_each_screen_only_in_its_window(self, tmp_path: Path) -> None:
        blue, green = tmp_path / "blue.png", tmp_path / "green.png"
        Image.new("RGB", (540, 960), (0, 0, 255)).save(blue)
        Image.new("RGB", (540, 960), (0, 255, 0)).save(green)
        filters, inputs, _, stream, _, _ = apply_step_visuals(
            (
                ["[0:v]null[v_base]"],
                ["-f", "lavfi", "-i", "color=c=red:s=540x960:r=10:d=3.5"],
                [],
                "[v_base]",
                [],
                False,
            ),
            [(blue, 1.0, 2.0), (green, 2.0, 3.0)],
        )
        out = tmp_path / "out.mp4"
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                *inputs,
                "-filter_complex", ";".join(filters), "-map", stream,
                "-pix_fmt", "yuv420p", str(out),
            ],
            check=True,
        )  # fmt: skip

        def colour(t: float) -> tuple[int, int, int]:
            raw = subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(out),
                    "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-",
                ],
                check=True,
                capture_output=True,
            ).stdout  # fmt: skip
            return raw[0], raw[1], raw[2]

        def dominant(t: float) -> int:
            rgb = colour(t)
            return rgb.index(max(rgb))

        assert dominant(0.5) == 0  # the footage before the screens
        assert dominant(1.5) == 2  # the first screen
        assert dominant(2.5) == 1  # the second
        assert dominant(3.3) == 0  # the footage after them


@pytest.mark.unit
@pytest.mark.req("REQ-VID-123")
def test_the_render_passes_its_screens_to_the_assembler() -> None:
    """Both hops, read at their call sites; the real path is a full render."""
    import inspect

    from src.video.assembler import core
    from src.video.producer import steps

    assemble = inspect.getsource(core.VideoAssembler.assemble_video)
    assert "apply_step_visuals(" in assemble
    # A screen carries the counter, so the cards are not drawn over it.
    assert "step_cards = None" in assemble
    assert "step_visuals=_step_visuals(ctx, total_duration)" in inspect.getsource(steps)


@pytest.mark.unit
@pytest.mark.req("REQ-VID-123")
def test_the_producer_draws_them_only_when_on(tmp_path: Path) -> None:
    from src.video.producer.steps import _step_visuals

    (tmp_path / "step_list.json").write_text(
        json.dumps({"start_screen": "Home screen", "steps": STEPS}), encoding="utf-8"
    )
    transcript = tmp_path / "whisper_transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "words": [
                            {
                                "word": w["word"],
                                "start": w["start_time"],
                                "end": w["end_time"],
                            }
                            for w in _spoken()
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def ctx(enabled: bool) -> types.SimpleNamespace:
        video = types.SimpleNamespace(
            step_visuals=StepVisualsSettings(enabled=enabled), resolution=(540, 960)
        )
        return types.SimpleNamespace(
            config=types.SimpleNamespace(video_settings=video),
            run_paths={
                "script_file": tmp_path / "script.txt",
                "whisper_transcript_file": transcript,
                "intermediate_base": tmp_path,
            },
        )

    assert _step_visuals(ctx(False), 15.0) is None
    drawn = _step_visuals(ctx(True), 15.0)
    assert drawn and drawn[-1][2] == 15.0


@pytest.mark.req("REQ-VID-123")
def test_the_bundled_config_turns_step_screens_on() -> None:
    from src.video.config import load_video_config_modular

    assert load_video_config_modular().video_settings.step_visuals.enabled is True
    # A config with no `step_visuals` section draws none.
    assert StepVisualsSettings().enabled is False
