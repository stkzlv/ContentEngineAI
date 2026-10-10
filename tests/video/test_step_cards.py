"""Step cards on a tutorial render (REQ-VID-124)."""

from __future__ import annotations

import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from src.video.assembler.overlay_builder import apply_step_card_overlay
from src.video.config.visual_models import GraphicsSettings
from src.video.step_cards import (
    StepCard,
    display_path,
    load_step_cards,
    plan_step_cards,
)


def _spoken(text: str, gap: float = 0.4) -> list[dict]:
    return [
        {
            "word": w,
            "start_time": round(i * gap, 2),
            "end_time": round(i * gap + 0.3, 2),
        }
        for i, w in enumerate(text.split())
    ]


def _plan(steps: list[dict], text: str, **kwargs) -> list[StepCard]:
    options = {"not_before": 0.0, "min_sec": 0.5, "max_sec": 6.0}
    options.update(kwargs)
    return plan_step_cards(steps, _spoken(text), **options)


BACKGROUND = (
    "Quick fix for apps running in the background. Open the Settings app. "
    "You'll see the main settings list. Tap General. The General Settings menu "
    "opens up. Tap Background App Refresh. The Background App Refresh screen "
    "appears. Tap Background App Refresh at the top. Select off. A check mark "
    "shows next to it."
)
BACKGROUND_STEPS = [
    {"ui_path": "Settings", "action": "Open the Settings app."},
    {"ui_path": "Settings > General", "action": "Tap General."},
    {"ui_path": "Settings > General > Background App Refresh", "action": "Tap it."},
    {
        "ui_path": "Settings > General > Background App Refresh > Background App Refresh",
        "action": "Tap Background App Refresh at the top of the menu.",
    },
    {"ui_path": "Background App Refresh > Off", "action": "Select Off."},
]


def _word_at(text: str, start: float, gap: float = 0.4) -> str:
    return text.split()[round(start / gap)]


@pytest.mark.unit
@pytest.mark.req("REQ-VID-124")
class TestWhenACardShows:
    def test_each_card_starts_when_its_step_is_spoken(self) -> None:
        cards = _plan(BACKGROUND_STEPS, BACKGROUND)

        assert [c.counter for c in cards] == [f"Step {n} of 5" for n in range(1, 6)]
        starts = [(_word_at(BACKGROUND, c.start), c.start) for c in cards]
        words = BACKGROUND.split()
        # Step 2 is "Tap General", not "The General Settings menu"; step 4 is
        # the second "Tap Background App Refresh", not the screen that appears.
        assert [w for w, _ in starts] == [
            "Settings",
            "General.",
            "Background",
            "Background",
            "off.",
        ]
        assert words[round(starts[3][1] / 0.4) - 1] == "Tap"

    def test_a_card_ends_when_the_next_starts(self) -> None:
        cards = _plan(BACKGROUND_STEPS, BACKGROUND)

        for card, following in zip(cards, cards[1:], strict=False):
            assert card.end == following.start

    def test_a_name_spoken_as_one_word_still_matches(self) -> None:
        cards = _plan(
            [{"ui_path": "Accessibility > Touch > Back Tap", "action": "Tap Back Tap"}],
            "Here's how to use it. Go to Accessibility. Then Touch. Tap BackTap. Done.",
        )

        assert len(cards) == 1
        assert cards[0].start == pytest.approx(11 * 0.4)  # "BackTap.", not "Go to"

    def test_the_first_sentence_never_starts_a_card(self) -> None:
        """It names the topic, which is often the last step's target."""
        cards = _plan(
            [{"ui_path": "Back Tap", "action": "Tap Back Tap"}],
            "Turn on BackTap to run shortcuts. Open Settings. Tap BackTap. It works.",
        )

        assert cards[0].start == pytest.approx(9 * 0.4)

    def test_a_step_not_spoken_gets_no_card(self) -> None:
        cards = _plan(
            [
                {"ui_path": "Settings", "action": "Open Settings"},
                {"ui_path": "Privacy", "action": "Tap Privacy"},
            ],
            "Intro here. Open Settings now and wait for it to load fully.",
        )

        assert [c.counter for c in cards] == ["Step 1 of 2"]

    def test_no_card_before_the_hook_ends_or_shorter_than_the_minimum(self) -> None:
        text = "Intro here. Open Settings. Tap General. Tap Privacy now please."
        steps = [
            {"ui_path": "Settings", "action": ""},
            {"ui_path": "General", "action": ""},
            {"ui_path": "Privacy", "action": ""},
        ]
        # Step 1 is spoken at 1.2 s and ends at 2.0 s: 0.5 s after the hook.
        cards = _plan(steps, text, not_before=1.5, min_sec=0.6)

        assert all(c.start >= 1.5 for c in cards)
        assert "Step 1 of 3" not in [c.counter for c in cards]

    def test_a_card_is_held_at_most_max_sec(self) -> None:
        cards = _plan(
            [{"ui_path": "Settings", "action": ""}],
            "Intro. Open Settings " + "and then wait " * 20,
            max_sec=3.0,
        )

        assert cards[0].end - cards[0].start == pytest.approx(3.0)

    def test_an_ampersand_matches_a_spoken_and(self) -> None:
        cards = _plan(
            [{"ui_path": "Settings > Bluetooth & devices", "action": "Open it"}],
            "Pair a mouse. Open Settings then Bluetooth and devices now. Done here.",
        )

        assert cards and cards[0].path == "Settings > Bluetooth & devices"
        assert cards[0].start == pytest.approx(6 * 0.4)

    def test_a_miss_is_not_found_in_the_recap(self) -> None:
        """Without a window the unmatched step lands on the recap at the end."""
        filler = "and wait a moment " * 25
        cards = _plan(
            [
                {"ui_path": "Settings", "action": "Open Settings"},
                {"ui_path": "Unmatched Menu", "action": "Tap it"},
                {"ui_path": "Storage", "action": "Tap Storage"},
            ],
            "Intro here. Open Settings. Tap Storage. "
            + filler
            + "Recap: Settings, Unmatched Menu, Storage.",
        )

        assert [c.counter for c in cards] == ["Step 1 of 3", "Step 3 of 3"]
        assert cards[1].start == pytest.approx(5 * 0.4)

    def test_a_step_far_after_the_last_is_still_found(self) -> None:
        """Real scripts put up to 42 words between two steps."""
        gap = "it is important that they are not still connected " * 5  # 45
        cards = _plan(
            [
                {"ui_path": "Settings", "action": ""},
                {"ui_path": "Add device", "action": ""},
            ],
            "Intro here. Open Settings. " + gap + "Then tap Add device. Done here.",
        )

        assert [c.counter for c in cards] == ["Step 1 of 2", "Step 2 of 2"]

    def test_a_miss_widens_the_next_window(self) -> None:
        gap = "it is important that they are not still connected " * 7  # 63
        cards = _plan(
            [
                {"ui_path": "Settings", "action": ""},
                {"ui_path": "Unspoken", "action": ""},
                {"ui_path": "Add device", "action": ""},
            ],
            "Intro here. Open Settings. " + gap + "Then tap Add device. Done here.",
        )

        assert [c.counter for c in cards] == ["Step 1 of 3", "Step 3 of 3"]

    def test_a_verb_three_words_back_wins(self) -> None:
        cards = _plan(
            [{"ui_path": "Password", "action": ""}],
            "Intro here. Pick the one you want the password for. "
            "Tap the word Password. Then wait a while.",
        )

        assert cards[0].start == pytest.approx(13 * 0.4)

    def test_a_placeholder_segment_uses_the_action(self) -> None:
        cards = _plan(
            [{"ui_path": "Apps > [App Name]", "action": "Select the app to clear"}],
            "Intro here. Now select the app you want. Then wait.",
        )

        assert cards and cards[0].path == "Apps > App Name"
        assert cards[0].start == pytest.approx(5 * 0.4)  # "app", the first content word

    def test_a_step_without_a_path_falls_back_to_its_action(self) -> None:
        cards = _plan(
            [{"ui_path": "", "action": "Restart the router"}],
            "Intro. Now restart the router and wait.",
        )

        assert cards[0].path == ""
        assert _word_at("Intro. Now restart the router and wait.", cards[0].start) == (
            "restart"
        )


@pytest.mark.unit
@pytest.mark.req("REQ-VID-124")
class TestThePathShown:
    def test_placeholders_and_repeats(self) -> None:
        assert display_path("[your name] > iCloud", 4) == "Your name > iCloud"
        assert display_path(
            "A > Background App Refresh > Background App Refresh", 4
        ) == ("A > Background App Refresh")

    def test_a_long_path_keeps_its_end(self) -> None:
        assert display_path("Settings > General > Background App Refresh > Off", 4) == (
            "... > Background App Refresh > Off"
        )


@pytest.mark.unit
@pytest.mark.req("REQ-VID-124")
class TestLoading:
    def test_missing_or_unreadable_step_list_means_no_cards(
        self, tmp_path: Path
    ) -> None:
        options = {"not_before": 0.0, "min_sec": 0.5, "max_sec": 6.0}
        spoken = _spoken("Intro. Open Settings.")

        assert load_step_cards(tmp_path / "none.json", spoken, **options) == []
        bad = tmp_path / "step_list.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_step_cards(bad, spoken, **options) == []
        bad.write_text(json.dumps({"steps": "x"}), encoding="utf-8")
        assert load_step_cards(bad, spoken, **options) == []
        assert load_step_cards(bad, None, **options) == []

    def test_no_timings_is_logged(self, tmp_path: Path, caplog) -> None:
        steps = tmp_path / "step_list.json"
        steps.write_text(json.dumps({"steps": [{"ui_path": "A"}]}), encoding="utf-8")

        with caplog.at_level("INFO"):
            cards = load_step_cards(
                steps, None, not_before=0.0, min_sec=0.5, max_sec=6.0
            )

        assert cards == []
        assert "no word timings" in caplog.text

    def test_the_producer_reads_the_list_beside_the_script(
        self, tmp_path: Path
    ) -> None:
        from src.video.producer.steps import _step_cards

        script = tmp_path / "script.txt"
        (tmp_path / "step_list.json").write_text(
            json.dumps({"steps": [{"ui_path": "Settings", "action": "Open it"}]}),
            encoding="utf-8",
        )
        transcript = tmp_path / "whisper_transcript.json"
        transcript.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "words": [
                                {"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
                                for i, w in enumerate(
                                    "Intro here. Open Settings and wait a while.".split()
                                )
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        def ctx(enabled: bool) -> types.SimpleNamespace:
            video = types.SimpleNamespace(
                graphics=GraphicsSettings(enabled=enabled),
                hook_overlay=types.SimpleNamespace(enabled=True, duration_sec=1.0),
            )
            return types.SimpleNamespace(
                config=types.SimpleNamespace(video_settings=video),
                run_paths={
                    "script_file": script,
                    "whisper_transcript_file": transcript,
                },
            )

        assert _step_cards(ctx(False)) is None
        cards = _step_cards(ctx(True))
        assert cards and cards[0].path == "Settings"
        assert cards[0].start >= 1.0


def _chain() -> list[str]:
    return ["[0:v]copy[v_out]"]


@pytest.mark.unit
@pytest.mark.req("REQ-VID-124")
class TestTheOverlay:
    CARDS = [StepCard("Step 1 of 2", "Settings > General", 0.5, 1.0)]

    def test_off_or_empty_leaves_the_chain(self, tmp_path: Path) -> None:
        off = GraphicsSettings(enabled=False)
        on = GraphicsSettings(enabled=True)

        assert apply_step_card_overlay(
            _chain(), off, self.CARDS, 96, 1080, tmp_path
        ) == (_chain())
        assert apply_step_card_overlay(_chain(), on, [], 96, 1080, tmp_path) == _chain()

    def test_two_lines_per_card_gated_to_its_window(self, tmp_path: Path) -> None:
        cards = [*self.CARDS, StepCard("Step 2 of 2", "", 1.2, 1.8)]
        chain = apply_step_card_overlay(
            _chain(), GraphicsSettings(enabled=True), cards, 96, 1080, tmp_path
        )

        assert chain[-1] == "[v_steps]copy[v_out]"
        drawtexts = chain[-2].split(";")
        assert len(drawtexts) == 3
        assert "between(t\\,0.500\\,1.000)" in drawtexts[0]
        assert "between(t\\,1.200\\,1.800)" in drawtexts[2]

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_ffmpeg_draws_the_card_only_in_its_window(self, tmp_path: Path) -> None:
        chain = apply_step_card_overlay(
            _chain(), GraphicsSettings(enabled=True), self.CARDS, 96, 1080, tmp_path
        )
        out = tmp_path / "out.mp4"
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=0x808080:s=1080x1920:r=10:d=1.5",
                "-filter_complex", ";".join(chain), "-map", "[v_out]",
                "-pix_fmt", "yuv420p", str(out),
            ],
            check=True,
        )  # fmt: skip

        def darkest(t: float) -> int:
            raw = subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(out),
                    "-frames:v", "1", "-vf", "format=gray", "-f", "rawvideo", "-",
                ],
                check=True,
                capture_output=True,
            ).stdout  # fmt: skip
            return min(raw)

        assert darkest(0.2) > 100  # plain grey before the card
        assert darkest(0.75) < 100  # the card's dark box
        assert darkest(1.3) > 100  # gone after it


@pytest.mark.unit
@pytest.mark.req("REQ-VID-124")
def test_the_render_passes_its_cards_to_the_assembler() -> None:
    """Both hops a card takes, read at their call sites.

    The script step's list into the assembler, and the assembler's argument
    into the filter chain; the real path is a full render.
    """
    import inspect

    from src.video.assembler import core
    from src.video.producer import steps

    assemble = inspect.getsource(core.VideoAssembler.assemble_video)
    assert "apply_step_card_overlay(" in assemble
    assert "step_cards or []" in assemble
    assert "step_cards=_step_cards(ctx)" in inspect.getsource(steps)
