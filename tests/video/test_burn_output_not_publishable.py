"""A truncated caption-burn output must not be publishable (#411).

Both burn intermediates rename onto the final video only after success,
which is right; what that does not cover is a kill that never reaches the
rename -- an OOM or a SIGTERM, where no `finally` runs. Written beside the
output, `video_<id>_<profile>_pycaps.mp4` matches the publisher's
`video_{asin}_*.mp4` discovery glob and can sort ahead of a second
profile's finished render. Same hazard the assembler's partial file was
moved into `temp/` for (#398); this covers the burn.
"""

import ast
import re
import tempfile
from pathlib import Path

from src.publisher.video_selector import sole_render_for_product


class TestTheBurnOutputsLiveInTemp:
    """Structural half: both derivations read the temp run path.

    Driving a real kill mid-burn needs a render; the guarantee is the
    placement, so the code is read instead, the way
    `tests/video/test_partial_render_not_kept.py` does for the assembler.
    """

    def test_every_burn_output_derives_from_intermediate_base(self):
        """Every occurrence, not the first: a second burn site added later
        with a sibling placement must fail this, and `source.index` only
        ever examined occurrence one.
        """
        source = Path("src/video/producer/steps.py").read_text()
        for marker in ("_pycaps.mp4", "_ffmpeg_burn.mp4"):
            hits = [m.start() for m in re.finditer(re.escape(marker), source)]
            assert hits, f"{marker} no longer appears; update this test"
            for idx in hits:
                window = source[max(0, idx - 400) : idx]
                assert 'run_paths["intermediate_base"]' in window, (
                    f"a {marker} intermediate near offset {idx} is not "
                    "derived from the temp run path"
                )

    def test_no_burn_output_is_a_sibling_of_the_final_render(self):
        """The two sibling-placement spellings: `with_name` on the final
        path, and `final.parent / f"...suffix"`.
        """
        tree = ast.parse(Path("src/video/producer/steps.py").read_text())

        def _carries_suffix(node: ast.AST) -> bool:
            dump = ast.dump(node)
            return "_pycaps" in dump or "_ffmpeg_burn" in dump

        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "with_name"
                and any(_carries_suffix(arg) for arg in node.args)
            )
            or (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Div)
                and _carries_suffix(node.right)
                and ".parent" in ast.dump(node.left)
            )
        ]
        assert not offenders, (
            f"burn intermediates placed beside the render at lines "
            f"{offenders}; they belong in the temp directory"
        )


class TestTheGlobCannotSelectIt:
    def test_a_stranded_burn_file_in_temp_is_not_discovered(self):
        """What the placement buys: the publisher cannot pick it up."""
        with tempfile.TemporaryDirectory() as d:
            product = Path(d) / "B0TEST"
            (product / "temp").mkdir(parents=True)
            (product / "temp" / "video_B0TEST_slideshow_stock_pycaps.mp4").write_bytes(
                b"truncated"
            )
            (product / "temp" / "video_B0TEST_stock_ffmpeg_burn.mp4").write_bytes(
                b"truncated"
            )
            (product / "video_B0TEST_video_sequential.mp4").write_bytes(b"ok")

            chosen = sole_render_for_product(product)
            assert chosen is not None
            assert chosen.name == "video_B0TEST_video_sequential.mp4"

    def test_a_sibling_burn_file_would_have_been_discovered(self):
        """The hazard is real: beside the output, the glob matches it.

        This is the pre-change layout. If discovery ever stops matching
        the burn suffixes, the structural tests above become moot and this
        documents why they exist.
        """
        with tempfile.TemporaryDirectory() as d:
            product = Path(d) / "B0TEST"
            product.mkdir(parents=True)
            (product / "video_B0TEST_slideshow_stock_pycaps.mp4").write_bytes(
                b"truncated"
            )
            (product / "video_B0TEST_video_sequential.mp4").write_bytes(b"ok")

            chosen = sole_render_for_product(product)
            # Sorted order puts "slideshow_stock_pycaps" ahead of
            # "video_sequential": the selector returns the truncated file.
            assert chosen is not None
            assert chosen.name.endswith("_pycaps.mp4"), (
                "discovery no longer selects a sibling burn file; if that "
                "is deliberate, the structural tests above can be revisited"
            )
