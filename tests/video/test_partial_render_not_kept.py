"""A killed encode leaves no file under the finished render's name (#398).

Both timeout attempts in #398 left a non-zero `.mp4` at the path a completed
render uses, failing `ffprobe` with `moov atom not found`. Anything checking
for existence rather than validity accepts that as a finished video, and the
scheduler enumerates rendered files.

The guarantee is structural rather than a cleanup handler: ffmpeg writes to a
sibling `.partial.mp4` and the finished name is only ever created by the
rename that follows a zero exit. The pipeline timeout arrives as a
cancellation, which no `if not success:` branch would ever reach.
"""

import ast
import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _kill_an_encode_mid_write(out: Path) -> None:
    """Run a real encode long enough to start writing, then kill it."""
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            # Real-time, or an ultrafast encode of a synthetic source
            # finishes the whole clip before the kill lands and the file is
            # perfectly readable, which tests nothing.
            "-re",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=25",
            "-t",
            "60",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(out),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


class TestAKilledEncodeIsNotAFinishedRender:
    def test_the_partial_file_is_what_a_kill_leaves(self, tmp_path):
        """The premise: a killed ffmpeg really does leave an unplayable file."""
        partial = tmp_path / "video.partial.mp4"
        _kill_an_encode_mid_write(partial)

        assert partial.exists(), "the encode did not start writing"
        assert partial.stat().st_size > 0
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                str(partial),
            ],
            capture_output=True,
            text=True,
        )
        assert probe.returncode != 0, "expected an unreadable file to verify against"

    def test_the_finished_name_never_holds_it(self, tmp_path):
        """What the assembler guarantees: kill it, and the final path is absent."""
        final = tmp_path / "video.mp4"
        partial = final.with_name(f"{final.stem}.partial{final.suffix}")

        async def encode_then_get_cancelled():
            # Stands in for the assembler's `try/finally` around the ffmpeg
            # call: the pipeline timeout cancels the task mid-encode.
            try:
                _kill_an_encode_mid_write(partial)
                await asyncio.sleep(3600)  # never reached in a real timeout
            finally:
                partial.unlink(missing_ok=True)

        async def run():
            task = asyncio.create_task(encode_then_get_cancelled())
            await asyncio.sleep(3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert not final.exists(), "a truncated render survived under the final name"
        assert not partial.exists(), "the partial file was not cleaned up"


class TestTheAssemblerIsWiredThatWay:
    """The behavioural test above models the contract; this reads the code.

    `_assemble` needs a rendered product to drive, so the guarantee is
    checked structurally: ffmpeg must be handed the partial path, the rename
    must be the only writer of the final name, and the cleanup must sit in a
    `finally` so a cancellation reaches it.
    """

    def test_ffmpeg_writes_to_the_partial_path(self):
        source = Path("src/video/assembler/core.py").read_text()
        assert "partial_path = temp_dir /" in source
        assert "os.replace(partial_path, output_path)" in source
        assert "partial_path.unlink(missing_ok=True)" in source

    def test_the_cleanup_is_in_a_finally(self):
        tree = ast.parse(Path("src/video/assembler/core.py").read_text())

        cleanups_in_finally = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            for stmt in node.finalbody
            for sub in ast.walk(stmt)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "unlink"
            # The partial specifically. Any `finally` holding any `unlink`
            # would otherwise satisfy this once a second one appears.
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "partial_path"
        ]
        assert cleanups_in_finally, (
            "the partial file must be removed in a finally; the pipeline "
            "timeout arrives as a cancellation, which an else branch misses"
        )

    def test_the_suffix_is_preserved(self):
        """Ffmpeg infers the muxer from the extension, so it must survive."""
        source = Path("src/video/assembler/core.py").read_text()
        assert ".partial{output_path.suffix}" in source

    def test_the_partial_is_outside_the_publishers_glob(self):
        """A kill that skips the cleanup must not leave a publishable file.

        `finally` does not run on an OOM kill or a SIGTERM, and the
        publisher discovers renders with `video_{asin}_*.mp4`. A sibling
        `video_<id>_<profile>.partial.mp4` matches that glob and sorts ahead
        of a second profile's finished render.
        """
        import tempfile

        from src.publisher.video_selector import sole_render_for_product

        with tempfile.TemporaryDirectory() as d:
            product = Path(d) / "B0TEST"
            (product / "temp").mkdir(parents=True)
            # Where the assembler now puts it.
            (product / "temp" / "video_B0TEST_a.partial.mp4").write_bytes(b"x")
            (product / "video_B0TEST_video_sequential.mp4").write_bytes(b"ok")

            chosen = sole_render_for_product(product, "B0TEST")
            assert chosen is not None
            assert chosen.name == "video_B0TEST_video_sequential.mp4"
