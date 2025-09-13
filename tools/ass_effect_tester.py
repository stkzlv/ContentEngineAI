#!/usr/bin/env python3
"""ASS Effect Testing Tool
Tests ASS subtitle effects in isolation to debug rendering issues.
"""

import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASSEffectTester:
    """Test ASS subtitle effects with minimal FFmpeg commands."""

    def __init__(self):
        self.test_dir = Path("outputs/ass_tests")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def create_test_video(self, duration=10, resolution=(1080, 1920)):
        """Create a simple test video background."""
        test_video = self.test_dir / "test_background.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=blue:size={resolution[0]}x{resolution[1]}:duration={duration}:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(test_video),
        ]

        logger.info(f"Creating test background video: {test_video}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Failed to create test video: {result.stderr}")
            return None

        return test_video

    def create_test_ass_file(self, filename, effects_type="all"):
        """Create ASS file with specific effects for testing."""
        ass_content = [
            "[Script Info]",
            "Title: ASS Effects Test",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "WrapStyle: 1",
            "",
            "[V4+ Styles]",
            (
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, "
                "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
            ),
            (
                "Style: Default,Arial,48,&H00FFFFFF,&H00FFFFFF,&H00000000,"
                "&H80000000,-1,0,1,3,0,2,10,10,200,1"
            ),
            "",
            "[Events]",
            (
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
                "MarginV, Effect, Text"
            ),
        ]

        # Test different effects based on type
        if effects_type == "fade" or effects_type == "all":
            ass_content.extend(
                [
                    (
                        "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,"
                        "{\\fad(1000,1000)}FADE TEST - This text should fade in and out"
                    ),
                    (
                        "Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,"
                        "{\\fad(500,500)}FAST FADE - Quicker fade effect"
                    ),
                ]
            )

        if effects_type == "karaoke" or effects_type == "all":
            ass_content.extend(
                [
                    (
                        "Dialogue: 0,0:00:02.00,0:00:05.00,Default,,0,0,0,,"
                        "{\\kf100}KARAOKE {\\kf100}TEST {\\kf100}EFFECT"
                    ),
                    (
                        "Dialogue: 0,0:00:06.00,0:00:09.00,Default,,0,0,0,,"
                        "{\\k50}Word {\\k50}by {\\k50}word {\\k50}timing"
                    ),
                ]
            )

        if effects_type == "color" or effects_type == "all":
            ass_content.extend(
                [
                    (
                        "Dialogue: 0,0:00:03.00,0:00:05.00,Default,,0,0,0,,"
                        "{\\c&H0000FF&}RED TEXT {\\c&H00FF00&}GREEN TEXT "
                        "{\\c&HFFFF00&}CYAN TEXT"
                    ),
                    "Dialogue: 0,0:00:07.00,0:00:09.00,Default,,0,0,0,,"
                    "{\\3c&HFF0000&\\1c&H00FFFF&}COLORED OUTLINE TEST",
                ]
            )

        if effects_type == "position" or effects_type == "all":
            ass_content.extend(
                [
                    "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,"
                    "{\\pos(540,200)}TOP POSITION",
                    "Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,"
                    "{\\pos(540,960)}MIDDLE POSITION",
                    "Dialogue: 0,0:00:07.00,0:00:09.00,Default,,0,0,0,,"
                    "{\\pos(540,1700)}BOTTOM POSITION",
                ]
            )

        ass_file = self.test_dir / filename
        with open(ass_file, "w", encoding="utf-8") as f:
            f.write("\n".join(ass_content))

        logger.info(f"Created test ASS file: {ass_file}")
        return ass_file

    def test_ass_rendering(self, test_video, ass_file, output_name):
        """Test ASS rendering with the ass filter."""
        output_video = self.test_dir / f"{output_name}.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(test_video),
            "-vf",
            f"ass='{ass_file}'",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            str(output_video),
        ]

        logger.info(f"Testing ASS rendering: {output_video}")
        logger.info(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"✅ ASS test successful: {output_video}")
            # Get video info
            info_cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-show_format",
                "-show_streams",
                str(output_video),
            ]
            subprocess.run(info_cmd, capture_output=True, text=True)
            logger.info("Video info: Duration and streams created successfully")
            return output_video
        else:
            logger.error(f"❌ ASS test failed: {result.stderr}")
            return None

    def run_all_tests(self):
        """Run comprehensive ASS effect tests."""
        logger.info("=== Starting ASS Effect Testing ===")

        # Create test video
        test_video = self.create_test_video()
        if not test_video:
            logger.error("Failed to create test video, aborting tests")
            return

        # Test different effect types
        test_cases = [
            ("fade", "Test fade in/out effects"),
            ("karaoke", "Test karaoke timing effects"),
            ("color", "Test color change effects"),
            ("position", "Test positioning effects"),
            ("all", "Test all effects combined"),
        ]

        results = {}

        for effect_type, description in test_cases:
            logger.info(f"\n--- Testing: {effect_type} ---")
            logger.info(description)

            # Create ASS file for this test
            ass_file = self.create_test_ass_file(f"test_{effect_type}.ass", effect_type)

            # Test rendering
            output = self.test_ass_rendering(
                test_video, ass_file, f"test_{effect_type}"
            )
            results[effect_type] = output is not None

        # Summary
        logger.info("\n=== Test Results Summary ===")
        for effect_type, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{effect_type}: {status}")

        if all(results.values()):
            logger.info("\n🎉 All ASS effects are working correctly!")
            logger.info(
                "Issue may be in the production pipeline, not ASS rendering itself."
            )
        else:
            logger.info(
                "\n🚨 Some ASS effects failed - FFmpeg ASS support issue detected"
            )

        logger.info(f"\nTest files saved to: {self.test_dir}")
        return results


def main():
    tester = ASSEffectTester()
    results = tester.run_all_tests()

    # Return exit code based on results
    if all(results.values()):
        return 0  # All tests passed
    else:
        return 1  # Some tests failed


if __name__ == "__main__":
    exit(main())
