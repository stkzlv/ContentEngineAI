"""Any ISO media file passes the video signature check (REQ-SCR-088).

The check matched two exact headers, so a valid MP4 with another box size or
brand was rejected after download. Listing videos read from the page's image
data carry such headers (an `M4V ` brand in a 28-byte box).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from src.scraper.amazon.media_validator import verify_video_file

pytestmark = [pytest.mark.unit, pytest.mark.req("REQ-SCR-088")]

# What ffprobe reports for a usable clip, so the check reaches the signature.
_PROBE = json.dumps(
    {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "duration": "22.0",
            }
        ],
        "format": {"duration": "22.0"},
    }
)


@pytest.fixture(autouse=True)
def _probe_says_usable(monkeypatch):
    monkeypatch.setattr(
        "src.scraper.amazon.media_validator.subprocess.run",
        lambda cmd, **_k: subprocess.CompletedProcess(cmd, 0, _PROBE, ""),
    )


@pytest.mark.parametrize(
    "header",
    [
        b"\x00\x00\x00\x1cftypM4V \x00\x00\x00\x01",
        b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00",
        b"\x00\x00\x00\x18ftypmp41\x00\x00\x00\x00",
        b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00",
    ],
)
def test_an_iso_media_header_has_a_valid_signature(tmp_path, header):
    path = tmp_path / "clip.mp4"
    path.write_bytes(header + b"\x00" * 64)

    result = verify_video_file(path)

    assert result.validation_data["has_valid_signature"] is True
    assert not any("signature" in issue for issue in result.issues)


def test_html_is_still_refused(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"<!DOCTYPE html><html>")

    result = verify_video_file(path)

    assert result.validation_data["has_valid_signature"] is False
