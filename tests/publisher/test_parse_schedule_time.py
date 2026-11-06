"""Explicit schedule times: offsets kept, bare times read in the schedule zone.

REQ-PUB-018, REQ-BAT-062.
"""

from datetime import UTC, datetime

import pytest

from src.publisher.schedule import parse_schedule_time


def test_offset_is_kept():
    dt = parse_schedule_time("2026-01-20 22:00:00+01:00", "UTC")
    assert dt.astimezone(UTC) == datetime(2026, 1, 20, 21, 0, tzinfo=UTC)


def test_z_suffix_is_utc():
    dt = parse_schedule_time("2026-01-20T22:00:00Z", "Europe/Madrid")
    assert dt.astimezone(UTC) == datetime(2026, 1, 20, 22, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("value", "utc_hour"),
    [
        ("2026-01-20 22:00:00", 21),  # winter, UTC+1
        ("2026-07-20T22:00", 20),  # summer, UTC+2
    ],
)
def test_bare_time_is_read_in_schedule_zone(value, utc_hour):
    dt = parse_schedule_time(value, "Europe/Madrid")
    assert dt.astimezone(UTC).hour == utc_hour


def test_bare_time_in_utc_zone_is_unchanged():
    dt = parse_schedule_time("2026-01-20 14:00", "UTC")
    assert dt.astimezone(UTC) == datetime(2026, 1, 20, 14, 0, tzinfo=UTC)


@pytest.mark.parametrize("value", ["tomorrow", "2026-13-01 10:00", ""])
def test_invalid_time_raises(value):
    with pytest.raises(ValueError, match="Invalid datetime format"):
        parse_schedule_time(value, "UTC")


def test_unknown_zone_raises():
    with pytest.raises(ValueError, match="Unknown schedule timezone"):
        parse_schedule_time("2026-01-20 14:00", "Mars/Olympus")


def test_resolver_logs_utc_and_schedule_zone(caplog):
    from src.publisher.schedule import resolve_schedule_time

    with caplog.at_level("INFO", logger="src.publisher.schedule"):
        dt = resolve_schedule_time("2026-01-20 22:00", "Europe/Madrid")
    assert dt.astimezone(UTC).hour == 21
    assert "2026-01-20 21:00 UTC (2026-01-20 22:00 Europe/Madrid)" in caplog.text


@pytest.mark.parametrize(
    "path", ["src/publisher/late/cli.py", "src/pipeline/phases/publishing.py"]
)
def test_single_and_batch_read_explicit_times_the_same_way(path):
    # Module/batch alignment: both resolve through the one reader, with the
    # recurring schedule's timezone.
    from src.utils.outputs_paths import get_project_root

    source = (get_project_root() / path).read_text()
    assert "resolve_schedule_time(" in source
    assert "schedule_config.timezone" in source
    assert "fromisoformat(schedule_time" not in source
