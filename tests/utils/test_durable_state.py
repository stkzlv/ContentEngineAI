"""Durable tracking state survives cleanup and lives under outputs/state (#441).

Five files are irreplaceable after loss: the publish history (backs the
duplicate guard), the registry JSON/CSV (rows for cleaned product dirs exist
nowhere else), the day-N metrics store (figures age out of the provider's
retention permanently), and the schedule. The cleanup's preserve-list missed
two of them, and the boundary between deletable and durable was a pattern
list rather than a directory.
"""

from __future__ import annotations

import time
from pathlib import Path

from src.utils.outputs_paths import STATE_DIR_NAME, durable_state_path


class TestTheResolver:
    def test_a_legacy_root_copy_is_migrated(self, tmp_path):
        legacy = tmp_path / "publish_history.json"
        legacy.write_text('{"posts": {}}', encoding="utf-8")

        resolved = durable_state_path(tmp_path, "publish_history.json")

        assert resolved == tmp_path / STATE_DIR_NAME / "publish_history.json"
        assert resolved.read_text(encoding="utf-8") == '{"posts": {}}'
        assert not legacy.exists(), "the root copy must move, not fork"

    def test_an_existing_state_copy_is_never_clobbered(self, tmp_path):
        """A migration must not overwrite the converged file with a stale
        root copy that something recreated (a rollback, a hand copy).
        """
        state = tmp_path / STATE_DIR_NAME
        state.mkdir()
        (state / "schedule.json").write_text("new", encoding="utf-8")
        (tmp_path / "schedule.json").write_text("stale", encoding="utf-8")

        resolved = durable_state_path(tmp_path, "schedule.json")

        assert resolved.read_text(encoding="utf-8") == "new"
        assert (tmp_path / "schedule.json").exists(), "the stray is left for a human"

    def test_a_missing_outputs_root_plants_no_directories(self, tmp_path):
        """Resolving a default path on a fresh clone (dry run, test
        collection) must not create outputs trees as a side effect.
        """
        absent = tmp_path / "nowhere"

        resolved = durable_state_path(absent, "post_metrics.json")

        assert resolved == absent / STATE_DIR_NAME / "post_metrics.json"
        assert not absent.exists()

    def test_an_existing_outputs_root_gets_the_state_dir(self, tmp_path):
        durable_state_path(tmp_path, "post_metrics.json")

        assert (tmp_path / STATE_DIR_NAME).is_dir()

    def test_a_read_only_tree_still_serves_the_legacy_copy(self, tmp_path):
        """Migration is an optimization for writers, not a precondition for
        reading: a backup snapshot mounted read-only must stay readable.
        """
        import stat

        legacy = tmp_path / "publish_history.json"
        legacy.write_text('{"posts": {}}', encoding="utf-8")
        tmp_path.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            resolved = durable_state_path(tmp_path, "publish_history.json")
        finally:
            tmp_path.chmod(stat.S_IRWXU)

        assert resolved == legacy, "a failed migration must fall back to the data"
        assert legacy.read_text(encoding="utf-8") == '{"posts": {}}'


class TestTheModulesResolveUnderState:
    """Each durable-state module funnels through one path function; all of
    them must agree on the state directory or readers and writers split.
    """

    def test_tracking(self, tmp_path):
        from src.publisher.tracking import get_tracking_path

        assert get_tracking_path(tmp_path).parent.name == STATE_DIR_NAME

    def test_registry_both_formats(self, tmp_path):
        from src.publisher.product_registry import get_registry_path

        assert get_registry_path(tmp_path, "json").parent.name == STATE_DIR_NAME
        assert get_registry_path(tmp_path, "csv").parent.name == STATE_DIR_NAME

    def test_metrics(self, tmp_path):
        from src.publisher.analytics import metrics_path

        assert metrics_path(tmp_path).parent.name == STATE_DIR_NAME

    def test_schedule_cleanup_reads_the_same_place(self, tmp_path):
        """cleanup.py builds the schedule path itself; it must match what
        ScheduleManager writes or occupied slots read as free.
        """
        import json

        from src.publisher.cleanup import get_schedule_entry

        state = tmp_path / STATE_DIR_NAME
        state.mkdir()
        (state / "schedule.json").write_text(
            json.dumps(
                {"entries": [{"product_id": "B0TEST00001", "platforms": ["youtube"]}]}
            ),
            encoding="utf-8",
        )

        entry = get_schedule_entry("B0TEST00001", "youtube", tmp_path)
        assert entry is not None, "cleanup.py read a different schedule location"

    def test_tracking_round_trip_migrates(self, tmp_path):
        """A tree from before the split reads back its history unchanged."""
        import json

        from src.publisher.tracking import is_already_published, load_tracking

        (tmp_path / "publish_history.json").write_text(
            json.dumps(
                {"posts": {"B0TEST00001:youtube": {"post_id": "p1"}}},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        data = load_tracking(tmp_path)
        assert "B0TEST00001:youtube" in data["posts"]
        assert is_already_published("B0TEST00001", "youtube", tmp_path)


class TestCleanupPreservesDurableState:
    def _config_on(self, tmp_path):
        from src.video.config_adapter import load_video_config_modular

        config = load_video_config_modular()
        config.global_output_root_path = tmp_path
        return config

    def test_every_durable_name_is_preserved_by_pattern(self):
        """The preserve-list covers all five names at the root (legacy trees)
        and everything under state/ -- asserted through the real matcher,
        not a re-implementation of it.
        """
        from src.video.config_adapter import load_video_config_modular

        config = load_video_config_modular()
        root = config.global_output_root_path
        durable = [
            "publish_history.json",
            "published_products.json",
            "published_products.csv",
            "schedule.json",
            "post_metrics.json",
        ]
        for name in durable:
            assert config._should_preserve(root / name), f"{name} unprotected at root"
            assert config._should_preserve(
                root / STATE_DIR_NAME / name
            ), f"{name} unprotected under state/"
        assert config._should_preserve(root / STATE_DIR_NAME)

    def test_a_real_cleanup_run_preserves_state_files(self, tmp_path):
        """Driven through the actual non-dry cleanup, because the shipped
        cleaner had never run for real and its preserve-list rotted
        unexercised.
        """
        state = tmp_path / STATE_DIR_NAME
        state.mkdir(parents=True)
        keep = state / "post_metrics.json"
        keep.write_text("[]", encoding="utf-8")
        junk = tmp_path / "stray.dat"
        junk.write_text("x", encoding="utf-8")
        old = time.time() - 60 * 60 * 24 * 400
        import os

        os.utime(keep, (old, old))
        os.utime(junk, (old, old))

        config = self._config_on(tmp_path)
        report = config.cleanup_outputs_directory(dry_run=False)

        assert keep.exists(), "cleanup removed a durable state file"
        assert not junk.exists(), "cleanup left the aged stray, so it never ran"
        assert report["statistics"]["files_removed"] >= 1
