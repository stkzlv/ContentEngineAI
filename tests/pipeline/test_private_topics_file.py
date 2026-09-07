"""The topic pool can live outside the repository.

A channel's topic titles are its editorial line, so the bundled
`config/pipeline.yaml` ships a generic pair and an installation points
`topics_file` -- or `PIPELINE_TOPICS_FILE`, which needs no tracked file
edited -- at a pool of its own. The env var exists because the shipped value
has to stay null: a committed path to a gitignored file fails on a fresh
clone.

A configured path that does not exist raises. Falling back to the bundled
block would render the wrong pool with nothing logged, which is the
silent-fallback class this repo refuses everywhere else.
"""

from __future__ import annotations

import argparse

import pytest
import yaml

from src.pipeline.config import load_global_batch_config
from src.video.producer.topic_input import TopicInputError

# `PIPELINE_TOPICS_FILE` is cleared for every test by the root conftest, so a
# machine that has a pool configured does not decide what these tests read.

BUNDLED = [
    {"title": "Bundled one", "description": "d", "keywords": ["k"]},
    {"title": "Bundled two", "description": "d", "keywords": ["k"]},
]
PRIVATE = [
    {"title": "Private one", "description": "d", "keywords": ["k"]},
    {"title": "Private two", "description": "d", "keywords": ["k"]},
    {"title": "Private three", "description": "d", "keywords": ["k"]},
]


def write_config(tmp_path, **global_batch) -> str:
    """A config directory holding `pipeline.yaml`, as the repo lays it out."""
    cfg = tmp_path / "config"
    cfg.mkdir(exist_ok=True)
    path = cfg / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "global_batch": {
                    "topics": BUNDLED,
                    "topics_per_run": 3,
                    **global_batch,
                }
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def write_pool(tmp_path, topics=None, name="topics.private.yaml"):
    cfg = tmp_path / "config"
    cfg.mkdir(exist_ok=True)
    path = cfg / name
    path.write_text(yaml.safe_dump(topics or PRIVATE), encoding="utf-8")
    return path


def titles(config) -> set[str]:
    return {t.title for t in config.topics}


@pytest.mark.unit
class TestTheKeyReplacesTheBundledBlock:
    def test_the_pool_is_read_from_the_named_file(self, tmp_path):
        write_pool(tmp_path)
        config = load_global_batch_config(
            argparse.Namespace(),
            write_config(tmp_path, topics_file="topics.private.yaml"),
        )
        assert titles(config) == {"Private one", "Private two", "Private three"}

    def test_the_bundled_block_is_not_merged_in(self, tmp_path):
        write_pool(tmp_path)
        config = load_global_batch_config(
            argparse.Namespace(),
            write_config(tmp_path, topics_file="topics.private.yaml"),
        )
        assert not titles(config) & {"Bundled one", "Bundled two"}

    def test_without_the_key_the_bundled_block_stands(self, tmp_path):
        write_pool(tmp_path)
        config = load_global_batch_config(argparse.Namespace(), write_config(tmp_path))
        assert titles(config) == {"Bundled one", "Bundled two"}

    def test_a_relative_path_resolves_against_the_config_directory(self, tmp_path):
        """Not against the working directory: the pool travels with the config."""
        write_pool(tmp_path, name="elsewhere.yaml")
        (tmp_path / "elsewhere.yaml").write_text(
            yaml.safe_dump([{"title": "Wrong directory"}]), encoding="utf-8"
        )
        config = load_global_batch_config(
            argparse.Namespace(), write_config(tmp_path, topics_file="elsewhere.yaml")
        )
        assert "Wrong directory" not in titles(config)

    def test_an_absolute_path_is_taken_as_given(self, tmp_path):
        pool = write_pool(tmp_path, name="absolute.yaml")
        config = load_global_batch_config(
            argparse.Namespace(), write_config(tmp_path, topics_file=str(pool))
        )
        assert titles(config) == {"Private one", "Private two", "Private three"}


@pytest.mark.unit
class TestTheEnvironmentVariableWins:
    def test_it_overrides_an_unset_key(self, tmp_path, monkeypatch):
        """The shipped key is null, so this is the usual way it is set."""
        write_pool(tmp_path)
        monkeypatch.setenv("PIPELINE_TOPICS_FILE", "topics.private.yaml")
        config = load_global_batch_config(argparse.Namespace(), write_config(tmp_path))
        assert titles(config) == {"Private one", "Private two", "Private three"}

    def test_it_overrides_a_set_key(self, tmp_path, monkeypatch):
        write_pool(tmp_path)
        write_pool(tmp_path, [{"title": "From the key"}], name="from_the_key.yaml")
        monkeypatch.setenv("PIPELINE_TOPICS_FILE", "topics.private.yaml")
        config = load_global_batch_config(
            argparse.Namespace(),
            write_config(tmp_path, topics_file="from_the_key.yaml"),
        )
        assert "From the key" not in titles(config)

    def test_an_empty_value_is_not_a_setting(self, tmp_path, monkeypatch):
        """A blank `PIPELINE_TOPICS_FILE=` in a .env must not mean a path."""
        monkeypatch.setenv("PIPELINE_TOPICS_FILE", "")
        config = load_global_batch_config(argparse.Namespace(), write_config(tmp_path))
        assert titles(config) == {"Bundled one", "Bundled two"}


@pytest.mark.unit
class TestAMissingPoolIsRefused:
    def test_a_named_file_that_is_absent_raises(self, tmp_path):
        with pytest.raises(TopicInputError, match="Could not read topics file"):
            load_global_batch_config(
                argparse.Namespace(), write_config(tmp_path, topics_file="gone.yaml")
            )

    def test_the_env_var_pointing_nowhere_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PIPELINE_TOPICS_FILE", "gone.yaml")
        with pytest.raises(TopicInputError, match="Could not read topics file"):
            load_global_batch_config(argparse.Namespace(), write_config(tmp_path))

    def test_a_malformed_pool_raises(self, tmp_path):
        cfg = tmp_path / "config"
        cfg.mkdir(exist_ok=True)
        (cfg / "bad.yaml").write_text("- title: [", encoding="utf-8")
        with pytest.raises(TopicInputError):
            load_global_batch_config(
                argparse.Namespace(), write_config(tmp_path, topics_file="bad.yaml")
            )

    def test_a_pool_entry_without_a_title_raises(self, tmp_path):
        write_pool(tmp_path, [{"description": "no title here"}])
        with pytest.raises(TopicInputError):
            load_global_batch_config(
                argparse.Namespace(),
                write_config(tmp_path, topics_file="topics.private.yaml"),
            )


@pytest.mark.unit
class TestTheShippedConfig:
    def test_the_key_ships_unset(self):
        """A committed path to a gitignored file breaks a fresh clone."""
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        shipped = yaml.safe_load((repo / "config" / "pipeline.yaml").read_text())
        assert shipped["global_batch"].get("topics_file") is None

    def test_the_example_pool_is_valid(self):
        from pathlib import Path

        from src.video.producer.topic_input import load_topics_file

        repo = Path(__file__).resolve().parents[2]
        specs = load_topics_file(repo / "config" / "topics.private.yaml.example")
        assert len(specs) >= 2
        assert all(spec.title for spec in specs)

    def test_the_private_pattern_is_gitignored(self):
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        assert "*.private.yaml" in (repo / ".gitignore").read_text()
