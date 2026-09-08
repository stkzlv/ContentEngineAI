"""The batch renderer's topic enumeration.

This is the part of `make topics-batch` that has broken twice, both times
silently enough that the run reported something plausible: once because NUL
bytes were routed through a command substitution, which strips them, and once
because the records were written to stdout, which the config chain has already
printed three lines to. Both are properties of the *handoff*, not of the topic
data, so they are what these tests pin.
"""

from pathlib import Path

import pytest
import yaml

from tools.enumerate_topics import enumerate_records, main, write_records

FIXTURE = [
    {"title": "Topic A", "description": "d1", "keywords": ["kw a"]},
    # Last entry deliberately has no keywords: its joined value is empty, and
    # NUL-*separated* output would end on a delimiter and lose it.
    {"title": "Topic B without keywords", "description": "d2"},
]


@pytest.fixture
def topics_file(tmp_path: Path) -> Path:
    path = tmp_path / "topics.yaml"
    path.write_text(yaml.safe_dump(FIXTURE, sort_keys=False))
    return path


class TestEnumeration:
    def test_every_topic_is_enumerated_including_a_keywordless_last_one(
        self, topics_file: Path
    ) -> None:
        records = enumerate_records(topics_file)

        # One root plus four fields each; the count is what the shell divides
        # by to get the topic total, so a dropped field silently loses a topic.
        assert len(records) == 1 + 4 * len(FIXTURE)
        assert records[2] == "Topic A"
        assert records[6] == "Topic B without keywords"
        assert records[8] == ""

    def test_the_product_id_is_the_one_the_producer_will_use(
        self, topics_file: Path
    ) -> None:
        """The shell locates each render by this id. A second implementation of
        the slug diverged on accents, on titles past the length cap and on
        titles normalising to nothing, and each divergence lost the render
        after its script step had been paid for.
        """
        from src.video.producer.topic_input import topic_product_id

        records = enumerate_records(topics_file)

        assert records[1] == topic_product_id("Topic A")
        assert records[5] == topic_product_id("Topic B without keywords")

    def test_the_root_comes_from_config_not_a_hardcoded_outputs(
        self, topics_file: Path
    ) -> None:
        from src.video.config import config

        assert enumerate_records(topics_file)[0] == str(config.global_output_root_path)


class TestTheHandoff:
    def test_every_field_is_terminated_not_separated(self, tmp_path: Path) -> None:
        """`mapfile -d ''` yields one element per terminator. Separating
        instead drops a trailing empty field, which is exactly a last topic
        with no keywords.
        """
        out = tmp_path / "records"
        write_records(["root", "id", "title", "desc", ""], out)

        raw = out.read_bytes()
        assert raw.endswith(b"\0")
        assert raw.split(b"\0")[:-1] == [b"root", b"id", b"title", b"desc", b""]

    def test_records_go_to_the_path_and_never_to_stdout(
        self, topics_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Importing the config chain prints to stdout, so stdout cannot carry
        the records: the first field would be that chatter plus the path, and
        every topic would then fail its directory check.
        """
        out = tmp_path / "records"

        assert main(["prog", str(topics_file), str(out)]) == 0

        assert out.read_bytes().split(b"\0")[0].decode() != ""
        assert "\0" not in capsys.readouterr().out

    def test_wrong_argument_count_is_refused(self) -> None:
        assert main(["prog", "only-one"]) == 2
