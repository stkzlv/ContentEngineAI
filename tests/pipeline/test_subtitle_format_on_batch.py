"""`--subtitle-format` reaches the batch, not only the standalone producer.

The Module/Batch Alignment Rule: the producer CLI and `global_batch`
re-implement the same logic rather than one calling the other, so a flag added
to one is silently absent from the other. `--subtitle-engine` and the three
pycaps flags were on both; `--subtitle-format` was on the producer alone, so a
batch run had no way to ask for a format at all and took the global value
whatever the operator wanted.

Asserted through the same three points its siblings pass through -- the parser,
the config resolution, and the dotted override handed to the producer -- because
the flag is useless if it stops at any of them, and each is a separate site.
"""

from __future__ import annotations

import pytest

from src.pipeline.global_batch import create_argument_parser


@pytest.fixture
def parser():
    return create_argument_parser()


class TestTheFlagIsAccepted:
    def test_it_parses(self, parser):
        assert parser.parse_args(["--subtitle-format", "srt"]).subtitle_format == "srt"

    def test_it_rejects_an_unknown_format(self, parser):
        """Same choices as the producer's, so the two cannot drift apart."""
        with pytest.raises(SystemExit):
            parser.parse_args(["--subtitle-format", "vtt"])

    def test_it_defaults_to_unset(self, parser):
        """Unset must be distinguishable from a value, or YAML can never win."""
        assert parser.parse_args([]).subtitle_format is None

    def test_it_matches_the_producer_choices(self, parser):
        from src.video.producer.cli import create_argument_parser as producer_parser

        def choices(p, flag):
            return next(
                a.choices for a in p._actions if flag in (a.option_strings or [])
            )

        assert choices(parser, "--subtitle-format") == choices(
            producer_parser(), "--subtitle-format"
        )


def load(tmp_path, yaml_value=None, **cli):
    """The real loader, driven through a YAML file as it reads one."""
    import argparse

    import yaml

    from src.pipeline.config import load_global_batch_config

    section: dict = {} if yaml_value is None else {"subtitle_format": yaml_value}
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump({"global_batch": section}), encoding="utf-8")

    return load_global_batch_config(argparse.Namespace(**cli), str(path))


class TestItReachesTheProducer:
    def test_the_cli_value_lands_on_the_config(self, tmp_path):
        assert load(tmp_path, subtitle_format="srt").subtitle_format == "srt"

    def test_yaml_supplies_it_when_the_flag_is_absent(self, tmp_path):
        assert load(tmp_path, yaml_value="ass").subtitle_format == "ass"

    def test_the_flag_beats_yaml(self, tmp_path):
        config = load(tmp_path, yaml_value="ass", subtitle_format="srt")

        assert config.subtitle_format == "srt"

    def test_neither_leaves_it_unset(self, tmp_path):
        """Unset must stay None, or the global value can never apply."""
        assert load(tmp_path).subtitle_format is None

    def test_it_becomes_the_dotted_override_the_producer_reads(self):
        """The last hop, and the one a partial wiring would drop silently."""
        import ast
        from pathlib import Path

        source = Path("src/pipeline/global_batch.py").read_text(encoding="utf-8")
        keys = {
            node.slice.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and node.slice.value.startswith("subtitle_settings.")
        }

        assert "subtitle_settings.subtitle_format" in keys, (
            "the flag is parsed and stored but never handed to the producer, "
            "so it does nothing"
        )
