"""Enumerate a topics file as NUL-terminated records for the batch renderer.

Written as a module rather than a heredoc inside the shell script because this
is the part that has broken twice: once by routing NULs through a command
substitution, which strips them, and once by writing to stdout, which the
config chain has already printed three lines to by the time this runs. Both
failures were silent in the sense that mattered -- the run reported no topics,
or reported every topic as missing its directory after paying for its script
step. As a module it is directly testable, and ``tests/tools`` does so.

Output is written to a path, never to stdout, and every field is terminated
rather than separated: a trailing empty field -- a last topic with no keywords
-- would otherwise end the stream on a delimiter and be dropped by ``mapfile``.

Record layout: the output root once, then four fields per topic (product id,
title, description, comma-joined keywords).
"""

from __future__ import annotations

import sys
from pathlib import Path


def enumerate_records(topics_file: Path) -> list[str]:
    """Return the output root followed by four fields per topic."""
    from src.video.config import config
    from src.video.producer.topic_input import load_topics_file, topic_product_id

    specs = load_topics_file(topics_file)
    records = [str(config.global_output_root_path)]
    for spec in specs:
        records += [
            topic_product_id(spec.title),
            spec.title,
            spec.description,
            ", ".join(spec.keywords),
        ]
    return records


def write_records(records: list[str], destination: Path) -> None:
    """Write NUL-*terminated* records, so a trailing empty field survives.

    A NUL inside a field is refused rather than written. It is reachable --
    a YAML escape in a double-quoted scalar produces one -- and it
    would terminate its field early, shifting every later field by one: the
    reader would take a description as a keyword list and a product id as a
    title, then render that. Failing here costs nothing; the alternative is a
    fully paid, silently wrong render.
    """
    for field in records:
        if "\0" in field:
            raise ValueError(
                "a topic field contains a NUL byte, which is the record "
                f"delimiter: {field!r}"
            )
    destination.write_bytes(b"".join(f.encode() + b"\0" for f in records))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <topics.yaml> <output-path>", file=sys.stderr)
        return 2
    write_records(enumerate_records(Path(argv[1])), Path(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
