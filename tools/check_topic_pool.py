"""Run the topic filter over the configured topic pool (REQ-VID-151).

Each topic gets the same grounded step-list call a render makes, and the
tool prints whether the topic would render or be dropped, and why. A render
drops a failing topic on its own; this shows which ones to take out of the
pool before their turn comes round.

Reads the pool the batch reads (`PIPELINE_TOPICS_FILE`, then `topics_file`,
then `topics:` in `config/pipeline.yaml`), or a topics file given as the
argument. Needs the LLM API key in the environment or `.env` (exits 2
without it). Exits 1 when any topic would be dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

# Invoked as `-m tools.check_topic_pool` from the repo root, where the cwd
# already supplies `src`; the insert covers a direct script invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def configured_pool(topics_file: Path | None) -> list[Any]:
    """The topics to check: the given file, or the batch's configured pool."""
    from src.video.producer.topic_input import load_topics_file

    if topics_file is not None:
        return load_topics_file(topics_file)
    import yaml

    from src.pipeline.config import DEFAULT_PIPELINE_CONFIG_PATH, _configured_topics

    raw = yaml.safe_load(DEFAULT_PIPELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    batch = (raw or {}).get("global_batch") or {}
    return _configured_topics(batch, DEFAULT_PIPELINE_CONFIG_PATH)


async def check_pool(specs: list[Any], api_key: str, settings: Any) -> list[str]:
    """One line per topic: `ok` or `drop`, the title, and the drop reason."""
    from src.ai.step_list import build_step_list, drop_reason

    lines = []
    for spec in specs:
        step_list = await build_step_list(
            spec.title, spec.description, api_key=api_key, settings=settings
        )
        reason = drop_reason(step_list, settings.max_steps)
        lines.append(
            f"drop  {spec.title}: {reason}" if reason else f"ok    {spec.title}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("topics_file", nargs="?", type=Path)
    args = parser.parse_args(argv)

    import os

    from dotenv import load_dotenv

    from src.utils.outputs_paths import get_project_root
    from src.video.config import config

    load_dotenv(get_project_root() / ".env")
    llm = config.llm_settings
    specs = configured_pool(args.topics_file)
    if not specs:
        print("No topics configured.")
        return 0
    api_key = os.environ.get(llm.api_key_env_var, "")
    if not api_key:
        # Without it every topic would read as dropped.
        print(f"{llm.api_key_env_var} is not set.", file=sys.stderr)
        return 2
    lines = asyncio.run(check_pool(specs, api_key, llm.topic_scripts.step_list))
    print("\n".join(lines))
    return 1 if any(line.startswith("drop") for line in lines) else 0


if __name__ == "__main__":
    raise SystemExit(main())
