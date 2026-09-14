"""Importing a module does not load the render or scrape stack (#470).

A package's `__init__` runs whenever any submodule is imported, so eager
re-exports there decide what every process pays: `src/pipeline/config.py`
importing `topic_input` pulled `orchestration` -> `steps` -> Whisper and torch,
and importing `models` pulled `browser_functions` -> Botasaurus and Chromium.
Measured before the fix, `import src.pipeline.global_batch` loaded 2,487
modules in 3.0s and `--help` took 3.8s; after, 703 modules in 0.5s.

It also made the function-local imports in the leaf modules pointless, which is
the trap this file exists to keep shut: a deferral is only a deferral if
nothing else has already imported the module.

Driven in subprocesses, because the suite's own conftest imports most of this
tree before any test runs.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from src.utils.outputs_paths import get_project_root

REPO = get_project_root()

# Everything here costs seconds to import, or drags a browser or a model file
# along with it. None is needed to parse arguments or read a config.
HEAVY = ("torch", "whisper", "botasaurus", "playwright", "TTS", "google.cloud")

PROBE = """
import sys
import {module}
heavy = [
    name
    for name in {heavy!r}
    if any(m == name or m.startswith(name + ".") for m in sys.modules)
]
print("heavy:" + ",".join(heavy))
print("modules:" + str(len(sys.modules)))
"""


def closure_of(module: str) -> tuple[list[str], int]:
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(module=module, heavy=HEAVY)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    # Read by prefix: the module list is empty on a healthy run, and a bare
    # blank line is indistinguishable from the provider chatter some imports
    # log on the way past.
    lines = result.stdout.splitlines()
    heavy = next(line for line in reversed(lines) if line.startswith("heavy:"))
    count = next(line for line in reversed(lines) if line.startswith("modules:"))
    return (
        [name for name in heavy[len("heavy:") :].split(",") if name],
        int(count[len("modules:") :]),
    )


@pytest.mark.parametrize(
    "module",
    [
        "src.pipeline.global_batch",
        "src.scraper.amazon.models",
        "src.video.producer.state",
        "src.publisher.late.cli",
    ],
)
def test_the_heavy_stack_stays_out(module: str):
    heavy, _ = closure_of(module)
    assert not heavy, (
        f"importing {module} loaded {heavy}; a package __init__ re-export or a "
        "module-level import is pulling the render/scrape stack into every "
        "process that touches this module"
    )


def test_the_batch_closure_stays_small():
    """A count, so a re-export creeping back is visible before it is heavy.

    The bound is roughly 1.5x the measured 703, which leaves room for ordinary
    growth while catching a re-introduced eager chain (the old number was
    2,487).
    """
    _, count = closure_of("src.pipeline.global_batch")
    assert count < 1100, f"import closure grew to {count} modules"


def test_the_lazy_re_exports_still_resolve():
    """The names have to keep working, or this is a breaking change."""
    import src.scraper.amazon as amazon
    import src.video.producer as producer

    assert amazon.ProductData.__name__ == "ProductData"
    assert producer.PipelineContext.__name__ == "PipelineContext"
    assert callable(producer.create_video_for_product)
    assert "ProductData" in dir(amazon)
    with pytest.raises(AttributeError):
        getattr(amazon, "no_such_name")  # noqa: B009 - the point is the lookup
