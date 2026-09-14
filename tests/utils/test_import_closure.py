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


@pytest.mark.parametrize(
    "module",
    ["src.ai.description_generator", "src.video.producer.steps"],
)
def test_a_first_import_of_the_cycle_succeeds(module: str):
    """Each of these was the head of an import cycle.

    `core_models` imports `PlatformMetadataSettings` from the
    platform-metadata package, whose `__init__` imported `base`, which imports
    `description_generator`, which imports the video config. Either module
    imported first died on a partially initialised module -- on main
    `description_generator` already did, and the producer's eager `__init__`
    was ordering around it for `steps`.

    A fresh interpreter is the whole point: the suite's conftest imports this
    tree before any test runs, so in-process nothing can reproduce it. These
    two cannot ride on the heavy-stack test above, which they legitimately
    fail.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"importing {module} first fails; an eager re-export in a package "
        f"__init__ has closed the cycle again:\n{result.stderr[-1500:]}"
    )


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


LAZY_PACKAGES = [
    "src.scraper.amazon",
    "src.video.producer",
    "src.ai.platform_metadata",
]


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_every_mapped_name_resolves(package: str):
    """The maps are the only new machinery here, and no gate reads them.

    mypy reads the `TYPE_CHECKING` block instead and ruff does not look, so a
    name pointed at a module that does not provide it passes lint, types and
    the rest of the suite, then raises `ImportError` on the one runtime path
    that uses it. For the two package-level maps nothing in the repo imports
    through them at all, so a wrong entry waits for an outside consumer; the
    platform-metadata map is reached on a render, by `steps.py` and by the
    batch generator, after the script, voiceover and assembly are paid for.

    An entry pointed at a module that merely re-imports the name is not
    caught here: it serves the identical object, and the maps legitimately
    re-export already (the media extractor's names are defined in
    `image_utils` and `video_extractor`), so there is no rule separating the
    two from the outside. What it would cost is what the test below measures.
    """
    import importlib

    module = importlib.import_module(package)
    wrong: list[str] = []
    for name, submodule in module._EXPORTS.items():
        target = importlib.import_module(f"{package}.{submodule}")
        if not hasattr(target, name):
            wrong.append(f"{name}: {submodule} does not provide it")
            continue
        if getattr(module, name) is not getattr(target, name):
            wrong.append(f"{name}: resolves to a different object than {submodule}'s")
    assert not wrong, f"{package} _EXPORTS entries are wrong: {wrong}"


# Names that must cost nothing heavy: the data models and the state helpers a
# config read, a batch plan or a resume reaches for. Listed by name rather
# than by the submodule they are mapped to, because the mapping is what is
# under test -- `ProductData` re-pointed at `scraper`, the spelling seventeen
# modules used before this branch, serves the same class and loads Botasaurus
# with it. The rest of both packages calls those libraries directly and is
# expected to load them.
_MUST_STAY_LIGHT = {
    "src.scraper.amazon": (
        "ProductData",
        "SearchParameters",
        "SerpProductInfo",
        "SearchParameterBuilder",
    ),
    "src.video.producer": (
        "PipelineContext",
        "PipelineError",
        "InsufficientMediaError",
        "VALID_STEPS",
        "get_video_run_paths",
        "_load_pipeline_state",
        "setup_logging",
        "validate_media_requirements",
        "load_artifacts_for_step",
    ),
}

RESOLVE_PROBE = """
import importlib, sys
package = importlib.import_module({package!r})
for name in {light_names!r}:
    getattr(package, name)
loaded = [
    n for n in {heavy!r}
    if any(m == n or m.startswith(n + ".") for m in sys.modules)
]
print("heavy:" + ",".join(loaded))
print("modules:" + str(len(sys.modules)))
"""


@pytest.mark.parametrize("package", ["src.scraper.amazon", "src.video.producer"])
def test_resolving_the_light_names_stays_light(package: str):
    """Resolving a name must cost only what that name needs.

    The map test above cannot tell a deliberate re-export from an entry
    pointed at a heavier module that happens to import the name, because both
    serve the identical object. This is the difference that matters: the
    lookup itself must not drag the render or scrape stack in.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            RESOLVE_PROBE.format(
                package=package,
                light_names=_MUST_STAY_LIGHT[package],
                heavy=HEAVY,
            ),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    heavy = next(
        line
        for line in reversed(result.stdout.splitlines())
        if line.startswith("heavy:")
    )
    loaded = [name for name in heavy[len("heavy:") :].split(",") if name]
    assert not loaded, (
        f"resolving {package}'s light names loaded {loaded}; an _EXPORTS entry "
        "points at a module that carries the heavy stack"
    )
