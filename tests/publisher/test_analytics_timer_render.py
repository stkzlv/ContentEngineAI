"""The rendered systemd units say what they must, and refuse when they cannot.

``install-timer.sh --render-only`` writes the units to a directory and touches
neither systemd nor $HOME, which is what lets these run in pytest without a
mocked ``systemctl``. Mocking it would test the mock.

Every assertion here anchors to a directive (``^ExecStart=``) rather than to a
bare word. The templates carry comments explaining why a directive is absent,
so an unanchored search finds the explanation and passes while the unit is
wrong.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "deploy" / "install-timer.sh"
SERVICE = "contentengineai-analytics.service"
TIMER = "contentengineai-analytics.timer"
FAILED = "contentengineai-analytics-failed.service"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or not INSTALLER.exists(),
    reason="needs bash and deploy/install-timer.sh",
)


def base_env(**overrides):
    """A complete setting environment, so no local file can reach the render.

    Every one of these keys must be set. The installer falls back to
    ``deploy/schedule.env`` for anything the environment does not carry, and
    that file is gitignored, so a partial environment makes these tests pass
    or fail by machine -- green on CI, red on a developer's box, for a reason
    that has nothing to do with the code.
    """
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(Path.home()),
        # Skip the `make print-python` probe: this interpreter is already the
        # project's, and the probe is not what these tests are about.
        "PYTHON": sys.executable,
        "REPO_DIR": str(REPO_ROOT),
        "ON_CALENDAR": "daily",
        "RANDOMIZED_DELAY_SEC": "900",
        "TIMEOUT_START_SEC": "30min",
        "NOTIFY_ON_FAILURE": "1",
    }
    env.update({k: str(v) for k, v in overrides.items()})
    return env


def render(out_dir, **env_overrides):
    """Render into out_dir. Returns the CompletedProcess."""
    return subprocess.run(
        ["bash", str(INSTALLER), "--render-only", str(out_dir)],
        capture_output=True,
        text=True,
        env=base_env(**env_overrides),
        cwd=str(REPO_ROOT),
    )


def directive(path, name):
    """Return the value of a systemd directive, ignoring comment lines."""
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith(f"{name}="):
            return stripped.split("=", 1)[1]
    return None


class TestRenderProducesUsableUnits:
    def test_all_three_units_are_written(self, tmp_path):
        result = render(tmp_path)

        assert result.returncode == 0, result.stderr
        assert {p.name for p in tmp_path.iterdir()} == {SERVICE, TIMER, FAILED}

    def test_no_placeholder_survives(self, tmp_path):
        """A half-rendered unit is worse than no unit.

        It installs cleanly and then fails, or silently never fires, once a
        day while the figures it exists to capture expire.
        """
        render(tmp_path)

        for unit in sorted(tmp_path.iterdir()):
            leftover = re.findall(r"@[A-Z][A-Z0-9_]*@", unit.read_text())
            assert not leftover, f"{unit.name} still holds {leftover}"

    def test_execstart_names_an_absolute_interpreter(self, tmp_path):
        """The documented trap, asserted.

        A user service has no pyenv shims on its PATH, and `poetry run python`
        resolves the base interpreter because virtualenvs.create is false.
        """
        render(tmp_path)

        exec_start = directive(tmp_path / SERVICE, "ExecStart")
        assert exec_start.startswith("/")
        assert "poetry" not in exec_start

    def test_execstart_carries_no_limit_flag(self, tmp_path):
        """The sweep size stays single-sourced in config/publisher.yaml.

        With a --limit here, editing the YAML would change `make analytics`
        and not the scheduled run, and nothing would report the divergence.
        """
        render(tmp_path)

        assert "--limit" not in directive(tmp_path / SERVICE, "ExecStart")

    def test_working_directory_is_the_repo(self, tmp_path):
        """The cwd decides where the figures land.

        --outputs-dir defaults to a relative path, so a wrong working
        directory writes them elsewhere and still exits 0.
        """
        render(tmp_path)

        assert directive(tmp_path / SERVICE, "WorkingDirectory") == str(REPO_ROOT)

    def test_start_timeout_is_set(self, tmp_path):
        """The start timeout is disabled by default for Type=oneshot units.

        Left disabled, a hung sweep activates forever, systemd drops every
        later firing, and the unit never reaches "failed" -- so the failure
        handler never runs either. A stuck sweep looks like a working one.
        """
        render(tmp_path)

        timeout = directive(tmp_path / SERVICE, "TimeoutStartSec")
        assert timeout not in (None, "", "infinity")

    def test_timer_is_persistent(self, tmp_path):
        """A window slept through must run on the next boot, not be skipped.

        Retention is finite, so a missed sweep is a permanent hole rather
        than a late reading.
        """
        render(tmp_path)

        assert directive(tmp_path / TIMER, "Persistent") == "true"

    def test_timer_has_a_non_empty_schedule(self, tmp_path):
        """The schedule must not render empty.

        An empty OnCalendar= resets the timer list (systemd.timer(5)), giving
        a timer that loads, enables, lists, and never fires.
        """
        render(tmp_path)

        assert directive(tmp_path / TIMER, "OnCalendar")


class TestRenderRefusesRatherThanInstallSomethingBroken:
    def test_unsubstituted_placeholder_aborts_and_writes_nothing(self, tmp_path):
        """Injecting an unknown placeholder must stop the render.

        ``REPO_DIR`` has to be passed explicitly. The copied installer derives
        its default from its own location, which here is a temp directory, so
        without it the script aborts at the earlier import check and both
        assertions below pass on the wrong abort -- leaving the placeholder
        guard untested while looking tested.
        """
        scratch = tmp_path / "deploy"
        shutil.copytree(REPO_ROOT / "deploy", scratch)
        template = scratch / f"{TIMER}.in"
        template.write_text(template.read_text() + "\nX=@NOT_A_REAL_KEY@\n")
        out = tmp_path / "out"

        result = subprocess.run(
            ["bash", str(scratch / "install-timer.sh"), "--render-only", str(out)],
            capture_output=True,
            text=True,
            env=base_env(),
            cwd=str(REPO_ROOT),
        )

        # Pin the reason, not just the exit code, so a different abort cannot
        # satisfy this test.
        assert "placeholder" in result.stderr, result.stderr

        assert result.returncode != 0
        assert not out.exists() or not list(out.iterdir())

    def test_interpreter_that_cannot_import_is_refused(self, tmp_path):
        """Existing is not enough; the check has to import.

        An interpreter that runs but cannot import the project is the exact
        documented failure, and it would surface once a day, unattended.
        """
        result = render(tmp_path, PYTHON="/usr/bin/python3")

        if result.returncode == 0:
            pytest.skip("system python3 can import the project here")
        assert "cannot import" in result.stderr
        assert not list(tmp_path.iterdir())


class TestFailureReportingIsOptional:
    def test_notify_off_omits_both_the_unit_and_the_hook(self, tmp_path):
        """A dangling OnFailure= would point at a unit that is not installed."""
        render(tmp_path, NOTIFY_ON_FAILURE=0)

        assert not (tmp_path / FAILED).exists()
        assert directive(tmp_path / SERVICE, "OnFailure") is None

    def test_notify_on_wires_the_hook_to_the_installed_unit(self, tmp_path):
        render(tmp_path, NOTIFY_ON_FAILURE=1)

        assert directive(tmp_path / SERVICE, "OnFailure") == FAILED
        assert (tmp_path / FAILED).exists()

    def test_the_handler_unit_also_bounds_its_start(self, tmp_path):
        """The handler is Type=oneshot too, so it has the same default.

        It blocks on nothing today, but it is the script most likely to grow
        a channel that can -- a webhook, a mail command -- and a failure
        reporter that activates forever is the worst place to find that out.
        """
        render(tmp_path)

        timeout = directive(tmp_path / FAILED, "TimeoutStartSec")
        assert timeout not in (None, "", "infinity")
