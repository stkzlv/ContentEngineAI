#!/usr/bin/env bash
# Render and install the daily analytics systemd user timer.
#
# Invoked by `make install-analytics-timer`. Deliberately not run through
# poetry: a user service inherits no login shell, so its PATH has no pyenv
# shims, and poetry.toml sets virtualenvs.create = false, which makes
# `poetry run python` resolve the base interpreter rather than the project
# environment. The unit must name the interpreter absolutely.
#
#   install-timer.sh                    render, install, enable, run once
#   install-timer.sh --uninstall        disable and remove the units
#   install-timer.sh --render-only DIR  render into DIR and stop
#   install-timer.sh --no-run           reinstall without sweeping
#
# --render-only touches neither systemd nor $HOME, which is what gives the
# renderer a test surface that does not need a mocked systemctl.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR_DEFAULT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
ENV_FILE="$SCRIPT_DIR/schedule.env"

UNIT_PREFIX="contentengineai-analytics"
SERVICE="$UNIT_PREFIX.service"
TIMER="$UNIT_PREFIX.timer"
FAILED="$UNIT_PREFIX-failed.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
NC=$'\033[0m'
die() {
    printf '%s\n' "${RED}Error: $*${NC}" >&2
    exit 1
}
info() { printf '%s\n' "${BLUE}$*${NC}"; }
ok() { printf '%s\n' "${GREEN}$*${NC}"; }
warn() { printf '%s\n' "${YELLOW}$*${NC}" >&2; }

RENDER_ONLY=""
NO_RUN=""
case "${1-}" in
--uninstall)
    systemctl --user disable --now "$TIMER" 2>/dev/null || true
    rm -f "$UNIT_DIR/$TIMER" "$UNIT_DIR/$SERVICE" "$UNIT_DIR/$FAILED"
    systemctl --user daemon-reload
    # Otherwise a unit that failed before removal lingers in --failed.
    systemctl --user reset-failed "$SERVICE" 2>/dev/null || true
    ok "Removed $TIMER, $SERVICE and $FAILED from $UNIT_DIR."
    ok "outputs/ and deploy/schedule.env are untouched."
    exit 0
    ;;
--render-only)
    RENDER_ONLY="${2-}"
    [ -n "$RENDER_ONLY" ] || die "--render-only needs a directory"
    ;;
--no-run)
    # Reinstall without sweeping. Editing the schedule needs a re-render and
    # a timer restart, and a sweep costs about one API request per measured
    # post; charging that for a one-line schedule change would push a few
    # edits past the provider's hourly cap for no reading worth having.
    NO_RUN=1
    ;;
"") ;;
*) die "unknown argument: $1" ;;
esac

# --- 1. machine-specific settings -------------------------------------------
# Precedence is environment, then the file, then the built-in default. Sourcing
# would otherwise clobber an exported value, which costs a one-off override
# (NOTIFY_ON_FAILURE=0 make install-analytics-timer) and, worse, makes the
# renderer's behaviour depend on whether a local file happens to exist: the
# tests inject through the environment and would pass or fail by machine.
SETTINGS="REPO_DIR PYTHON ON_CALENDAR RANDOMIZED_DELAY_SEC TIMEOUT_START_SEC NOTIFY_ON_FAILURE"
for name in $SETTINGS; do
    eval "exported_$name=\${$name-}"
done

if [ -f "$ENV_FILE" ]; then
    # shellcheck source=/dev/null
    . "$ENV_FILE"
else
    info "No deploy/schedule.env; using defaults for every value."
    info "Copy deploy/schedule.env.example to override any of them."
fi

for name in $SETTINGS; do
    eval "was_exported=\$exported_$name"
    if [ -n "$was_exported" ]; then
        eval "$name=\$was_exported"
    fi
done

REPO_DIR="${REPO_DIR:-$REPO_DIR_DEFAULT}"
if [ "$REPO_DIR" != "$REPO_DIR_DEFAULT" ]; then
    warn "REPO_DIR is $REPO_DIR but this script lives in $REPO_DIR_DEFAULT."
    warn "Using the configured value; the timer will sweep that checkout."
fi
ON_CALENDAR="${ON_CALENDAR:-daily}"
RANDOMIZED_DELAY_SEC="${RANDOMIZED_DELAY_SEC:-900}"
TIMEOUT_START_SEC="${TIMEOUT_START_SEC:-30min}"
NOTIFY_ON_FAILURE="${NOTIFY_ON_FAILURE:-1}"

# --- 2. the interpreter ------------------------------------------------------
# Ask the Makefile rather than keeping a second copy of its candidate list, so
# this stays correct when .python-version changes.
if [ -z "${PYTHON:-}" ]; then
    PYTHON="$(make -s -C "$REPO_DIR" print-python 2>/dev/null || true)"
fi
[ -n "$PYTHON" ] || die "No project interpreter found. Run 'poetry install', or
set PYTHON= in deploy/schedule.env ('poetry env info -p' prints the path)."
[ -x "$PYTHON" ] || die "PYTHON=$PYTHON is not an executable file."

info "Checking that the interpreter can import the project..."
if ! (cd "$REPO_DIR" && "$PYTHON" -c 'import src.publisher.late') >/dev/null 2>&1; then
    die "$PYTHON cannot import src.publisher.late from $REPO_DIR.

That is exactly what the unit would hit: every day, unattended, while the
figures it exists to capture age past the provider's five-week retention
horizon. Install the project into that interpreter, or point PYTHON= at the
right one. Testing that the file exists is not enough, which is why this
check imports."
fi

# --- 3. reject values the renderer cannot carry -----------------------------
for name in REPO_DIR PYTHON ON_CALENDAR RANDOMIZED_DELAY_SEC TIMEOUT_START_SEC; do
    value="${!name}"
    [ -n "$value" ] || die "$name is empty in deploy/schedule.env"
    case "$value" in
    *'|'* | *$'\n'*)
        die "$name contains '|' or a newline, which the renderer cannot
substitute safely: $value"
        ;;
    esac
done

# --- 4. render ---------------------------------------------------------------
if [ "$NOTIFY_ON_FAILURE" = "1" ]; then
    ONFAILURE_LINE="OnFailure=$FAILED"
else
    ONFAILURE_LINE=""
fi

render() { # render <template> <dest>
    sed \
        -e "s|@REPO_DIR@|$REPO_DIR|g" \
        -e "s|@PYTHON@|$PYTHON|g" \
        -e "s|@ON_CALENDAR@|$ON_CALENDAR|g" \
        -e "s|@RANDOMIZED_DELAY_SEC@|$RANDOMIZED_DELAY_SEC|g" \
        -e "s|@TIMEOUT_START_SEC@|$TIMEOUT_START_SEC|g" \
        -e "s|@ONFAILURE_LINE@|$ONFAILURE_LINE|g" \
        "$1" >"$2"
    # An unsubstituted placeholder must never reach systemd. Some would make
    # the unit fail to load, which is survivable; but a blank OnCalendar=
    # resets the timer list (systemd.timer(5)), giving a timer that loads,
    # enables, appears in list-timers and never fires. Refuse here instead.
    if grep -n '@[A-Z][A-Z0-9_]*@' "$2"; then
        rm -f "$2"
        die "unsubstituted placeholder in $(basename "$1"), shown above.
Refusing to install a unit that would fail, or silently not fire, every day."
    fi
}

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
render "$SCRIPT_DIR/$SERVICE.in" "$STAGE/$SERVICE"
render "$SCRIPT_DIR/$TIMER.in" "$STAGE/$TIMER"
if [ "$NOTIFY_ON_FAILURE" = "1" ]; then
    render "$SCRIPT_DIR/$FAILED.in" "$STAGE/$FAILED"
fi

if [ -n "$RENDER_ONLY" ]; then
    mkdir -p "$RENDER_ONLY"
    for f in "$STAGE"/*; do install -m 0644 "$f" "$RENDER_ONLY/"; done
    ok "Rendered into $RENDER_ONLY. Nothing installed; systemd untouched."
    exit 0
fi

# --- 5. install --------------------------------------------------------------
mkdir -p "$UNIT_DIR"
for f in "$STAGE"/*; do install -m 0644 "$f" "$UNIT_DIR/"; done
if [ "$NOTIFY_ON_FAILURE" != "1" ]; then rm -f "$UNIT_DIR/$FAILED"; fi
info "Wrote units to $UNIT_DIR"

systemctl --user daemon-reload
systemctl --user enable "$TIMER" >/dev/null
# daemon-reload does not re-arm a running timer with a changed OnCalendar, and
# `enable --now` on an active timer is a no-op. Re-running after editing
# schedule.env is the normal case, so restart unconditionally: otherwise a
# schedule change looks applied while the old one stays in force until reboot.
systemctl --user restart "$TIMER"

if [ "$(loginctl show-user "$(id -un)" --value -p Linger 2>/dev/null || echo no)" != "yes" ]; then
    warn "Lingering is off for $(id -un): the timer runs only while you are
logged in, and a missed window cannot be recovered once the figures expire.
Enable it with:  sudo loginctl enable-linger $(id -un)"
fi

# --- 6. prove it actually runs ------------------------------------------------
if [ -n "$NO_RUN" ]; then
    systemctl --user list-timers "$TIMER" --no-pager
    ok "Reinstalled without sweeping. The schedule above is now in force."
    exit 0
fi

METRICS="$REPO_DIR/outputs/post_metrics.json"
before=0
if [ -f "$METRICS" ]; then before="$(stat -c %Y "$METRICS")"; fi

info "Triggering one run now. Type=oneshot, so this blocks until it finishes."
run_ok=1
systemctl --user start "$SERVICE" || run_ok=0

after=0
if [ -f "$METRICS" ]; then after="$(stat -c %Y "$METRICS")"; fi

if [ "$run_ok" -eq 0 ]; then
    systemctl --user status "$SERVICE" --no-pager -l -n 30 || true
    die "The first run failed; status above. Full log:
    journalctl --user -u $SERVICE -n 100 --no-pager"
fi
if [ "$after" -le "$before" ]; then
    die "The service exited 0 but $METRICS did not change.

The sweep writes that file on every run, so an unchanged mtime means it wrote
somewhere else, almost certainly from a wrong working directory. Check:
    systemctl --user cat $SERVICE
    journalctl --user -u $SERVICE -n 50 --no-pager"
fi

ok "First run succeeded. $METRICS updated at $(date -d "@$after" '+%F %T')."
systemctl --user list-timers "$TIMER" --no-pager
ok "Installed. Check on it later with 'make analytics-timer-status'."
