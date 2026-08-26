#!/usr/bin/env bash
# OnFailure handler for the analytics timer.
#
# No `set -e`, and an unconditional `exit 0` at the end. This runs as a
# systemd user service: if it exits non-zero, the failure *report* becomes a
# failed unit of its own and the original failure is left in the journal for
# nobody to read. Every channel below is independently best-effort.
set -uo pipefail

UNIT="${1:-contentengineai-analytics.service}"
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
LOG_DIR="$REPO_DIR/outputs/logs"
LOG="$LOG_DIR/analytics-failures.log"

when="$(date '+%Y-%m-%d %H:%M:%S %z')"
result="$(systemctl --user show "$UNIT" --property=Result --value 2>/dev/null)"
status="$(systemctl --user show "$UNIT" --property=ExecMainStatus --value 2>/dev/null)"
summary="analytics sweep failed (result=${result:-unknown} exit=${status:-?})"

# 1. The journal. Works headless, survives a reboot, and lands under the same
#    journalctl query as the failing unit's own output.
printf '%s\n' "$summary" |
    systemd-cat -t contentengineai-analytics -p err 2>/dev/null

# 2. A file beside publisher.log. This is the copy still there tomorrow
#    morning, so a sweep that failed weeks ago while nobody was at the machine
#    is not invisible. outputs/ is gitignored, so it never enters the tree.
if mkdir -p "$LOG_DIR" 2>/dev/null; then
    {
        printf '%s  %s\n' "$when" "$summary"
        systemctl --user status "$UNIT" --no-pager -l -n 20 2>/dev/null |
            sed 's/^/    /'
        printf '\n'
    } >>"$LOG" 2>/dev/null
fi

# 3. Desktop notification, best effort and last. The user manager's
#    DBUS_SESSION_BUS_ADDRESS is set by the graphical session at login, so
#    this works from a user service while someone is logged in. With lingering
#    enabled the manager also runs before any login and after logout, where it
#    does not. Its failure must not become the report's failure.
if command -v notify-send >/dev/null 2>&1; then
    notify-send -u critical -a ContentEngineAI \
        "ContentEngineAI analytics failed" \
        "$summary. See outputs/logs/analytics-failures.log" \
        >/dev/null 2>&1
fi

exit 0
