# deploy

Ops artifacts for running ContentEngineAI on a schedule. Nothing here is
imported by the application; these files are rendered and installed onto a
machine.

| File | What it is |
|---|---|
| `schedule.env.example` | Committed sample. Copy to `schedule.env`, which is gitignored |
| `install-timer.sh` | Renders the units, installs them, enables the timer, runs one sweep |
| `analytics-failure-notify.sh` | `OnFailure=` handler: journal, log file, desktop notification |
| `contentengineai-analytics.service.in` | Service template |
| `contentengineai-analytics.timer.in` | Timer template |
| `contentengineai-analytics-failed.service.in` | Failure-handler template |

Precedence for every setting is environment, then `schedule.env`, then the
built-in default, so `NOTIFY_ON_FAILURE=0 make install-analytics-timer` works
without editing the file.

Usage and the reasoning behind the settings are in
[the publisher docs](../docs/publisher.md), under `Command: analytics`.

```bash
make install-analytics-timer      # install, enable, and prove it runs
make analytics-timer-status       # last run, next run, recorded failures
make uninstall-analytics-timer    # remove the units, keep the figures
./deploy/install-timer.sh --no-run   # re-arm after a schedule edit, no sweep
```

## Why the units are rendered rather than parameterised

systemd forbids variable expansion in the `ExecStart` executable path, so an
`EnvironmentFile=` cannot supply the interpreter. `OnCalendar=` lives in the
timer unit, which reads no environment file at all. Both values have to be
substituted before the unit is written.

Placeholders are `@NAME@`, substituted with `sed`. `envsubst` is not used: it
needs `gettext-base`, and it renders an unset variable as the empty string. An
empty `OnCalendar=` resets the timer list rather than erroring, which yields a
timer that loads, enables, appears in `list-timers`, and never fires. The
renderer greps its own output for a surviving placeholder and refuses to install
if it finds one.

## The sweep size is not here

`analytics.limit` lives in `config/publisher.yaml`, and the rendered `ExecStart`
passes no `--limit`. One decision, one home: editing the YAML changes both the
manual run and the scheduled one, with no reinstall and no chance of the two
disagreeing.

## A note on credentials

The sweep reads `LATE_API_KEY` from the project's `.env`, which is resolved
relative to the source file rather than the working directory. `load_dotenv`
does not override a variable that is already set, so a key exported into the
systemd user manager's environment would outrank `.env` and the timer would
authenticate as a different account than `make analytics` does. Keep the key in
`.env` only.
