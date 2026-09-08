#!/usr/bin/env bash
# Render a list of topics, one pipeline step per process.
#
# `pipeline_timeout_sec` is one budget covering all eight steps of a topic,
# and a Whisper pass can consume most of it, so assembly is reached with
# nothing left (issues #398, #402). Each `--step` call is its own process and
# gets its own budget. `--topics-file` does not help here: it applies the same
# timeout per record, so every topic in its list has the same problem. What
# changes is the per-step process, not the list. Assembly also has a separate
# limit, `final_assembly_timeout_sec`, which this script does not touch --
# raise that one if assembly is what times out.
#
# TOPICS is the same YAML `--topics-file` accepts. The project's own loader
# reads it, so validation, the slug and the product id all come from one place:
# a second implementation here diverged on accents, on titles over the 60-char
# slug cap and on titles that normalise to nothing, and each divergence lost
# the render *after* the paid script step had run.
set -uo pipefail

TOPICS_FILE=${TOPICS:?set TOPICS=<topics.yaml>}
PROFILE=${PROFILE:-slideshow_stock}
PY=${LOWPRI_PYTHON:?no project interpreter}
STEPS="gather_visuals generate_description create_voiceover download_music
       generate_subtitles assemble_video burn_pycaps_subtitles"

# Resolve the topics path before moving, then run from the repo root. Two
# reasons. A relative `tools/...` path only resolves from the root, so running
# this without make otherwise failed with a message blaming the topics file.
# And a script *path* puts that script's directory on sys.path, not the cwd,
# so `import src` fell through to the venv's editable-install .pth -- which
# names whichever checkout the venv was built from. In a worktree the
# enumeration then read one checkout's output root while the producer, invoked
# with -m, rendered into another's, and every topic failed its directory check
# after its script step had been paid for. `-m` on both sides keeps them
# resolving from the same tree.
case "$TOPICS_FILE" in /*) ;; *) TOPICS_FILE="$PWD/$TOPICS_FILE" ;; esac
cd "$(dirname "$0")/.." || { echo "cannot reach the repo root" >&2; exit 1; }

[ -r "$TOPICS_FILE" ] || { echo "cannot read $TOPICS_FILE" >&2; exit 1; }
# The recipe guards this too, but the script is executable and takes its input
# from the environment, so it can be run without make. Without the guard, a box
# where ffprobe is absent reports every good render as a failure.
command -v ffprobe >/dev/null 2>&1 || { echo "ffprobe not found" >&2; exit 1; }

# Enumerate through the project's loader: product id, title, description and
# keywords per topic, NUL-delimited so no title can break the framing. The
# output root comes from config rather than a hardcoded "outputs".
# Via a temp file, not $(...): command substitution strips NUL bytes -- bash
# cannot hold a NUL in a variable at all -- so routing the records through one
# collapses every field into the first and the run reports "no topics".
records_file=$(mktemp) || { echo "cannot create temp file" >&2; exit 1; }
trap 'rm -f "$records_file"' EXIT

# The records go to a path, NOT to stdout: importing the config chain prints
# three lines to stdout before any of this runs, so capturing stdout would put
# them in the first field. The enumeration lives in tools/enumerate_topics.py
# rather than inline here because it is the part that has broken twice, and as
# a module it has a test.
"$PY" -m tools.enumerate_topics "$TOPICS_FILE" "$records_file"
PY_EXIT=$?
[ "$PY_EXIT" -eq 0 ] || { echo "could not read topics from $TOPICS_FILE" >&2; exit 1; }

mapfile -d '' -t fields < "$records_file"
root=${fields[0]}
total=$(( (${#fields[@]} - 1) / 4 ))
[ "$total" -gt 0 ] || { echo "no topics in $TOPICS_FILE" >&2; exit 1; }

ok=0; failed=0; summary=()

# Every producer call reads from /dev/null: a child inheriting this shell's
# stdin swallows whatever is feeding the loop.
run() { "$PY" -m src.video.producer "$@" --debug < /dev/null; }

for ((i = 0; i < total; i++)); do
  pid=${fields[$((i * 4 + 1))]}
  title=${fields[$((i * 4 + 2))]}
  desc=${fields[$((i * 4 + 3))]}
  kw=${fields[$((i * 4 + 4))]}
  echo "=== [$((i + 1))/$total] $title"

  if ! run "$PROFILE" --topic "$title" --topic-description "$desc" \
           --topic-keywords "$kw" --step generate_script; then
    summary+=("FAIL  $title (generate_script)"); failed=$((failed + 1)); continue
  fi

  dir="$root/$pid"
  if [ ! -d "$dir" ]; then
    summary+=("FAIL  $title (no output dir at $dir)"); failed=$((failed + 1)); continue
  fi

  step_failed=""
  for s in $STEPS; do
    if ! run "$dir/data.json" "$PROFILE" --step "$s"; then step_failed=$s; break; fi
  done
  if [ -n "$step_failed" ]; then
    summary+=("FAIL  $title ($step_failed)"); failed=$((failed + 1)); continue
  fi

  # Check the artifact, not the exit code: a timeout leaves a truncated .mp4
  # under the finished render's name, non-zero in size and failing ffprobe.
  # This profile's render: a topic rendered under two profiles keeps both, and
  # the alphabetically first may not be the one this run produced. The glob is
  # loose enough for both shapes the config allows -- video_<id>_<profile>.mp4
  # in the bundled file, video_<profile>.mp4 in the model default.
  mp4s=( "$dir"/video_*"$PROFILE".mp4 )
  mp4=${mp4s[0]}
  if [ ! -f "$mp4" ]; then
    summary+=("FAIL  $title (no mp4)"); failed=$((failed + 1)); continue
  fi
  # Separate "ffprobe could not run" from "the file is bad", or a box where
  # FFmpeg is off PATH reports every good render as a failure.
  if ! dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$mp4"); then
    summary+=("FAIL  $title (ffprobe failed on $(basename "$mp4"))"); failed=$((failed + 1)); continue
  fi
  summary+=("OK    $title (${dur}s)"); ok=$((ok + 1))
done

echo
echo "=== summary: $ok rendered, $failed failed"
printf '  %s\n' "${summary[@]}"
[ "$failed" -eq 0 ]
