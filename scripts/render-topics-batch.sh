#!/usr/bin/env bash
# Render a list of topics, one pipeline step per process.
#
# Why not one producer run per topic: `pipeline_timeout_sec` is a single budget
# covering every step, and a Whisper pass can consume most of it, so assembly
# is reached with nothing left (issues #398, #402). Each `--step` call is its
# own process and gets its own budget. `--topics-file` renders a list in one
# run and so shares that single budget; this target is for when that does not
# fit. Assembly also has a separate limit, `final_assembly_timeout_sec`, which
# this script does not touch -- raise that one if assembly is what times out.
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

[ -r "$TOPICS_FILE" ] || { echo "cannot read $TOPICS_FILE" >&2; exit 1; }

# Enumerate through the project's loader: product id, title, description and
# keywords per topic, NUL-delimited so no title can break the framing. The
# output root comes from config rather than a hardcoded "outputs".
# Via a temp file, not $(...): command substitution strips NUL bytes -- bash
# cannot hold a NUL in a variable at all -- so routing the records through one
# collapses every field into the first and the run reports "no topics".
records_file=$(mktemp) || { echo "cannot create temp file" >&2; exit 1; }
trap 'rm -f "$records_file"' EXIT

"$PY" - "$TOPICS_FILE" > "$records_file" <<'PY'
import sys
from pathlib import Path
from src.video.config import config
from src.video.producer.topic_input import load_topics_file, topic_product_id

specs = load_topics_file(Path(sys.argv[1]))
root = config.global_output_root_path
out = [str(root)]
for s in specs:
    out += [topic_product_id(s.title), s.title, s.description, ", ".join(s.keywords)]
sys.stdout.write("\0".join(out))
PY
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
  mp4s=( "$dir"/*.mp4 )
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
