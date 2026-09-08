#!/usr/bin/env bash
# Render a list of topics, one pipeline step per process.
#
# Why not just run the producer once per topic: `pipeline_timeout_sec` is a
# single budget covering every step, and a Whisper pass on a ~60s voiceover
# can consume most or all of it, so assembly is reached with nothing left.
# Each `--step` call is its own process and gets its own budget. Assembly also
# has a second, independent limit (`final_assembly_timeout_sec`), which this
# script does not change -- raise it in config if assembly is what times out.
#
# Input: a file of `title|description|comma,separated,keywords` lines.
# Blank lines and lines starting with # are ignored.
set -uo pipefail

TOPICS_FILE=${TOPICS:?set TOPICS=<file>}
PROFILE=${PROFILE:-slideshow_stock}
PY=${LOWPRI_PYTHON:?no project interpreter}
NICE=${NICE_LEVEL:-15}
STEPS="gather_visuals generate_description create_voiceover download_music
       generate_subtitles assemble_video burn_pycaps_subtitles"

[ -r "$TOPICS_FILE" ] || { echo "cannot read $TOPICS_FILE" >&2; exit 1; }

ok=0; failed=0; declare -a summary=()

# Every producer call reads from /dev/null. Without it the child inherits this
# loop's stdin and swallows the rest of the topic file, so only the first topic
# renders and the loop still reports success.
run() { nice -n "$NICE" "$PY" -m src.video.producer "$@" --debug < /dev/null; }

# fd 3 for the same reason, belt and braces.
while IFS='|' read -r title desc kw <&3; do
  case "$title" in ''|'#'*) continue ;; esac
  echo "=== $title"

  if ! run "$PROFILE" --topic "$title" --topic-description "$desc" \
           --topic-keywords "$kw" --step generate_script; then
    summary+=("FAIL  $title (generate_script)"); failed=$((failed+1)); continue
  fi

  # Derive the directory from the title. Taking the newest topic-* dir instead
  # renders this topic's steps into a previous topic's directory whenever
  # generate_script short-circuits on an existing script.
  slug=$(printf %s "$title" | tr '[:upper:]' '[:lower:]' \
         | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g')
  matches=( "outputs/topic-${slug}"-*/ )
  dir=${matches[0]%/}
  if [ ! -d "$dir" ]; then
    summary+=("FAIL  $title (no output dir for slug '$slug')"); failed=$((failed+1)); continue
  fi

  step_failed=""
  for s in $STEPS; do
    if ! run "$dir/data.json" "$PROFILE" --step "$s"; then step_failed=$s; break; fi
  done
  if [ -n "$step_failed" ]; then
    summary+=("FAIL  $title ($step_failed)"); failed=$((failed+1)); continue
  fi

  # Check the artifact, not the exit code: a timeout leaves a truncated .mp4
  # under the finished render's name, non-zero in size and failing ffprobe.
  mp4s=( "$dir"/*.mp4 )
  mp4=${mp4s[0]}
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$mp4" 2>/dev/null)
  if [ -f "$mp4" ] && [ -n "$dur" ]; then
    summary+=("OK    $title (${dur}s)"); ok=$((ok+1))
  else
    summary+=("FAIL  $title (no valid mp4)"); failed=$((failed+1))
  fi
done 3< "$TOPICS_FILE"

echo
echo "=== summary: $ok rendered, $failed failed"
printf '  %s\n' "${summary[@]}"
[ "$failed" -eq 0 ]
