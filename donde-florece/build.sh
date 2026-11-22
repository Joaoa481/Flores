#!/bin/sh
# One command: Donde florece mi cielo (Firefly Pointillism). Picture (Python, 60 fps) → score + ambience → master.
#   sh build.sh               WORKERS=4 by default
# Needs the Lemo-Opuscar library with its core tier, the VSCO 2 CE, VCSL and FreePats samples, and ffmpeg 7+.
set -e
LIB=${LIB:-/root/lemo-opuscar}
HERE=$(cd "$(dirname "$0")" && pwd)
export LIB
[ -f "$LIB/core/render/mux.sh" ] || { echo "set LIB to the library folder"; exit 1; }
PY="$LIB/.venv/bin/python"
export WORKERS=${WORKERS:-4}
mkdir -p "$HERE/out"
[ -f "$HERE/fonts/Lora.ttf" ] || curl -sfL -o "$HERE/fonts/Lora.ttf" "https://raw.githubusercontent.com/google/fonts/main/ofl/lora/Lora%5Bwght%5D.ttf"
"$PY" "$HERE/film.py" render 0 45 "$HERE/out/video.mp4"
"$PY" "$HERE/mix.py"
cd "$LIB"
sh core/render/mux.sh "$HERE/out/video.mp4" "$HERE/out/mix.wav" "$HERE/donde-florece.mp4" 60 0
ffmpeg -v error -y -ss 40.5 -i "$HERE/donde-florece.mp4" -frames:v 1 -q:v 3 "$HERE/poster.jpg"
echo "done: $HERE/donde-florece.mp4"
