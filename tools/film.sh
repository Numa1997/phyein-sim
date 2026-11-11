#!/usr/bin/env bash
# film.sh <case_dir> <field> <out.mp4> [vmin vmax [component [width height [colormap]]]]
# frames with ParaView (pvbatch under xvfb), then H.264 with ffmpeg: yuv420p, +faststart (plays in a browser while loading),
# 24 frames/s, even dimensions. Also writes <out>.png (a still from the last frame) and keeps the frames in <out>_frames/.
set -euo pipefail
case_dir=$1; field=$2; out=$3; shift 3
frames="${out%.mp4}_frames"; rm -rf "$frames"
xvfb-run -a pvbatch "$PHYEIN_SIM/tools/film.py" "$case_dir" "$field" "$frames" "$@"
ffmpeg -y -loglevel error -framerate "${FPS:-24}" -i "$frames/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 23 -preset veryfast \
       -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -movflags +faststart "$out"
cp "$(ls "$frames"/frame_*.png | tail -1)" "${out%.mp4}.png"
echo "film: $out ($(du -k "$out" | cut -f1) kB), still: ${out%.mp4}.png"
