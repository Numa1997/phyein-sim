#!/usr/bin/env python3
"""web_export.py <run_dir> <out_dir> [--film film.mp4] [--name NAME] [--max-mb 20]

Turns one finished run into the three things a website page needs, and nothing else:
  <out_dir>/<name>.mp4         the film (copied; must be H.264, yuv420p, faststart: tools/film.sh makes it so)
  <out_dir>/<name>.png         the poster / still (the film's last frame, if film.sh made one)
  <out_dir>/<name>.json        ONE small data file: parameters, measured error and verdict, benchmark and simulated curves,
                               and provenance (image, commit, date), so every number on the page can be traced and re-checked
The page draws its plot from the json; it never solves anything in the visitor's browser.
Looks for <run_dir>/result.json, curves.json (written by the case's check.py), and the film next to or inside <run_dir>.
Exit 1 if a file exceeds --max-mb (the static host refuses files over 25 MiB) or the verdict file is missing.
"""
import argparse, json, os, shutil, sys, datetime

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("out_dir")
ap.add_argument("--film", default=None); ap.add_argument("--name", default=None); ap.add_argument("--max-mb", type=float, default=20.0)
a = ap.parse_args()
run = a.run_dir.rstrip("/"); name = a.name or os.path.basename(run)
os.makedirs(a.out_dir, exist_ok=True)

res_f = os.path.join(run, "result.json")
if not os.path.exists(res_f): sys.exit(f"no result.json in {run}: the case did not finish its check")
res = json.load(open(res_f))
cur_f = os.path.join(run, "curves.json"); curves = json.load(open(cur_f)) if os.path.exists(cur_f) else None

film = a.film
if film is None:
    for c in (os.path.join(os.path.dirname(run), os.path.basename(run) + ".mp4"), os.path.join(run, "film.mp4"), os.path.join(os.path.dirname(run), "film.mp4")):
        if os.path.exists(c): film = c; break
out = {"name": name, "result": res, "curves": curves,
       "provenance": {"image": os.environ.get("PHYEIN_IMAGE", "unknown"), "commit": os.environ.get("PHYEIN_COMMIT", "unknown"),
                      "exported": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                      "note": "model run with the stated grid and parameters; compare the benchmark curve, not a measurement"}}
files = []
jf = os.path.join(a.out_dir, name + ".json"); json.dump(out, open(jf, "w"), separators=(",", ":")); files.append(jf)
if film:
    fm = os.path.join(a.out_dir, name + ".mp4"); shutil.copy(film, fm); files.append(fm)
    png = os.path.splitext(film)[0] + ".png"
    if os.path.exists(png): pf = os.path.join(a.out_dir, name + ".png"); shutil.copy(png, pf); files.append(pf)
else:
    print("note: no film found (give --film); only the data file was written")
bad = 0
for f in files:
    mb = os.path.getsize(f) / 2**20; flag = "" if mb <= a.max_mb else "   <-- TOO LARGE"
    bad += mb > a.max_mb; print(f"{f}  {mb:.3f} MiB{flag}")
print("verdict:", "PASS" if res.get("pass") else "FAIL", "| graded" if res.get("graded", True) else "| NOT GRADED")
sys.exit(1 if bad else 0)
