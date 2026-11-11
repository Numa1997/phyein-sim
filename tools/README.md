# tools

| file | use |
|---|---|
| `foam_run.sh <case-dir> [NAME=value ...]` | copy a case, override its `params`, run `Allrun` then `check.py`; prints wall time; exit code is the benchmark verdict. `OUT=<dir>` sets the run folder, `TOL=` the tolerance |
| `fem_run.sh <case-dir> [NAME=value ...]` | the same for the finite-element cases in `cases/fem/` (each is one `run.py` that accepts `name=value`) |
| `film.sh <case-dir> <field> <out.mp4> [vmin vmax [component [width height [colormap]]]]` | ParaView `pvbatch` + ffmpeg: H.264 yuv420p faststart film with a text bar, 24 frames/s (`FPS=` changes it); also writes the last frame as `<out>.png` |
| `film.py` | the ParaView script behind `film.sh` |
| `web_export.py <run-dir> <out-dir> --name <id>` | film + still + one JSON for the website; refuses files over `--max-mb` (the host stops at 25 MiB) |

Inside the images these are on the PATH. Overrides use the names in each case's `params` file.
