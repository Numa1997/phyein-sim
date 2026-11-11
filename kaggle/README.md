# phyein-sim on Kaggle

Kaggle notebooks cannot run Docker, so the images cannot be used there. Instead `bootstrap.sh` installs the same releases natively
(OpenFOAM v2512 from the publisher's Ubuntu repository; DOLFINx 0.11, SLEPc, Gmsh, CalculiX from conda-forge) and the **same case files,
runners and checks** run unchanged. A Kaggle result is therefore graded by the same `check.py` / `result.json` as a GitHub result.

**When Kaggle is the right machine.** A Kaggle CPU session has 4 cores, about 30 GB of memory and up to 12 hours; a GitHub runner has 4 cores, 16 GB
and 6 hours. So: one long or memory-hungry run → Kaggle; many short runs at once (sweeps) → GitHub (20 in parallel). The GPU helps only JAX
(`cases/jax/`); OpenFOAM and DOLFINx do not use it, so leave the GPU off for them and spend no GPU quota.

**How.** `kernel.py` is the whole notebook: edit the six parameters at its top (owner, commit, tool, case, parameter sets, ranks), save it as a
Kaggle *script* with Internet on. It prints the hardware, installs, runs, and leaves `run<i>/` (results, logs, `result.json`), `timing.json` and
`commit.txt` under `/kaggle/working`, which is the notebook's downloadable output. Claude can launch it through the Kaggle connector (skill
`kaggle-gpu-run`); by hand it is: kaggle.com → Create → New Notebook → File → Editor Type → Script, paste, Settings → Internet on → Save Version → Save & Run All.

**Not on Kaggle:** ParaView films. Download the run folder and make the film in the `phyein-foam` image (`film.sh`), or on GitHub.

**In a shell on any Ubuntu machine without Docker:** `bash kaggle/bootstrap.sh foam && source kaggle/env.sh foam && foam_run.sh cases/foam/cavity N=32`.
