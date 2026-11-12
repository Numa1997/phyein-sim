---
name: physics-simulation-toolchain
description: Run the user's free physics-simulation toolchain (phyein-sim): choose OpenFOAM, FEM, topology optimisation or JAX for a physics question, run pinned Docker images on free GitHub runners or Kaggle, film and publish results to the website.
---

# phyein-sim: one toolchain for the user's simulations

Use this skill whenever a physics question has to be *computed*: flow, free surfaces, sloshing, vortex shedding, electrostatics and beam optics, vibration modes,
heat, optimal shapes, optimal control, or any parameter sweep of these. It chooses the tool, runs it for free, checks it against a known answer, and hands a film,
stills and a small data file to the website. Method details are in the companion skills: `openfoam-cfd`, `fem-multiphysics`, `topology-optimisation`
(and `jax-differentiable-physics`, `kaggle-gpu-run` for the GPU). Read the repository's `HANDOVER.md` before building anything new: it lists what exists.

## 1. Where everything lives, and what it costs (nothing)

- **Code and cases**: the public GitHub repository `phyein-sim` on the user's account (`<owner>` = the login in lower case, `gh api user --jq .login | tr A-Z a-z`).
  Public, because GitHub's standard runners are free without limit only for public repositories: 4 vCPU, 16 GB RAM, 6 h per job, 20 jobs at once, no GPU.
- **Images** (GitHub Container Registry, free for public images), each built by its own workflow and pushed only if its self-test passes:
  - `ghcr.io/<owner>/phyein-foam`: OpenFOAM ESI v2512 (openfoam.com), ParaView 5.11 `pvbatch`, ffmpeg, Gmsh/PyVista in `/opt/venv`. Self-test 13 s.
  - `ghcr.io/<owner>/phyein-fem`: DOLFINx 0.11 (FEniCSx), SLEPc, Gmsh 4.15, CalculiX 2.21, ParaView, ffmpeg. Self-test 124 s.
  - `ghcr.io/<owner>/phyein-topopt`: the structural topology-optimisation code (repository `phyein-topopt`, still valid).
  Use `:<commit-sha>` when a result must be reproducible; write the sha into the results.
- **Kaggle** (free GPU, 12 h per session, weekly quota shared by the account): only for JAX work that wants a GPU. Notebook and clicks: `cases/jax/swingup/KAGGLE.md`.
- **Cloudflare** serves the website files only (Workers Builds publishes the repository `SiteTest` from `main`). Nothing is solved in a visitor's browser.
- **Rule**: nothing may cost money. If a step would (paid runner, larger runner, GPU runner, paid storage), stop and say so.

## 2. Choose the tool from the physics

| The question is about | Tool | Case that proves it | Measured |
|---|---|---|---|
| Incompressible flow, vortex shedding, drag | OpenFOAM `icoFoam` | `foam/cavity`, `foam/cylinder` | cavity vs Ghia 1982: 0.0042 of U (3 s); cylinder St −2.9 %, C_D +0.8 % (339 s, 2 ranks) |
| Free surface: dam break, sloshing, filling, waves | OpenFOAM `interFoam` (VOF) | `foam/dambreak`, `foam/slosh` | dam break surge 1.73 vs 1.71 Martin & Moyce (55 s); sloshing +0.77 % cylinder, +0.33 % glass vs page (106 s, 129 s) |
| Electrostatics, electron optics, Poisson-type fields | FEniCSx (DOLFINx) | `fem/lens` | focal length −0.01 % vs Davisson & Calbick (39 s) |
| Vibration modes of beams and rotors; stress | FEniCSx + SLEPc, cross-check CalculiX | `fem/beam` | modes +0.13, −0.60, −1.75 %, CalculiX agrees 0.02 % (9 s) |
| Heat, diffusion, any time-dependent PDE | FEniCSx θ-scheme | `fem/heat` | max error 1.3·10⁻⁵ of T₀ (2 s) |
| Where to put material (solid), where to put liquid (channels) | `phyein-topopt` (solid); `fem/fluidtopo` (Stokes) | MBB, chair; straight channel | solid: MBB within 10⁻⁴ of the paper; fluid: −0.9 % vs Poiseuille (27 s) |
| Optimal control, many parameter sets at once, derivatives | JAX | `jax/swingup` | gradient vs finite differences 4·10⁻⁸ (64-bit); 4096 optimisations in 154 s on a Kaggle T4 (≈10× the CPU) |
| Not covered yet | see section 8 and `HANDOVER.md` | | |

Rule of thumb: a **field in space** (u(x,t), φ(x), T(x,t)) → OpenFOAM or FEniCSx; a **shape** → topology optimisation; a **trajectory or a design vector with a
derivative** → JAX. Every new problem first gets its *simplest analytic or published case* (a limit with a known answer), then the real geometry.

## 3. Run it

**In a session** (cloud workspace, 2 cores: small runs and tests only):
1. Docker daemon: if `docker info` fails, `nohup dockerd > /tmp/dockerd.log 2>&1 &` and wait about 8 s.
2. `docker pull ghcr.io/<owner>/phyein-foam:latest` (or `phyein-fem`). Self-test: `docker run --rm <image>` prints `SELFTEST PASS`.
3. One case, results in `./out`: `docker run --rm -v "$PWD/out:/out" -e OUT=/out/run -e NP=2 <image> bash -c "foam_run.sh /opt/phyein-sim/cases/foam/cavity N=32"`
   (`foam_run.sh` and `fem_run.sh` are on the image's PATH; override any parameter as `NAME=value`; `OUT`: where the run goes; `NP`: MPI ranks; `TOL`: tolerance).
4. A film of a finished foam case: `film.sh /out/run alpha.water /out/film.mp4 0 1` (field, output, colour range). Writes `film.mp4` (H.264, 960×540, yuv420p), `film.png`.
5. Building an image inside a session: the proxy's certificate must be passed as a build secret:
   `docker build --network host --secret id=ca,src=$HOME/.ccr/ca-bundle.crt -f images/foam/Dockerfile -t phyein-foam:local .`

**On GitHub Actions** (free CPU, the default for real runs): workflow `run-case`, inputs `tool` (foam or fem; structural topology optimisation uses the `run.yml` workflow of `phyein-topopt`, section 3b), `case`, `set` (`Na=40 eps=0.03`),
`sweep` (`Na=20,40,80;Re=100,400`: Cartesian product, one parallel job each, at most 20 at once, 60 total), `film` (`alpha.water:0:1`), `np`, `label`.
From a Claude session use the REST endpoint (GraphQL, hence `gh workflow run`, is unreachable there):
```sh
R=repos/<owner>/phyein-sim
gh api -X POST $R/actions/workflows/run-case.yml/dispatches -f ref=main -f "inputs[tool]=foam" -f "inputs[case]=dambreak" \
   -f "inputs[sweep]=Na=10,20,40" -f "inputs[film]=alpha.water:0:1" -f "inputs[label]=dam-grid"
gh api "$R/actions/workflows/run-case.yml/runs?per_page=1" --jq '.workflow_runs[0] | "\(.id) \(.status) \(.created_at)"'
gh run download <id> -R <owner>/phyein-sim -D out/
```
A run whose case misses its benchmark ends **red** (the step exits with the case's verdict) but still uploads log, `result.json` and film. GitHub timestamps are UTC: convert
before telling the user how long something has run. Starting a run acts on the user's account: give the time estimate and get a yes.

**Sizing before you launch** (say the number to the user; over 10 minutes needs an honest estimate): time the coarsest grid, scale by the cell count to the power 4/3
for explicit 3D (cells × time steps), add film time (about 1 s per 10 frames for 960×540). Measured on GitHub's runners (12 Nov 2026): serial cases take the same time as in the sandbox (dam break 57 s vs 55 s, sloshing 110 s vs 106 s);
the 4-rank cylinder took 194 s against 339 s on 2 ranks (1.75×); FEM cases were 1.3–1.7× faster. Add about 1 minute per job for pulling the image. A job that needs more than 16 GB or 6 h is killed and returns nothing: split or coarsen.

**On Kaggle** (no Docker there; one long or memory-hungry run: 4 cores, 31 GB, 12 h, Ubuntu 24.04, Python 3.13, NumPy 2.1, measured 12 Nov 2026). `kaggle/bootstrap.sh <foam|fem>`
installs the same releases natively (OpenFOAM v2512 from dl.openfoam.com; DOLFINx 0.11 from conda-forge with micromamba), `source kaggle/env.sh <tool>` puts the same runners on the PATH, and
`kaggle/kernel.py` is the whole script notebook (edit OWNER, REF, TOOL, CASE, SETS, NP; Internet on, GPU off). Launch it with the Kaggle connector as the `kaggle-gpu-run` skill describes.
The GPU helps JAX only; do not spend GPU quota on OpenFOAM or DOLFINx. No ParaView on Kaggle: film the downloaded run in the foam image. Sweeps belong on GitHub (20 at once).
**Proven on Kaggle** (12 Nov 2026): install 60 s (foam) and 130 s (fem); cavity 0.0042 and beam modes identical to GitHub. The account must be
phone-verified, or the session has no internet and nothing installs.

### 3b. Structural topology optimisation (repository `phyein-topopt`, image `phyein-topopt`; this replaces the old `topopt-toolchain` skill)

The solid-shape code stays in its own repository and image (`ghcr.io/<owner>/phyein-topopt`, Python 3.13, NumPy/SciPy, pypardiso + MKL as the reference solver, pyamg, scikit-image,
trimesh, PyVista, JAX CPU, ffmpeg). The optimiser uses the matrix-free multigrid-preconditioned CG (`topopt/mgcg.py`). Its workflow is `run.yml` (named `run-case` there).
- In a session: `docker run --rm ghcr.io/<owner>/phyein-topopt:latest` runs the acceptance checks (about 1 minute, prints `ALL PASS`: MBB within 10⁻⁴ of the paper, cantilever within 1 % of
  Timoshenko at 16 elements, gradient within 10⁻¹⁰, MGCG within 10⁻⁶ of the direct solve). One case:
  `docker run --rm -v "$PWD/out:/work" <image> sh -c "cd /opt/phyein-topopt && python examples/chair.py --n 12 --volfrac 0.03 --out /work/chair12 && python examples/chair_check.py /work/chair12 && python examples/chair_animate.py /work/chair12"`
- On GitHub: `gh api -X POST repos/<owner>/phyein-topopt/actions/workflows/run.yml/dispatches -f ref=main -f "inputs[case]=chair" -f "inputs[n]=20" -f "inputs[volfrac]=0.03" -f "inputs[rmin]=2.0" -f "inputs[penal]=3.0" -f "inputs[label]=chair20" -f "inputs[animate]=yes" -f "inputs[maxit]=300"`
  (`maxit=3` times a grid; `animate=no` skips film for sweeps; up to 20 runs at once).
- **Size first.** Measured (MGCG, PA12, ν = 0.43): chair at 13 824 voxels: 8.7 s per iteration on the runner (4.7 s on a 2-core session), 30–45 min per run; at 64 000 voxels 27 s per
  iteration on the runner, 1.5–2.3 h plus 20–30 min for the film; runs take 160–300 iterations. A job over 16 GB or 6 h is killed.
- Results to publish: `chair_load_path.mp4` (H.264, faststart), `still_front.png`, `still_back.png`, `chair_stress.glb` (three.js viewer, pinned local copy, rotate on drag), `chair.stl`,
  `chair_run.json`, `render_summary.json`, `design_check.json`; every figure labelled "approximate voxel FEA · design proposal, not a structural verification".
- Kaggle for structural runs: notebooks cannot run Docker; `git clone` the repository, `pip install -r requirements.txt`, run, collect `/kaggle/working`. A JAX port of the multigrid solver for the
  GPU is planned, unmeasured.
- Two lessons: a new GHCR package accepts the workflow's token only once the image carries `org.opencontainers.image.source`; pushing from a session needs the Claude GitHub App to have access to the
  repository (GitHub → Settings → Applications → Claude → Configure), attaching alone is not enough.

## 4. Check before you believe (every study)

1. State the benchmark and its source (paper, table, analytic formula) *before* the run, and the tolerance with its reason.
2. Do not grade at parameters the reference does not cover (the glass-sloshing frequency of the Guinness page is valid at 125 mm fill only: `check.py` reports
   "NOT GRADED" elsewhere).
3. Convergence: two or three resolutions, observed order versus theory. A result that moves by more than the tolerance between the two finest grids is not converged.
4. Report the **measured** error with its sign. A mesh-limited bias is stated, not tuned away (cylinder St −2.9 % at the default 14 000 cells).
5. After a change to a solver, case or image, `docker run --rm <image>` must print `SELFTEST PASS` before you report success.

## 5. Deliver to the website (never solve in the browser)

Every simulation page gets three files, none larger than a few MB, plus the numbers:
- the **film**: `film.mp4`, H.264, `yuv420p`, `-movflags +faststart`, 960×540, a few MB; autoplay muted, loop, a poster image (`film.png`);
- the **stills**: one or two PNGs at the moments that carry the physics (the surge front; the peak of the standing wave; the wake);
- the **data file**: a small JSON (tens of kB) with the parameters, the benchmark curve and the simulated curve, the measured error and the image commit sha.
  The page draws its plot from this file; the numbers on the page come from it, not from retyping (skill `phyein-usul-data-panel`).
Where and how: follow the website repository's own rules first (`CLAUDE.md`, `AGENTS.md`, skills `phyein-usul-physics-content`, `physics-simulation-study`,
`phyein-usul-smooth-site`, `phyein-usul-i18n`). Work on a branch; check the page at 360, 390, 1024 and 1440 px in a real browser; merge to `main`; Cloudflare publishes.
Label every figure with what it is (a model run with the stated grid, not a measurement). The recipe with file names is in `docs/website-delivery.md`.

## 6. Outward actions need an explicit yes, asked once and all at once

Creating or renaming repositories, pushing, starting Actions runs, making packages public, merging to the website, using Kaggle GPU hours: each needs the user's yes
for that exact action. A yes covers only what it names. Ask for all you will need in one message, with what, why, effect, cost ("none"), and whether it can be undone.

## 7. Failure modes that cost hours (all met while building this)

| Symptom | Cause | Fix |
|---|---|---|
| Self-test exits 139 or 1 with no output | OpenFOAM's `bashrc` reads the *command's arguments* as its settings (`FOAM_SETTINGS`) | `entry.sh` clears `$@` while sourcing, then runs the command |
| `pvbatch` segfaults loading `vtkmodules` | pip's VTK 9.7 and ParaView's own VTK 9.1 share one Python | pip packages only in `/opt/venv`; system `python3` stays clean |
| `WM_PROJECT_SITE: unbound variable` | `set -u` while sourcing OpenFOAM's `bashrc` | no `-u`; do not source twice |
| `docker build` cannot reach PyPI in a session | proxy certificate | `--network host --secret id=ca,src=$HOME/.ccr/ca-bundle.crt` |
| Push to GHCR "denied" | image lacks `org.opencontainers.image.source` label | the workflow sets it from the repository URL |
| `np.trapz` missing | NumPy 2 renamed it `np.trapezoid`, the image has 1.26 | `_trapz = getattr(np, "trapezoid", None) or np.trapz` |
| Film: domain tiny, bar labels invisible | camera reset on first render; white text on white | render once, then set camera; black bar text (fixed in `film.py`) |
| A run is green but the benchmark was missed | exit code swallowed (`\|\| true`) | the step exits with the case's verdict; artifacts upload `if: always()` |
| First push of a NEW ghcr.io package: "error from registry: denied" after all layers uploaded | unknown (label and permissions were correct) | re-run the workflow once; the second push succeeded |
| Kaggle: `Temporary failure resolving ...`, nothing installs | session has no internet: account not phone-verified, or Internet switch off | the user verifies the phone number in Kaggle's settings; then Internet on |
| Creating a public repository from a session is refused | the session's safety layer | the user creates it at github.com/new (4 clicks) and gives the Claude GitHub App access |
| Daemon died mid-session | sandbox limits | restart `dockerd`; images persist |
| `rm -rf` of a run folder refused by the safety check | a guard against deleting too much | do not work around it; use a fresh folder and tell the user |

## 8. What the toolchain cannot do (yet)

Moving-body and overset meshes are not set up (the cart in Fig. 12 uses a prescribed *tilt* of the liquid, not the cart's motion); turbulence models are untested
(the cylinder runs at Re = 100, laminar); no compressible or reacting flow; no contact or plasticity (CalculiX could, unvalidated); no magnetostatic lens; no fluid-structure
coupling (a rod swinging in liquid); no GPU for OpenFOAM or FEniCSx (the free GPU is Kaggle's, via JAX only); no 3D fluid topology optimisation (2D Stokes only).
`HANDOVER.md` lists each project's status and the sentence to start it.
