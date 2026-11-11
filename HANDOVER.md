# HANDOVER: phyein-sim

State on 11 November 2026, written for whoever continues (a person or a Claude session). For method, load the skill
`physics-simulation-toolchain` (it points to `openfoam-cfd`, `fem-multiphysics`, `topology-optimisation`, `jax-differentiable-physics`,
`kaggle-gpu-run`). For putting a result on the website, read `docs/website-delivery.md`.

**Honest state (12 November 2026):** the repository is public at github.com/Numa1997/phyein-sim. Both images were built by GitHub Actions,
passed their self-tests there (foam 12 s, fem 105 s) and are public on ghcr.io. Every worked example below was then run on GitHub's standard
runners through `run-case` and reproduced the sandbox numbers digit for digit. Kaggle: the native install path exists (`kaggle/`) but is **not yet
proven on Kaggle**: the test session had no internet (see "Kaggle" below). The JAX GPU run has not been done.

## What this is

One toolchain for the user's physics simulations: pinned Docker images, a workflow that runs one case or a whole sweep in parallel for
free, validated worked examples with measured errors and times, film plus stills plus one small data file for the website, and skills
that let a fresh session do all of this without asking again. Every result is a **model run with a stated grid and parameters, checked
against a known benchmark**; it is not a measurement.

## Repository choice (two sentences)

A new repository `phyein-sim` holds the general toolchain, because renaming `phyein-topopt` would break saved skills, the plugin, the image
label and every Actions link that point at it. The old repository stays as it is (structural topology optimisation lives on there) and gets a
one-line pointer to the new one.

## What exists

| part | where | what |
|---|---|---|
| `foam` image | `images/foam/` | OpenFOAM ESI v2512 (base pinned by digest), ParaView 5.11.2 `pvbatch` with xvfb, ffmpeg 6.1.1, gmsh 4.15.2, pyvista 0.49.2, numpy 1.26.4, scipy 1.11.4. Self-test: cavity at N=32 plus a film, 13 s. |
| `fem` image | `images/fem/` | DOLFINx 0.11 with SLEPc, Gmsh, CalculiX 2.21, ParaView. Self-test: lens (aperture and einzel), beam, heat, fluid topology optimisation, 124 s. |
| JAX | `cases/jax/swingup/` | swing-up control by gradient descent, a Kaggle notebook and `KAGGLE.md` (the clicks). CPU test done; GPU run not done. |
| cases | `cases/foam/{cavity,dambreak,slosh,cylinder}`, `cases/fem/{lens,beam,heat,fluidtopo}` | each has a `params` file or arguments (everything tunable is a named parameter) and a `check.py` / result check that prints a verdict and writes `result.json` and `curves.json` |
| runners | `tools/foam_run.sh`, `tools/fem_run.sh` | `NAME=value` overrides, wall time, exit code = benchmark verdict |
| film | `tools/film.py`, `tools/film.sh` | ParaView → H.264 yuv420p faststart; text bar; camera fitted to the bounds |
| web | `tools/web_export.py`, `docs/website-delivery.md` | film + still + one JSON, size guard (25 MiB host limit) |
| Kaggle | `kaggle/bootstrap.sh`, `env.sh`, `kernel.py`, `README.md` | native install of the same releases where Docker is not allowed; same cases and checks |
| workflows | `.github/workflows/image-foam.yml`, `image-fem.yml`, `run-case.yml` | build → self-test → push to ghcr.io; `run-case` runs one case or a sweep (max 20 at once), uploads artifacts even on failure, exits with the case verdict |
| skills | `skills/` | `physics-simulation-toolchain`, `openfoam-cfd`, `fem-multiphysics` (new); `topology-optimisation` (new sections 2.15 and 2.16 in `NEW_SECTIONS.md`); the master replaces `topopt-toolchain` |

Structural topology optimisation (SIMP, the chair, MBB) stays in `phyein-topopt` with its own image and workflow `run.yml`; see that
repository's `HANDOVER.md`. `run-case` here covers `foam` and `fem` only.

## Measured results

Errors are identical in the sandbox and on GitHub. Times: wall-clock of the case itself (image pull, about 1 minute per job, and film come on top).

| case | benchmark | measured error | sandbox, 2 cores | GitHub runner, 4 cores |
|---|---|---|---|---|
| lid-driven cavity N=32 (N=64) | Ghia, Ghia & Shin 1982 | 0.0042 of U (0.0049) | 3 s (40 s) | 4 s (46 s) |
| dam break, Na=20 | Martin & Moyce 1952 | surge time within 0.196 (scaled units); front speed 1.73 vs 1.71 | 55 s | 57 s |
| sloshing, cylinder | Lamb f11 | +0.77 % in frequency (+0.03 % with surface tension) | 106 s | 110 s |
| sloshing, tulip glass, h = 125 mm | the website's own value | +0.33 % | 129 s | 115 s |
| cylinder wake, Re = 100 | Williamson / Henderson | Strouhal −2.9 %, C_D +0.8 % | 339 s | 194 s |
| electron lens, aperture | Davisson & Calbick | −0.01 % | 39 s | 23 s |
| electron lens, einzel | paraxial ray equation on the same field | −0.20 % (no independent published value yet) | 48 s | 29 s |
| beam modes | Euler–Bernoulli; CalculiX cross-check 0.02 % | +0.13 %, −0.60 %, −1.75 % | 9 s | 7 s |
| heat | analytic solution | 1.3·10⁻⁵ | 2 s | 1 s |
| fluid topology optimisation | Poiseuille P = 12 μ L Q² / w³ | −0.9 % (double channel −2 %) | 27 s (32 s) | 26 s (26 s) |
| JAX swing-up, 256 optimisations, float32 | none (gradient checked against finite differences) | works | 95 s on CPU | not run |

Every example is far below the 15-minute limit on its stated hardware (the GitHub runner). The runner is about as fast as the sandbox per core; only
cases that use 4 MPI ranks (cylinder) gain. A finer dam break (Na=40, 4 times the cells) took 728 s on the runner and gave the same answer (0.197; front speed 1.72): the
difference from the paper is therefore not a grid effect. Artifacts are large for free-surface cases (about 90 MB zipped for the dam break) because all time steps are kept for re-filming.

Not yet verified: the Kaggle path on Kaggle, the GPU timing, whether the −0.9 % of the fluid optimisation separates into its two causes (finite α_max
leak and discretisation), the einzel lens against an independent published value.

## Kaggle

Kaggle cannot run Docker. `kaggle/bootstrap.sh` installs the same releases natively and `kaggle/kernel.py` is a ready script notebook that clones
this repository at a chosen commit, installs, runs the case with the same runner and check, and leaves `result.json` and `timing.json` as output
(`kaggle/README.md`). Measured facts about a Kaggle CPU session (12 November 2026): Ubuntu 24.04, Python 3.13, NumPy 2.1, 4 cores, 31 GB, 12 h.
Use Kaggle for one long or memory-hungry run; use GitHub for sweeps. The GPU helps JAX only.
**Blocker found:** the first test session had no internet although the switch was saved as on (`Temporary failure resolving ...`); Kaggle gives
internet and GPU only to phone-verified accounts. Until the account is verified nothing can be installed there. After verification: rerun the test
(`kernel.py` with `TOOL="foam"`, `CASE="cavity"`), then record the install and run times here.

## The projects: tool, example, time, look, sentence

For each project: the tool, the worked example that exists or is planned, how long it runs and where, how it would look on the website,
and the exact sentence to give a future session. "Built" means validated in the table above. "Planned" has an **estimate**, not a measurement.

1. **Fig. 12, a pint of Guinness on a one-wheel cart.** OpenFOAM `slosh` (tank and tulip glass), built: 106 s and 129 s, GitHub runner. The glass
   profile is in `cases/foam/slosh/profile.csv` (fill 0.125 m). The cart's own motion is not built (the liquid is tilted, not carried). Website: the film of the
   surface, the standing-wave curve against Lamb. *Sentence:* "Run the `slosh` case with `profile=1 h=0.125` for the Guinness tulip glass and export it for Fig. 12."
2. **Project 06, rods in liquid.** OpenFOAM `cylinder` (drag and vortex shedding), built: 339 s. A swinging rod needs fluid–structure coupling or a moving mesh: not built.
   *Sentence:* "Run the `cylinder` case at the project 06 Reynolds number and report St and C_D; a swinging rod needs a moving-mesh case, which does not exist yet."
3. **Pendulum control and active suspension.** JAX `swingup`, built on CPU (95 s for 256 optimisations); GPU run needs one yes per run. Website: the swing-up film and the
   controller-gain sweep plot. *Sentence:* "Use `cases/jax/swingup` and the kaggle-gpu-run skill; start with the 32-bit CPU check."
4. **Droplet collection (thesis).** JAX Monte Carlo of coalescence checked against the Golovin kernel: planned, estimate a few minutes on CPU, seconds on a GPU. Not built.
   *Sentence:* "Build `cases/jax/droplets`: Monte Carlo coalescence against the Golovin analytic solution."
5. **Rotor pendula.** FEM `beam` for the bending modes, built: 9 s. Rotating stiffening (centrifugal) is not added. *Sentence:* "Run `beam`, then add the centrifugal stiffness term as a parameter and compare with the rotor pendula page."
6. **Hydraulics.** FEM `fluidtopo` (channels in a design space, Stokes), built: 27 s, 2D only. *Sentence:* "Run `fluidtopo` with `case=double`, `Vf` as the liquid volume fraction."
7. **Electron-beam column.** FEM `lens`, aperture and einzel, built: 39 s and 48 s. Magnetic lens not built. *Sentence:* "Run `lens` with `geometry=einzel` and `Vm` set; report the focal length against the paraxial value."
8. **Impact.** CalculiX Hertz contact against the Hertz solution: planned, estimate under 5 minutes (contact is untested here). Not built.
9. **Wind turbine.** OpenFOAM actuator disc against the Betz limit (16/27 ≈ 0.593): planned, estimate 10 to 20 minutes on 4 cores; may exceed 15 minutes on a coarse grid, to be measured. Not built.
10. **Water waves.** OpenFOAM numerical wave tank against linear (Airy) dispersion ω² = g k tanh(k h): planned, estimate about 10 minutes. Not built.
11. **Wave interference.** FEM Helmholtz double slit against the Fraunhofer fringe spacing: planned, estimate under 2 minutes. Not built.
12. **Driven oscillator.** Needs no tool beyond the website's own page; nothing to add.

## What the toolchain still cannot do

Moving bodies and overset meshes (no cart that moves, no swinging rod in liquid); turbulence models (untested; the cylinder is laminar at Re = 100); compressible
and reacting flow; contact and plasticity (CalculiX could, unvalidated); a magnetostatic (magnetic) lens; fluid–structure coupling; GPU for OpenFOAM or
FEniCSx (the free GPU is Kaggle's, for JAX only); 3D fluid topology optimisation (2D Stokes only); runs longer than six hours (GitHub's limit) or on more than 4 cores.

## Open items

- A folder `runs/cyl_f15` could not be deleted by the sandbox (a safety rule blocked the delete). It is scratch output and is excluded by `.gitignore`.
- Done with the owner's yes on 11–12 November 2026: repository created (by the owner; the session may not create public repositories), pushed, images built, nine `run-case` runs,
  pointer line in `phyein-topopt`. Not done: a SiteTest page, the Kaggle GPU run. Each further outward action needs its own yes.
- The first push of each new ghcr.io package was refused ("denied") after the layers had uploaded; a second run of the workflow succeeded. Cause unknown; re-run once.
- Skills: the run commands of the master and fem skills were followed literally in clean containers and reproduced their numbers; a separate fresh chat session has not been tried.

## Failure modes met while building (short list; full list in the master skill, section 7)

OpenFOAM's `bashrc` treats script arguments as settings (clear `$@` before sourcing); `set -u` breaks it; pip packages into the system Python
clash with ParaView's VTK (use the venv at `/opt/venv`); NumPy 1 vs 2 (`trapz`/`trapezoid`); float32 finite-difference step must not be tiny; a free
outlet in fluid optimisation is a loophole (prescribe it); a workflow step that swallows the exit code hides failures; GHCR needs the
`org.opencontainers.image.source` label.
