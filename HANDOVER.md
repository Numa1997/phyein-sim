# HANDOVER: phyein-sim

State on 11 November 2026, written for whoever continues (a person or a Claude session). For method, load the skill
`physics-simulation-toolchain` (it points to `openfoam-cfd`, `fem-multiphysics`, `topology-optimisation`, `jax-differentiable-physics`,
`kaggle-gpu-run`). For putting a result on the website, read `docs/website-delivery.md`.

**Honest state:** everything below was built and measured in a local sandbox. At the time of writing nothing was pushed to GitHub, no
image was built by GitHub Actions, and nothing was published. "Verified" means "run here, inside the image, with the number shown". Where a
number comes from GitHub or Kaggle it says so; until then those lines read "not yet run".

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
| workflows | `.github/workflows/image-foam.yml`, `image-fem.yml`, `run-case.yml` | build → self-test → push to ghcr.io; `run-case` runs one case or a sweep (max 20 at once), uploads artifacts even on failure, exits with the case verdict |
| skills | `skills/` | `physics-simulation-toolchain`, `openfoam-cfd`, `fem-multiphysics` (new); `topology-optimisation` (new sections 2.15 and 2.16 in `NEW_SECTIONS.md`); the master replaces `topopt-toolchain` |

Structural topology optimisation (SIMP, the chair, MBB) stays in `phyein-topopt` with its own image and workflow `run.yml`; see that
repository's `HANDOVER.md`. `run-case` here covers `foam` and `fem` only.

## Measured results (sandbox, 2 cores, inside the image)

| case | benchmark | measured | time |
|---|---|---|---|
| lid-driven cavity N=32 | Ghia, Ghia & Shin 1982 | max difference 0.0042 (N=64: 0.0049) | 3 s (N=64: about 40 s) |
| dam break | Martin & Moyce 1952 | max difference of the surge time 0.196 (scaled units); front 1.73 vs 1.71 | 55 s |
| sloshing, cylinder | Lamb f11 with surface tension | +0.77 % in frequency | 106 s |
| sloshing, tulip glass | same, at its own fill | +0.33 % | 129 s |
| cylinder wake, Re = 100 | Williamson / Henderson | Strouhal −2.9 %, C_D +0.8 % | 339 s |
| electron lens, aperture | Davisson & Calbick | −0.01 % | 39 s |
| electron lens, einzel | paraxial ray equation | −0.20 % (not yet against an independent published value) | 48 s |
| beam modes | Euler–Bernoulli, CalculiX | +0.13 %, −0.60 %, −1.75 % (modes 1–3); CalculiX vs DOLFINx 0.02 % | 9 s |
| heat | analytic solution | 1.3·10⁻⁵ | 2 s |
| fluid topology optimisation | Poiseuille P = 12 μ L Q² / w³ | −0.9 %; double channel −2 % | 27 s; 32 s |
| JAX swing-up, 256 optimisations, float32 | — | works; 64-bit would be about 32 times slower on a T4 | 95 s on CPU |

All are under the 15-minute limit on 2 cores; GitHub's public runners have 4 cores, so they should be faster (**not yet measured there**).
Not yet verified: any GitHub Actions run, any ghcr push, the GPU timing, whether the −0.9 % of the fluid optimisation is separated into its two
causes (finite α_max leak and discretisation), the einzel lens against an independent published value.

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
- Every outward action (create repository, push, start Actions, make packages public, pointer push to `phyein-topopt`, SiteTest branch and merge, each Kaggle run)
  needs the owner's explicit yes and had none at the time of writing.
- Skills were written and checked against the repository, but a fresh-session end-to-end test is still to be done and recorded here.

## Failure modes met while building (short list; full list in the master skill, section 7)

OpenFOAM's `bashrc` treats script arguments as settings (clear `$@` before sourcing); `set -u` breaks it; pip packages into the system Python
clash with ParaView's VTK (use the venv at `/opt/venv`); NumPy 1 vs 2 (`trapz`/`trapezoid`); float32 finite-difference step must not be tiny; a free
outlet in fluid optimisation is a loophole (prescribe it); a workflow step that swallows the exit code hides failures; GHCR needs the
`org.opencontainers.image.source` label.
