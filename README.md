# phyein-sim

Free, reproducible physics simulations: pinned Docker images (OpenFOAM, finite elements with DOLFINx and CalculiX, ParaView, ffmpeg),
validated worked examples, a GitHub Actions workflow that runs one case or a whole parameter sweep in parallel, and a small export that
turns a finished run into a film, a still and one JSON file for a website. Nothing here costs money: GitHub's public runners, ghcr.io and
(for JAX on a GPU) Kaggle's free tier.

Start with `HANDOVER.md` (what exists, measured errors and times, status of each project) and the skill `physics-simulation-toolchain`
in `skills/`. Putting a result on the website: `docs/website-delivery.md`. Every result is a model run checked against a benchmark,
never a measurement.

    images/   foam/  fem/        Dockerfiles, self-tests (run on every build)
    cases/    foam/  fem/  jax/  one folder per worked example, with its parameters and its benchmark check
    tools/                       runners, film, web export
    skills/                      the skills that let a fresh session use all of this
    .github/workflows/           image-foam, image-fem, run-case

Structural topology optimisation lives on in `phyein-topopt` (its own image and workflow).
