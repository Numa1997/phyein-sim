---
name: openfoam-cfd
description: OpenFOAM (ESI v2512) for the user's flow problems: cavity, dam break, sloshing in a tank or tulip glass, cylinder vortex street. Derivations, symbolic parameters, benchmarks with measured errors and times, failure modes.
---

# OpenFOAM CFD for the user's projects

Companion of `physics-simulation-toolchain` (images, running, delivery). Image: `ghcr.io/<owner>/phyein-foam` (OpenFOAM ESI v2512 from openfoam.com, ParaView `pvbatch`,
ffmpeg). Why ESI: its distribution includes the moving-body and overset solvers we may need for carts and swinging rods (not yet set up here).
Cases live in `cases/foam/<case>/`; **every physical and numerical choice is one named line in the case's `params` file**; change it with `NAME=value` on the command line.

## 1. The physics in one page (what OpenFOAM solves)

**Incompressible flow.** Picture honey poured slowly: its velocity **u**(**x**,t) (m/s) and pressure p (Pa, here per unit density) obey mass and momentum balance on any
fixed piece of space. Mass: ∇·**u** = 0. Momentum: ∂**u**/∂t + (**u**·∇)**u** = −∇p + ν∇²**u** + **g**, with ν the kinematic viscosity (m²/s). The only dimensionless number
is the Reynolds number Re = UL/ν (inertia over viscosity). Cases keep **Re** and the geometry as the parameters, and ν is derived (`nu #eval{ $U*$L/$Re }`).

**Two liquids, one interface (volume of fluid, VOF).** Let α(**x**,t) ∈ [0,1] be the fraction of a cell filled with water. It is carried by the flow:
∂α/∂t + ∇·(α**u**) = 0 (OpenFOAM adds an artificial compression flux so the interface stays about 2–3 cells thick). One momentum equation is solved for the mixture, with
ρ = αρ_w + (1−α)ρ_a and likewise μ, plus surface tension as a force at the interface: **f**_σ = σ κ ∇α (σ = 0.07 N/m, κ = −∇·(∇α/|∇α|) the curvature).
Picture: the water is a coloured fog, α its density; gravity pulls the heavy fog down, surface tension pulls the interface taut like a skin.

**Time step.** Explicit transport needs the Courant number Co = |**u**|Δt/Δx ≤ 1 (a fluid particle crosses less than one cell per step). Cavity and cylinder set Δt from Co;
`interFoam` also needs the *interface* Courant number `maxAlphaCo`; the dictionaries set `maxCo`, `maxAlphaCo`, `maxDeltaT` even with a fixed step (it stops with
"keyword maxAlphaCo not found" otherwise).

## 2. The four validated cases

Times are wall-clock on 2 cores inside the image, measured 11 November 2026 (a runner has 4 vCPU: expect somewhat faster; not measured).

### 2.1 Lid-driven cavity vs Ghia et al. 1982 (`cavity`, icoFoam) — the solver check
Square box of side L with the lid sliding at U. Parameters: `L U Re N Co nTurn` (Re 100, 400, 1000 have tabulated profiles in `check.py`).
Benchmark: Ghia, Ghia & Shin 1982, J. Comput. Phys. 48:387, Table I, u along the vertical centreline. **Measured**: N = 32, Re = 100: max |u − u_Ghia|/U = **0.0042**
(3 s); N = 64: 0.0049 (≈ 40 s). The error does not fall with N because the tabulated values themselves carry about 3 digits, and the lid corner singularity limits the order.
Tolerance 0.03. Use: smallest end-to-end test of an install.

### 2.2 Dam break vs Martin & Moyce 1952 (`dambreak`, interFoam, 2D)
A water column of width a and height H₀ = n²·a (the paper's n² is the height-to-width ratio; `params` calls it `n2`, and n = √n2) collapses on a dry floor.
Non-dimensionalise as the paper: Z = z/a (front position), T = n t √(g/a). For a given n² the front follows one universal curve Z(T). The end time in seconds is therefore
t_end = T_end √(a/g)/n, with n = √n2 (not n2: a slip of exactly this kind ended the first version of the case early, at T ≈ 6.7 instead of 9.5). Parameters: `a n2 Na Lxa Lya Tend Co` + fluid properties.
Benchmark: Martin & Moyce 1952, Phil. Trans. R. Soc. A 244:312–324, Table 2 (n² = 2), read at the first cell row `yRow`. The paper's time zero is ambiguous (the gate lifts
in finite time), so `check.py` shifts time so that Z = 1.44 occurs at T = 1.19 (the paper's own normalisation point) and compares Z ≤ 4.6.
**Measured**: Na = 20 (cells across a), n² = 2: max |ΔT| = **0.196** (tolerance 0.25), rms 0.151; late surge speed dZ/dT = **1.73** against the paper's 1.71 (+1.2 %), 55 s
(75 s with the film). The margin to the tolerance is thin: do not coarsen below Na = 20. The paper itself notes retardation at small scale (surface tension and viscosity) for Z > 5.

### 2.3 Sloshing in a cylinder and in the tulip glass (`slosh`, interFoam, 3D)
Liquid depth h in a vertical cylinder of radius R, released from a small tilt of the free surface (z = h + εx). Derivation of the benchmark: for small waves the flow is
irrotational, **u** = ∇φ with ∇²φ = 0. Separate φ = F(r) cos θ cosh(k(z+h)) e^{−iωt}: F obeys Bessel's equation, the wall condition F′(R) = 0 picks k = ξ₁₁/R with
ξ₁₁ = 1.8412 the first zero of J₁′ (a standing wave with one nodal diameter). The linearised free-surface condition ∂²φ/∂t² = −g∂φ/∂z gives
ω² = (g ξ₁₁/R) tanh(ξ₁₁h/R), i.e. **f₁₁ = (1/2π) √(gξ₁₁/R · tanh(ξ₁₁h/R))** (Lamb, *Hydrodynamics*, §257). Surface tension adds σk³/ρ to gk (a +0.75 % rise for R = 40 mm). Picture:
the liquid rocks like a seesaw whose pivot line is a diameter; the restoring "spring" is gravity, the "mass" is the liquid that has to move sideways.
Parameters: `R h Hair eps profile Nc Nr Nz nPeriods dt sample`. The 3D mesh is an O-grid (square in the middle, four curved blocks round it) so that the wall is round.
`check.py` fits a damped cosine to the surface height at the wall.
**Measured** (cylinder R = 40 mm, h = 100 mm): f = 3.408 Hz vs theory 3.382 Hz: **+0.77 %**, vs the capillary-corrected 3.407 Hz: +0.03 %; damping 0.09 s⁻¹; 106 s.
**Tulip glass** (`profile=1`: the radius follows R(z) from `profile.csv`, taken from the website's Fig. 12 glass profile): the page's own value is 3.421 Hz at **h = 125 mm**;
**measured** 3.432 Hz, **+0.33 %**, 129 s. The page's value exists at 125 mm only; run the glass as `profile=1 h=0.125`; at any other fill `check.py` prints NOT GRADED.
This is the building block for Fig. 12: liquid in a moving glass (the cart). The cart's motion itself (a prescribed solid-body translation of the mesh) is **not** built.

### 2.4 Cylinder vortex street (`cylinder`, icoFoam, 2D, Re = 100)
Flow U past a cylinder of diameter D. Above Re ≈ 47 the steady wake loses stability and sheds alternating vortices. Dimensionless results: Strouhal number St = fD/U
(f the shedding frequency, from the lift coefficient), mean drag coefficient C_D = F_x/(½ρU²D). Parameters: `D U Re Lu Ld Hh b Nf N1 N2 Nu Nd Nh Gr Gd Gu dt tEnd tAvg`;
`Nf` multiplies every cell count. Mesh: 12 blocks (an O-grid round the cylinder, wake and inlet blocks), blockage D/H = 1/30. Run on 2 or 4 ranks (`NP`).
Benchmarks: Williamson 1989, J. Fluid Mech. 206:579 (St = 0.164); Henderson 1995, J. Fluid Mech. 285:69 (C_D = 1.35). `check.py` uses St_ref = 0.165.
**Measured** (default 14 000 cells, 339 s on 2 ranks): St = 0.1603 (**−2.9 %**), C_D = 1.361 (**+0.8 %**), C_L amplitude 0.34, 9 shedding periods averaged; tolerance 5 %.
The −2.9 % is a mesh-limited bias, not a bug: a finer wake mesh (Gd = 12, Gr = 15) gave −2.6 % and cost 14.5 min on contended cores. State the bias; do not tune it away.
Use this for project 06 (drag on rods in liquid): for a rod of diameter D at Re = UD/ν, C_D and St come straight from here.

## 3. Procedure for a new case (the order that works)

1. **Write down the physics and the analytic or published reference first** (formula, table, source, tolerance and why that tolerance). Define every symbol with SI units.
2. **Make the geometry parametric**: dimensions as named lengths in `params`, derived values as `#eval{...}`. Negative numbers must be written `#eval{-$x}`.
3. **Start from the nearest validated case** (copy its folder). `<case>` in dictionaries is replaced by the runner with the run's folder (`#include "<case>/params"`).
4. **Coarse run first** (seconds), then check CFL, boundedness (α between 0 and 1), mass conservation (∫α dV constant), then refine twice and compare with the reference.
5. Write `check.py` that prints one line with the measured error and the tolerance and exits non-zero on failure.
6. Film (`film.sh`) and stills; save a small JSON with parameters, reference curve, simulated curve, error, image sha.
7. Only then run sweeps on GitHub (`sweep=` input).

## 4. Failure modes met here

| Symptom | Cause | Fix |
|---|---|---|
| `maxAlphaCo not found` | interFoam wants it even with a fixed step | set `maxCo`, `maxAlphaCo`, `maxDeltaT` in `controlDict` |
| `sets` function object syntax error | missing `;` after `start`/`end` vectors | write `start (x y z); end (x y z);` |
| blockMesh "inside-out" blocks | vertex order gives a left-handed block | reorder so local x, y, z are right-handed; inward blocks use ratio 1/Gr |
| "Symmetry plane not planar" | side wall of a curved mesh | patch type `symmetry`, not `symmetryPlane` |
| Negative number in a dictionary read as a keyword | `-$x` is not an expression | `#eval{-$x}` |
| Tag line fails in the runner | `tr` range error | `tr -c 'A-Za-z0-9=._\n-' '_'` |
| Strouhal about 3 % low at Re = 100 | mesh-limited wake resolution | document the bias; the 5 % tolerance is deliberate |
| Glass frequency "−2.6 %" | page value applied at h = 100 mm | glass is graded only at h = 125 mm |
| Film shows a tiny domain / invisible labels | camera reset on first render; white bar text | fixed in `film.py`; render once, then set the camera |
| Case ends earlier than its stated T_end | end time computed with n² instead of n = √n² | `tEnd = Tend √(a/g)/sqrt(n2)`; always print the dimensionless end time the run really reached |
| Everything crashes with exit 139, no message | OpenFOAM `bashrc` read the command line | use the image's `entry.sh` (it clears `$@`) |

## 5. Projects this serves

Fig. 12 (liquid in the Guinness glass on a cart): `slosh profile=1 h=0.125` today; the moving cart needs a dynamic-mesh case (planned). Project 06 (rods in liquid):
`cylinder` for drag and shedding; a rod swinging in liquid needs fluid–structure coupling (planned, not built). Hydraulics, impact, wind turbine, water waves: see `HANDOVER.md`.
