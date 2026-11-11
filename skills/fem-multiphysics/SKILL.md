---
name: fem-multiphysics
description: Finite elements (FEniCSx, SLEPc, CalculiX) for the user's problems: electron lens, vibration modes of beams and rotors, heat, and fluid topology optimisation. Weak forms, symbolic parameters, benchmarks with measured errors, failure modes.
---

# Finite elements for the user's projects

Companion of `physics-simulation-toolchain`. Image: `ghcr.io/<owner>/phyein-fem` (DOLFINx 0.11 = FEniCSx, PETSc, SLEPc, Gmsh 4.15.2, CalculiX 2.21, ParaView, ffmpeg). Cases are
single Python scripts in `cases/fem/<case>/run.py`; **every parameter is an argument** (`fem_run.sh <case> NAME=value` becomes `--NAME value`); each writes `result.json`
with the measured error and exits non-zero if it misses its benchmark. Self-test of the image: 124 s.

## 1. The method in one page

A PDE is turned into a finite sum. Picture a stretched rubber sheet pinned at its edge: instead of knowing the height everywhere, we record it at the corners of small
triangles and let it vary linearly (or quadratically, "P2") inside each. **Weak form**: multiply the PDE by any test function v that vanishes where the value is fixed,
integrate over the domain Ω, integrate the highest derivative by parts so that it moves onto v and the boundary conditions appear naturally. For −∇·(k∇u) = f it reads
∫ k ∇u·∇v dx = ∫ f v dx for all v. Writing u = Σ u_j φ_j and choosing v = φ_i gives one linear equation per node: **K u = f**, K_ij = ∫ k ∇φ_j·∇φ_i dx.
Time-dependent: ρc ∂T/∂t = ∇·(k∇T) becomes M dT/dt + K T = 0; the θ-scheme (θ = ½ Crank–Nicolson, 1 backward Euler) steps it. Vibration: M ü + K u = 0 with u ∝ e^{iωt}
gives the generalised eigenproblem **K x = ω² M x** (SLEPc, shift-and-invert). Error theory: for degree-q elements the error in the field falls as h^{q+1} (the energy norm as h^q).

## 2. The four validated cases (times on 2 cores inside the image, 11 November 2026)

### 2.1 Electron lens (`lens`) vs Davisson & Calbick 1931 — for the electron-beam column
Electrons of charge −e travel along z through a rotationally symmetric field. With the potential φ(r,z) measured from the cathode, kinetic energy = eφ, and away from charge
Laplace holds: (1/r)∂_r(r ∂_rφ) + ∂²_zφ = 0, weak form ∫ r ∇φ·∇v dr dz = 0 (the factor r is the area element of the axisymmetric problem). The paraxial ray equation
d/dz(√φ r′) = −φ″ r/(4√φ) follows from the force −eE and conservation of energy for rays close to the axis. For a thin plate with a round hole separating a uniform
field E₁ from E₂, r is constant across the plate and integrating gives **1/f = (E₂ − E₁)/(4V)** (V = potential at the plate). Picture: the field lines bulge through the
hole; the bulge on the side of the stronger field deflects electrons towards the axis, like a glass lens bends light.
Parameters: `geometry` (aperture | einzel), `ra` (m), `V` (V), `E1`, `E2` (V/m; keep E₂·ra ≪ V: thin-lens limit), `Vm` (V, einzel middle tube), `gap`, `tube`, `r0` (launch height in units of ra; default 0.05 for aperture, 0.1 for einzel), `hmin`, `tol`. Defaults: `Vm` = 5000 V, `V` = 1000 V, `E2` = 2·10⁴ V/m.
The script traces a ray through the full numerical field (no paraxial approximation) and compares. **Measured**: *aperture*: f_theory = 200.00 mm, f_traced = 200.03 mm, error −0.01 %,
4594 triangles, 39 s. *Einzel* (three tubes, outer at V = 1000 V, middle at V_m = 5000 V; the traced ray against the paraxial equation integrated on the FEM axis potential):
the ray crosses the axis at z = 3.400 a (traced) vs 3.406 a (paraxial): **−0.20 %**, 8097 triangles, 48 s. Two traps found by measuring: (1) V_m = 1500 V (the first default tried; the default is now 5000 V) is a weak lens whose
focus lies beyond the box (the case returned NaN); a lens must focus inside the computational box. (2) The traced-minus-paraxial difference is *spherical aberration*, ∝ r₀²:
−0.68 % at r₀ = 0.2 a, −0.20 % at 0.1 a (ratio 3.4, expected 4); below r₀ ≈ 0.05 a the absolute noise of the interpolated radial field dominates and the difference jumps to +10 %.
Hence the defaults r₀ = 0.05 (aperture) and 0.1 (einzel). Not tested against an independent published einzel focal length: the comparison is between two routes through the same FEM potential.
Not built: magnetostatic lens (vector potential A_θ; focal length from ∫B_z² dz), space-charge, aberrations.

### 2.2 Vibration modes (`beam`) vs Euler–Bernoulli, with CalculiX — for rotors and shafts
A cantilever (clamped at one end) of length L, section b × h, Young's modulus E, density ρ. Euler–Bernoulli beam theory (EI ∂⁴w/∂x⁴ + ρA ∂²w/∂t² = 0, w ∝ e^{iωt})
gives ω_n = (β_nL)² √(EI/(ρAL⁴)) with β_nL = 1.8751, 4.6941, 7.8548, 10.9955. Picture: a ruler pinned to a desk edge; the first mode swings with no node, the second has one.
The FEM solves the full 3D elasticity (quadratic hexahedra, `nx ny nz` cells), not the beam model, so the difference measures the beam theory's neglect of shear and
rotary inertia plus the discretisation. Parameters: `L b h E nu rho nx ny nz nmodes tol ccx`. **Measured** (L = 0.5 m, 20 × 20 mm steel, 640 hexahedra, 19 683 dof):
bending mode 1: 66.93 Hz vs 66.84 Hz (**+0.13 %**), mode 2: **−0.60 %**, mode 3: **−1.75 %** (the higher the mode, the larger the shear correction: that is physics, not error),
axial mode 1 **+0.15 %**; 9 s. Cross-check: **CalculiX C3D20** agrees with FEniCSx to **0.02 %** on all modes.
For a rotor (project: rotor pendula): add the centrifugal stiffening K_g Ω² and gyroscopic coupling; **not built**. First step when asked: free modes of the shaft here, then Campbell diagram.

### 2.3 Heat (`heat`) vs the exact decaying mode — the template for any time-dependent PDE
ρc ∂T/∂t = ∇·(k∇T) on a plate L × L with edges at T = 0 and T(x,y,0) = T₀ sin(πx/L) sin(πy/L), an eigenfunction of the Laplacian with eigenvalue −2π²/L². Exact:
T = T₀ e^{−2απ²t/L²} sin sin, with α = k/(ρc). Parameters: `L k rho c T0 N steps theta degree taus tol`. **Measured**: max error **1.3·10⁻⁵ T₀** (tolerance 10⁻³),
2 s; observed orders under joint refinement 3.6 and 3.2 (P2 in space, Crank–Nicolson in time: at these small errors the spatial and temporal terms trade places).
Use: diffusion of droplets' vapour, thermal rotors, any scalar transport; replace the exact mode by a manufactured solution for new problems.

### 2.4 Fluid topology optimisation (`fluidtopo`) vs Poiseuille — for hydraulics
**Problem** (Borrvall & Petersson 2003, Int. J. Numer. Meth. Fluids 41:77–107). A box Ω (length Lx, height Ly) holds a given fraction V_f of liquid (viscosity μ); the rest is
solid. Find where the liquid goes so that the power lost to viscous friction is least. Liquid at low Reynolds number obeys Stokes: −μ∇²**u** + ∇p = 0, ∇·**u** = 0.
**Porosity trick.** Let γ(x) ∈ [0,1] be the fluid fraction. Add a Darcy drag −α(γ)**u** with α(γ) = α_max + (α_min − α_max) γ(1+q)/(γ+q), α_min = 0, α_max large
(the solid is a medium so resistive that nothing flows through it); q ≥ 0 tunes the convexity (continuation from q = 0.1 to 1). Picture: a sponge that is clogged where γ = 0.
**Variational form**: **u** minimises the dissipated power J = ½∫(μ|∇**u**|² + α(γ)|**u**|²)dx over divergence-free fields with the prescribed inlet/outlet profiles
(pressure = multiplier of ∇·**u** = 0; mixed Taylor–Hood P2–P1 elements).
**Sensitivity without an adjoint.** Because **u** is a minimiser (self-adjoint problem), the envelope theorem gives dJ/dγ = ½ α′(γ)|**u**|², with
α′ = −α_max(1+q)q/(γ+q)² < 0: more liquid always helps, where the flow is fast it helps most. An optimality-criteria update with a bisection on the volume constraint moves γ.
Parameters: `case` (straight | double), `Lx Ly mu U Vf amax_fac q q_end nx ny iters move tol outlet`; α_max = `amax_fac`·μ/Ly².
**Benchmark.** Inlet = plane Poiseuille window of width w = V_f·Ly, walls elsewhere: the best design with V_f = w/Ly is a straight channel, whose power is exactly
**P = Δp·Q = 12 μ Lx Q²/w³** (Poiseuille: u = 6(Q/w³) y(w−y), Δp = 12 μ Lx Q/w³). **Measured** (60 × 60, 80 iterations, α_max = 10⁵ μ/Ly², 27 s): P = 13.214 vs 13.333:
**−0.9 %** (tolerance 3 %), grey (non-0/1) cells 0.1 %. Case `double` (two inlet windows) finds **two clean channels**, P = 52.2 vs 53.3 for two Poiseuille channels (−2 %), 32 s.
Two traps, both found by measuring the channel width the optimiser produced (`design.npz`): (1) **the outlet must be prescribed** (`outlet=dirichlet`, as in the paper, plus one pinned
pressure value): with a free outlet the optimiser flares the exit to spend volume where the velocity profile costs nothing, and P comes out 14 % too low or 35 % too high depending
on α_max. (2) Finite α_max lets liquid leak a depth ≈ 1/√(α_max/μ) into the "solid", widening the channel by about 2/√(α_max/μ): α_max = 10⁴ μ/Ly² widens a 0.4 m channel by 5 %,
which lowers P by 14 % (P ∝ w⁻³); 10⁵ shrinks the widening to 1.6 %, which alone would lower P by about 5 %: the measured −0.9 % is the sum of this and the
discretisation (a transition layer one or two cells thick), not an effect I have separated. Do not read −0.9 % as the method's accuracy; read it as "within 1 % at this resolution".
Not built: 3D, pressure-drop-constrained forms, bends and manifolds from real hydraulic specifications. Next step for hydraulics: Y-junction and bend with two outlets.

## 3. Procedure for a new FEM case

1. Physical set-up with SI units; the PDE; the **weak form written out** in the docstring; boundary conditions and which are essential (fixed) and natural.
2. The **reference**: an exact solution, a manufactured one, or a published number. For a new equation use the method of manufactured solutions (pick u, compute f, solve, compare).
3. All numbers as arguments with their units in the comment; derived values computed, not typed.
4. Mesh with Gmsh (or `create_rectangle`); two or three refinements; print the **observed order** and compare it with h^{q+1}.
5. `result.json` with parameters, measured errors, tolerance, `pass`. Exit code from `pass`.
6. Cross-check with a second code where one exists (CalculiX for elasticity).
7. Film or stills with ParaView/PyVista (`film.py` is for OpenFOAM data; PyVista in the fem image reads FEniCSx output).

## 4. Failure modes met here

| Symptom | Cause | Fix |
|---|---|---|
| CalculiX "exit 201: more than 16 entries on a line" | C3D20 element has 20 nodes | continue the element line (16 + 4 nodes) |
| CalculiX frequencies read as 0.0 | regex took the wrong column of the `E I G E N V A L U E` table | match the 4th numeric column (cycles/s) |
| Fluid TO power wrong by 14 % or 35 %, channel flares at the exit | free outlet / finite α_max leak | prescribe the outlet; α_max ≥ 10⁵ μ/Ly²; measure the produced width |
| float32 gradient check returns `inf` | finite-difference step 10⁻⁶ below float32 resolution | step ≈ 3·10⁻³ in float32 |
| `np.trapz` missing | NumPy 2 | use `trapezoid` with fallback |
| Eigenvalues come out as spurious zeros | free body: rigid-body modes | clamp or expect 6 zeros (3D) |
| pip VTK clashes with ParaView | two VTK builds in one Python | keep pip packages in a venv |

## 5. Projects this serves

Electron-beam column: `lens` (aperture, einzel) today; magnetic lens and aberrations planned. Rotor pendula and shafts: `beam` today; rotating terms planned. Hydraulics: `fluidtopo`.
Impact (Hertz contact via CalculiX), wave interference (Helmholtz, double slit) and wind turbine: not built; see `HANDOVER.md`.
