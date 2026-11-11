### 2.15 Where to put liquid instead of solid: fluid topology optimisation (Borrvall & Petersson 2003)

The same idea, a density field γ(x) ∈ [0,1] that a penalty pushes to 0 or 1, works for channels. Picture: a block of sponge in which the liquid may only travel where γ = 1; elsewhere
a Darcy drag −α(γ)**u**, α(γ) = α_max + (α_min − α_max) γ(1+q)/(γ+q), clogs the pores (α_min = 0, α_max ≫ μ/L²). The objective is the **dissipated power** J = ½∫(μ|∇**u**|² + α|**u**|²)dx
(what the pump pays), the constraint the liquid volume fraction V_f, and because **u** itself minimises J the sensitivity needs no adjoint: dJ/dγ = ½ α′(γ)|**u**|² (negative: liquid helps most where it flows fastest).
Solid ↔ fluid dictionary: compliance C = **f**·**u** ↔ power J; stiffness E(ρ) = ρ^p E₀ ↔ resistance α(γ) with convexity q (use q from 0.1 up to 1 by continuation, like raising p); the filter and the
optimality-criteria update carry over unchanged. Derivation, parameters and the validated case are in the skill `fem-multiphysics`, section 2.4 (`cases/fem/fluidtopo` in `phyein-sim`).
**Measured** (Stokes, 60 × 60 elements, 80 iterations, 27 s on 2 cores): straight channel within **0.9 %** of the Poiseuille power 12μL Q²/w³; two inlet windows give two clean channels (−2 %).
**Decide**: use it for manifolds, junctions and cold plates at low Reynolds number (Re ≲ 1); at higher Re the Stokes design is only a starting point (inertia changes the optimum: Navier–Stokes forms exist, are
not built here). **Traps**: prescribe the velocity on the outlet too (a free outlet is exploited: the optimiser flares the exit), keep α_max ≥ 10⁵ μ/L², and always measure the channel width the optimiser produced.
Hydraulic parts are manufactured by casting or additive manufacturing, so a 2D result is a section, not a part: say so.

### 2.16 Mounts and chassis for the carts (Fig. 12, project 06) — PLAN, not built

Use the structural code as it is (`phyein-topopt`, voxel hex8, SIMP, MGCG), adding one load-case file `examples/cart.py` in the style of `examples/chair.py` (all parameters as arguments, passive pads under loads and supports,
a docstring with the hand statics). Physics of the loads, in SI: (i) weight of the glass with its liquid, m g, applied through the holder pads; (ii) inertia of that mass during the cart's acceleration, m a, with a taken from the
control trajectory (the JAX swing-up or the cart's real profile), as a second load case with weight w₂ (multi-load compliance, section 2.7); (iii) the sloshing force on the holder, taken as the peak wall force from `phyein-sim`
`cases/foam/slosh` (a force history exists in the interFoam run via `forces`, not yet wired). Supports: the axle seats. Constraint: V_f. Check afterwards: first bending frequency of the proposal (`fem/beam`-style eigenproblem
on the smoothed, re-meshed shape) must lie above the excitation frequencies (sloshing 3.4 Hz for the glass; motor and wheel frequencies), otherwise the stiffest static design can still ring.
Time: from the measured table, 13 824 voxels is 30–45 min on a runner (8.7 s per iteration, 160–300 iterations); 64 000 voxels 1.5–2.3 h. Start at the coarse grid. First sentence for a future session:
"Use the physics-simulation-toolchain and topology-optimisation skills to design the chassis of the Fig. 12 cart as a new load case, starting with weight and braking inertia, then check its first mode above 10 Hz."
