#!/usr/bin/env python3
"""Fluid topology optimisation, Stokes flow (Borrvall & Petersson 2003, Int. J. Numer. Meth. Fluids 41, 77-107) in FEniCSx.

PHYSICAL SET-UP.  A box Omega (length Lx, height Ly, unit depth) is to be filled with a given fraction Vf of liquid (viscosity mu) and the
rest solid, so that liquid entering through the inlet windows on the left leaves through the open right side with the least power lost to
viscous friction.  Liquid flow at low Reynolds number obeys Stokes' equations  -mu laplacian(u) + grad p = 0, div u = 0.
POROSITY TRICK.  Let gamma(x) in [0,1] be the fluid fraction (1 = liquid, 0 = solid).  A solid is modelled as a very porous-resistant medium:
a Darcy drag force -alpha(gamma) u is added,  alpha(gamma) = alpha_max + (alpha_min - alpha_max) gamma (1+q)/(gamma+q)   (BP eq. 3.2; q tunes convexity).
alpha_min = 0 in liquid; alpha_max large (1e4 mu / Ly^2 here) in solid.
VARIATIONAL FORM.  The velocity minimises the dissipated power  J(u, gamma) = 1/2 integral( mu |grad u|^2 + alpha(gamma) |u|^2 ) dx  over
divergence-free fields with the given inlet profile; the pressure is the multiplier of div u = 0 (weak form: mixed P2-P1, Taylor-Hood).
SENSITIVITY.  Because u minimises J, the derivative of the minimum with respect to gamma is the partial derivative at fixed u (envelope theorem):
        dJ/dgamma = 1/2 alpha'(gamma) |u|^2,     alpha'(gamma) = (alpha_min - alpha_max)(1+q) q/(gamma+q)^2  < 0.
No adjoint solve is needed (the problem is self-adjoint).  Optimality-criteria update with a volume bisection (as in Bendsoe-Sigmund) moves gamma.
BENCHMARK (case straight).  The inlet is a plane-Poiseuille window of width w = Vf Ly centred in the box and the other inlet parts are walls: the
best design with Vf = w/Ly is a straight channel of width w, whose power is exactly  P = Delta p Q = 12 mu Lx Q^2 / w^3  (Q = flow per unit depth,
Delta p = 12 mu Lx Q/w^3 from the Poiseuille profile u = 6 (Q/w^3) y (w - y)).  J here is half of the power P.  Case `double` has two inlet
windows and two outlets-in-line (two channels expected).

usage: python3 run.py [--name value ...]    writes result.json, design.png, design.npz; exit 0 = PASS
"""
import argparse, json, sys
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl, basix
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import petsc as fpetsc

P = dict(case="straight",          # straight | double
         Lx=1.0, Ly=1.0,           # m
         mu=1.0, U=1.0,            # Pa s, peak inlet speed m/s (scales out of the design)
         Vf=0.4,                   # liquid volume fraction
         amax_fac=1e5,             # alpha_max = amax_fac * mu / Ly^2
         q=0.1, q_end=1.0,         # BP convexity parameter, raised from q to q_end over the iterations (continuation)
         outlet="dirichlet",       # dirichlet (as Borrvall-Petersson: same windows on the right side) | free (traction-free exit)
         nx=60, ny=60, iters=80, move=0.15, tol=0.03)
ap = argparse.ArgumentParser()
for k, v in P.items(): ap.add_argument("--" + k, type=type(v), default=v)
P = vars(ap.parse_args())
Lx, Ly, mu, U, Vf = P["Lx"], P["Ly"], P["mu"], P["U"], P["Vf"]
amax = P["amax_fac"] * mu / Ly**2

msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]], [P["nx"], P["ny"]], dmesh.CellType.triangle, diagonal=dmesh.DiagonalType.crossed)
el_u = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(2,)); el_p = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
W = fem.functionspace(msh, basix.ufl.mixed_element([el_u, el_p]))
Vu, _ = W.sub(0).collapse(); G = fem.functionspace(msh, ("Lagrange", 1))
gam = fem.Function(G); gam.x.array[:] = Vf
qc = fem.Constant(msh, PETSc.ScalarType(P["q"]))
alpha = lambda g: amax * (1 - g * (1 + qc) / (g + qc))                      # alpha_min = 0
dalpha = lambda g: -amax * (1 + qc) * qc / (g + qc) ** 2

# inlet windows: plane Poiseuille profiles on the left side, walls elsewhere; outlet (right side): same windows (Dirichlet, as Borrvall-Petersson), pressure pinned; --outlet free gives a traction-free exit, which the optimiser exploits by flaring the channel
if P["case"] == "straight":
    wins = [(0.5 * Ly - 0.5 * Vf * Ly, 0.5 * Ly + 0.5 * Vf * Ly)]
else:
    w = Vf * Ly / 2                                                          # two windows, total width Vf*Ly
    wins = [(Ly / 4 - w / 2, Ly / 4 + w / 2), (3 * Ly / 4 - w / 2, 3 * Ly / 4 + w / 2)]
def inlet(x):
    ux = np.zeros_like(x[0])
    for a, b in wins:
        s = (x[1] - a) / (b - a); ux += np.where((s > 0) & (s < 1), 4 * U * s * (1 - s), 0.0)
    return np.vstack([ux, np.zeros_like(ux)])
u_in = fem.Function(Vu); u_in.interpolate(inlet)
walls = fem.Function(Vu); walls.x.array[:] = 0
fdim = msh.topology.dim - 1
left = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0))
tb = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0) | np.isclose(x[1], Ly))
bc_in = fem.dirichletbc(u_in, fem.locate_dofs_topological((W.sub(0), Vu), fdim, left), W.sub(0))
bc_w = fem.dirichletbc(walls, fem.locate_dofs_topological((W.sub(0), Vu), fdim, tb), W.sub(0))
bcs = [bc_in, bc_w]
if P["outlet"] == "dirichlet":
    right = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], Lx))
    bcs.append(fem.dirichletbc(u_in, fem.locate_dofs_topological((W.sub(0), Vu), fdim, right), W.sub(0)))
    Qp, _ = W.sub(1).collapse(); p0 = fem.Function(Qp); p0.x.array[:] = 0.0
    pin = fem.locate_dofs_geometrical((W.sub(1), Qp), lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
    bcs.append(fem.dirichletbc(p0, pin, W.sub(1)))            # pressure is defined up to a constant: pin it

(u, p), (v, qt) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
a = (mu * ufl.inner(ufl.grad(u), ufl.grad(v)) + alpha(gam) * ufl.inner(u, v) - p * ufl.div(v) - qt * ufl.div(u)) * ufl.dx
Lf = ufl.inner(fem.Constant(msh, PETSc.ScalarType((0.0, 0.0))), v) * ufl.dx
sol = fem.Function(W)
prob = fpetsc.LinearProblem(a, Lf, u=sol, bcs=bcs, petsc_options_prefix="stokes_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
uh = sol.sub(0)
Jform = fem.form(0.5 * (mu * ufl.inner(ufl.grad(uh), ufl.grad(uh)) + alpha(gam) * ufl.inner(uh, uh)) * ufl.dx)
gs = ufl.TestFunction(G)
gform = fem.form(0.5 * dalpha(gam) * ufl.inner(uh, uh) * gs * ufl.dx)
mform = fem.form(1.0 * gs * ufl.dx)
vol_nodes = fpetsc.assemble_vector(mform); vol_nodes.assemble(); vn = vol_nodes.array.copy()
vol_tot = msh.comm.allreduce(vn.sum()); target = Vf * vol_tot

def oc(g, dens, move):
    lo, hi = 0.0, 1e12
    for _ in range(80):
        lam = 0.5 * (lo + hi)
        new = np.clip(dens * np.sqrt(np.maximum(-g, 1e-30) / lam), np.maximum(dens - move, 0.0), np.minimum(dens + move, 1.0))
        if (new * vn).sum() > target: lo = lam
        else: hi = lam
    return new

hist = []
for it in range(P["iters"]):
    qc.value = P["q"] * (P["q_end"] / P["q"]) ** (it / max(P["iters"] - 1, 1))
    prob.solve()
    J = msh.comm.allreduce(fem.assemble_scalar(Jform))
    gvec = fpetsc.assemble_vector(gform); gvec.assemble()
    g = gvec.array / vn                                                         # per unit volume (lumped)
    hist.append(J)
    gam.x.array[:] = oc(g, gam.x.array.copy(), P["move"])
    if it % 10 == 0: print(f"iter {it:3d}  q={float(qc.value):.2f}  power P = 2J = {2*J:.5g}  volume {(gam.x.array*vn).sum()/vol_tot:.3f}")
prob.solve(); J = msh.comm.allreduce(fem.assemble_scalar(Jform)); Pw = 2 * J
gray = float(((gam.x.array > 0.05) & (gam.x.array < 0.95)).mean())
res = {"parameters": {k: P[k] for k in P}, "power_P": float(Pw), "gray_fraction": gray, "history_J": [float(h) for h in hist[:: max(1, len(hist) // 40)]]}
if P["case"] == "straight":
    w = Vf * Ly; Q = U * w * 2 / 3                                              # mean of 4U s(1-s) over the window is 2U/3
    Pth = 12 * mu * Lx * Q**2 / w**3
    res.update(power_theory=float(Pth), rel_error=float(Pw / Pth - 1), tolerance=P["tol"], reference="plane Poiseuille channel: P = 12 mu L Q^2 / w^3",
               **{"pass": bool(abs(Pw / Pth - 1) < P["tol"])})
    print(f"straight pipe, Vf={Vf}: power {Pw:.5g} vs Poiseuille {Pth:.5g}: error {100*(Pw/Pth-1):+.2f} % (tol {100*P['tol']:g} %), "
          f"grey fraction {100*gray:.1f} % -> {'PASS' if res['pass'] else 'FAIL'}")
else:
    res["pass"] = True; print(f"double pipe: power {Pw:.5g}, grey fraction {100*gray:.1f} %")
json.dump(res, open("result.json", "w"), indent=1)
# design picture
xy = G.tabulate_dof_coordinates(); np.savez_compressed("design.npz", x=xy[:, 0], y=xy[:, 1], gamma=gam.x.array)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.2, 4.4 * Ly / Lx + 0.6))
tc = ax.tricontourf(xy[:, 0], xy[:, 1], gam.x.array, levels=np.linspace(0, 1, 21), cmap="gray"); plt.colorbar(tc, label=r"fluid fraction $\gamma$")
ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); fig.tight_layout(); fig.savefig("design.png", dpi=130)
sys.exit(0 if res["pass"] else 1)
