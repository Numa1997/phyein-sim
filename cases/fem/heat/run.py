#!/usr/bin/env python3
"""Transient heat conduction in a square plate (FEniCSx), against the exact decaying mode.

Set-up.  Plate L x L, thickness small (2D), conductivity k, density rho, specific heat c; thermal diffusivity alpha = k/(rho c).
Energy balance on a small piece:  rho c dT/dt = div(k grad T)   ->   dT/dt = alpha laplacian T.
Edges held at T = 0 (a heat sink).  Initial T = T0 sin(pi x/L) sin(pi y/L) is an eigenfunction of the Laplacian with eigenvalue
-2 (pi/L)^2, so the exact solution stays the same shape and decays:  T(x,y,t) = T0 exp(-2 alpha (pi/L)^2 t) sin sin.
Weak form (theta scheme):  (T^{n+1} - T^n) v/dt + alpha [theta grad T^{n+1} + (1-theta) grad T^n].grad v = 0, integrated over the plate.
theta = 1/2 is Crank-Nicolson (second order in time), theta = 1 backward Euler.  The script also reports the observed order of
convergence in space from three meshes.

usage: python3 run.py [--name value ...]   exit 0 = PASS
"""
import argparse, json, sys
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import petsc as fpetsc

P = dict(L=0.1, k=50.0, rho=7850.0, c=460.0, T0=100.0,   # m, W/(m K), kg/m^3, J/(kg K), K : steel plate
         N=32, steps=50, theta=0.5, degree=2, taus=1.0,  # mesh cells per side, time steps, scheme, element degree, end time in decay times
         tol=1e-3)
ap = argparse.ArgumentParser()
for k_, v_ in P.items(): ap.add_argument("--" + k_, type=type(v_), default=v_)
P = vars(ap.parse_args())

def solve(N, steps):
    L, alpha = P["L"], P["k"] / (P["rho"] * P["c"])
    tau = 1.0 / (2 * alpha * (np.pi / L) ** 2)           # decay time of the mode
    tEnd = P["taus"] * tau; dt = tEnd / steps
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, N, N)  # unit square, scaled by L below (x -> L x)
    V = fem.functionspace(msh, ("Lagrange", P["degree"]))
    Tn, Tnew = fem.Function(V), fem.Function(V)
    exact = lambda x, t: P["T0"] * np.exp(-t / tau) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    Tn.interpolate(lambda x: exact(x, 0.0))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    th = P["theta"]; D = alpha / L**2                    # unit-square coordinates: laplacian scales as 1/L^2
    a = fem.form(u * v * ufl.dx + dt * D * th * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    Lf = fem.form(Tn * v * ufl.dx - dt * D * (1 - th) * ufl.inner(ufl.grad(Tn), ufl.grad(v)) * ufl.dx)
    bdofs = fem.locate_dofs_topological(V, 1, dmesh.exterior_facet_indices(msh.topology)) if False else None
    msh.topology.create_connectivity(1, 2)
    facets = dmesh.exterior_facet_indices(msh.topology)
    bc = fem.dirichletbc(0.0, fem.locate_dofs_topological(V, 1, facets), V)
    A = fpetsc.assemble_matrix(a, bcs=[bc]); A.assemble()
    from petsc4py import PETSc
    ksp = PETSc.KSP().create(msh.comm); ksp.setOperators(A); ksp.setType("preonly"); ksp.getPC().setType("lu")
    b = A.createVecRight(); 
    for n in range(steps):
        with b.localForm() as lb: lb.set(0)
        fpetsc.assemble_vector(b, Lf); fpetsc.apply_lifting(b, [a], [[bc]]); b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fpetsc.set_bc(b, [bc]); ksp.solve(b, Tnew.x.petsc_vec); Tnew.x.scatter_forward(); Tn.x.array[:] = Tnew.x.array
    Te = fem.Function(V); Te.interpolate(lambda x: exact(x, tEnd))
    err = np.abs(Tn.x.array - Te.x.array).max() / P["T0"]
    return err, tEnd, Tn

err, tEnd, Tn = solve(P["N"], P["steps"])
errs = [solve(n, 4 * n)[0] for n in (4, 8, 16)]          # space and time refined together (degree 2, theta 1/2: error ~ h^3 ... dt^2)
orders = [float(np.log2(errs[i] / errs[i + 1])) for i in range(2)]
res = {"parameters": {k: P[k] for k in P}, "t_end_s": float(tEnd), "max_error_over_T0": float(err), "tolerance": P["tol"],
       "refinement_errors": [float(e) for e in errs], "observed_orders": orders,
       "reference": "exact decaying eigenmode", "pass": bool(err < P["tol"])}
json.dump(res, open("result.json", "w"), indent=1)
print(f"heat: N={P['N']}, {P['steps']} steps (theta={P['theta']}, P{P['degree']}), end time {tEnd:.2f} s: max error {err:.2e} T0 (tol {P['tol']:g}); "
      f"observed orders under joint refinement {orders[0]:.2f}, {orders[1]:.2f} -> {'PASS' if res['pass'] else 'FAIL'}")
sys.exit(0 if res["pass"] else 1)
