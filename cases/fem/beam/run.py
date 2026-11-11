#!/usr/bin/env python3
"""Free vibration of a clamped-free (cantilever) beam in 3D linear elasticity (FEniCSx, SLEPc), checked against Euler-Bernoulli
beam theory, and cross-checked with CalculiX on an identical hexahedral mesh.

Set-up.  A prismatic bar, length L along x, cross-section b (y) by h (z), clamped at x = 0.  Small displacements u(x, t) = U(x) cos(w t) obey
        div sigma(U) + rho w^2 U = 0,   sigma = 2 mu eps + lambda tr(eps) I,   eps = sym grad U,
and the weak form  K U = w^2 M U  with  K_ij = integral sigma(phi_j):eps(phi_i) dx,  M_ij = rho integral phi_j.phi_i dx.
Euler-Bernoulli (thin beam, no shear or rotary inertia): bending angular frequencies
        w_n = (beta_n L)^2 sqrt(E I /(rho A)) / L^2,  cos(beta L) cosh(beta L) = -1,  beta_n L = 1.8751, 4.6941, 7.8548, 10.9955,
I = h b^3/12 for motion in y, I = b h^3/12 for motion in z (a square section gives equal pairs).  Axial: w = (pi/2) sqrt(E/rho)/L.
Timoshenko effects (shear, rotary inertia) lower the frequencies by about 1 % (mode 1) to 8 % (mode 4) for L/h = 25; the comparison is for
the first two modes, where Euler-Bernoulli is the right reference.

usage: python3 run.py [--name value ...]   writes result.json; exit 0 = PASS
"""
import argparse, json, sys, shutil, subprocess, os, re
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import ufl
from dolfinx import fem, mesh as dmesh, geometry
from dolfinx.fem import petsc as fpetsc

P = dict(L=0.5, b=0.02, h=0.02, E=210e9, nu=0.3, rho=7850.0,   # m, m, m, Pa, -, kg/m^3 (steel)
         nx=40, ny=4, nz=4,                                     # cells along L, b, h (hexahedra, quadratic)
         nmodes=12, tol=0.02, ccx=1)
ap = argparse.ArgumentParser()
for k, v in P.items(): ap.add_argument("--" + k, type=type(v), default=v)
P = vars(ap.parse_args())
L, b, h, E, nu, rho = (P[k] for k in ("L", "b", "h", "E", "nu", "rho"))
mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

msh = dmesh.create_box(MPI.COMM_WORLD, [[0, 0, 0], [L, b, h]], [P["nx"], P["ny"], P["nz"]], dmesh.CellType.hexahedron)
V = fem.functionspace(msh, ("Lagrange", 2, (3,)))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
eps = lambda w: ufl.sym(ufl.grad(w))
sig = lambda w: 2 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(3)
k_form = fem.form(ufl.inner(sig(u), eps(v)) * ufl.dx)
m_form = fem.form(rho * ufl.inner(u, v) * ufl.dx)
clamp = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
bc = fem.dirichletbc(np.zeros(3), clamp, V)
K = fpetsc.assemble_matrix(k_form, bcs=[bc], diag=1.0); K.assemble()
M = fpetsc.assemble_matrix(m_form, bcs=[bc], diag=0.0); M.assemble()      # constrained rows give w^2 = infinity, never selected
print(f"hexahedra: {msh.topology.index_map(3).size_local}, dofs: {V.dofmap.index_map.size_global * 3}")

# lowest eigenvalues: shift-invert about a small positive value
eps_s = SLEPc.EPS().create(MPI.COMM_WORLD)
eps_s.setOperators(K, M); eps_s.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps_s.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE); eps_s.setTarget((2 * np.pi * 10.0) ** 2)
eps_s.setDimensions(P["nmodes"], PETSc.DECIDE, PETSc.DECIDE); eps_s.setTolerances(1e-10, 200)
st = eps_s.getST(); st.setType(SLEPc.ST.Type.SINVERT)
ksp = st.getKSP(); ksp.setType("preonly"); ksp.getPC().setType("lu")
eps_s.solve()
n = eps_s.getConverged()
vr = K.createVecRight()
modes = []
tree = geometry.bb_tree(msh, 3)
xs = np.linspace(0.05 * L, L * 0.999, 40)
cl = np.zeros((40, 3)); cl[:, 0] = xs; cl[:, 1] = b / 2; cl[:, 2] = h / 2
cells = [geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, cl), cl).links(i)[0] for i in range(40)]
uf = fem.Function(V)
for i in range(n):
    lam_i = eps_s.getEigenpair(i, vr).real
    if lam_i <= 0 or lam_i > 1e12: continue
    uf.x.array[:] = vr.array_r
    d = uf.eval(cl, np.array(cells, dtype=np.int32))          # displacement along the centre line
    c = np.abs(d).max(axis=0)                                  # (ux, uy, uz) amplitude on the axis
    kind = "torsion" if c.max() < 0.05 * np.abs(uf.x.array).max() else ["axial", "bending-y", "bending-z"][int(np.argmax(c))]
    modes.append((np.sqrt(lam_i) / (2 * np.pi), kind))
modes.sort()

beta = [1.8751041, 4.6940911, 7.8547574, 10.9955407]
Iy, Iz, A = h * b**3 / 12, b * h**3 / 12, b * h                # I about z (motion in y) and about y (motion in z)
f_eb = lambda bl, I: bl**2 * np.sqrt(E * I / (rho * A)) / L**2 / (2 * np.pi)
rows = []
for kind, I in (("bending-y", Iy), ("bending-z", Iz)):
    fem_f = [f for f, k in modes if k == kind]
    for j, f in enumerate(fem_f[:3]):
        rows.append({"kind": kind, "mode": j + 1, "f_fem_Hz": f, "f_euler_bernoulli_Hz": f_eb(beta[j], I), "rel_error": f / f_eb(beta[j], I) - 1})
ax = [f for f, k in modes if k == "axial"]
if ax: rows.append({"kind": "axial", "mode": 1, "f_fem_Hz": ax[0], "f_euler_bernoulli_Hz": np.sqrt(E / rho) / (4 * L), "rel_error": ax[0] / (np.sqrt(E / rho) / (4 * L)) - 1})
for r in rows: print("%-10s mode %d: FEM %8.2f Hz, beam theory %8.2f Hz, %+.2f %%" % (r["kind"], r["mode"], r["f_fem_Hz"], r["f_euler_bernoulli_Hz"], 100 * r["rel_error"]))
first2 = [abs(r["rel_error"]) for r in rows if r["kind"].startswith("bending") and r["mode"] <= 2]
res = {"parameters": {k: P[k] for k in P}, "modes": rows, "reference": "Euler-Bernoulli cantilever, beta_n L = 1.8751, 4.6941",
       "tolerance": P["tol"], "max_error_modes_1_2": float(max(first2)), "pass": bool(max(first2) < P["tol"])}

# ---- CalculiX cross-check on the same grid of 20-node bricks (C3D20), eigenfrequencies *FREQUENCY -------------------------
if P["ccx"] and shutil.which("ccx"):
    nx, ny, nz = P["nx"], P["ny"], P["nz"]
    idx = lambda i, j, k: 1 + i + (2 * nx + 1) * (j + (2 * ny + 1) * k)       # lattice of half-steps
    lines = ["*NODE"]
    keep = set()
    for k in range(2 * nz + 1):
        for j in range(2 * ny + 1):
            for i in range(2 * nx + 1):
                if (i % 2) + (j % 2) + (k % 2) <= 1 or True: pass
                lines.append(f"{idx(i,j,k)}, {i*L/(2*nx):.9g}, {j*b/(2*ny):.9g}, {k*h/(2*nz):.9g}")
    lines.append("*ELEMENT, TYPE=C3D20, ELSET=EALL")
    e = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                I, J, K_ = 2 * i, 2 * j, 2 * k
                c = [(0,0,0),(2,0,0),(2,2,0),(0,2,0),(0,0,2),(2,0,2),(2,2,2),(0,2,2),
                     (1,0,0),(2,1,0),(1,2,0),(0,1,0),(1,0,2),(2,1,2),(1,2,2),(0,1,2),(0,0,1),(2,0,1),(2,2,1),(0,2,1)]
                nd = [str(idx(I + a_, J + b_, K_ + c_)) for a_, b_, c_ in c]
                lines.append(f"{e}, " + ", ".join(nd[:15]) + ",")                  # ccx: at most 16 entries per line
                lines.append(", ".join(nd[15:])); e += 1
    fixed = [idx(0, j, k) for k in range(2 * nz + 1) for j in range(2 * ny + 1)]
    lines += ["*NSET, NSET=FIX"] + [", ".join(map(str, fixed[q:q + 8])) for q in range(0, len(fixed), 8)]
    lines += ["*MATERIAL, NAME=STEEL", "*ELASTIC", f"{E}, {nu}", "*DENSITY", f"{rho}", "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL",
              "*BOUNDARY", "FIX, 1, 3", "*STEP", "*FREQUENCY, SOLVER=SPOOLES", "12", "*NODE FILE", "U", "*END STEP"]
    open("beam.inp", "w").write("\n".join(lines) + "\n")
    try:
        subprocess.run(["ccx", "beam"], check=True, capture_output=True, timeout=600)
        txt = open("beam.dat").read()
        fr = [float(m.group(1)) for m in re.finditer(r"^\s+\d+\s+[-\d.E+]+\s+[-\d.E+]+\s+([-\d.E+]+)\s+[-\d.E+]+\s*$", txt, re.M)]
        res["calculix_Hz"] = fr[:12]
        fem_all = [f for f, _ in modes][:len(fr[:12])]
        res["calculix_vs_fem_max_rel"] = float(max(abs(a_ / b_ - 1) for a_, b_ in zip(fr, fem_all) if b_ > 0))
        print("CalculiX C3D20 frequencies (Hz):", ", ".join(f"{x:.1f}" for x in fr[:8]), f"| max difference to FEniCSx {100*res['calculix_vs_fem_max_rel']:.2f} %")
    except Exception as ex:
        res["calculix_error"] = str(ex)[:200]; print("CalculiX cross-check failed:", ex)
json.dump(res, open("result.json", "w"), indent=1)
print("beam modes ->", "PASS" if res["pass"] else "FAIL")
sys.exit(0 if res["pass"] else 1)
