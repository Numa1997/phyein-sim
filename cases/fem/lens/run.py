#!/usr/bin/env python3
"""Electrostatic electron lens: axisymmetric Laplace equation (FEniCSx) + electron trajectories.

Physical set-up.  Electrons travel along z through a rotationally symmetric electrostatic field.  With the potential phi(r, z)
measured from the cathode (so the kinetic energy is e*phi) the field is E = -grad phi and the force is -e E = +e grad phi.
Away from charge phi obeys Laplace's equation; with rotational symmetry
        (1/r) d/dr( r dphi/dr ) + d2phi/dz2 = 0,
whose weak form on the (r, z) half plane is   integral r grad(phi).grad(v) dr dz = 0   (the factor r is the area element).

Case `aperture` (analytic benchmark, Davisson & Calbick 1931, Phys. Rev. 38, 585):  a thin plate with a round hole of radius ra
at z = 0 separates a region with a uniform field E1 (left) from one with E2 (right).  The paraxial ray equation
        d/dz( sqrt(phi) r' ) = - phi''/(4 sqrt(phi)) r
integrated across the thin lens (r constant, sqrt(phi) = sqrt(V) inside) gives  1/f = (E2 - E1)/(4 V),   V = potential at the plate.
A ray entering parallel at height r0 therefore leaves with  sqrt(phi) r' = - r0 (E2 - E1)/(4 sqrt(V)).  The script compares this
with the slope obtained from the traced ray (full field, no paraxial approximation).

Case `einzel`: three coaxial tubes (radius a, gap g, lengths L); outer two at V, middle at Vm.  A parallel ray is traced; the
focal distance is compared with the paraxial ray equation integrated on the axis potential (an independent route to the same number).

usage: python3 run.py [--name value ...]   writes result.json, field.png, rays.png; exit 0 = PASS
"""
import argparse, json, sys
import numpy as np
from mpi4py import MPI
import gmsh
import ufl
from dolfinx import fem, mesh as dmesh, geometry
from dolfinx.fem import petsc as fpetsc
from dolfinx.io import gmsh as gmshio
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline

# ---- parameters (every number that can be changed) ------------------------------------------------------------------
P = dict(geometry="aperture",   # aperture | einzel
         ra=1e-3,               # m, hole radius (aperture) or tube radius a (einzel)
         V=1000.0,              # V, potential of the beam where it meets the lens (aperture: at the plate)
         E1=0.0,                # V/m, field left of the plate (aperture)
         E2=2.0e4,              # V/m, field right of the plate (aperture); keep E2*ra << V (thin-lens limit)
         Vm=5000.0,             # V, potential of the middle tube (einzel); outer tubes at V (1500 V gives a focus outside the box)
         gap=0.2,               # einzel: gap between tubes in units of a
         tube=2.0,              # einzel: tube length in units of a
         r0=None,               # launch height of the ray in units of ra; default 0.05 (aperture) or 0.1 (einzel: aberration ~ r0^2 above 0.1, field noise below 0.05)
         hmin=0.025,            # smallest mesh size in units of ra (at the edges)
         tol=0.03,              # accepted relative error
         )
ap = argparse.ArgumentParser()
for k, v in P.items(): ap.add_argument("--" + k, type=(float if v is None else type(v)), default=v)
P = vars(ap.parse_args())
if P["r0"] is None: P["r0"] = 0.05 if P["geometry"] == "aperture" else 0.1
ra, V = P["ra"], P["V"]
EM = 1.75882001076e11            # e/m_e in C/kg

# ---- geometry and mesh (Gmsh) ---------------------------------------------------------------------------------------
gmsh.initialize(["-v", "0"]); gmsh.model.add("lens"); occ = gmsh.model.occ
if P["geometry"] == "aperture":
    dL = dR = 15 * ra; Rmax = 15 * ra; t = 0.05 * ra
    box = occ.addRectangle(-dL, 0, 0, dL + dR, Rmax)
    plate = occ.addRectangle(-t / 2, ra, 0, t, Rmax - ra + 1e-9)
    occ.cut([(2, box)], [(2, plate)])
    zl, zr = -dL, dR
    walls = {"left": V - P["E1"] * dL, "right": V + P["E2"] * dR, "plate": V}
else:
    a = ra; g = P["gap"] * a; Lt = P["tube"] * a; w = 0.04 * a
    zl, zr = -(1.5 * Lt + g + 4 * a), (1.5 * Lt + g + 4 * a); Rmax = 3 * a
    box = occ.addRectangle(zl, 0, 0, zr - zl, Rmax)
    tubes = []
    z0s = [-1.5 * Lt - g, -0.5 * Lt, 0.5 * Lt + g]          # tube starts: [z0, z0+Lt] with the middle tube centred on 0
    z0s = [-1.5 * Lt - g, -0.5 * Lt, 0.5 * Lt + g]
    for z0 in z0s:
        tubes.append((2, occ.addRectangle(z0, a, 0, Lt, w)))
    occ.cut([(2, box)], tubes)
    walls = {"left": V, "right": V}
occ.synchronize()
# size field: fine near every edge of an electrode, coarse far away
hmax = 1.5 * ra
pts = [e for e in gmsh.model.getEntities(0)]
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "PointsList", [p[1] for p in pts if abs(gmsh.model.getValue(0, p[1], [])[1]) > 0 or True])
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", P["hmin"] * ra)
gmsh.model.mesh.field.setNumber(2, "SizeMax", hmax)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.2 * ra)
gmsh.model.mesh.field.setNumber(2, "DistMax", 6 * ra)
gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0); gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
for s in gmsh.model.getEntities(2): pass
gmsh.model.addPhysicalGroup(2, [s[1] for s in gmsh.model.getEntities(2)], 1)
gmsh.model.mesh.generate(2)
res = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
msh = res.mesh if hasattr(res, "mesh") else res[0]
gmsh.finalize()

# ---- FEM: Dirichlet on electrodes, natural (zero-flux) on the axis and the outer cylinder -----------------------------
V2 = fem.functionspace(msh, ("Lagrange", 2))
x = ufl.SpatialCoordinate(msh)
u, v = ufl.TrialFunction(V2), ufl.TestFunction(V2)
a_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * x[0 + 1] * ufl.dx     # x[1] is r (the mesh's second coordinate)
L_form = fem.Constant(msh, 0.0) * v * x[1] * ufl.dx
eps = 1e-9 * ra
bcs = []
def add_bc(marker, value):
    dofs = fem.locate_dofs_geometrical(V2, marker)
    bc_f = fem.Function(V2); bc_f.x.array[:] = value
    bcs.append(fem.dirichletbc(bc_f, dofs))
if P["geometry"] == "aperture":
    add_bc(lambda p: np.isclose(p[0], zl), walls["left"])
    add_bc(lambda p: np.isclose(p[0], zr), walls["right"])
    add_bc(lambda p: (np.abs(p[0]) <= t / 2 + eps) & (p[1] >= ra - eps) & (p[1] < Rmax - eps), walls["plate"])
else:
    add_bc(lambda p: np.isclose(p[0], zl), V); add_bc(lambda p: np.isclose(p[0], zr), V)
    for k, z0 in enumerate(z0s):
        val = P["Vm"] if k == 1 else V
        add_bc(lambda p, z0=z0: (p[0] >= z0 - eps) & (p[0] <= z0 + Lt + eps) & (p[1] >= a - eps) & (p[1] <= a + w + eps), val)
problem = fpetsc.LinearProblem(a_form, L_form, bcs=bcs, petsc_options_prefix="lens_", petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-12})
phi = problem.solve()
print(f"mesh: {msh.topology.index_map(2).size_local} triangles, {V2.dofmap.index_map.size_global} P2 dofs")

# ---- potential on a regular grid (cell lookup) and a smooth spline -------------------------------------------------------
nz, nr = 1400, 360
zg = np.linspace(zl, zr, nz); rg = np.linspace(0, Rmax, nr)
Zg, Rg = np.meshgrid(zg, rg, indexing="ij")
pts3 = np.zeros((Zg.size, 3)); pts3[:, 0] = Zg.ravel(); pts3[:, 1] = Rg.ravel()
tree = geometry.bb_tree(msh, 2)
cand = geometry.compute_collisions_points(tree, pts3)
coll = geometry.compute_colliding_cells(msh, cand, pts3)
cells = np.array([coll.links(i)[0] if len(coll.links(i)) else -1 for i in range(pts3.shape[0])])
ok = cells >= 0
vals = np.full(Zg.size, np.nan)
vals[ok] = phi.eval(pts3[ok], cells[ok]).ravel()
PH = vals.reshape(Zg.shape)
fill = np.nanmean(PH); PH = np.where(np.isnan(PH), fill, PH)           # inside electrodes: never visited by rays
spl = RectBivariateSpline(zg, rg, PH, kx=5, ky=5, s=0)
def phi_f(z, r): return float(spl.ev(z, r))
def grad_f(z, r): return float(spl.ev(z, r, dx=1)), float(spl.ev(z, r, dy=1))

# ---- trajectory: m d2x/dt2 = + e grad(phi) (electron), start with the energy e*phi at the start point -------------------
def trace(r_start, z_start, z_end, n_out=400):
    ph0 = phi_f(z_start, 0.0)
    v0 = np.sqrt(2 * EM * ph0)
    def rhs(t, y):
        z, r, vz, vr = y
        gz, gr = grad_f(z, max(r, 0.0))
        return [vz, vr, EM * gz, EM * gr]
    ev_end = lambda t, y: y[0] - z_end; ev_end.terminal = True
    ev_axis = lambda t, y: y[1]; ev_axis.terminal = False; ev_axis.direction = -1
    T = 4 * (z_end - z_start) / v0
    sol = solve_ivp(rhs, [0, T], [z_start, r_start, v0, 0.0], events=[ev_end, ev_axis], rtol=1e-10, atol=1e-14, dense_output=True, max_step=T / 4000)
    t = np.linspace(0, sol.t[-1], n_out); y = sol.sol(t)
    return y, sol, ev_axis

def slope_N(sol, z_at):
    """sqrt(phi) r' at height z_at along the traced ray (paraxial invariant in a field-free or uniform-field region)."""
    for k in range(len(sol.t) - 1):
        pass
    tt = np.linspace(0, sol.t[-1], 20000); yy = sol.sol(tt)
    i = np.searchsorted(yy[0], z_at)
    z, r, vz, vr = yy[:, i]
    return np.sqrt(phi_f(z, r)) * vr / vz, r

result = {"geometry": P["geometry"], "parameters": {k: P[k] for k in P}}
r0 = P["r0"] * ra
if P["geometry"] == "aperture":
    z_start = -10 * ra; z_end = 8 * ra
    y, sol, _ = trace(r0, z_start, z_end)
    z_meas = 5 * ra
    N_num, r_at = slope_N(sol, z_meas)
    E1, E2 = P["E1"], P["E2"]
    N_th = -r0 * (E2 - E1) / (4 * np.sqrt(V))            # thin-lens prediction (Davisson & Calbick), parallel entry
    # the entry is parallel only if the field E1 does not bend the ray before the plate; for E1 != 0 the entry slope is not zero:
    # correct by integrating the same paraxial equation on the left (N is conserved in a uniform field).
    N_in, _ = slope_N(sol, -3 * ra)
    dN_th = -r0 * (E2 - E1) / (4 * np.sqrt(V))
    dN_num = N_num - N_in
    f_th = 4 * V / (E2 - E1); f_num = f_th * dN_th / dN_num
    err = dN_num / dN_th - 1
    result.update(f_theory_m=f_th, f_traced_m=f_num, rel_error=float(err), dN_num=float(dN_num), dN_theory=float(dN_th),
                  reference="Davisson & Calbick 1931: 1/f = (E2 - E1)/(4 V)", tolerance=P["tol"], **{"pass": bool(abs(err) < P["tol"])})
    print(f"aperture lens: f_theory = {f_th*1e3:.2f} mm, f_traced = {f_num*1e3:.2f} mm, error {100*err:+.2f} % (tol {100*P['tol']:g} %) -> {'PASS' if result['pass'] else 'FAIL'}")
    rays = [(y, "ray")]
else:
    a = ra
    # (1) full tracing of a parallel ray at r0 -> where it crosses the axis
    z_start = zl + 1.0 * a; z_end = zr - 0.5 * a
    y, sol, ev_axis = trace(r0, z_start, z_end)
    zc = sol.t_events[1]
    z_cross = float(sol.sol(zc[0])[0]) if len(zc) else float("nan")
    # (2) paraxial ray equation on the axis potential: d/dz(sqrt(phi) r') = -phi''/(4 sqrt(phi)) r, integrated with the FEM axis potential
    za = np.linspace(zl, zr, 4000); pa = np.array([phi_f(z, 0.0) for z in za])
    dp = np.gradient(pa, za); d2p = np.gradient(dp, za)
    pf = lambda z: np.interp(z, za, pa); d1 = lambda z: np.interp(z, za, dp); d2 = lambda z: np.interp(z, za, d2p)
    def par(z, s):                                     # s = (r, r'):  r'' = -phi'/(2 phi) r' - phi''/(4 phi) r
        r, rp = s; ph = pf(z)
        return [rp, -d1(z) / (2 * ph) * rp - d2(z) / (4 * ph) * r]
    solp = solve_ivp(par, [z_start, z_end], [r0, 0.0], rtol=1e-10, atol=1e-14, dense_output=True,
                     events=lambda z, s: s[0], max_step=a / 50)
    z_cross_par = float(solp.t_events[0][0]) if len(solp.t_events[0]) else float("nan")
    err = z_cross / z_cross_par - 1 if z_cross == z_cross and z_cross_par == z_cross_par else float("nan")
    result.update(z_cross_traced_m=z_cross, z_cross_paraxial_m=z_cross_par, rel_difference=float(err), tolerance=P["tol"],
                  reference="paraxial ray equation (Busch) integrated on the FEM axis potential",
                  **{"pass": bool(abs(err) < P["tol"])})
    print(f"einzel lens: ray crosses the axis at z = {z_cross/a:.3f} a (traced) vs {z_cross_par/a:.3f} a (paraxial equation), "
          f"difference {100*err:+.2f} % (tol {100*P['tol']:g} %) -> {'PASS' if result['pass'] else 'FAIL'}")
    rays = [(y, "ray")]

json.dump(result, open("result.json", "w"), indent=1)
np.savez_compressed("fields.npz", z=zg, r=rg, phi=PH, ray=rays[0][0])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 3.6))
c = ax.contourf(zg * 1e3, rg * 1e3, PH.T, 40, cmap="viridis"); plt.colorbar(c, label="potential (V)")
ax.contour(zg * 1e3, rg * 1e3, PH.T, 18, colors="w", linewidths=0.4)
ax.plot(rays[0][0][0] * 1e3, rays[0][0][1] * 1e3, "r-", lw=1.5, label="electron")
ax.set_xlabel("z (mm)"); ax.set_ylabel("r (mm)"); ax.legend(loc="upper right")
if P["geometry"] == "aperture": ax.set_xlim(-8 * ra * 1e3, 10 * ra * 1e3); ax.set_ylim(0, 6 * ra * 1e3)
fig.tight_layout(); fig.savefig("field.png", dpi=130)
sys.exit(0 if result["pass"] else 1)
