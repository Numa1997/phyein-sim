#!/usr/bin/env python3
"""Benchmark: Strouhal number and mean drag of the laminar vortex street behind a circular cylinder at Re = 100.
St from the lift coefficient (mean spacing of its up-crossings), CD averaged over whole periods after t > tAvg.
References: Williamson (1989) J. Fluid Mech. 206, 579 (St = 0.164 at Re = 100, unbounded); Henderson (1995)
J. Fluid Mech. 285, 69 (CD about 1.35 at Re = 100 in 2D).  A side wall at 15 D (blockage 3.3 %) raises both by a few %.
usage: check.py <case_dir> [tol_St] [tol_CD]    exit 0 = PASS
"""
import sys, os, json, glob
import numpy as np

ST_REF, CD_REF = 0.165, 1.35

def read_params(case):
    out = {}
    for line in open(os.path.join(case, "params")):
        line = line.split("//")[0].strip().rstrip(";"); parts = line.split(None, 1)
        if len(parts) == 2 and not parts[1].startswith("#eval"):
            try: out[parts[0]] = float(parts[1])
            except ValueError: pass
    return out

def main():
    case = sys.argv[1]
    tolS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05    # relative
    tolC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    p = read_params(case)
    f = sorted(glob.glob(os.path.join(case, "postProcessing", "forces", "*", "coefficient.dat")))[-1]
    rows = [l.split() for l in open(f) if not l.startswith("#") and l.strip()]
    # columns: Time Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) CmPitch ...
    t = np.array([float(r[0]) for r in rows]); cd = np.array([float(r[1]) for r in rows]); cl = np.array([float(r[4]) for r in rows])
    m = t > p["tAvg"]
    t, cd, cl = t[m], cd[m], cl[m]
    cl0 = cl - cl.mean()
    up = np.where((cl0[:-1] < 0) & (cl0[1:] >= 0))[0]            # upward zero crossings, linearly interpolated
    tc = t[up] - cl0[up] * (t[up + 1] - t[up]) / (cl0[up + 1] - cl0[up])
    if len(tc) < 4: print("too few shedding periods after tAvg"); return 2
    T = np.mean(np.diff(tc)); St = p["D"] / (T * p["U"])
    sel = (t >= tc[0]) & (t <= tc[-1])                           # whole periods only
    CD = cd[sel].mean(); CLrms = cl0[sel].std() * np.sqrt(2)
    res = {"St": float(St), "CD_mean": float(CD), "CL_amplitude": float(CLrms), "periods": int(len(tc) - 1),
           "St_ref": ST_REF, "CD_ref": CD_REF, "tol_St_rel": tolS, "tol_CD_rel": tolC, "Re": p["Re"], "blockage": 1 / p["Hh"],
           "reference": "Williamson 1989 (St), Henderson 1995 (CD)",
           "pass": bool(abs(St / ST_REF - 1) < tolS and abs(CD / CD_REF - 1) < tolC)}
    json.dump(res, open(os.path.join(case, "result.json"), "w"), indent=1)
    k = max(1, len(t) // 400)                            # curves for the website: lift and drag coefficients against time
    json.dump({"x_label": "t U/D", "simulated": {"x": np.round(t[::k], 4).tolist(), "C_L": np.round(cl[::k], 5).tolist(), "C_D": np.round(cd[::k], 5).tolist()},
               "benchmark": {"St": ST_REF, "C_D": CD_REF, "source": "Williamson 1989; Henderson 1995"}}, open(os.path.join(case, "curves.json"), "w"))
    print(f"cylinder Re={p['Re']:g}: St = {St:.4f} (ref {ST_REF}, {100*(St/ST_REF-1):+.1f} %), CD = {CD:.3f} (ref {CD_REF}, {100*(CD/CD_REF-1):+.1f} %), "
          f"CL amplitude {CLrms:.2f}, {len(tc)-1} periods -> {'PASS' if res['pass'] else 'FAIL'}")
    return 0 if res["pass"] else 1

if __name__ == "__main__":
    sys.exit(main())
