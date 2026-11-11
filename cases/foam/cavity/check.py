#!/usr/bin/env python3
"""Benchmark: u along the vertical centre line of the lid-driven cavity against Ghia, Ghia & Shin (1982),
J. Comput. Phys. 48, 387-411, Table I (u-velocity along the vertical line through the geometric centre).
Reference values transcribed from a public copy of that table; the Re = 100 column was compared with the
values quoted by several published studies, and the run itself acts as a second check (a mistyped
entry would stand out as an outlier against a mesh-converged solution).

usage: check.py <case_dir> [tolerance]      exit 0 = PASS (max |u - u_Ghia| / U below tolerance)
"""
import sys, json, glob, os
import numpy as np

# y/L, then u/U for Re = 100, 400, 1000 (Ghia et al. 1982, Table I)
GHIA_Y = [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000]
GHIA_U = {
    100:  [1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.0],
    400:  [1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.0],
    1000: [1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.0],
}

def read_params(case):
    out = {}
    for line in open(os.path.join(case, "params")):
        line = line.split("//")[0].strip().rstrip(";")
        if not line or line.startswith(("/", "\\", "F", "}")):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and not parts[1].startswith("#eval"):
            try: out[parts[0]] = float(parts[1])
            except ValueError: pass
    return out

def main():
    case = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    p = read_params(case)
    Re, U, L = int(round(p["Re"])), p["U"], p["L"]
    if Re not in GHIA_U:
        print(f"no Ghia column for Re = {Re}; available: {sorted(GHIA_U)}"); return 2
    f = sorted(glob.glob(os.path.join(case, "postProcessing", "centreLines", "*", "vertical_U.xy")),
               key=lambda s: float(s.split(os.sep)[-2]))[-1]
    d = np.loadtxt(f)
    y, u = d[:, 0] / L, d[:, 1] / U
    uc = np.interp(GHIA_Y, y, u)
    err = np.abs(uc - np.array(GHIA_U[Re]))[1:-1]          # drop the two end points (set by the boundary conditions)
    worst = int(np.argmax(err)) + 1
    res = {"Re": Re, "N": int(p["N"]), "max_abs_error": float(err.max()), "rms_error": float(np.sqrt((err**2).mean())),
           "worst_y": GHIA_Y[worst], "tolerance": tol, "reference": "Ghia, Ghia & Shin 1982, Table I",
           "pass": bool(err.max() < tol)}
    json.dump(res, open(os.path.join(case, "result.json"), "w"), indent=1)
    print(f"cavity Re={Re} N={res['N']}: max |u-u_Ghia|/U = {res['max_abs_error']:.4f} (at y/L={res['worst_y']}), rms {res['rms_error']:.4f}, tolerance {tol} -> {'PASS' if res['pass'] else 'FAIL'}")
    return 0 if res["pass"] else 1

if __name__ == "__main__":
    sys.exit(main())
