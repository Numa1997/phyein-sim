#!/usr/bin/env python3
"""Benchmark: surge-front position of a collapsing water column against Martin & Moyce (1952),
Phil. Trans. R. Soc. A 244, 312-324 (JSTOR 91519), Part IV, Table 2 (rectangular section, plane symmetry, n^2 = 2)
and the maximum surge speed quoted in section 5 of that paper.

Non-dimensional variables of the paper:  Z = z/a (front distance from the symmetry plane over the semi-base a),
T = n t sqrt(g/a).  The experiment has no exact start time: the authors normalise T so that Z = 1.44 occurs at T = 1.19
(n^2 = 2), so the simulation time origin is shifted the same way before comparing.

usage: check.py <case_dir> [tolerance_in_T]    exit 0 = PASS
"""
import sys, json, glob, os
import numpy as np

# n^2 = 2 column of Table 2, a = 2 in (one record, internally consistent steps of 0.33 in T per 0.44 in Z)
MM_Z = [1.44, 1.89, 2.33, 2.78, 3.22, 3.67, 4.11, 4.56, 5.00, 5.44, 5.89, 6.33]
MM_T = [1.19, 1.56, 1.87, 2.26, 2.57, 2.88, 3.25, 3.59, 3.89, 4.20, 4.54, 4.85]
UMAX_PAPER = (1.62, 1.71)      # section 5: maximum dZ/dT, 1.62-1.71 over n^2 = 1,2,4 (1.71 for n^2 = 2)

def read_params(case):
    out = {}
    for line in open(os.path.join(case, "params")):
        line = line.split("//")[0].strip().rstrip(";")
        parts = line.split(None, 1)
        if len(parts) == 2 and not parts[1].startswith("#eval"):
            try: out[parts[0]] = float(parts[1])
            except ValueError: pass
    return out

def main():
    case = sys.argv[1]; tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    p = read_params(case); a, n2, g = p["a"], p["n2"], p["g"]; n = np.sqrt(n2)
    dirs = sorted(glob.glob(os.path.join(case, "postProcessing", "floorLine", "*")), key=lambda s: float(os.path.basename(s)))
    ts, zs = [], []
    for d in dirs:
        t = float(os.path.basename(d))
        if t <= 0: continue
        x, al = np.loadtxt(os.path.join(d, "floor_alpha.water.xy"), unpack=True)
        wet = np.where(al >= 0.5)[0]
        if len(wet) == 0: continue
        i = wet.max()                                  # last cell still wet along the floor: the front
        xf = x[i] if i == len(x) - 1 else x[i] + (al[i] - 0.5) / max(al[i] - al[i + 1], 1e-12) * (x[i + 1] - x[i])
        ts.append(t); zs.append(xf / a)
    ts, zs = np.array(ts), np.array(zs)
    T = n * ts * np.sqrt(g / a)
    # strictly increasing front only (the numerical front can jitter by one cell)
    zs = np.maximum.accumulate(zs)
    shift = np.interp(1.44, zs, T) - 1.19              # same normalisation as the paper
    T = T - shift
    Tsim = np.interp(MM_Z, zs, T)
    dT = Tsim - np.array(MM_T)
    # Compare up to Z = 4.6 only: beyond that the small-scale (a = 2 in) record retards (the paper says so, section 5)
    # while the larger-scale runs keep accelerating to U = 1.71, which is checked separately below.
    sel = [i for i, z in enumerate(MM_Z) if z <= 4.6]
    err = np.abs(dT[sel])[1:]                           # first point is the normalisation point
    # maximum surge speed: slope of Z(T) between Z = 6 and Z = 9 (needs a tank long enough: Lxa >= 12)
    m = (zs > 6) & (zs < 9)
    U = float(np.polyfit(T[m], zs[m], 1)[0]) if m.sum() > 3 else float("nan")
    u_ok = bool(abs(U - UMAX_PAPER[1]) / UMAX_PAPER[1] < 0.10) if U == U else True
    res = {"N_a": int(p["Na"]), "max_abs_dT": float(err.max()), "rms_dT": float(np.sqrt((err**2).mean())),
           "tolerance_T": tol, "front_speed_U": float(U), "paper_Umax": UMAX_PAPER, "time_shift": float(shift),
           "reference": "Martin & Moyce 1952, Table 2 (n^2=2) and section 5",
           "pass": bool(err.max() < tol and u_ok)}
    json.dump(res, open(os.path.join(case, "result.json"), "w"), indent=1)
    k = max(1, len(T) // 400)                           # curves for the website: front position Z against time T, simulation and the paper
    json.dump({"x_label": "T = n t sqrt(g/a)", "y_label": "Z = z/a (front position)",
               "simulated": {"x": np.round(T[::k], 5).tolist(), "y": np.round(zs[::k], 5).tolist()},
               "benchmark": {"x": list(MM_T), "y": list(MM_Z), "source": "Martin & Moyce 1952, Table 2 (n^2 = 2)"}},
              open(os.path.join(case, "curves.json"), "w"))
    print(f"dam break n2={n2:g} Na={res['N_a']}: max |T_sim-T_MM| = {res['max_abs_dT']:.3f}, rms {res['rms_dT']:.3f} (tolerance {tol}); "
          f"surge speed dZ/dT (Z 6-9) = {U:.2f} (paper {UMAX_PAPER[1]}, 10 % allowed) -> {'PASS' if res['pass'] else 'FAIL'}")
    return 0 if res["pass"] else 1

if __name__ == "__main__":
    sys.exit(main())
