#!/usr/bin/env python3
"""Benchmark: first antisymmetric sloshing frequency f11 of a liquid in a vertical circular tank against linear
potential-flow theory  f11 = (1/2pi) sqrt( g xi11/R tanh(xi11 h/R) ), xi11 = 1.8412 (first zero of J1'),
for the plain cylinder (profile 0). For the glass profile (profile 1) there is no closed form: the reference is the
Guinness page's own value for the glass (fPage in params, 3.421 Hz at 125 mm fill).

The simulated frequency is a damped-cosine fit to the surface height at x = 0.9 R (alpha integrated along z).
usage: check.py <case_dir> [tolerance]    exit 0 = PASS
"""
import sys, os, json, glob
import numpy as np
_trapz = getattr(__import__('numpy'), 'trapezoid', None) or __import__('numpy').trapz   # NumPy 2 renamed trapz
from scipy.optimize import curve_fit

def read_params(case):
    out = {}
    for line in open(os.path.join(case, "params")):
        line = line.split("//")[0].strip().rstrip(";"); parts = line.split(None, 1)
        if len(parts) == 2 and not parts[1].startswith("#eval"):
            try: out[parts[0]] = float(parts[1])
            except ValueError: pass
    return out

def main():
    case = sys.argv[1]; tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    p = read_params(case); g, R, h, xi = p["g"], p["R"], p["h"], p["xi11"]
    f_lin = np.sqrt(g * xi / R * np.tanh(xi * h / R)) / (2 * np.pi)
    glass = p.get("profile", 0) >= 1
    graded = True
    if glass:
        f_lin = p["fPage"]                                       # glass: compare with the page's value (no closed form) ...
        graded = abs(h - 0.125) < 1e-4                           # ... which exists only for the 125 mm fill; at any other fill nothing is graded
    k = xi / R                                                   # wave number of the mode
    f_cap = np.sqrt((g * k + p["sigma"] * k**3 / p["rhoW"]) * np.tanh(k * h)) / (2 * np.pi)   # with surface tension (rhoA << rhoW)
    ts, hs = [], []
    for d in sorted(glob.glob(os.path.join(case, "postProcessing", "wallLine", "*")), key=lambda s: float(os.path.basename(s))):
        z, al = np.loadtxt(os.path.join(d, "line_alpha.water.xy"), unpack=True)
        ts.append(float(os.path.basename(d))); hs.append(_trapz(al, z))
    t, y = np.array(ts), np.array(hs); y = y - y.mean()
    model = lambda t, A, gam, f, ph: A * np.exp(-gam * t) * np.cos(2 * np.pi * f * t + ph)
    popt, _ = curve_fit(model, t, y, p0=[y[0], 1.0, f_lin, 0.0])
    f_sim = abs(popt[2]); err = (f_sim - f_lin) / f_lin
    res = {"f_sim_Hz": float(f_sim), "f_linear_Hz": float(f_lin), "rel_error": float(err), "f_capillary_Hz": float(f_cap), "damping_per_s": float(popt[1]),
           "amplitude_m": float(abs(popt[0])), "tolerance": tol, "reference": "Guinness page 3.421 Hz" if p.get("profile",0)>=1 else "linear sloshing theory (Lamb)",
           "profile": int(p.get("profile", 0)), "graded": graded, "pass": bool(abs(err) < tol) if graded else True}
    json.dump(res, open(os.path.join(case, "result.json"), "w"), indent=1)
    k = max(1, len(t) // 400)                            # curves for the website: surface height at the wall against time, simulation and the fitted damped cosine
    json.dump({"x_label": "t (s)", "y_label": "surface height deviation (m)",
               "simulated": {"x": np.round(t[::k], 5).tolist(), "y": np.round(y[::k], 7).tolist()},
               "fit": {"x": np.round(t[::k], 5).tolist(), "y": np.round(model(t[::k], *popt), 7).tolist(), "f_Hz": float(f_sim)},
               "benchmark": {"f_Hz": float(f_lin), "graded": graded, "source": "Lamb, Hydrodynamics, 257 (cylinder); Guinness page 3.421 Hz at 125 mm (glass)"}},
              open(os.path.join(case, "curves.json"), "w"))
    print(f"slosh R={R*1e3:g} mm h={h*1e3:g} mm: f_sim = {f_sim:.3f} Hz, reference {f_lin:.3f} Hz, error {100*err:+.2f} % "
          f"(with surface tension {f_cap:.3f} Hz) (tolerance {100*tol:g} %), damping {popt[1]:.2f} 1/s -> "
          + (("PASS" if res["pass"] else "FAIL") if graded else "NOT GRADED (the page's 3.421 Hz is valid only at h = 125 mm; the 'reference' shown is not comparable)"))
    return 0 if res["pass"] else 1

if __name__ == "__main__":
    sys.exit(main())
