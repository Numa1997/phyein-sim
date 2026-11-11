#!/usr/bin/env python3
"""After blockMesh: (1) optionally scale the mesh radius with R(z) from profile.csv (glass), then
(2) write alpha.water for a planar free surface z = h + eps x (fractional in the cells it cuts).
usage: init_fields.py <case_dir>   (run `postProcess -func writeCellCentres -time 0` first when profile=0; this script runs it itself otherwise)"""
import sys, os, re, subprocess
import numpy as np

case = sys.argv[1]
P = {}
for line in open(os.path.join(case, "params")):
    line = line.split("//")[0].strip().rstrip(";"); parts = line.split(None, 1)
    if len(parts) == 2 and not parts[1].startswith("#eval"):
        try: P[parts[0]] = float(parts[1])
        except ValueError: pass
R, h, eps, H, Nz = P["R"], P["h"], P["eps"], P["h"] + P["Hair"], round((P["h"] + P["Hair"]) / 0.0025)

def read_list(path, ncomp):
    txt = open(path).read()
    m = re.search(r"internalField\s+nonuniform\s+List<\w+>\s*\n?\s*(\d+)\s*\(", txt) or re.search(r"\n(\d+)\s*\n\(", txt)
    n = int(m.group(1)); body = txt[m.end():]
    nums = re.findall(r"[-+0-9.eE]+", body.split("\n)\n")[0].replace("(", " ").replace(")", " "))
    return np.array(nums[: n * ncomp], float).reshape(n, ncomp)

if P.get("profile", 0) >= 1:                           # radial profile R(z): rewrite the points file
    prof = np.loadtxt(os.path.join(case, "profile.csv"), delimiter=",", comments="#")   # columns z [m], R [m]
    pf = os.path.join(case, "constant", "polyMesh", "points")
    txt = open(pf).read(); m = re.search(r"\n(\d+)\s*\n\(", txt); n = int(m.group(1))
    head, rest = txt[: m.end()], txt[m.end():]
    body, tail = rest.split("\n)\n", 1)
    pts = np.array(re.findall(r"[-+0-9.eE]+", body.replace("(", " ").replace(")", " ")), float).reshape(n, 3)
    f = np.interp(pts[:, 2], prof[:, 0], prof[:, 1]) / R
    pts[:, 0] *= f; pts[:, 1] *= f
    open(pf, "w").write(head + "\n".join("(%.10g %.10g %.10g)" % tuple(p) for p in pts) + "\n)\n" + tail)

subprocess.run(["postProcess", "-func", "writeCellCentres", "-time", "0", "-case", case], check=True,
               stdout=open(os.path.join(case, "log.cellCentres"), "w"), stderr=subprocess.STDOUT)
C = read_list(os.path.join(case, "0", "C"), 3)
dz = H / Nz
a = np.clip((h + eps * C[:, 0] - C[:, 2]) / dz + 0.5, 0.0, 1.0)
a[np.abs(a) < 1e-9] = 0.0
f = os.path.join(case, "0", "alpha.water")
txt = open(f).read()
txt = re.sub(r"internalField\s+uniform\s+0;", "internalField nonuniform List<scalar>\n%d\n(\n%s\n)\n;" % (len(a), "\n".join("%.6g" % v for v in a)), txt)
open(f, "w").write(txt)
print("alpha.water written: %d cells, liquid volume fraction %.4f" % (len(a), a.mean()))
