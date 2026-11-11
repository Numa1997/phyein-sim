# phyein-sim on Kaggle: one case (or a small sweep) run natively, graded by the case's own check.
# Save this text as a Kaggle *script* notebook with Internet ON (GPU off: OpenFOAM and DOLFINx do not use it). Edit only the block below.
OWNER = "numa1997"            # GitHub login that holds phyein-sim
REF   = "main"                # branch or, for a reproducible result, a full commit sha
TOOL  = "foam"                # foam | fem
CASE  = "cavity"              # folder below cases/<TOOL>/
SETS  = ["N=32"]              # one entry per run: "NAME=value NAME=value"; several entries = a sweep, run one after another
NP    = 4                     # MPI ranks for foam cases (a Kaggle CPU session has 4 cores)
# ----------------------------------------------------------------------------------------------------------------------
import json, os, platform, shutil, subprocess, time
W = "/kaggle/working"; R = "/kaggle/tmp/phyein-sim"
def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(["bash", "-c", cmd], **kw)
t = {"start": time.time()}
print("cores:", os.cpu_count(), "| python:", platform.python_version(), "| machine:", platform.platform())
sh("grep -m1 'model name' /proc/cpuinfo; free -g | sed -n 2p")
os.makedirs("/kaggle/tmp", exist_ok=True)
if sh(f"rm -rf {R} && git clone -q https://github.com/{OWNER}/phyein-sim {R} && cd {R} && git checkout -q {REF} && git rev-parse HEAD > {W}/commit.txt").returncode:
    raise SystemExit("clone failed: is Internet switched on for this notebook, and is the repository public?")
if sh(f"bash {R}/kaggle/bootstrap.sh {TOOL} 2>&1 | tail -5").returncode: raise SystemExit("install failed (see the lines above)")
t["installed"] = time.time()
runs = []
for i, s in enumerate(SETS):
    out = f"{W}/run{i}"; t0 = time.time()
    rc = sh(f"source {R}/kaggle/env.sh {TOOL}; export OUT={out} NP={NP}; cd /kaggle/tmp && {TOOL}_run.sh {R}/cases/{TOOL}/{CASE} {s} 2>&1 | tail -4").returncode
    res = json.load(open(f"{out}/result.json")) if os.path.exists(f"{out}/result.json") else None
    for junk in ("processor0", "processor1", "processor2", "processor3", "constant/polyMesh"): shutil.rmtree(f"{out}/{junk}", ignore_errors=True)
    runs.append({"set": s, "exit": rc, "wall_s": round(time.time() - t0, 1), "result": res})
    json.dump({"tool": TOOL, "case": CASE, "np": NP, "cores": os.cpu_count(), "install_s": round(t["installed"] - t["start"], 1), "runs": runs},
              open(f"{W}/timing.json", "w"), indent=1)          # rewritten after every run: a timeout still leaves the finished ones
print(json.dumps(runs, indent=1))
if any(r["exit"] for r in runs): raise SystemExit("at least one run missed its benchmark or crashed (timing.json has the details)")
