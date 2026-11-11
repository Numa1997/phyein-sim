#!/usr/bin/env bash
# Self-test of the phyein-fem image: versions, then four real runs with their benchmarks (about 2 minutes in all):
#   aperture electron lens (Davisson & Calbick), cantilever beam modes (Euler-Bernoulli, CalculiX cross-check), heat decay (exact),
#   fluid topology optimisation (Borrvall & Petersson; straight channel = Poiseuille power).
set -euo pipefail
export PHYEIN_SIM=${PHYEIN_SIM:-/opt/phyein-sim}
python3 -c "import dolfinx, gmsh, petsc4py, slepc4py, pyvista; print('dolfinx', dolfinx.__version__, '| gmsh', gmsh.__version__, '| pyvista', pyvista.__version__)"
ccx -v 2>&1 | head -2 || true
pvbatch --version 2>&1 | head -1; ffmpeg -version | head -1
work=$(mktemp -d); cd "$work"
for c in lens beam heat fluidtopo; do
  echo "== $c"
  OUT="$work/$c" "$PHYEIN_SIM/tools/fem_run.sh" "$PHYEIN_SIM/cases/fem/$c" 2>&1 | grep -vE "^Info|^Warning" | tail -12
done
echo "== einzel"
OUT="$work/einzel" "$PHYEIN_SIM/tools/fem_run.sh" "$PHYEIN_SIM/cases/fem/lens" geometry=einzel 2>&1 | grep -vE "^Info|^Warning" | tail -3
python3 - <<'PY'
import json, glob, sys
for f in sorted(glob.glob("*/result.json")):
    r = json.load(open(f)); print(f, "pass" if r["pass"] else "FAIL")
    if not r["pass"]: sys.exit(1)
PY
test -n "$(command -v ccx)" && python3 -c "import json; r=json.load(open('beam/result.json')); assert 'calculix_Hz' in r, 'CalculiX cross-check did not run'"
echo "SELFTEST PASS"
