#!/usr/bin/env bash
# Self-test of the phyein-foam image: tools present, then a real pipeline run (cavity vs Ghia, N = 32, about 20 s)
# and a headless film of that result. Exit code 0 only if everything works.
set -eo pipefail   # no -u: OpenFOAM's own bashrc reads unset variables
command -v icoFoam >/dev/null || { set +u; source /usr/lib/openfoam/openfoam2512/etc/bashrc; }
export PHYEIN_SIM=${PHYEIN_SIM:-/opt/phyein-sim}
echo "== versions"
echo "OpenFOAM: $(foamVersion 2>/dev/null || echo "$WM_PROJECT_VERSION")"; interFoam -help | head -1 >/dev/null && echo "interFoam ok"; icoFoam -help >/dev/null && echo "icoFoam ok"
python3 -c "import numpy, scipy, matplotlib; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
/opt/venv/bin/python -c "import gmsh; print('gmsh', gmsh.__version__)"
/opt/venv/bin/python -c "import pyvista; print('pyvista', pyvista.__version__)"
pvbatch --version 2>&1 | head -1
ffmpeg -version | head -1
echo "== cavity N=32 against Ghia (Re 100)"
work=$(mktemp -d); cd "$work"
OUT="$work/cavity" TOL=0.03 "$PHYEIN_SIM/tools/foam_run.sh" "$PHYEIN_SIM/cases/foam/cavity" N=32 nTurn=30
echo "== film"
# write one more field for the film (final time exists); view the speed
postProcess -case "$work/cavity" -func mag\(U\) -latestTime > /dev/null 2>&1 || true
"$PHYEIN_SIM/tools/film.sh" "$work/cavity" U "$work/cavity.mp4" 0 1
test -s "$work/cavity.mp4"
echo "SELFTEST PASS"
