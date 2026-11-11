#!/usr/bin/env bash
# bootstrap.sh <foam|fem> : install one tool NATIVELY on a machine that cannot run Docker (a Kaggle notebook: Ubuntu, root, internet on).
# The cases, runners and checks are the same files the Docker images use, so a Kaggle result is graded by the same check.py / result.json.
#   foam: OpenFOAM ESI v2512 from the publisher's own Debian/Ubuntu repository (dl.openfoam.com), the same release as the phyein-foam image.
#   fem : DOLFINx 0.11 + SLEPc + Gmsh + CalculiX + PyVista from conda-forge into /opt/phyein-fem with micromamba (the same DOLFINx release as phyein-fem).
# Afterwards:  source kaggle/env.sh <foam|fem>   then   foam_run.sh cases/foam/cavity N=32   or   fem_run.sh cases/fem/heat
# Not installed here: ParaView (films). Make the film from the downloaded case on GitHub or in a session (tools/film.sh in the foam image).
# Why Kaggle at all: 4 CPU cores, about 30 GB RAM and 12 h per session (GitHub: 16 GB, 6 h); the GPU does NOT speed up OpenFOAM or DOLFINx.
set -eo pipefail
tool=${1:?usage: bootstrap.sh foam|fem}
here=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
export DEBIAN_FRONTEND=noninteractive
t0=$(date +%s)
case "$tool" in
  foam)
    if [ ! -f /usr/lib/openfoam/openfoam2512/etc/bashrc ]; then
      apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates gnupg > /dev/null
      # the publisher's repository, added by hand (key + one source line) instead of piping a script from the web into a shell
      curl -fsSL https://dl.openfoam.com/pubkey.gpg | gpg --dearmor --yes -o /etc/apt/trusted.gpg.d/openfoam.gpg
      code=$(sed -ne 's/^UBUNTU_CODENAME=//p' /etc/os-release); [ -n "$code" ] || code=$(sed -ne 's/^VERSION_CODENAME=//p' /etc/os-release)
      echo "deb [arch=$(dpkg --print-architecture)] https://dl.openfoam.com/repos/deb $code main" > /etc/apt/sources.list.d/openfoam.list
      apt-get update -qq && apt-get install -y -qq --no-install-recommends openfoam2512-default > /dev/null
    fi
    python3 -c "import numpy, scipy" 2>/dev/null || pip install -q numpy scipy
    ;;
  fem)
    if [ ! -x /opt/phyein-fem/bin/python ]; then
      apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates bzip2 libgl1 libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 > /dev/null
      mkdir -p /opt/mm && curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/2.3.3 | tar -xj -C /opt/mm bin/micromamba
      /opt/mm/bin/micromamba create -y -q -p /opt/phyein-fem -c conda-forge \
          python=3.12 "fenics-dolfinx=0.11.*" petsc4py slepc4py mpi4py python-gmsh calculix "numpy<3" scipy matplotlib pyvista
    fi
    ;;
  *) echo "unknown tool $tool (foam or fem)"; exit 2;;
esac
echo "bootstrap $tool: $(( $(date +%s) - t0 )) s. Next: source $here/kaggle/env.sh $tool"
