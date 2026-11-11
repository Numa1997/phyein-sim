# source kaggle/env.sh <foam|fem>   (after kaggle/bootstrap.sh): puts the runners and the tool on PATH in THIS shell.
_t=${1:?usage: source env.sh foam|fem}
export PHYEIN_SIM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH=$PHYEIN_SIM/tools:$PATH MPLBACKEND=Agg OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_rmaps_base_oversubscribe=1 PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
case "$_t" in
  foam) _a=("$@"); set --                      # OpenFOAM's bashrc reads the caller's arguments as settings: clear them first
        set +u; . /usr/lib/openfoam/openfoam2512/etc/bashrc > /dev/null 2>&1; set -- "${_a[@]}"; unset _a;;
  fem)  export PATH=/opt/phyein-fem/bin:$PATH PYVISTA_OFF_SCREEN=true;;
esac
unset _t
