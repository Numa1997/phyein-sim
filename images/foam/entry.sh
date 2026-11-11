#!/bin/bash
# Source OpenFOAM, then run whatever was asked. Works for any uid.
# OpenFOAM's bashrc reads its positional parameters as settings (FOAM_SETTINGS), so the command
# is put aside while it is sourced; otherwise the command line changes the environment it builds.
set +u
cmd=("$@"); set --
. /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1
exec "${cmd[@]}"
