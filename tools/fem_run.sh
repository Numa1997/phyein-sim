#!/usr/bin/env bash
# fem_run.sh <case-dir> [NAME=value ...]
# Copies a case from cases/fem/<name> to a scratch directory and runs its run.py with the overrides given as --NAME value.
# run.py writes result.json (with "pass": true/false) and its figures there.  Prints wall-clock time.  OUT=<dir> sets the run directory.
set -euo pipefail
src=$(readlink -f "$1"); shift
name=$(basename "$src")
tag=$(printf '%s-' "$@" | tr -c 'A-Za-z0-9=._\n-' '_' | sed 's/-$//')
out=${OUT:-$PWD/runs/${name}${tag:+-$tag}}
rm -rf "$out"; mkdir -p "$(dirname "$out")"; cp -r "$src" "$out"
args=(); for kv in "$@"; do args+=("--${kv%%=*}" "${kv#*=}"); done
cd "$out"
t0=$(date +%s)
rc=0; python3 run.py "${args[@]}" 2>&1 | tee log.run || rc=${PIPESTATUS[0]}
echo "run time: $(( $(date +%s) - t0 )) s  ($name ${*:-defaults})"
echo "$(( $(date +%s) - t0 ))" > walltime_s
exit $rc
