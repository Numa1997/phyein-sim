#!/usr/bin/env bash
# foam_run.sh <case-dir> [NAME=value ...]
# Copies a case from cases/foam/<name> to a scratch directory, overrides parameters in its `params` file,
# runs the case's Allrun, then its check.py (if any). Prints wall-clock time. Result files stay in the scratch directory.
#   OUT=<dir>   where to put the run (default: ./runs/<name>-<overrides>)
set -euo pipefail
src=$(readlink -f "$1"); shift
name=$(basename "$src")
tag=$(printf '%s-' "$@" | tr -c 'A-Za-z0-9=._\n-' '_' | sed 's/-$//')
out=${OUT:-$PWD/runs/${name}${tag:+-$tag}}
rm -rf "$out"; mkdir -p "$(dirname "$out")"; cp -r "$src" "$out"
for kv in "$@"; do                                   # NAME=value -> replace the number in "NAME  value;"
  k=${kv%%=*}; v=${kv#*=}
  grep -qE "^$k[[:space:]]" "$out/params" || { echo "no parameter '$k' in $src/params" >&2; exit 2; }
  sed -i -E "s|^($k[[:space:]]+)[^;]*;|\1$v;|" "$out/params"
done
grep -rl '<case>' "$out" | xargs -r sed -i "s|<case>|$out|g"   # case files include "<case>/params"
cd "$out"
t0=$(date +%s)
bash ./Allrun > log.Allrun 2>&1 || { echo "Allrun failed, see $out/log.Allrun"; tail -20 log.Allrun; exit 1; }
t1=$(date +%s)
echo "run time: $((t1-t0)) s  ($name ${*:-defaults})"
if [ -f check.py ]; then python3 check.py "$out" ${TOL:-} || rc=$?; fi
echo "$((t1-t0))" > walltime_s
exit ${rc:-0}
