#!/bin/bash
# SPDX-License-Identifier: MIT
# Advance every build in the job list by as much as fits in one ~165 s call.
# usage: chunk.sh CASES_JSON SET_NAME [variants...]   variants like file-source nominal-integer
# Builds go to the VM's local disk ($HOME/build), because DuckDB and SQLite misbehave on the mounted
# Windows folder; copy_out.sh copies finished builds to C:\vitalDB.
set -u
CASES=$1; SET=$2; shift 2
# Machine-specific paths come from scripts/local.env (not published; copy local.env.example) or the environment.
[ -f "$(dirname "$0")/local.env" ] && . "$(dirname "$0")/local.env"
R="${VITALDB_ROOT:?set VITALDB_ROOT to the VitalDB release folder (see scripts/local.env.example)}"
PY="${VITALDB_PYTHON:-python}"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(dirname "$PWD")/vitaldb-bench-atriumdb${PYTHONPATH:+:$PYTHONPATH}"  # GPL AtriumDB package
pkg() { [ "$1" = atriumdb ] && echo vitaldb_bench_atriumdb || echo vitaldb_bench; }
END=$(( $(date +%s) + 160 ))
for v in "$@"; do
  timing=${v%-*}; values=${v#*-}
  for db in timenet atriumdb; do
    left=$(( END - $(date +%s) ))
    [ $left -lt 30 ] && exit 0
    out=$HOME/build/$db/$SET-$v
    extra=""; [ $db = timenet ] && extra="--partition-size 2"
    budget=$(( left - 25 ))
    [ $db = atriumdb ] && budget=$(( left - 50 ))
    [ $budget -lt 10 ] && continue
    timeout $(( left )) $PY -W ignore -m $(pkg $db).build $db --root "$R" --cases "$CASES" --out "$out" \
      --timing $timing --values $values --time-budget $budget $extra 2>&1 | tail -1 | sed "s|^|$db $v: |"
  done
done
