#!/bin/bash
# SPDX-License-Identifier: MIT
# Advance the 30-case test matrix by one ~170 s call: up to $SLOTS unfinished builds run side by side.
# usage: t30.sh [name-filter-regex]      jobs: scripts/t30_jobs.txt    builds: $HOME/build/t30/<name>
# A finished build gets a DONE file; measure/copy-out happen in t30_measure.sh.
set -u
FILTER=${1:-.}
SLOTS=${SLOTS:-2}
# Machine-specific paths come from scripts/local.env (not published; copy local.env.example) or the environment.
[ -f "$(dirname "$0")/local.env" ] && . "$(dirname "$0")/local.env"
R="${VITALDB_ROOT:?set VITALDB_ROOT to the VitalDB release folder (see scripts/local.env.example)}"
PY="${VITALDB_PYTHON:-python}"
CASES=${VITALDB_WORK:?set VITALDB_WORK in scripts/local.env}/bench/cases_test30.json
mkdir -p $HOME/build/t30
cd "$(dirname "$0")/.."
export PYTHONPATH="$(dirname "$PWD")/vitaldb-bench-atriumdb${PYTHONPATH:+:$PYTHONPATH}"  # GPL AtriumDB package
pkg() { [ "$1" = atriumdb ] && echo vitaldb_bench_atriumdb || echo vitaldb_bench; }
n=0
while IFS='|' read -r name db args; do
  name=$(echo $name); db=$(echo $db)
  [[ -z "$name" || "$name" == \#* ]] && continue
  [[ "$name" =~ $FILTER ]] || continue
  out=$HOME/build/t30/$name
  [ -e "$out/DONE" ] && continue
  [ -e "$HOME/results/t30/$name/COPIED" ] && continue  # built, measured and moved to C:\vitalDB\test_dbs
  [ $n -ge $SLOTS ] && break
  n=$((n+1))
  budget=${BUDGET:-125}; [ $db = timenet ] && budget=$(( budget - 5 ))
  ( timeout $(( budget + 43 )) $PY -W ignore -m $(pkg $db).build $db --root "$R" --cases "$CASES" --out "$out" --time-budget $budget $args 2>&1 \
      | tail -1 > "$out.last"; line=$(cat "$out.last"); echo "$name: $line"
    if echo "$line" | grep -Eq '"(cases_left|partitions_left)": 0[,}]'; then touch "$out/DONE"; fi ) &
done < scripts/t30_jobs.txt
wait
[ $n -eq 0 ] && echo "nothing left for filter $FILTER"
