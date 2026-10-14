#!/bin/bash
# SPDX-License-Identifier: MIT
# Measure finished test builds within one ~170 s call: size -> fidelity (resumable) -> warm reads.
# usage: t30_measure.sh [name-filter-regex]       results: $HOME/results/t30/<name>/
set -u
FILTER=${1:-.}
# Machine-specific paths come from scripts/local.env (not published; copy local.env.example) or the environment.
[ -f "$(dirname "$0")/local.env" ] && . "$(dirname "$0")/local.env"
R="${VITALDB_ROOT:?set VITALDB_ROOT to the VitalDB release folder (see scripts/local.env.example)}"
PY="${VITALDB_PYTHON:-python}"
CASES=${VITALDB_WORK:?set VITALDB_WORK in scripts/local.env}/bench/cases_test30.json
cd "$(dirname "$0")/.."
export PYTHONPATH="$(dirname "$PWD")/vitaldb-bench-atriumdb${PYTHONPATH:+:$PYTHONPATH}"  # GPL AtriumDB package
pkg() { [ "$1" = atriumdb ] && echo vitaldb_bench_atriumdb || echo vitaldb_bench; }
END=$(( $(date +%s) + 165 ))
for out in $HOME/build/t30/*/; do
  out=${out%/}; name=$(basename $out)
  [[ "$name" =~ $FILTER ]] || continue
  [ -e "$out/DONE" ] || continue
  res=$HOME/results/t30/$name; mkdir -p $res
  [ -e "$res/MEASURED" ] && continue
  left=$(( END - $(date +%s) )); [ $left -lt 20 ] && break
  db=atriumdb; [ -d "$out/dataset" ] || db=timenet
  timing=nominal; [[ "$name" == *resync* ]] && timing=resync; [[ "$name" == *-file ]] && timing=file
  if [ ! -e "$res/size.json" ]; then
    $PY tools/t30_measure.py "$out" --out "$res/size.json" | cut -c1-300
  fi
  left=$(( END - $(date +%s) )); [ $left -lt 20 ] && break
  if [ ! -e "$res/FIDDONE" ]; then
    timeout $left $PY -W ignore -m $(pkg $db).fidelity --root "$R" --$db "$out" --out "$res/fid.json" \
      $( [ $db = atriumdb ] && echo --exclude 4649 ) --timing $timing --time-budget $(( left - 45 )) 2>&1 | tail -1 | sed "s|^|$name fid: |" | cut -c1-200
    $PY -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(d['cases']+len([x for x in d['excluded'].split(',') if x])<d['of'])" "$res/fid.json" 2>/dev/null \
      && touch "$res/FIDDONE" || break
  fi
  if [[ "$name" != N* && ! -e "$res/reads.jsonl" ]]; then
    left=$(( END - $(date +%s) )); [ $left -lt 90 ] && break
    timeout $left $PY -W ignore -m $(pkg $db).perf reads --db $db --build "$out" --root "$R" --cases "$CASES" \
      --out "$res/reads.jsonl" --n-cases 10 --windows 10 2>&1 | tail -1 | cut -c1-300
    [ -e "$res/reads.jsonl" ] || break
  fi
  touch "$res/MEASURED"; echo "$name measured"
done
