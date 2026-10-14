#!/bin/bash
# SPDX-License-Identifier: MIT
# Advance one database's builds for several variants at once, one process per variant (the VM has 2
# cores), within one ~170 s call. Variants: TIMING-VALUES[-zLEVEL[-bBLOCK]], e.g. nominal-integer-z22-b8192.
# Ingest times measured this way are under contention; clean ingest timings come from chunk.sh.
set -u
CASES=$1; SET=$2; DB=$3; shift 3
# Machine-specific paths come from scripts/local.env (not published; copy local.env.example) or the environment.
[ -f "$(dirname "$0")/local.env" ] && . "$(dirname "$0")/local.env"
R="${VITALDB_ROOT:?set VITALDB_ROOT to the VitalDB release folder (see scripts/local.env.example)}"
PY="${VITALDB_PYTHON:-python}"
PART=${PART:-5}
cd "$(dirname "$0")/.."
export PYTHONPATH="$(dirname "$PWD")/vitaldb-bench-atriumdb${PYTHONPATH:+:$PYTHONPATH}"  # GPL AtriumDB package
pkg() { [ "$1" = atriumdb ] && echo vitaldb_bench_atriumdb || echo vitaldb_bench; }
for v in "$@"; do
  IFS=- read -r timing values z b <<< "$v"
  extra=""; [ $DB = timenet ] && extra="--partition-size $PART"
  [ -n "${z:-}" ] && extra="$extra --zstd-level ${z#z}"
  [ -n "${b:-}" ] && extra="$extra --block-size ${b#b}"
  budget=125; [ $DB = timenet ] && budget=120
  ( timeout 168 $PY -W ignore -m $(pkg $DB).build $DB --root "$R" --cases "$CASES" --out "$HOME/build/$DB/$SET-$v" \
      --timing $timing --values $values --time-budget $budget $extra 2>&1 | tail -1 | sed "s|^|$DB $v: |" ) &
done
wait
