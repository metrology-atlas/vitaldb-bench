#!/usr/bin/env bash
# Run a tool's "scan" over every recording in P parallel parts (WSL; P=8 by default, P=4 where WSL cannot hold 8).
# Extra arguments are passed to every part. The tool is tools/NAME.py, or examples/NAME.py where tools/ has none.
#   bash tools/run_tool_scan.sh direct_rate_intervals
#   bash tools/run_tool_scan.sh primus_sync_accuracy --modes $W/results/sync_full/snuadc_mode_label.json \
#        --ports $W/results/sync_full/snuadc_mode.json --step-rates $W/results/pat/snuadc_step_rates.json
# Writes $W/NAME/part_*.jsonl and log_*.txt (or $OUT/... when OUT is set); rerunning resumes (finished recordings
# are skipped).
# The tool must take: scan --root R --cases LIST --out PART [extra arguments].
set -euo pipefail
NAME=${1:?tool script name without .py}; shift
# Machine-specific paths come from the environment or scripts/local.env (copy scripts/local.env.example; git
# ignores the copy); the environment wins. W is the work folder (VITALDB_WORK), R the VitalDB release (VITALDB_ROOT).
_W=${W:-}; _R=${R:-}; _PP=${PYTHONPATH:-}
ENVF="$(cd "$(dirname "$0")/.." && pwd)/scripts/local.env"
[ -f "$ENVF" ] && . "$ENVF"
W=${_W:-${VITALDB_WORK:-}}; R=${_R:-${VITALDB_ROOT:-}}; export PYTHONPATH=${_PP:-${VITALDB_PYTHONPATH:-}}
: "${W:?set W or VITALDB_WORK to your work folder (see scripts/local.env.example)}"
: "${R:?set R or VITALDB_ROOT to the VitalDB release folder (see scripts/local.env.example)}"
O=${OUT:-$W/$NAME}
P=${P:-8}
cd "$(dirname "$0")/.."
SCRIPT=tools/$NAME.py
[ -f "$SCRIPT" ] || SCRIPT=examples/$NAME.py
PY=${PY:-$(command -v python || command -v python3 || true)}
if [ -z "$PY" ] || ! "$PY" -c "import numpy, scipy" 2>/dev/null; then
  echo "No Python with numpy and scipy found: activate the analysis environment first (your (vb) prompt), or set PY=..."; exit 1
fi
mkdir -p "$O"
N=$(ls "$R"/vital_files/*.vital | wc -l); echo "$N recordings; output in $O"
start=$(date +%s)
for i in $(seq 0 $((P - 1))); do
  PART=$(ls "$R"/vital_files/*.vital | sed 's#.*/0*##; s#\.vital##' | awk -v i=$i -v p=$P 'NR % p == i' | paste -sd,)
  "$PY" -W ignore "$SCRIPT" scan --root "$R" --cases "$PART" --out "$O/part_$i.jsonl" "$@" > "$O/log_$i.txt" 2>&1 &
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  sleep 60; echo "$(( ($(date +%s) - start) / 60 )) min: $(cat "$O"/part_*.jsonl 2>/dev/null | wc -l) of $N recordings"
done
wait
echo "done in $(( ($(date +%s) - start) / 60 )) min: $(cat "$O"/part_*.jsonl | wc -l) of $N recordings"
