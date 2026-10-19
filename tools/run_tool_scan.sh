#!/usr/bin/env bash
# Run a tool's "scan" over every recording in 8 parallel parts (WSL). Extra arguments are passed to every part.
#   bash tools/run_tool_scan.sh direct_rate_intervals
#   bash tools/run_tool_scan.sh primus_sync_accuracy --modes $W/results/sync_full/snuadc_mode_label.json \
#        --ports $W/results/sync_full/snuadc_mode.json --step-rates $W/results/pat/snuadc_step_rates.json
# Writes $W/NAME/part_*.jsonl and log_*.txt (or $OUT/... when OUT is set); rerunning resumes (finished recordings
# are skipped).
# The tool must take: scan --root R --cases LIST --out PART [extra arguments].
set -euo pipefail
NAME=${1:?tool script name without .py}; shift
# W (work folder), R (VitalDB release) and PYTHONPATH come from the environment or scripts/run.env
# (copy scripts/run.env.example); the environment wins.
_W=${W:-}; _R=${R:-}; _PP=${PYTHONPATH:-}
ENVF="$(cd "$(dirname "$0")/.." && pwd)/scripts/run.env"
[ -f "$ENVF" ] && . "$ENVF"
W=${_W:-${W:-}}; R=${_R:-${R:-}}; export PYTHONPATH=${_PP:-${PYTHONPATH:-}}
: "${W:?set W to your work folder, or create scripts/run.env (see scripts/run.env.example)}"
: "${R:?set R to the VitalDB release folder, or create scripts/run.env (see scripts/run.env.example)}"
O=${OUT:-$W/$NAME}
cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python || command -v python3 || true)}
if [ -z "$PY" ] || ! "$PY" -c "import numpy, scipy" 2>/dev/null; then
  echo "No Python with numpy and scipy found: activate the analysis environment first (your (vb) prompt), or set PY=..."; exit 1
fi
mkdir -p "$O"
N=$(ls "$R"/vital_files/*.vital | wc -l); echo "$N recordings; output in $O"
start=$(date +%s)
for i in 0 1 2 3 4 5 6 7; do
  PART=$(ls "$R"/vital_files/*.vital | sed 's#.*/0*##; s#\.vital##' | awk -v i=$i 'NR % 8 == i' | paste -sd,)
  "$PY" -W ignore tools/$NAME.py scan --root "$R" --cases "$PART" --out "$O/part_$i.jsonl" "$@" > "$O/log_$i.txt" 2>&1 &
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  sleep 60; echo "$(( ($(date +%s) - start) / 60 )) min: $(cat "$O"/part_*.jsonl 2>/dev/null | wc -l) of $N recordings"
done
wait
echo "done in $(( ($(date +%s) - start) / 60 )) min: $(cat "$O"/part_*.jsonl | wc -l) of $N recordings"
