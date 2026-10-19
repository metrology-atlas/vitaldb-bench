#!/usr/bin/env bash
# Whole-release ECG-EEG burst validation of the stamp model, in 8 parallel parts (about 12-15 min in WSL).
#   bash tools/run_burst_validation.sh VARIANT OUTDIR      e.g.  bash tools/run_burst_validation.sh model3 burst_validation_m3
# OUTDIR is under the work folder W (scripts/run.env). Then aggregate (needs matplotlib):
#   python tools/burst_validation_agg.py --inp $W/OUTDIR --modes ... --ports ... --variant VARIANT
set -euo pipefail
V=${1:?variant (model, model2, model3)}; OUTNAME=${2:?output folder under the work folder}
# W (work folder), R (VitalDB release) and PYTHONPATH come from the environment or scripts/run.env
# (copy scripts/run.env.example); the environment wins.
_W=${W:-}; _R=${R:-}; _PP=${PYTHONPATH:-}
ENVF="$(cd "$(dirname "$0")/.." && pwd)/scripts/run.env"
[ -f "$ENVF" ] && . "$ENVF"
W=${_W:-${W:-}}; R=${_R:-${R:-}}; export PYTHONPATH=${_PP:-${PYTHONPATH:-}}
: "${W:?set W to your work folder, or create scripts/run.env (see scripts/run.env.example)}"
: "${R:?set R to the VitalDB release folder, or create scripts/run.env (see scripts/run.env.example)}"
O=$W/$OUTNAME
cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python || command -v python3 || true)}
if [ -z "$PY" ] || ! "$PY" -c "import numpy, scipy" 2>/dev/null; then
  echo "No Python with numpy and scipy found. Activate the analysis environment first (e.g. conda activate vb),"
  echo "or set PY=/path/to/python. Environments found:"; (conda env list 2>/dev/null || true); ls -d ~/*/bin/python ~/.venvs/*/bin/python ~/venvs/*/bin/python 2>/dev/null || true
  exit 1
fi
mkdir -p "$O"
start=$(date +%s)
for i in 0 1 2 3 4 5 6 7; do
  "$PY" -W ignore tools/sa_corrected.py --root "$R" --cases "$(cat $W/pat_census/cases_$i.txt)" \
    --units $W/results/sync_full/units.csv --step-rates $W/results/pat/snuadc_step_rates.json \
    --bis-rates $W/results/pat/bis_port_rates.json --modes $W/results/sync_full/snuadc_mode_label.json \
    --ports $W/results/sync_full/snuadc_mode.json --variants "$V" --out "$O/part_$i.jsonl" > "$O/log_$i.txt" 2>&1 &
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  sleep 60; echo "$(( ($(date +%s) - start) / 60 )) min: $(cat "$O"/part_*.jsonl 2>/dev/null | wc -l) of 6388 recordings"
done
wait
echo "done in $(( ($(date +%s) - start) / 60 )) min: $(cat "$O"/part_*.jsonl | wc -l) recordings (expect 6388)"
