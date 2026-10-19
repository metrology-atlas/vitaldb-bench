#!/usr/bin/env bash
# Worked example "PPV by breath": scan every recording with Primus and SNUADC in 8 parallel parts (WSL, about 30-40 min).
#   bash examples/run_ppv_by_breath.sh            (writes $W/examples/ppv_by_breath/part_*.jsonl)
# Then aggregate (needs matplotlib):  python examples/ppv_by_breath.py agg $W/examples/ppv_by_breath/part_*.jsonl --out DIR
set -euo pipefail
# Machine-specific paths come from the environment or scripts/local.env (copy scripts/local.env.example; git
# ignores the copy); the environment wins. W is the work folder (VITALDB_WORK), R the VitalDB release (VITALDB_ROOT).
_W=${W:-}; _R=${R:-}; _PP=${PYTHONPATH:-}
ENVF="$(cd "$(dirname "$0")/.." && pwd)/scripts/local.env"
[ -f "$ENVF" ] && . "$ENVF"
W=${_W:-${VITALDB_WORK:-}}; R=${_R:-${VITALDB_ROOT:-}}; export PYTHONPATH=${_PP:-${VITALDB_PYTHONPATH:-}}
: "${W:?set W or VITALDB_WORK to your work folder (see scripts/local.env.example)}"
: "${R:?set R or VITALDB_ROOT to the VitalDB release folder (see scripts/local.env.example)}"
O=$W/examples/ppv_by_breath
cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python || command -v python3 || true)}
if [ -z "$PY" ] || ! "$PY" -c "import numpy, scipy" 2>/dev/null; then
  echo "No Python with numpy and scipy found. Activate the analysis environment first (e.g. conda activate vb),"
  echo "or set PY=/path/to/python. Environments found:"; (conda env list 2>/dev/null || true); ls -d ~/*/bin/python ~/.venvs/*/bin/python ~/venvs/*/bin/python 2>/dev/null || true
  exit 1
fi
mkdir -p "$O"
CASES=$("$PY" - "$W" <<'P'
import json, sys
d = json.load(open(sys.argv[1] + "/results/sync_full/snuadc_mode.json"))
print(",".join(str(c) for c in sorted(int(k) for k, v in d.items() if v and {"Primus", "SNUADC"} <= set(v.get("devices") or []))))
P
)
N=$(echo "$CASES" | tr ',' '\n' | wc -l); echo "$N recordings with Primus and SNUADC"
start=$(date +%s)
for i in 0 1 2 3 4 5 6 7; do
  PART=$(echo "$CASES" | tr ',' '\n' | awk -v i=$i 'NR % 8 == i' | paste -sd,)
  "$PY" -W ignore examples/ppv_by_breath.py scan --root "$R" --cases "$PART" \
    --modes $W/results/sync_full/snuadc_mode_label.json --ports $W/results/sync_full/snuadc_mode.json \
    --step-rates $W/results/pat/snuadc_step_rates.json --out "$O/part_$i.jsonl" > "$O/log_$i.txt" 2>&1 &
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  sleep 60; echo "$(( ($(date +%s) - start) / 60 )) min: $(cat "$O"/part_*.jsonl 2>/dev/null | wc -l) of $N recordings"
done
wait
echo "done in $(( ($(date +%s) - start) / 60 )) min: $(cat "$O"/part_*.jsonl | wc -l) of $N recordings"
