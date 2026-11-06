#!/usr/bin/env bash
# Run a worked example's (or a study tool's) "scan" over every recording in P parallel parts (WSL; P=8 by default,
# P=4 where WSL cannot hold 8), then aggregate where matplotlib exists.
#   bash examples/run_scan.sh hypotension_burden          (writes $W/examples/hypotension_burden/part_*.jsonl)
#   P=4 O=$W/examples/NAME_v2 bash examples/run_scan.sh NAME   (4 parts, another output folder)
# The example script must take: scan --root R --cases LIST --out PART.  Aggregate afterwards with:
#   python examples/NAME.py agg $W/examples/NAME/part_*.jsonl --out $W/examples/NAME/agg
set -euo pipefail
NAME=${1:?example script name without .py}
# Machine-specific paths come from the environment or scripts/local.env (copy scripts/local.env.example; git
# ignores the copy); the environment wins. W is the work folder (VITALDB_WORK), R the VitalDB release (VITALDB_ROOT).
_W=${W:-}; _R=${R:-}; _PP=${PYTHONPATH:-}
ENVF="$(cd "$(dirname "$0")/.." && pwd)/scripts/local.env"
[ -f "$ENVF" ] && . "$ENVF"
W=${_W:-${VITALDB_WORK:-}}; R=${_R:-${VITALDB_ROOT:-}}; export W; export PYTHONPATH=${_PP:-${VITALDB_PYTHONPATH:-}}
: "${W:?set W or VITALDB_WORK to your work folder (see scripts/local.env.example)}"
: "${R:?set R or VITALDB_ROOT to the VitalDB release folder (see scripts/local.env.example)}"
# a worked example (examples/NAME.py) or a study tool (tools/NAME.py); output goes to $W/examples/NAME either way
SCRIPT=examples/$NAME.py
[ -f "$(dirname "$0")/$NAME.py" ] || SCRIPT=tools/$NAME.py
O=${O:-$W/examples/$NAME}
P=${P:-8}
cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python || command -v python3 || true)}
if [ -z "$PY" ] || ! "$PY" -c "import numpy, scipy" 2>/dev/null; then
  echo "No Python with numpy and scipy found: activate the analysis environment first (your (vb) prompt), or set PY=..."; exit 1
fi
mkdir -p "$O"
N=$(ls "$R"/vital_files/*.vital | wc -l); echo "$N recordings; output in $O"
start=$(date +%s)
for i in $(seq 0 $((P - 1))); do
  PART=$(ls "$R"/vital_files/*.vital | sed 's#.*/0*##; s#\.vital##' | awk -v i=$i -v p=$P 'NR % p == i' | paste -sd,)
  "$PY" -W ignore "$SCRIPT" scan --root "$R" --cases "$PART" --out "$O/part_$i.jsonl" > "$O/log_$i.txt" 2>&1 &
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  sleep 60; echo "$(( ($(date +%s) - start) / 60 )) min: $(cat "$O"/part_*.jsonl 2>/dev/null | wc -l) of $N recordings"
done
wait
echo "done in $(( ($(date +%s) - start) / 60 )) min: $(cat "$O"/part_*.jsonl | wc -l) of $N recordings"
