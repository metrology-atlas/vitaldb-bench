#!/usr/bin/env bash
# Regenerate the Solar8000 HR delay labels with the HR rule found by tools/hr_derivation.py (1 Nov 2026):
# 60 / mean of the last 8 R-R intervals within 15% of their median, truncated (was 60 / median R-R over 6 s).
#   bash scripts/run_hr_relabel.sh            (WSL, analysis environment active; resumes if stopped)
# 1. HR-only rescan of every recording with tools/numeric_delay_variability.py --only HR (8 parts, ~2-3 s each)
# 2. tools/numeric_delay_labels.py on the rescan: new HR delay rows and HR delay groups
# 3. merge: the existing numeric_delay_labels.csv with its HR rows replaced
# Writes $W/results/hr_relabel/ only; results/pat/ is left as it is until the new labels are reviewed.
# Needs the TimeNet timing model updated first (HR algorithm text, window, MODEL_VERSION), since rows carry them.
set -euo pipefail
_W=${W:-}; _R=${R:-}; _PP=${PYTHONPATH:-}
ENVF="$(cd "$(dirname "$0")/.." && pwd)/scripts/local.env"
[ -f "$ENVF" ] && . "$ENVF"
W=${_W:-${VITALDB_WORK:-}}; R=${_R:-${VITALDB_ROOT:-}}; export W; export PYTHONPATH=${_PP:-${VITALDB_PYTHONPATH:-}}
: "${W:?set W or VITALDB_WORK}"; : "${R:?set R or VITALDB_ROOT}"
cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python || command -v python3)}
"$PY" -c "from vitaldb_reader.timing_model import NUMERIC_LAG as L; import sys; a=L['Solar8000/HR'].algorithm; print('timing model HR:', a); sys.exit(0 if 'mean' in a and '8' in a else 1)" \
  || { echo "Update the TimeNet timing model's HR entry first"; exit 1; }
S=$W/examples/hr_relabel; O=$W/results/hr_relabel; mkdir -p "$S" "$O"
N=$(ls "$R"/vital_files/*.vital | wc -l); echo "$N recordings; scan in $S"
start=$(date +%s)
for i in 0 1 2 3 4 5 6 7; do
  PART=$(ls "$R"/vital_files/*.vital | sed 's#.*/0*##; s#\.vital##' | awk -v i=$i 'NR % 8 == i' | paste -sd,)
  "$PY" -W ignore tools/numeric_delay_variability.py scan --root "$R" --cases "$PART" --only HR \
    --units "$W/results/sync_full/units.csv" --step-rates "$W/results/pat/snuadc_step_rates.json" \
    --out "$S/part_$i.jsonl" > "$S/log_$i.txt" 2>&1 &
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  sleep 60; echo "$(( ($(date +%s) - start) / 60 )) min: $(cat "$S"/part_*.jsonl 2>/dev/null | wc -l) of $N recordings"
done
wait
echo "scan done in $(( ($(date +%s) - start) / 60 )) min"
# Solar8000 ports, carried over from the existing group labels, for the groups' port summary
"$PY" - "$W/results/pat/hr_delay_group_labels.csv" "$O/ports.jsonl" <<'PYEOF'
import csv, json, sys
with open(sys.argv[2], "w") as fh:
    for r in csv.DictReader(open(sys.argv[1])):
        fh.write(json.dumps({"case": int(r["case"]), "devices": {"Solar8000": [None, r["solar_port"] or None]}}) + "\n")
PYEOF
"$PY" -W ignore tools/numeric_delay_labels.py --scan "$S"/part_*.jsonl --root "$R" --devices "$O/ports.jsonl" \
  --out "$O/hr_only" --summary "$O/hr_only_summary.json"
"$PY" - "$W/results/pat/numeric_delay_labels.csv" "$O/hr_only/numeric_delay_labels.csv" "$O/numeric_delay_labels.csv" <<'PYEOF'
import csv, sys
old, new, out = sys.argv[1:]
o = list(csv.DictReader(open(old))); n = list(csv.DictReader(open(new)))
rows = [r for r in o if r["track"] != "Solar8000/HR"] + n
rows.sort(key=lambda r: (int(r["case"]), r["track"]))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(o[0].keys()), extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(f"merged: {len(o) - sum(r['track'] == 'Solar8000/HR' for r in o)} non-HR rows kept, {len(n)} HR rows new")
PYEOF
cp "$O/hr_only/hr_delay_group_labels.csv" "$O/hr_delay_group_labels.csv"
echo "done: $O/numeric_delay_labels.csv, $O/hr_delay_group_labels.csv, $O/hr_only_summary.json"
