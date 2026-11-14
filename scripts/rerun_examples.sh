#!/usr/bin/env bash
# Rerun one worked example from the committed code, with a run record for every command, into a fresh folder.
# A worked example is marked current only when every output its page uses comes from such a run (metrology-atlas
# README, "Checks"). Run in WSL, one example at a time:
#   bash scripts/rerun_examples.sh list                     the examples and roughly how long each takes
#   TEST=1 bash scripts/rerun_examples.sh NAME              a quick test on a few recordings, into examples_v3_test/
#   bash scripts/rerun_examples.sh NAME                     the full run, into $W/examples_v3/NAME
#   nohup bash scripts/rerun_examples.sh NAME > $W/examples_v3/NAME.log 2>&1 &     (in the background)
# It refuses to run with uncommitted changes in vitaldb-bench, physiosync or vitaldb-reader (the run records would
# not point at code anyone can check out), into a folder that already has outputs, or with less than MIN_FREE_GB
# (default 5) free on the work drive. Scans use at most P=4 parts at once.
set -euo pipefail
NAME=${1:?"example name, or list"}
B="$(cd "$(dirname "$0")/.." && pwd)"
_W=${W:-}; _R=${R:-}; _PP=${PYTHONPATH:-}
[ -f "$B/scripts/local.env" ] && . "$B/scripts/local.env"
W=${_W:-${VITALDB_WORK:-}}; R=${_R:-${VITALDB_ROOT:-}}; export W
: "${W:?set W or VITALDB_WORK}"; : "${R:?set R or VITALDB_ROOT}"
CODE=$(cd "$B/../.." && pwd)                     # the folder holding metrology-atlas, physiosync, vitaldb-reader
export PYTHONPATH=$B/tools:${_PP:-${VITALDB_PYTHONPATH:-}}
PY=${PY:-${VITALDB_PYTHON:-$(command -v python || command -v python3)}}
PROV=$CODE/metrology-atlas/metrology_atlas/provenance.py
P=4; MIN_FREE_GB=${MIN_FREE_GB:-5}
RES_TAG=v0.6.0                                   # the vitaldb-resolution release the example pages pin
TEST=${TEST:-0}

case "$NAME" in
  list) cat <<'L'
cuff_vs_arterial          scan of every recording, 4 parts (about 10 min)
hypotension_burden        scan of every recording, 4 parts (about 30 min)
induction_hypotension     scan of every recording, 4 parts (about 20 min)
pat_vs_bp                 scan of every recording, 4 parts (about 30 min); needs the slip_rebuild_fit run
ppv_by_breath             scan of the recordings with Primus and SNUADC, 4 parts (about 50 min)
pulse_arrival_time        121 recordings, rebuilt and detect (about an hour)
blood_gas_gradients       Monte Carlo per gas and the timing scan, in chunks (about 1-2 h)
fluid_balance             clinical table only (a few minutes)
oliguria_labels           clinical table only (under a minute)
regression_with_rounding  clinical table only (under a minute)
urine_output_rate         clinical table only (a few minutes)
delta_pat_hist            not an example: beat-to-beat change in PAT, every recording, 4 parts (about 25 min)
slip_rebuild_fit          not an example: the whole-release rebuilt PPG fit the two PAT examples read, 6 parts (about 30 min)
L
  exit 0;;
esac

cd "$B"
SCRIPT=examples/$NAME.py; [ "$NAME" = delta_pat_hist ] && SCRIPT=tools/$NAME.py
[ "$NAME" = slip_rebuild_fit ] && SCRIPT=tools/slip_rebuild.py
[ -f "$SCRIPT" ] || { echo "no $SCRIPT"; exit 1; }
for repo in "$B" "$CODE/physiosync" "$CODE/vitaldb-reader"; do
  if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
    echo "uncommitted changes in $repo: commit first, so the run records point at code anyone can check out"; exit 1
  fi
done
"$PY" -c "import numpy, scipy, matplotlib, pandas, sklearn" 2>/dev/null || {
  echo "$PY lacks numpy, scipy, matplotlib, pandas or scikit-learn: install them first, e.g."
  echo "  $PY -m pip install numpy scipy matplotlib pandas scikit-learn"; exit 1; }
if [ "$TEST" = 1 ]; then O=$W/examples_v3_test/$NAME; else O=$W/examples_v3/$NAME; fi
if [ -d "$O" ] && [ -n "$(ls -A "$O" 2>/dev/null)" ] && [ "$TEST" != 1 ]; then
  echo "$O already has outputs; this script never writes over a run"; exit 1
fi
mkdir -p "$O"
disk() {
  local free; free=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
  echo "$(date +%H:%M) $free GB free on the work drive"
  [ "$free" -ge "$MIN_FREE_GB" ] || { echo "less than $MIN_FREE_GB GB free: stopped"; exit 1; }
}
rec() {                                          # run one command with a run record; its output also to $O.steps.log (beside the folder, so no run record lists it)
  echo "== $(date +%H:%M) $*" >> "$O.steps.log"
  "$PY" "$PROV" record --out "$O" -- "$PY" -W ignore "$@" 2>&1 | tee -a "$O.steps.log"
  return "${PIPESTATUS[0]}"
}
resolution() {                                   # the pinned vitaldb-resolution, exported once outside the repo
  VR=$HOME/vitaldb-resolution-$RES_TAG
  [ -d "$VR" ] || { mkdir -p "$VR"; git -C "$CODE/vitaldb/vitaldb-resolution" archive "$RES_TAG" | tar -x -C "$VR"; }
  echo "$VR"
}
all_cases() { ls "$R"/vital_files/*.vital | sed 's#.*/0*##; s#\.vital##'; }
test_cases() { if [ "$TEST" = 1 ]; then head -${1:-6}; else cat; fi; }
scan_parts() {                                   # scan_parts SCRIPT_ARGS... : the case list on stdin, P parts at once
  local cases; cases=$(cat)
  local n; n=$(echo "$cases" | wc -l); echo "$(date +%H:%M) scan of $n recordings, $P parts"
  for i in $(seq 0 $((P - 1))); do
    local part; part=$(echo "$cases" | awk -v i=$i -v p=$P 'NR % p == i' | paste -sd,)
    rec "$SCRIPT" scan --root "$R" --cases "$part" "$@" --out "$O/part_$i.jsonl" > "$O/log_$i.txt" 2>&1 &
  done
  wait
  if grep -l Traceback "$O"/log_*.txt > /dev/null 2>&1; then echo "a part failed: see $O/log_*.txt"; exit 1; fi
  echo "$(date +%H:%M) scan done: $(cat "$O"/part_*.jsonl | wc -l) of $n recordings"
}

disk
case "$NAME" in
  cuff_vs_arterial|hypotension_burden)
    all_cases | test_cases | scan_parts
    rec "examples/$NAME.py" agg "$O"/part_*.jsonl --out "$O/agg";;
  induction_hypotension)
    all_cases | test_cases | scan_parts
    rec "examples/$NAME.py" agg "$O"/part_*.jsonl --root "$R" --out "$O/agg";;
  pat_vs_bp)
    FIT=${PAT_BP_FIT:-$W/examples_v3/slip_rebuild_fit/fit.jsonl}; [ -e "$FIT" ] || { echo "missing $FIT"; exit 1; }
    all_cases | test_cases | scan_parts --fit "$FIT"
    rec "examples/$NAME.py" agg "$O"/part_*.jsonl --out "$O/agg";;
  ppv_by_breath)
    "$PY" - "$W" <<'PYC' | tr ',' '\n' | test_cases | scan_parts --modes "$W/results/sync_full/snuadc_mode_label.json" \
        --ports "$W/results/sync_full/snuadc_mode.json" --step-rates "$W/results/pat/snuadc_step_rates.json"
import json, sys
d = json.load(open(sys.argv[1] + "/results/sync_full/snuadc_mode.json"))
print(",".join(str(c) for c in sorted(int(k) for k, v in d.items() if v and {"Primus", "SNUADC"} <= set(v.get("devices") or []))))
PYC
    rec "examples/$NAME.py" agg "$O"/part_*.jsonl --out "$O/agg";;
  pulse_arrival_time)
    FIT=${PAT_FIT:-$W/examples_v3/slip_rebuild_fit/fit.jsonl}; [ -e "$FIT" ] || { echo "missing $FIT"; exit 1; }
    CASES=$(sed -n '/^CASES="/,/"$/p' scripts/run_pat_example.sh | tr -dc '0-9 \n' | tr '\n' ' ')
    FIG=5975; [ "$TEST" = 1 ] && { CASES="2173 3607"; FIG=""; }
    mkdir -p "${O}_detect"
    PAT_METHOD=rebuilt rec examples/pulse_arrival_time.py --root "$R" --cases $CASES ${FIG:+--figure $FIG} --out "$O" \
      --fit "$FIT" --time-budget 1e9 > "$O/log_rebuilt.txt" 2>&1 &
    PAT_METHOD=detect "$PY" "$PROV" record --out "${O}_detect" -- "$PY" -W ignore examples/pulse_arrival_time.py \
      --root "$R" --cases $CASES --out "${O}_detect" --fit "$FIT" --time-budget 1e9 > "$O/log_detect.txt" 2>&1 &
    wait
    if grep -l Traceback "$O"/log_*.txt > /dev/null 2>&1; then echo "failed: see $O/log_*.txt"; exit 1; fi
    rec examples/pulse_arrival_time.py --out "$O" --cohort-figure
    "$PY" "$PROV" record --out "${O}_detect" -- "$PY" -W ignore examples/pulse_arrival_time.py --out "${O}_detect" --cohort-figure
    rec examples/pat_example_stats.py "$O" --compare "${O}_detect" --case 5975 --json "$O/page_stats.json";;
  blood_gas_gradients)
    VR=$(resolution); D=$W/results/pat/lab_draw_labels.csv
    N=$("$PY" -c "import pandas as pd; d=pd.read_csv('$D'); d=d[(d.panel=='blood_gas')&(d.label=='lab_draw:single')]; print(d.case.nunique())")
    [ "$TEST" = 1 ] && N=8
    STEP=$(( (N + P - 1) / P )); echo "$(date +%H:%M) $N cases, $P chunks of $STEP"
    for i in $(seq 0 $((P - 1))); do A=$((i * STEP)); Z=$(( (i + 1) * STEP < N ? (i + 1) * STEP : N ))
      rec examples/$NAME.py run --resolution "$VR" --root "$R" --draws-csv "$D" --out "$O" --cases "$A:$Z" > "$O/log_run_$i.txt" 2>&1 &
    done; wait
    for i in $(seq 0 $((P - 1))); do A=$((i * STEP)); Z=$(( (i + 1) * STEP < N ? (i + 1) * STEP : N ))
      rec examples/$NAME.py lagscan --root "$R" --draws-csv "$D" --out "$O" --cases "$A:$Z" > "$O/log_lag_$i.txt" 2>&1 &
    done; wait
    if grep -l Traceback "$O"/log_*.txt > /dev/null 2>&1; then echo "failed: see $O/log_*.txt"; exit 1; fi
    if [ "$TEST" = 1 ]; then rec examples/$NAME.py summarise --out "$O" --root "$R"
    else rec examples/$NAME.py summarise --out "$O" --root "$R" --draws-csv "$D" --figure 136-1 16-3; fi
    rec examples/$NAME.py lagsummary --out "$O"
    rec examples/co2_gradient_spread.py "$O/parts" --out "$O/co2_gradient_spread.json";;
  slip_rebuild_fit)
    # The fit step of scripts/run_slip_rebuild_full.sh (5 Nov, into $W/slip_rebuild/full3), now with run records. It
    # reuses the beat-quality folder of 4 Nov ($W/slip_rebuild/quality, hashed as an input); 6 parts as on 5 Nov, so
    # the joined fit.jsonl lists the recordings in the same order and can be compared with full3 byte for byte.
    Q=$W/slip_rebuild/quality; NP=6; LIMIT=0; [ "$TEST" = 1 ] && { NP=1; LIMIT=5; }
    for f in $Q $W/sawtooth_char/scan_full $W/sawtooth_stats/recordings.csv $W/results/sync_full/snuadc_mode_label.json \
             $W/label_quality_final/summary.json; do [ -e "$f" ] || { echo "missing input: $f"; exit 1; }; done
    rec -c 'import json, sys
from pathlib import Path
cases = sorted(int(p.stem) for p in Path(sys.argv[1]).glob("*.npz"))
cases = cases[: int(sys.argv[3])] if int(sys.argv[3]) else cases
Path(sys.argv[2]).write_text(json.dumps({"cases": cases}) + "\n")
print(len(cases), "recordings")' "$W/sawtooth_char/scan_full" "$O/cases.json" "$LIMIT" || { echo "case list failed"; exit 1; }
    echo "$(date +%H:%M) fit, $NP parts"
    for p in $(seq 0 $((NP - 1))); do
      rec "$SCRIPT" fit --scan "$W/sawtooth_char/scan_full" --quality "$Q" --cases "$O/cases.json" \
        --rates "$W/sawtooth_stats/recordings.csv" --modes "$W/results/sync_full/snuadc_mode_label.json" \
        --rule "$W/label_quality_final/summary.json" --out "$O/fit_p$p.jsonl" --part $p --of $NP > "$O/log_fit_$p.txt" 2>&1 &
    done
    wait
    if grep -l Traceback "$O"/log_fit_*.txt > /dev/null 2>&1; then echo "a fit part failed: see $O/log_fit_*.txt"; exit 1; fi
    [ "$(ls "$O"/fit_p*.jsonl | wc -l)" -eq "$NP" ] || { echo "only $(ls "$O"/fit_p*.jsonl | wc -l) of $NP parts wrote a fit"; exit 1; }
    rec -c 'import sys
from pathlib import Path
parts = [Path(p) for p in sys.argv[2:]]
Path(sys.argv[1]).write_text("".join(p.read_text() for p in parts))
rows = sum(1 for p in parts for _ in p.open())
print(rows, "recordings in", sys.argv[1])' "$O/fit.jsonl" $(for p in $(seq 0 $((NP - 1))); do echo "$O/fit_p$p.jsonl"; done)
    "$PY" -c "import json,sys; e=sum('error' in json.loads(l) for l in open(sys.argv[1])); print(e, 'recordings with an error row')" "$O/fit.jsonl";;
  delta_pat_hist)
    all_cases | test_cases | scan_parts
    rec "$SCRIPT" agg "$O"/part_*.jsonl --out "$O/agg";;
  fluid_balance)
    VR=$(resolution); rec examples/$NAME.py --resolution "$VR" --root "$R" --out "$O" --figure 6156 4167 5253;;
  oliguria_labels|regression_with_rounding)
    VR=$(resolution); rec examples/$NAME.py --resolution "$VR" --root "$R" --out "$O" --draws 200;;
  urine_output_rate)
    VR=$(resolution); rec examples/$NAME.py --resolution "$VR" --root "$R" --out "$O" --figure 5111 111 113 305;;
  *) echo "unknown example $NAME (bash scripts/rerun_examples.sh list)"; exit 1;;
esac
disk
echo "$(date +%H:%M) $NAME done: outputs and run records in $O"
