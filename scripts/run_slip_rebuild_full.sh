#!/bin/bash
# Slip rebuild over the whole release (5 Nov): beat quality, the fit, the tiers. Run in WSL.
# First the short foreground test (5 recordings, about a minute):
#   TEST=1 bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_slip_rebuild_full.sh
# then the full run in the background:
#   mkdir -p /mnt/c/vitalDB/work/slip_rebuild/full && nohup bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_slip_rebuild_full.sh > /mnt/c/vitalDB/work/slip_rebuild/full/run.log 2>&1 &
# Check: tail -3 /mnt/c/vitalDB/work/slip_rebuild/full/run.log
# Stop:  pkill -f run_slip_rebuild_full ; pkill -f slip_rebuild.py      (everything resumes: rerun the same command)
# Steps:
#   1. the case list: every recording of the full-release scan ($W/sawtooth_char/scan_full, about 5,700)
#   2. quality: re-reads ECG and PLETH per recording (about 1.2 s each per core; N=6: about 20 min); the 324 already done
#      are kept
#   3. fit (about 3-5 s each per core; N=6: about 50-80 min), then the parts joined
#   4. eval against the labels and injections (seconds), and the tiers and gallery (a few minutes)
# Each part needs about 0.5 GB of memory; lower N if parts are killed.
W=${W:-/mnt/c/vitalDB/work}
B=${B:-/mnt/c/code/vitaldb/vitaldb-bench}
PY=${PY:-$HOME/venv/bin/python}
N=${N:-6}
cd $B
. $B/scripts/local.env
export PYTHONPATH=$B/tools:$VITALDB_PYTHONPATH:$PYTHONPATH
if [ "${TEST:-0}" = 1 ]; then
  O=${O:-$W/slip_rebuild/full_test}; N=1; LIMIT=5
else
  O=${O:-$W/slip_rebuild/full}; LIMIT=0
fi
Q=$W/slip_rebuild/quality
mkdir -p $O
"$PY" -c "import numpy, scipy, matplotlib, physiosync, vitaldb_reader, sawtooth_char, label_quality, slip_rebuild, slip_tiers" \
  || { echo "Python environment not ready ($PY)"; exit 1; }
for f in $W/sawtooth_stats/recordings.csv $W/results/sync_full/snuadc_mode_label.json $W/label_quality_final/summary.json \
         $W/sawtooth_char/labels/key.json $W/label_eval_final/label_verdicts.json $W/sawtooth_char/full/slips.jsonl; do
  [ -f "$f" ] || { echo "missing input: $f"; exit 1; }
done
check_parts() {  # step-name log-glob out-glob expected-parts
  if grep -l Traceback $2 > /dev/null 2>&1; then echo "$1 failed: see $(grep -l Traceback $2 | head -1)"; exit 1; fi
  local n=$(ls $3 2>/dev/null | wc -l)
  if [ "$n" -lt "$4" ]; then echo "$1: $n of $4 parts wrote output; see the logs $2"; exit 1; fi
}

echo "$(date +%H:%M) 1. case list"
"$PY" - "$W/sawtooth_char/scan_full" "$O/cases.json" "$LIMIT" <<'EOF' || { echo "case list failed"; exit 1; }
import json, sys
from pathlib import Path
cases = sorted(int(p.stem) for p in Path(sys.argv[1]).glob("*.npz"))
lim = int(sys.argv[3])
if lim:
    cases = cases[:lim]
Path(sys.argv[2]).write_text(json.dumps({"cases": cases}) + "\n")
print(len(cases), "recordings")
EOF
echo "$(date +%H:%M) 2. quality, $N parts"
for p in $(seq 0 $((N - 1))); do
  "$PY" tools/slip_rebuild.py quality --root "$VITALDB_ROOT" --scan $W/sawtooth_char/scan_full --cases $O/cases.json \
    --out $Q --part $p --of $N > $O/log_quality_$p.txt 2>&1 &
done
wait
check_parts "quality" "$O/log_quality_*.txt" "$O/log_quality_[0-9]*.txt" $N
tail -q -n1 $O/log_quality_*.txt | tail -1
echo "$(date +%H:%M) 3. fit, $N parts"
for p in $(seq 0 $((N - 1))); do
  "$PY" tools/slip_rebuild.py fit --scan $W/sawtooth_char/scan_full --quality $Q --cases $O/cases.json \
    --rates $W/sawtooth_stats/recordings.csv --modes $W/results/sync_full/snuadc_mode_label.json \
    --rule $W/label_quality_final/summary.json --out $O/fit_p$p.jsonl --part $p --of $N > $O/log_fit_$p.txt 2>&1 &
done
wait
check_parts "fit" "$O/log_fit_*.txt" "$O/fit_p[0-9]*.jsonl" $N
cat $O/fit_p[0-9]*.jsonl > $O/fit.jsonl
echo "$(date +%H:%M) 4. eval and tiers: $(wc -l < $O/fit.jsonl) recordings"
for cl in "" "--clean-hosts"; do
  "$PY" tools/slip_rebuild.py eval $cl --fit $O/fit.jsonl --key $W/sawtooth_char/labels/key.json \
    --verdicts $W/label_eval_final/label_verdicts.json --slips $W/sawtooth_char/full/slips.jsonl \
    --scan $W/sawtooth_char/scan_full --out $O/eval${cl:+_clean} > $O/log_eval${cl:+_clean}.txt 2>&1 \
    || { echo "eval failed: see $O/log_eval${cl:+_clean}.txt"; exit 1; }
done
"$PY" tools/slip_tiers.py --fit $O/fit.jsonl --scan $W/sawtooth_char/scan_full --out $O/tiers > $O/log_tiers.txt 2>&1 \
  || { echo "tiers failed: see $O/log_tiers.txt"; exit 1; }
echo "$(date +%H:%M) done: $O/fit.jsonl, $O/eval/summary.json, $O/tiers/summary.json"
