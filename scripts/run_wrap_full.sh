#!/bin/bash
# Wrap test on the whole release (3 Nov). Run in WSL:
#   nohup bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_wrap_full.sh > /mnt/c/vitalDB/work/wrap_full/run.log 2>&1 &
# Steps (all resume: rerun the same command to carry on; `pkill -f run_wrap_full` and `pkill -f monitor_lines` stop it):
#   1. monitor_lines scan over the full-release sawtooth scan (about 2.4 s per recording on one core; 5,700
#      recordings: about 30-40 min with N=8 parts in parallel on 8 cores, longer on fewer)
#   2. monitor_lines agg (the 240 Hz line and the 30 Hz family over the release; seconds)
#   3. wrap_test select: every recording with the 240 Hz line in ART and PLETH, its longest stretch (about 1 s each)
#   4. wrap_test run on every selected recording spanning at least one wrap (about 1 s each), with the out-of-sample
#      control against the six of the first run
# Check: tail -3 /mnt/c/vitalDB/work/wrap_full/run.log ; output in /mnt/c/vitalDB/work/wrap_full/
W=${W:-/mnt/c/vitalDB/work}
B=${B:-/mnt/c/code/vitaldb/vitaldb-bench}
O=$W/wrap_full
. $B/scripts/local.env
export PYTHONPATH=$B/tools:$VITALDB_PYTHONPATH
PY=${PY:-$HOME/venv/bin/python}
N=${N:-8}
cd $B
"$PY" -c "import numpy, scipy, physiosync" || { echo "Python environment not ready ($PY)"; exit 1; }
mkdir -p $O

echo "$(date +%H:%M) 1. monitor_lines scan, $N parts"
for p in $(seq 0 $((N - 1))); do
  "$PY" tools/monitor_lines.py scan --root "$VITALDB_ROOT" --scan $W/sawtooth_char/scan_full --out $O/lines_$p.jsonl \
    --part $p --of $N --time-budget 1000000 > $O/log_lines_$p.txt 2>&1 &
done
wait
cat $O/lines_*.jsonl > $O/lines.jsonl
echo "$(date +%H:%M) 2. monitor_lines agg: $(wc -l < $O/lines.jsonl) recordings"
"$PY" tools/monitor_lines.py agg --lines $O/lines.jsonl --recordings $W/sawtooth_char/full/recordings.csv --out $O/lines_agg
echo "$(date +%H:%M) 3. wrap_test select"
"$PY" tools/wrap_test.py select --lines $O/lines.jsonl --root "$VITALDB_ROOT" --out $O/files.json
echo "$(date +%H:%M) 4. wrap_test run"
"$PY" tools/wrap_test.py run --files $O/files.json --root "$VITALDB_ROOT" --out $O/run --min-wraps 1 \
  --first $W/wrap_test/files.json --time-budget 1000000 | tail -1
echo "$(date +%H:%M) done: $O/run/wrap_test.json"
