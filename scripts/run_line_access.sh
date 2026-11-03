#!/bin/bash
# Line-access summary (3 Nov): rebuilds the inputs of the line-access-summary figure. Run in WSL:
#   nohup bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_line_access.sh > /mnt/c/vitalDB/work/line_access/run.log 2>&1 &
# Steps (all resume: rerun the same command to carry on; Ctrl-C or `pkill -f run_line_access` stops it):
#   1. beats_export on the 266 benchmark cases (about 3 s each, 3 parts in parallel: ~5 min)
#   2. art_export on those beats (~4 min)
#   3. line_access_scan on the 1,440 cases the first scan labelled, with the benchmark beats and ART for the
#      arterial-side steps (about 4 s per case, N parts in parallel: ~30 min with N=3)
#   4. the parts' labels joined, lab_draw_labels, and the figure (seconds)
# Check: tail -3 /mnt/c/vitalDB/work/line_access/run.log ; output in /mnt/c/vitalDB/work/line_access/
W=${W:-/mnt/c/vitalDB/work}
B=${B:-/mnt/c/code/vitaldb/vitaldb-bench}
L=$W/line_access
. $B/scripts/local.env
export PYTHONPATH=$B/tools:$VITALDB_PYTHONPATH
PY=${PY:-$HOME/venv/bin/python}
N=${N:-3}
cd $B
"$PY" -c "import numpy, scipy, pandas, physiosync" || { echo "Python environment not ready ($PY)"; exit 1; }
mkdir -p $L/beats $L/art

echo "$(date +%H:%M) 1. beats_export"
for p in $(seq 0 $((N - 1))); do
  "$PY" tools/beats_export.py --root "$VITALDB_ROOT" --cases $L/cases_bench.json --out $L/beats --part $p --of $N --time-budget 100000 > $L/log_beats_$p.txt 2>&1 &
done
wait
echo "$(date +%H:%M) 2. art_export"
for p in $(seq 0 $((N - 1))); do
  "$PY" tools/art_export.py --root "$VITALDB_ROOT" --beats $L/beats --out $L/art --part $p --of $N --time-budget 100000 > $L/log_art_$p.txt 2>&1 &
done
wait
echo "$(date +%H:%M) 3. line_access_scan"
for p in $(seq 0 $((N - 1))); do
  "$PY" tools/line_access_scan.py --root "$VITALDB_ROOT" --cases $L/cases_art.json --out $L/scan_$p.jsonl \
    --labels $L/labels_$p.csv --beats $L/beats --art $L/art --part $p --of $N > $L/log_scan_$p.txt 2>&1 &
done
wait
echo "$(date +%H:%M) 4. labels, draws, figure"
awk 'FNR == 1 && NR != 1 { next } { print }' $L/labels_*.csv > $L/art_line_labels.csv
"$PY" tools/lab_draw_labels.py --lab "$VITALDB_ROOT/lab_data.csv" --art-labels $L/art_line_labels.csv \
  --out-draw $L/lab_draw_labels.csv --out-time $L/lab_time_labels.csv
"$PY" tools/line_access_summary.py figure --scan "$L/scan_*.jsonl" --labels $L/art_line_labels.csv \
  --draws $L/lab_draw_labels.csv --out $L
echo "$(date +%H:%M) done: $(wc -l < $L/art_line_labels.csv) label rows; figure $L/line-access-summary.png"
