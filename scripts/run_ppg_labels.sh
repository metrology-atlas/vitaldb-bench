#!/bin/bash
# PPG timing labels for atlas v2 (5 Nov): the rebuilt correction (physiosync 0.3.3) on the 266 benchmark recordings.
# Run in WSL. Test first (3 recordings, about a minute):
#   TEST=1 bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_ppg_labels.sh
# then the full run in the background:
#   mkdir -p /mnt/c/vitalDB/work/ppg_labels_v2 && nohup bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_ppg_labels.sh > /mnt/c/vitalDB/work/ppg_labels_v2/run.log 2>&1 &
# Check: tail -3 /mnt/c/vitalDB/work/ppg_labels_v2/run.log
# Stop:  pkill -f run_ppg_labels ; pkill -f foot_export.py ; pkill -f measurement_export.py   (rerun to resume step 1)
# Steps:
#   1. foot_export: the PLETH foot of every benchmark beat ($W/line_access/beats; about 3 s per recording, N parts)
#   2. measurement_export --rebuilt-fit: records and labels per part (the whole-release fit $W/slip_rebuild/full3),
#      offsets from the setups ($W/results/pat/setup_offsets.json), feet as the PPG fiducial
#   3. the parts joined into ppg_timing_labels.csv, ppg_sawtooth_labels.csv and ppg_offset_labels.csv, then
#      tools/label_stats.py for the atlas pages
W=${W:-/mnt/c/vitalDB/work}
B=${B:-/mnt/c/code/vitaldb/vitaldb-bench}
PY=${PY:-$HOME/venv/bin/python}
N=${N:-3}
cd $B
. $B/scripts/local.env
export PYTHONPATH=$B/tools:$VITALDB_PYTHONPATH:$PYTHONPATH
if [ "${TEST:-0}" = 1 ]; then O=${O:-$W/ppg_labels_v2_test}; else O=${O:-$W/ppg_labels_v2}; fi
L=$W/line_access
mkdir -p $O/foot $O/records $O/parts
"$PY" -c "import numpy, scipy, physiosync, vitaldb_reader; assert physiosync.__version__ >= '0.3.3', physiosync.__version__" \
  || { echo "Python environment not ready ($PY), or physiosync older than 0.3.3"; exit 1; }
for f in $L/beats $L/art $W/slip_rebuild/full3/fit.jsonl $W/ppg_correction_check/priors266.json \
         $W/results/pat/setup_offsets.json $W/results/sync_full/units.csv; do
  [ -e "$f" ] || { echo "missing input: $f"; exit 1; }
done
BEATS=$L/beats
if [ "${TEST:-0}" = 1 ]; then
  mkdir -p $O/beats_test && for c in 0013 0029 0052; do cp -n $L/beats/$c.npz $O/beats_test/; done
  BEATS=$O/beats_test; N=1
fi

echo "$(date +%H:%M) 1. foot_export, $N parts"
for p in $(seq 0 $((N - 1))); do
  "$PY" tools/foot_export.py --root "$VITALDB_ROOT" --beats $BEATS --out $O/foot --part $p --of $N --time-budget 100000 \
    > $O/log_foot_$p.txt 2>&1 &
done
wait
if grep -l Traceback $O/log_foot_*.txt > /dev/null 2>&1; then echo "foot_export failed: see $O/log_foot_*.txt"; exit 1; fi
echo "$(ls $O/foot | wc -l) recordings with feet"

echo "$(date +%H:%M) 2. measurement_export (rebuilt), $N parts"
for p in $(seq 0 $((N - 1))); do
  rm -f $O/parts/*_$p.csv 2>/dev/null
  "$PY" tools/measurement_export.py --beats $BEATS --art $L/art --priors $W/ppg_correction_check/priors266.json \
    --fiducial foot --foot $O/foot --setups $W/results/sync_full/units.csv --setup-offsets $W/results/pat/setup_offsets.json \
    --rebuilt-fit $W/slip_rebuild/full3/fit.jsonl --out-dir $O/records --labels $O/parts/timing_$p.csv \
    --saw-labels $O/parts/sawtooth_$p.csv --offset-labels $O/parts/offset_$p.csv --part $p --of $N \
    > $O/log_export_$p.txt 2>&1 &
done
wait
if grep -l Traceback $O/log_export_*.txt > /dev/null 2>&1; then echo "measurement_export failed: see $O/log_export_*.txt"; exit 1; fi

echo "$(date +%H:%M) 3. join and statistics"
for k in timing sawtooth offset; do
  out=$O/ppg_${k}_labels.csv
  head -1 $O/parts/${k}_0.csv > $out
  for f in $O/parts/${k}_*.csv; do tail -n +2 $f >> $out; done
done
grep -h '"skip"\|"error"' $O/log_export_*.txt > $O/skipped.jsonl
"$PY" tools/label_stats.py --timing $O/ppg_timing_labels.csv --sawtooth $O/ppg_sawtooth_labels.csv \
  --ends $W/results/pat/numeric_delay_labels.csv --out $O/label_stats.json || { echo "label_stats failed"; exit 1; }
echo "$(date +%H:%M) done: $(ls $O/records | wc -l) records; $(tail -n +2 $O/ppg_timing_labels.csv | wc -l) timing labels; skipped: $(wc -l < $O/skipped.jsonl)"
