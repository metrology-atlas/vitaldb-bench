#!/bin/bash
# Power spectra of every waveform over the whole release (metrology-atlas: the spectrum figure on each signal page and
# the study of the waveforms' frequency content). Run in WSL. Test first (3 recordings, a few minutes):
#   TEST=1 bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_spectra.sh
# then the full run in the background:
#   mkdir -p /mnt/c/vitalDB/work/spectra && nohup bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_spectra.sh > /mnt/c/vitalDB/work/spectra/run.log 2>&1 &
# Check: tail -3 /mnt/c/vitalDB/work/spectra/run.log
# Stop:  pkill -f run_spectra ; pkill -f spectra_scan.py   (rerun to resume: each part skips the recordings it has)
# Steps: 1. synth (the fit, notch and delay checks); 2. scan in N parts ($W/spectra/part_K.jsonl and part_K_mNNN.npz).
# The summaries and figures (tools/spectra_agg.py) are run afterwards.
W=${W:-/mnt/c/vitalDB/work}
B=${B:-/mnt/c/code/vitaldb/vitaldb-bench}
N=${N:-8}
cd $B
. $B/scripts/local.env
PY=${PY:-$VITALDB_PYTHON}
export PYTHONPATH=$B/tools:$VITALDB_PYTHONPATH:$PYTHONPATH
"$PY" -c "import numpy, scipy, vitaldb_reader" || { echo "Python environment not ready ($PY)"; exit 1; }
if [ "${TEST:-0}" = 1 ]; then O=${O:-$W/spectra_test}; else O=${O:-$W/spectra}; fi
mkdir -p $O
NREC=$(ls "$VITALDB_ROOT"/vital_files/*.vital | wc -l)

echo "$(date +%H:%M) 1. synth"
"$PY" -W ignore tools/spectra_scan.py synth --out $O/synth.json > $O/log_synth.txt 2>&1 \
  || { echo "synth failed: see $O/log_synth.txt"; exit 1; }

if [ "${TEST:-0}" = 1 ]; then
  echo "$(date +%H:%M) 2. scan, test: 3 recordings"
  "$PY" -W ignore tools/spectra_scan.py scan --root "$VITALDB_ROOT" --out $O/part_0 --part 0 --of $((NREC / 3)) \
    > $O/log_0.txt 2>&1
  grep -q Traceback $O/log_0.txt && { echo "scan failed: see $O/log_0.txt"; exit 1; }
  echo "$(date +%H:%M) test done: $(wc -l < $O/part_0.jsonl) recordings, $(grep -c '"error"' $O/part_0.jsonl) errors"
  exit 0
fi

echo "$(date +%H:%M) 2. scan, $N parts, $NREC recordings"
start=$(date +%s)
for p in $(seq 0 $((N - 1))); do
  "$PY" -W ignore tools/spectra_scan.py scan --root "$VITALDB_ROOT" --out $O/part_$p --part $p --of $N \
    > $O/log_$p.txt 2>&1 &
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  sleep 300; echo "$(( ($(date +%s) - start) / 60 )) min: $(cat $O/part_*.jsonl 2>/dev/null | wc -l) of $NREC recordings"
done
wait
if grep -l Traceback $O/log_*.txt > /dev/null 2>&1; then echo "scan failed: see $O/log_*.txt"; exit 1; fi
echo "$(date +%H:%M) done in $(( ($(date +%s) - start) / 60 )) min: $(cat $O/part_*.jsonl | wc -l) of $NREC recordings;" \
  "$(cat $O/part_*.jsonl | grep -c '"error"') errors"
