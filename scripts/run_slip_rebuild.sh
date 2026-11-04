#!/bin/bash
# Slip-finding rebuild: beat quality, then the fit with each piece switched off in turn, then the evaluation (4 Nov).
# Run in WSL. First the short foreground test (3 recordings, about a minute):
#   TEST=1 bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_slip_rebuild.sh
# then the full run in the background:
#   mkdir -p /mnt/c/vitalDB/work/slip_rebuild/ablation && nohup bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_slip_rebuild.sh > /mnt/c/vitalDB/work/slip_rebuild/ablation/run.log 2>&1 &
# Check: tail -3 /mnt/c/vitalDB/work/slip_rebuild/ablation/run.log
# Stop:  pkill -f run_slip_rebuild ; pkill -f slip_rebuild.py      (everything resumes: rerun the same command)
# Steps:
#   1. quality on the 324 selected recordings ($W/slip_rebuild/cases.json); already done, so seconds (resume)
#   2. fit, one variant at a time, N parts in parallel (about 1.5-3 s per recording per core: with N=6, 2-3 min a variant)
#        full      everything on (shape veto at grid slips only where the pulse clearly did not move)
#        rule_veto the shape rule must confirm grid slips
#        no_resp   breathing not fitted
#        no_gate   bad beats kept
#        r_peak    R peaks as detected (no template refinement)
#        no_prior  no setup prior (broad search only)
#   3. eval of each variant (seconds)
# About 15-20 min in all with N=6. Each part needs about 0.5 GB of memory; lower N if parts are killed.
W=${W:-/mnt/c/vitalDB/work}
B=${B:-/mnt/c/code/vitaldb/vitaldb-bench}
PY=${PY:-$HOME/venv/bin/python}
N=${N:-6}
cd $B
. $B/scripts/local.env
export PYTHONPATH=$B/tools:$VITALDB_PYTHONPATH:$PYTHONPATH
if [ "${TEST:-0}" = 1 ]; then
  O=${O:-$W/slip_rebuild/test}; CASES=$W/slip_rebuild/dev/cases.json; VARIANTS=${VARIANTS:-full}; N=1; Q=$O/quality
else
  O=${O:-$W/slip_rebuild/ablation}; CASES=$W/slip_rebuild/cases.json; Q=$W/slip_rebuild/quality
  VARIANTS=${VARIANTS:-"full rule_veto no_resp no_gate r_peak no_prior"}
fi
mkdir -p $O
# preflight: every tool this script runs must import
"$PY" -c "import numpy, scipy, physiosync, vitaldb_reader, sawtooth_char, label_quality, slip_rebuild" || { echo "Python environment not ready ($PY)"; exit 1; }
for f in $CASES $W/sawtooth_stats/recordings.csv $W/results/sync_full/snuadc_mode_label.json $W/label_quality_final/summary.json \
         $W/sawtooth_char/labels/key.json $W/label_eval_final/label_verdicts.json $W/sawtooth_char/full/slips.jsonl; do
  [ -f "$f" ] || { echo "missing input: $f"; exit 1; }
done
# after a parallel step: stop if any part's log holds a traceback, or a part wrote no output
check_parts() {  # step-name log-glob out-glob expected-parts
  if grep -l Traceback $2 > /dev/null 2>&1; then echo "$1 failed: see $(grep -l Traceback $2 | head -1)"; exit 1; fi
  local n=$(ls $3 2>/dev/null | wc -l)
  if [ "$n" -lt "$4" ]; then echo "$1: $n of $4 parts wrote output; see the logs $2"; exit 1; fi
}
opts() {
  case $1 in
    full) echo "--grid-veto clear" ;;
    rule_veto) echo "--grid-veto rule" ;;
    no_resp) echo "--grid-veto clear --resp off" ;;
    no_gate) echo "--grid-veto clear --gate off" ;;
    r_peak) echo "--grid-veto clear --r peak" ;;
    no_prior) echo "--grid-veto clear --prior none" ;;
    *) echo "unknown variant $1" >&2; exit 1 ;;
  esac
}

echo "$(date +%H:%M) 1. quality, $N parts"
for p in $(seq 0 $((N - 1))); do
  "$PY" tools/slip_rebuild.py quality --root "$VITALDB_ROOT" --scan $W/sawtooth_char/scan_full --cases $CASES --out $Q \
    --part $p --of $N > $O/log_quality_$p.txt 2>&1 &
done
wait
check_parts "quality" "$O/log_quality_*.txt" "$O/log_quality_[0-9]*.txt" $N
tail -q -n1 $O/log_quality_*.txt | tail -1
for v in $VARIANTS; do
  echo "$(date +%H:%M) 2. fit $v ($(opts $v)), $N parts"
  for p in $(seq 0 $((N - 1))); do
    "$PY" tools/slip_rebuild.py fit --scan $W/sawtooth_char/scan_full --quality $Q --rates $W/sawtooth_stats/recordings.csv \
      --modes $W/results/sync_full/snuadc_mode_label.json --rule $W/label_quality_final/summary.json \
      --out $O/${v}_p$p.jsonl --part $p --of $N $(opts $v) > $O/log_${v}_$p.txt 2>&1 &
  done
  wait
  check_parts "fit $v" "$O/log_${v}_*.txt" "$O/${v}_p[0-9]*.jsonl" $N
  cat $O/${v}_p[0-9]*.jsonl > $O/$v.jsonl
  echo "$(date +%H:%M) 3. eval $v: $(wc -l < $O/$v.jsonl) recordings"
  "$PY" tools/slip_rebuild.py eval --fit $O/$v.jsonl --key $W/sawtooth_char/labels/key.json \
    --verdicts $W/label_eval_final/label_verdicts.json --slips $W/sawtooth_char/full/slips.jsonl \
    --scan $W/sawtooth_char/scan_full --out $O/eval_$v > $O/log_eval_$v.txt 2>&1 || { echo "eval $v failed: see $O/log_eval_$v.txt"; exit 1; }
done
echo "$(date +%H:%M) done: $O/eval_*/summary.json"
