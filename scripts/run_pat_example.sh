#!/bin/bash
# Worked example pulse arrival time (atlas v2, 5 Nov): the 121 recordings of pat4, rerun with the example's window
# inside the whole-release fit's stretch; rebuilt (the atlas default) and detect on the same windows.
# Run in WSL. Test first (2 recordings, about a minute):
#   TEST=1 bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_pat_example.sh
# then the full run in the background (about an hour):
#   mkdir -p /mnt/c/vitalDB/work/examples/pat5 && nohup bash /mnt/c/code/vitaldb/vitaldb-bench/scripts/run_pat_example.sh > /mnt/c/vitalDB/work/examples/pat5/run.log 2>&1 &
# Check: tail -3 /mnt/c/vitalDB/work/examples/pat5/run.log
# Stop:  pkill -f run_pat_example ; pkill -f pulse_arrival_time.py   (rerun to resume)
W=${W:-/mnt/c/vitalDB/work}
B=${B:-/mnt/c/code/vitaldb/vitaldb-bench}
PY=${PY:-$HOME/venv/bin/python}
cd $B
. $B/scripts/local.env
export W PYTHONPATH=$B/tools:$VITALDB_PYTHONPATH:$PYTHONPATH
CASES="9 31 66 152 203 236 265 272 337 352 355 402 421 463 482 487 529 536 572 595 717 725 883 932 933 937 1003 1076
1078 1270 1274 1281 1304 1368 1380 1479 1559 1682 1719 1740 1741 1836 1876 1957 1977 2100 2173 2187 2384 2421 2434 2440
2456 2482 2499 2697 2716 2731 2742 2862 2929 2954 3069 3218 3274 3282 3342 3346 3394 3438 3470 3489 3508 3607 3622 3714
3722 3751 3871 3969 3999 4057 4063 4103 4119 4122 4231 4273 4282 4342 4415 4546 4572 4611 4636 4687 4700 4714 4853 4952
5057 5118 5190 5205 5218 5266 5577 5588 5603 5739 5847 5871 5902 5954 5975 6018 6205 6251 6276 6308 6367"
FIG=5975
if [ "${TEST:-0}" = 1 ]; then CASES="2173 3607"; FIG=""; O=$W/examples/pat5_test; else O=$W/examples/pat5; fi
"$PY" -c "import physiosync; assert physiosync.__version__ >= '0.3.3', physiosync.__version__" \
  || { echo "physiosync older than 0.3.3 ($PY)"; exit 1; }
[ -e $W/slip_rebuild/full3/fit.jsonl ] || { echo "missing $W/slip_rebuild/full3/fit.jsonl"; exit 1; }
mkdir -p $O $O/../$(basename $O)_detect
echo "$(date +%H:%M) rebuilt and detect, $(echo $CASES | wc -w) recordings"
PAT_METHOD=rebuilt "$PY" examples/pulse_arrival_time.py --root "$VITALDB_ROOT" --cases $CASES --figure $FIG --out $O \
  --time-budget 1e9 > $O/log_rebuilt.txt 2>&1 &
PAT_METHOD=detect "$PY" examples/pulse_arrival_time.py --root "$VITALDB_ROOT" --cases $CASES --out ${O}_detect \
  --time-budget 1e9 > $O/log_detect.txt 2>&1 &
wait
if grep -l Traceback $O/log_*.txt > /dev/null 2>&1; then echo "failed: see $O/log_*.txt"; exit 1; fi
echo "$(date +%H:%M) figures"
"$PY" examples/pulse_arrival_time.py --out $O --cohort-figure > $O/figures_rebuilt.txt 2>&1 || { echo "figures failed"; exit 1; }
"$PY" examples/pulse_arrival_time.py --out ${O}_detect --cohort-figure > $O/figures_detect.txt 2>&1 || { echo "figures failed"; exit 1; }
echo "$(date +%H:%M) done"; tail -4 $O/figures_rebuilt.txt
