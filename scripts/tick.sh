#!/bin/bash
# SPDX-License-Identifier: MIT
# One ~170 s step of the whole test matrix: AtriumDB builds in one slot; measurements (then TimeNet and
# numeric-only builds with whatever time is left) in the other; copy-out of measured builds alongside.
# HOLD lists builds kept on the VM after measuring (for the cold-read study).
cd "$(dirname "$0")"
T0=$(date +%s)
HOLD=${HOLD:-"A01-base A02-b4096 A03-b16384 A04-b65536 A05-b131072 A06-z3 A07-z9"}
SLOTS=1 bash t30.sh '^A' &
( bash t30_measure.sh '.'
  left=$(( T0 + 168 - $(date +%s) ))
  if [ $left -gt 100 ]; then SLOTS=1 BUDGET=$(( left - 45 )) bash t30.sh '^[TN]'; fi ) &
( for res in $HOME/results/t30/*/; do
    name=$(basename $res)
    [ -e $res/MEASURED ] && [ ! -e $res/COPIED ] && [[ " $HOLD " != *" $name "* ]] || continue
    [ $(( T0 + 100 - $(date +%s) )) -gt 0 ] || break
    bash t30_out.sh $name
  done ) &
wait
df -h $HOME | tail -1 | awk '{print "free", $4}'
