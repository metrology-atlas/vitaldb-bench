#!/bin/bash
# SPDX-License-Identifier: MIT
# Copy measured test builds to C:\vitalDB\test_dbs\<db>\<name>, check the file count, then free the VM copy.
# usage: t30_out.sh name...   (only builds with a MEASURED marker)
[ -f "$(dirname "$0")/local.env" ] && . "$(dirname "$0")/local.env"
set -u
for name in "$@"; do
  src=$HOME/build/t30/$name; res=$HOME/results/t30/$name
  [ -e "$res/MEASURED" ] || { echo "$name: not measured yet"; continue; }
  db=atriumdb; [ -d "$src/dataset" ] || db=timenet
  dst=${VITALDB_TEST_DBS:?set VITALDB_TEST_DBS in scripts/local.env}/$db/$name
  mkdir -p "$dst"
  timeout 160 cp -r "$src/." "$dst/" || { echo "$name: copy incomplete"; continue; }
  a=$(find "$src" -type f | wc -l); b=$(find "$dst" -type f | wc -l)
  sa=$(du -sb "$src" | cut -f1)
  if [ "$a" = "$b" ]; then rm -rf "$src"; touch "$res/COPIED"; echo "$name: $a files, $sa bytes -> test_dbs/$db"; else echo "$name: count mismatch $a vs $b"; fi
done
