# SPDX-License-Identifier: MIT
"""Do AtriumDB's waveform intervals match the real gaps in the source? Aggregates only.

    python tools/t30_gaps.py --root R --cases cases.json --index BUILD/dataset/meta/index.db --out gaps.json

For every waveform track: the source's runs (records that join to within half a sample), merged where
the next run starts no later than 1.5 sample periods after the previous one ends (overlaps and
re-sync jumps are not gaps; a missing sample is), against the number of intervals AtriumDB's index
holds for that track.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

import numpy as np

from timenet_connectors.datasets.vitaldb.vitaldb.timing import runs
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import TRACK_WAVE, read_vital


def expected_intervals(track) -> tuple[int, int]:
    period = 1.0 / float(track.srate)
    spans = []
    for a, b in runs(track):
        start = track.record_times[a]
        end = track.record_times[b - 1] + track.record_sizes[b - 1] * period
        spans.append((start, end))
    spans.sort()
    groups, cur_end = 0, -np.inf
    for s, e in spans:
        if s > cur_end + 1.5 * period:
            groups += 1
            cur_end = e
        else:
            cur_end = max(cur_end, e)
    return len(spans), groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    db = sqlite3.connect(f"file:{a.index}?mode=ro", uri=True)
    counts = {(m, d): n for m, d, n in db.execute(
        "select m.tag, d.tag, count(*) from interval_index i join measure m on m.id=i.measure_id "
        "join device d on d.id=i.device_id group by 1, 2")}
    diff = Counter()
    tot = Counter()
    worst = []
    for c in json.loads(Path(a.cases).read_text())["cases"]:
        case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital")
        for name, tr in case.tracks.items():
            if tr.track_type != TRACK_WAVE or not tr.record_times.size:
                continue
            n_runs, n_exp = expected_intervals(tr)
            got = counts.get((name, f"vitaldb-{c:04d}"), 0)
            tot["tracks"] += 1
            tot["runs"] += n_runs
            tot["expected"] += n_exp
            tot["atriumdb"] += got
            diff[got - n_exp] += 1
            if got != n_exp:
                worst.append((c, name, n_runs, n_exp, got))
    res = {"totals": dict(tot), "atriumdb_minus_expected": dict(sorted(diff.items())), "mismatches": worst[:40]}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("totals", "atriumdb_minus_expected")}))


if __name__ == "__main__":
    main()
