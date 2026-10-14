# SPDX-License-Identifier: MIT
"""Gap and overlap distribution of waveform tracks, as each database stores them. Aggregates only.

    python tools/gapdist.py BUILD [BUILD ...] --out gaps.json

For every waveform track of every case, the stored pieces (AtriumDB: rows of the interval index;
TimeNet: regular-axis Signals) are sorted by start, and the step from the furthest end so far (last
sample + one period) to the start of the next piece is classed, in sample periods and seconds:
overlap (< -0.5 periods), touching (|step| <= 0.5 periods), a gap of 0.5-1.5 periods (one sample
missing), up to 1 s, 1-10 s, 10-60 s, 1-10 min, > 10 min. Reads only the SQLite index / DuckDB control
files, never values.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np

BINS = [("overlap > 1 s", -np.inf, -1.0), ("overlap <= 1 s", -1.0, None), ("touching", None, None),
        ("1 sample missing", None, None), ("gap < 1 s", None, 1.0), ("1-10 s", 1.0, 10.0), ("10-60 s", 10.0, 60.0),
        ("1-10 min", 60.0, 600.0), ("> 10 min", 600.0, np.inf)]


def classify(step_s: float, period_s: float) -> str:
    p = step_s / period_s
    if p < -0.5:
        return "overlap > 1 s" if step_s < -1.0 else "overlap <= 1 s"
    if p <= 0.5:
        return "touching"
    if p <= 1.5:
        return "1 sample missing"
    for name, lo, hi in BINS[4:]:
        if (lo is None or step_s > lo) and step_s <= hi:
            return name
    return "> 10 min"


def pieces_atriumdb(build: Path) -> dict:
    c = sqlite3.connect(f"file:{build / 'dataset' / 'meta' / 'index.db'}?mode=ro", uri=True)
    out = defaultdict(list)
    rows = c.execute("select m.tag, m.freq_nhz, d.tag, i.start_time_n, i.end_time_n from interval_index i "
                     "join measure m on m.id=i.measure_id join device d on d.id=i.device_id where m.freq_nhz != 500000000")
    for tag, f, dev, s, e in rows:
        period = 1e9 / f  # seconds
        # AtriumDB's interval end is the last sample's time; add one period for the end of the piece
        out[(dev, tag)].append((s / 1e9, e / 1e9 + period, period))
    return out


def pieces_timenet(build: Path) -> dict:
    import duckdb  # noqa: PLC0415

    out = defaultdict(list)
    for v in sorted(build.glob("part-*/vitaldb/vitaldb/1.0.0")):
        c = duckdb.connect(str(v / "control.duckdb"), read_only=True)
        rows = c.execute(
            "select r.record_id, src.name || '/' || s.name, a.period_numerator_us, a.period_denominator, a.offset_us, "
            "a.start_index, s.n_values from signals s join axes a using (axis_key) join sources src using (source_key) "
            "join records r on r.record_key = src.record_key where a.axis_type = 'regular'").fetchall()
        for rec, name, num, den, off, si, n in rows:
            period_us = num / den
            first = off + (si * num) // den
            last = off + ((si + n - 1) * num) // den
            out[(rec, name)].append((first / 1e6, (last + period_us) / 1e6, period_us / 1e6))
        c.close()
    return out


def distribution(pieces: dict) -> dict:
    counts: Counter = Counter()
    by_track: dict = defaultdict(Counter)
    steps_s: list[float] = []
    gap_time = 0.0
    n_tracks = n_pieces = 0
    for (_, name), ps in pieces.items():
        ps.sort()
        period = float(np.median([x[2] for x in ps]))  # nominal, or close to it (resync/file timing vary per piece)
        n_tracks += 1
        n_pieces += len(ps)
        reach = ps[0][1]  # furthest end so far: a piece nested inside a longer one is not a gap
        for s1, e1, _ in ps[1:]:
            step = s1 - reach
            reach = max(reach, e1)
            k = classify(step, period)
            counts[k] += 1
            by_track[name.split("/")[0]][k] += 1
            steps_s.append(step)
            if step > 0:
                gap_time += step
    st = np.array(steps_s)
    q = (lambda a, p: float(np.quantile(a, p)) if a.size else None)
    pos, neg = st[st > 0], st[st < 0]
    return {"tracks": n_tracks, "pieces": n_pieces, "boundaries": int(st.size),
            "classes": {name: counts.get(name, 0) for name, *_ in BINS},
            "by_device": {d: {name: c.get(name, 0) for name, *_ in BINS} for d, c in sorted(by_track.items())},
            "gap_s_quantiles": {p: q(pos, p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "overlap_s_quantiles": {p: q(-neg, p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "total_gap_hours": gap_time / 3600, "steps_s": sorted(round(x, 6) for x in steps_s)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("builds", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for b in map(Path, a.builds):
        pieces = pieces_atriumdb(b) if (b / "dataset").exists() else pieces_timenet(b)
        res[b.name] = distribution(pieces)
        r = res[b.name]
        print(b.name, r["tracks"], "tracks", r["pieces"], "pieces", r["classes"])
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
