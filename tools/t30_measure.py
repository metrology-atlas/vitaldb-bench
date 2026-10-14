# SPDX-License-Identifier: MIT
"""Size and index measurements of one test build, without opening either SDK.

    python tools/t30_measure.py BUILD_DIR --out size.json

AtriumDB builds (``BUILD_DIR/dataset/meta/index.db``) are measured from the SQLite index and the
``.tsc`` files; TimeNet builds (``BUILD_DIR/part-*``) from the Parquet footers. Output is aggregate
only: byte counts, value counts and index sizes, split by waveform / numeric / string.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import os
from pathlib import Path
import sqlite3

NUMERIC_FREQ_NHZ = 500_000_000  # the adapter registers every numeric track at 0.5 Hz


def _files_bytes(root: Path, pattern: str) -> int:
    return sum(os.path.getsize(p) for p in glob.glob(str(root / pattern), recursive=True))


def atriumdb(build: Path) -> dict:
    d = build / "dataset"
    c = sqlite3.connect(f"file:{d / 'meta' / 'index.db'}?mode=ro", uri=True)
    kind = {mid: ("numeric" if f == NUMERIC_FREQ_NHZ else "wave") for mid, f in c.execute("select id, freq_nhz from measure")}
    tag = dict(c.execute("select id, tag from measure"))
    tsc = defaultdict(int)
    for p in glob.glob(str(d / "tsc" / "*" / "*" / "*.tsc")):
        tsc[int(Path(p).parent.name)] += os.path.getsize(p)
    out: dict = {"db": "atriumdb", "total_bytes": _files_bytes(d, "**/*"), "index_db_bytes": os.path.getsize(d / "meta" / "index.db"),
                 "kinds": {}}
    for k in ("wave", "numeric"):
        mids = [m for m in kind if kind[m] == k]
        nv = nb = nblk = 0
        for mid in mids:
            for n, b, cnt in c.execute("select coalesce(sum(num_values),0), coalesce(sum(num_bytes),0), count(*) from block_index where measure_id=?", (mid,)):
                nv += n
                nb += b
                nblk += cnt
        nint = sum(c.execute("select count(*) from interval_index where measure_id=?", (m,)).fetchone()[0] for m in mids)
        tb = sum(tsc[m] for m in mids)
        out["kinds"][k] = {"values": nv, "blocks": nblk, "block_bytes": nb, "tsc_bytes": tb, "intervals": nint,
                           "bits_per_value": tb * 8 / nv if nv else None}
    out["labels"] = c.execute("select count(*) from label").fetchone()[0]
    per: dict = {}
    for mid, t in tag.items():
        n, b = c.execute("select coalesce(sum(num_values),0), coalesce(sum(num_bytes),0) from block_index where measure_id=?", (mid,)).fetchone()
        if n:
            per[t] = {"values": n, "tsc_bytes": tsc[mid], "bits_per_value": tsc[mid] * 8 / n,
                      "intervals": c.execute("select count(*) from interval_index where measure_id=?", (mid,)).fetchone()[0]}
    out["measures"] = dict(sorted(per.items(), key=lambda kv: -kv[1]["values"])[:25])
    return out


def timenet(build: Path) -> dict:
    import pyarrow.parquet as pq  # noqa: PLC0415

    kinds: dict = defaultdict(lambda: defaultdict(int))
    spec: dict = defaultdict(lambda: defaultdict(int))
    parts = sorted(build.glob("part-*/vitaldb/vitaldb/1.0.0"))
    import duckdb  # noqa: PLC0415

    axis_kind: dict[str, str] = {}
    n_offsets = n_signals = 0
    for v in parts:
        c = duckdb.connect(str(v / "control.duckdb"), read_only=True)
        for st, at, dt in c.execute("select distinct s.spec_type, a.axis_type, s.dtype from signals s join axes a using (axis_key)").fetchall():
            axis_kind[st] = "string" if dt == "str" else ("wave" if "regular" == str(at).lower() else "numeric")
        n_offsets += c.execute("select count(*) from axis_offsets").fetchone()[0]
        n_signals += c.execute("select count(*) from signals").fetchone()[0]
        c.close()
    for v in parts:
        for p in glob.glob(str(v / "time_series" / "*.parquet")):
            md = pq.ParquetFile(p).metadata
            for i in range(md.num_row_groups):
                rg = md.row_group(i)
                cols = {rg.column(j).path_in_schema: rg.column(j) for j in range(rg.num_columns)}
                vcol = next(c for k, c in cols.items() if k.startswith("values"))
                tcol = next(c for k, c in cols.items() if k.startswith("time_offsets"))
                st = cols["spec_type"].statistics
                key = st.min if st is not None and st.min == st.max else "mixed"
                k = axis_kind.get(key, "mixed")
                n_values = vcol.num_values
                for agg in (kinds[k], spec[key]):
                    agg["values"] += n_values
                    agg["value_bytes"] += vcol.total_compressed_size
                    agg["time_bytes"] += tcol.total_compressed_size
                    agg["other_bytes"] += sum(c.total_compressed_size for kk, c in cols.items() if c is not vcol and c is not tcol)
                    agg["row_groups"] += 1
    zarr_bytes = sum(_files_bytes(v, "values/**/*") + _files_bytes(v, "*.zarr/**/*") for v in parts)
    control = sum(os.path.getsize(v / "control.duckdb") for v in parts if (v / "control.duckdb").exists())
    out = {"db": "timenet", "partitions": len(parts), "total_bytes": _files_bytes(build, "part-*/**/*"),
           "control_bytes": control, "zarr_bytes": zarr_bytes, "signals": n_signals, "irregular_time_offsets": n_offsets,
           "axis_types": sorted(set(axis_kind.values())),
           "kinds": {k: dict(v) | {"bits_per_value": v["value_bytes"] * 8 / v["values"] if v["values"] else None} for k, v in kinds.items()},
           "spec_types": dict(sorted(((k, dict(v)) for k, v in spec.items()), key=lambda kv: -kv[1]["values"])[:25])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build = Path(a.build)
    res = atriumdb(build) if (build / "dataset").exists() else timenet(build)
    res["build"] = build.name
    log = build / "build_log.jsonl"
    if log.exists():
        rows = [json.loads(x) for x in log.read_text().splitlines()]
        res["ingest_seconds"] = sum(r.get("seconds", 0) for r in rows)
        res["maxrss_mb"] = max((r.get("maxrss_mb", 0) for r in rows), default=None)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k in ("build", "total_bytes", "kinds", "ingest_seconds")}))


if __name__ == "__main__":
    main()
