# SPDX-License-Identifier: MIT
"""Resumable builds of the TimeNet database, in time-boxed chunks.

Each run does as much as fits in ``--time-budget`` seconds and stops cleanly between units of work, so a
build can be driven by repeated short commands. Progress lives in ``build_log.jsonl`` beside the output.

    python -m vitaldb_bench.build select  --root R --out cases.json --n 200 --seed 0
    python -m vitaldb_bench.build timenet --root R --cases cases.json --out DIR --timing file --values source

TimeNet: cases are written in partitions of ``--partition-size`` cases, each its own registry root
(``DIR/part-0000``), because the TimeF writer holds a whole dataset in memory until it commits.
AtriumDB builds live in the separate GPL-3.0 package vitaldb-bench-atriumdb
(``python -m vitaldb_bench_atriumdb.build``), which reuses the helpers here; this module never imports it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import time
from pathlib import Path


def _log(out: Path, entry: dict) -> None:
    with (out / "build_log.jsonl").open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _done(out: Path, kind: str) -> dict:
    log = out / "build_log.jsonl"
    if not log.exists():
        return {}
    done = {}
    for line in log.read_text().splitlines():
        e = json.loads(line)
        if e.get("kind") == kind and e.get("status") == "ok":
            done[e["unit"]] = e
    return done


def cmd_select(a) -> None:
    """Pick a reproducible sample of cases, stratified by file size so short and long cases both appear."""
    files = sorted((Path(a.root) / "vital_files").glob("*.vital"))
    fresh = time.time() - 300
    files = [p for p in files if p.stat().st_mtime < fresh]
    files.sort(key=lambda p: p.stat().st_size)
    rng = random.Random(a.seed)
    strata = [files[i * len(files) // a.strata : (i + 1) * len(files) // a.strata] for i in range(a.strata)]
    per = a.n // a.strata
    chosen = sorted(int(p.stem) for s in strata for p in rng.sample(s, min(per, len(s))))
    Path(a.out).write_text(json.dumps({"cases": chosen, "seed": a.seed, "strata": a.strata, "pool": len(files)}))
    print(f"selected {len(chosen)} of {len(files)} cases -> {a.out}")


def _cases(a) -> list[int]:
    return json.loads(Path(a.cases).read_text())["cases"]


def cmd_timenet(a) -> None:
    os.environ.update(VITALDB_ROOT=a.root, VITALDB_TIMING=a.timing, VITALDB_VALUES=a.values,
                      VITALDB_TOLERANCE_US=str(a.tolerance_us))
    from timenet_connectors.datasets.vitaldb.vitaldb.connector import CONNECTOR  # noqa: PLC0415

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = _cases(a)
    parts = [cases[i : i + a.partition_size] for i in range(0, len(cases), a.partition_size)]
    done = _done(out, "timenet")
    root = Path(a.root) / "vital_files"

    def size_of(part: list[int]) -> int:
        return sum((root / f"{c:04d}.vital").stat().st_size for c in part)

    # seconds per source byte, from earlier partitions: predicts whether the next one fits the budget
    rates = [e["seconds"] / e["bytes"] for e in done.values() if e.get("bytes")]
    t_start = time.time()
    for i, part in enumerate(parts):
        unit = f"part-{i:04d}"
        if unit in done or set(part) & a.exclude_cases:
            continue
        estimate = 1.3 * max(rates) * size_of(part) if rates else 40.0
        if time.time() - t_start > 5 and time.time() - t_start + estimate > a.time_budget:
            continue  # a smaller partition later in the list may still fit
        os.environ["VITALDB_CASES"] = ",".join(map(str, part))
        t0 = time.perf_counter()
        connector = CONNECTOR()
        refs = connector.download(out)
        t1 = time.perf_counter()
        dataset = connector.convert(refs)
        t2 = time.perf_counter()
        _write_timenet(dataset, out / unit, a)
        t3 = time.perf_counter()
        rates.append((t3 - t0) / size_of(part))
        _log(out, {"kind": "timenet", "unit": unit, "status": "ok", "cases": part, "seconds": t3 - t0, "bytes": size_of(part),
                   "convert_s": t2 - t1, "write_s": t3 - t2, "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
    left = sum(1 for i, p in enumerate(parts) if f"part-{i:04d}" not in _done(out, "timenet") and not set(p) & a.exclude_cases)
    print(json.dumps({"partitions_left": left, "of": len(parts), "seconds": round(time.time() - t_start, 1)}))


def _write_timenet(dataset, root: Path, a) -> None:
    """Write one partition with the writer options under test (all default to the TimeF defaults)."""
    from timenet.writer import TimeFWriter  # noqa: PLC0415

    kw: dict = {"values_backend": a.backend}
    if a.encoding:
        kw["value_encoding"] = a.encoding
    if a.level is not None:
        kw["compression_level"] = a.level
    if a.chunk_bytes:
        kw["chunk_max_bytes"] = a.chunk_bytes
    if a.compression:
        kw["compression"] = a.compression
    if dataset.schema is None:
        dataset.derive_schema()
    with TimeFWriter(root, dataset, **kw) as writer:
        writer.write()


def add_case_args(b) -> None:
    """Arguments shared by every per-case build (TimeNet here, AtriumDB in vitaldb-bench-atriumdb)."""
    b.add_argument("--root", required=True)
    b.add_argument("--cases", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--timing", choices=("file", "nominal", "resync"), default="nominal")
    b.add_argument("--values", choices=("source", "integer"), default="source")
    b.add_argument("--tolerance-us", type=float, default=1.0)
    b.add_argument("--time-budget", type=float, default=150)
    b.add_argument("--exclude-cases", type=lambda x: {int(c) for c in x.split(",") if c}, default=set(),
                   help="comma list of cases to leave out (TimeNet: their whole partition)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--strata", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("timenet")
    add_case_args(b)
    b.add_argument("--partition-size", type=int, default=5)
    b.add_argument("--backend", choices=("parquet", "zarr"), default="parquet")
    b.add_argument("--encoding", choices=("auto", "dictionary", "byte_stream_split", "plain"), default=None)
    b.add_argument("--level", type=int, default=None, help="values compression level")
    b.add_argument("--compression", default=None, help="values codec (zarr: Blosc inner codec, e.g. zstd)")
    b.add_argument("--chunk-bytes", type=int, default=None, help="max uncompressed bytes per values chunk")
    a = ap.parse_args()
    {"select": cmd_select, "timenet": cmd_timenet}[a.cmd](a)


if __name__ == "__main__":
    main()
