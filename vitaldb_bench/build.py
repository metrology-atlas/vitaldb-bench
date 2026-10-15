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

from vitaldb_bench.label_families import FAMILIES
from vitaldb_bench.timenet_labels import label_dataset, label_version, load_labels


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
    """Pick a reproducible sample of cases, stratified by file size (the ``select`` command).

    The .vital files are sorted by size and cut into ``a.strata`` equal strata; ``a.n // a.strata`` cases are
    drawn from each with ``random.Random(a.seed)``, so short and long cases both appear and the same seed
    gives the same sample. Files modified in the last five minutes are skipped (a copy still in progress).

    Args:
        a: Parsed arguments: ``root`` (the VitalDB release), ``out`` (JSON to write), ``n``, ``strata``
            and ``seed``.
    """
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
    """Build TimeNet partitions until the time budget runs out (the ``timenet`` command).

    Cases are written ``a.partition_size`` at a time, each partition its own TimeNet registry root. A
    partition already logged as done is skipped, so repeated calls finish the build. The time a partition
    will take is predicted from earlier ones (seconds per source byte); a partition that would overrun the
    budget is left for the next call.

    Args:
        a: Parsed arguments of the ``timenet`` subcommand (see :func:`add_case_args` and :func:`main`).
    """
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

    labels, families = None, list(FAMILIES)
    if a.results:
        labels = load_labels(a.results, families)
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
        if labels is not None:          # derived labels written with the cases (--results)
            label_dataset(dataset, labels[0], families, labels[1])
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


def cmd_timenet_labels(a) -> None:
    """Write the derived labels into an existing TimeNet build (``timenet-labels``).

    Each partition's newest version is copied on write to ``--version`` with the labels as record
    annotations (:mod:`vitaldb_bench.timenet_labels`). With ``--work``, each partition is labelled in that
    local folder and the new version copied back, for builds on a mounted folder where the writer's DuckDB
    control file misbehaves. Resumable: each labelled partition is logged.

    Args:
        a: Parsed arguments of the ``timenet-labels`` subcommand.
    """
    import shutil  # noqa: PLC0415

    from vitaldb_bench.stores import newest_version  # noqa: PLC0415

    out = Path(a.out)
    fams = list(FAMILIES) if a.families == "all" else [f for f in a.families.split(",") if f]
    done = {u for u, e in _done(out, "timenet-labels").items() if e.get("version") == a.version}
    parts = sorted({e["unit"] for e in _done(out, "timenet").values()})
    labels = load_labels(a.results, fams)
    t_start, slowest = time.time(), 0.0
    for unit in parts:
        if unit in done:
            continue
        if time.time() - t_start + max(2 * slowest, 10) > a.time_budget:
            break
        t0 = time.time()
        base = newest_version(out / unit / "vitaldb" / "vitaldb")
        if base.name == a.version:      # an earlier run wrote it but was stopped before logging: redo from 1.0.0
            base = base.parent / "1.0.0"
        if a.work:
            work = Path(a.work) / unit
            shutil.rmtree(work, ignore_errors=True)
            shutil.copytree(base, work / "vitaldb" / "vitaldb" / base.name)
            stats = label_version(work / "vitaldb" / "vitaldb" / base.name, work, a.results, fams, a.version, labels)
            dest = out / unit / "vitaldb" / "vitaldb" / a.version
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(Path(stats["version_dir"]), dest)
            shutil.rmtree(work, ignore_errors=True)
        else:
            stats = label_version(base, out / unit, a.results, fams, a.version, labels)
        slowest = max(slowest, time.time() - t0)
        _log(out, {"kind": "timenet-labels", "unit": unit, "status": "ok", "version": a.version,
                   "base": base.name, "seconds": time.time() - t0, "records": stats["records"],
                   "labels": stats["labels"], "skipped": stats["skipped"]})
    left = sum(1 for u in parts if u not in {x for x, e in _done(out, "timenet-labels").items()
                                              if e.get("version") == a.version})
    print(json.dumps({"partitions_left": left, "of": len(parts), "seconds": round(time.time() - t_start, 1)}))


def cmd_label_tables(a) -> None:
    """Write the label tables beside a TimeNet build (``label-tables``).

    ``<build>/label_params/<family>.parquet`` for the derived label families (the AtriumDB layout), and
    ``clock_model.parquet`` with ``clock_errors.parquet`` from the clock fits (:mod:`vitaldb_bench.clock_model`).
    Nothing in the build itself changes; rewriting the tables is how a model update reaches a build.

    Args:
        a: Parsed arguments of the ``label-tables`` subcommand.
    """
    from vitaldb_bench import clock_model  # noqa: PLC0415
    from vitaldb_bench.stores import TimeNetStore  # noqa: PLC0415

    out, done = Path(a.out), {}
    if a.results:
        fams = None if a.families == "all" else [f for f in a.families.split(",") if f]
        done["families"] = clock_model.write_family_tables(out, a.results, fams)
    if a.clock_fits:
        cases = set(TimeNetStore(out).caseids)
        fits = {c: f for c, f in clock_model.load_fits(*a.clock_fits).items() if c in cases}
        pop = clock_model.load_fits(*a.population) if a.population else {}
        pop.update(clock_model.load_fits(*a.clock_fits))
        done["clock"] = clock_model.write_tables(out, fits, pop)
    print(json.dumps(done))


def add_case_args(b) -> None:
    """Add the arguments every per-case build shares (TimeNet here, AtriumDB in vitaldb-bench-atriumdb).

    Args:
        b: The ``argparse`` subparser to add ``--root``, ``--cases``, ``--out``, ``--timing``, ``--values``,
            ``--tolerance-us``, ``--time-budget`` and ``--exclude-cases`` to.
    """
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
    """Command-line entry point: ``python -m vitaldb_bench.build select|timenet ...``."""
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
    b.add_argument("--results", default=None, help="folder with the label CSVs: write the derived labels too")
    lb = sub.add_parser("timenet-labels", help="derived labels into an existing TimeNet build (copy-on-write)")
    lb.add_argument("--out", required=True, help="the TimeNet build folder")
    lb.add_argument("--results", required=True, help="folder with the label CSVs (and label_versions.json)")
    lb.add_argument("--families", default="all", help="comma-separated label families, or 'all'")
    lb.add_argument("--version", default="1.1.0", help="version number of the labelled copy of each partition")
    lb.add_argument("--work", default=None, help="local folder to label each partition in (for mounted builds)")
    lb.add_argument("--time-budget", type=float, default=150)
    tb = sub.add_parser("label-tables", help="label and clock-model tables beside a TimeNet build")
    tb.add_argument("--out", required=True, help="the TimeNet build folder")
    tb.add_argument("--results", default=None, help="folder with the label CSVs: write the family tables")
    tb.add_argument("--families", default="all", help="comma-separated label families, or 'all'")
    tb.add_argument("--clock-fits", nargs="*", default=[], help="tools/clock_model_fit.py output (JSONL)")
    tb.add_argument("--population", nargs="*", default=[],
                    help="more fits (or sa_corrected output) to estimate the constants and errors from")
    a = ap.parse_args()
    {"select": cmd_select, "timenet": cmd_timenet, "timenet-labels": cmd_timenet_labels,
     "label-tables": cmd_label_tables}[a.cmd](a)


if __name__ == "__main__":
    main()
