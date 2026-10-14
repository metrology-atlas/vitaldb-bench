# SPDX-License-Identifier: MIT
"""Performance and downstream-equivalence benchmarks for the two builds of the same cases.

    python -m vitaldb_bench_atriumdb.perf size     --timenet DIR --atriumdb DIR --root R --out size.json
    python -m vitaldb_bench_atriumdb.perf reads    --db timenet|atriumdb --build DIR --root R --out reads.jsonl [--seed 0] [--n 50]
    python -m vitaldb_bench_atriumdb.perf map      --timenet DIR --atriumdb DIR --root R --out map.json

``reads`` runs in a fresh process per call, so its first pass is as cold as the VM allows (the OS page
cache on the host cannot be dropped from here; "cold" means first access by a new process). It measures,
on a seeded random set of cases and windows:

- open: opening the store;
- full: reading one whole waveform track (SNUADC/ECG_II) of a case;
- window: 10 s random windows of SNUADC/ECG_II, first pass and repeat (warm);
- aligned: one 60 s window of five tracks at once (ECG_II, ART, PLETH, Primus/CO2, Solar8000/HR);
- peak resident memory.

AtriumDB stores are not opened here: this module never imports AtriumDB code. The GPL package
vitaldb-bench-atriumdb registers its store in :data:`STORES` and calls :func:`main`
(``python -m vitaldb_bench_atriumdb.perf ...``); without it, ``--db atriumdb`` and ``map`` stop with a pointer.

``map`` computes mean arterial pressure (the mean of SNUADC/ART physical values) over consecutive 60 s
windows from the reader, from TimeNet and from AtriumDB, and reports how far apart they are.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

WAVE = "SNUADC/ECG_II"
ALIGNED = ("SNUADC/ECG_II", "SNUADC/ART", "SNUADC/PLETH", "Primus/CO2", "Solar8000/HR")


def _du(path: Path) -> int:
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def cmd_size(a) -> None:
    built = sorted(int(e["unit"]) for e in map(json.loads, (Path(a.atriumdb) / "build_log.jsonl").read_text().splitlines())
                   if e.get("kind") == "atriumdb" and e.get("status") == "ok")
    tn_cases = sorted(c for e in map(json.loads, (Path(a.timenet) / "build_log.jsonl").read_text().splitlines())
                      if e.get("status") == "ok" for c in e["cases"])
    src = sum((Path(a.root) / "vital_files" / f"{c:04d}.vital").stat().st_size for c in tn_cases)
    logs = {}
    for name, d in (("timenet", a.timenet), ("atriumdb", a.atriumdb)):
        entries = [e for e in map(json.loads, (Path(d) / "build_log.jsonl").read_text().splitlines()) if "seconds" in e]
        logs[name] = {"build_seconds_total": sum(e["seconds"] for e in entries),
                      "units": sum(e.get("status") == "ok" for e in entries),
                      "peak_rss_mb": max(e["maxrss_mb"] for e in entries)}
    res = {"cases_timenet": len(tn_cases), "cases_atriumdb": len(built), "source_vital_bytes": src,
           "timenet_bytes": _du(Path(a.timenet)), "atriumdb_bytes": _du(Path(a.atriumdb)), **{f"{k}_build": v for k, v in logs.items()}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def _timenet(build: str):
    from vitaldb_bench.stores import TimeNetStore  # noqa: PLC0415

    return TimeNetStore(build)


# store openers by database name; vitaldb-bench-atriumdb adds "atriumdb"
STORES: dict = {"timenet": _timenet}


def _open(db: str, build: str):
    if db not in STORES:
        raise SystemExit(f"no {db} store: run this through python -m vitaldb_bench_atriumdb.perf")
    return STORES[db](build)


def _window(db: str, store, caseid: int, origin_ns: int, dtname: str, start_us: int, end_us: int) -> int:
    if db == "timenet":
        v, _ = store.window(caseid, dtname, start_us, end_us)
    else:
        if store.measure(dtname) is None:
            return 0
        v, _ = store.physical(caseid, dtname, origin_ns, start_us, end_us)
    return len(v)


def cmd_reads(a) -> None:
    from fractions import Fraction  # noqa: PLC0415

    rng = random.Random(a.seed)
    t0 = time.perf_counter()
    store = _open(a.db, a.build)
    t_open = time.perf_counter() - t0
    built = sorted(json.loads(Path(a.cases).read_text())["cases"])
    cases = rng.sample(built, min(a.n_cases, len(built)))
    plan = []
    for c in cases:
        case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={WAVE})
        tr = case.tracks.get(WAVE)
        if tr is not None and tr.record_times.size:
            plan.append((c, tr))
    rows = []
    for c, tr in plan:
        full_case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names=set(ALIGNED) | {"EVENT"})
        origin = full_case.origin
        origin_ns = round(Fraction(origin) * 10**9)
        first_us = int((tr.record_times[0] - origin) * 1e6)
        last_us = int((tr.record_times[-1] - origin) * 1e6)
        def full_read() -> tuple[float, int]:
            t = time.perf_counter()
            if a.db == "timenet":
                n = len(store.track(c, WAVE).raw)
            else:
                n = len(store.physical(c, WAVE, origin_ns, 0, last_us + 10**7)[0])
            return time.perf_counter() - t, n

        if a.order == "full-first":
            t_full, n_full = full_read()
        starts = [rng.randint(first_us, max(first_us, last_us - 10**7)) for _ in range(a.windows)]
        cold, warm = [], []
        for s in starts:
            t = time.perf_counter()
            _window(a.db, store, c, origin_ns, WAVE, s, s + 10**7)
            cold.append(time.perf_counter() - t)
        for s in starts:
            t = time.perf_counter()
            _window(a.db, store, c, origin_ns, WAVE, s, s + 10**7)
            warm.append(time.perf_counter() - t)
        s = rng.randint(first_us, max(first_us, last_us - 6 * 10**7))
        t = time.perf_counter()
        n_aligned = sum(_window(a.db, store, c, origin_ns, d, s, s + 6 * 10**7) for d in ALIGNED)
        t_aligned = time.perf_counter() - t
        if a.order != "full-first":
            t_full, n_full = full_read()
        rows.append({"case": c, "full_s": t_full, "full_n": n_full, "window_first_s": cold, "window_repeat_s": warm,
                     "aligned_s": t_aligned, "aligned_n": n_aligned})
    out = {"db": a.db, "order": a.order, "open_s": t_open, "rows": rows, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    with Path(a.out).open("a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps({"db": a.db, "order": a.order, "open_s": round(t_open, 2), "cases": len(rows),
                      "window_first_of_case_ms_median": float(np.median([r["window_first_s"][0] for r in rows]) * 1e3) if rows else None,
                      "full_s_median": float(np.median([r["full_s"] for r in rows])) if rows else None,
                      "window_first_ms_median": float(np.median([x for r in rows for x in r["window_first_s"]]) * 1e3) if rows else None,
                      "window_repeat_ms_median": float(np.median([x for r in rows for x in r["window_repeat_s"]]) * 1e3) if rows else None,
                      "aligned_ms_median": float(np.median([r["aligned_s"] for r in rows]) * 1e3) if rows else None,
                      "peak_rss_mb": out["peak_rss_mb"]}))


def cmd_map(a) -> None:
    from fractions import Fraction  # noqa: PLC0415

    tn, adb = _open("timenet", a.timenet), _open("atriumdb", a.atriumdb)
    state = Path(a.out).with_suffix(".jsonl")
    done = {json.loads(x)["case"] for x in state.read_text().splitlines()} if state.exists() else set()
    t0 = time.time()
    with state.open("a") as fh:
        for c in tn.caseids:
            if c in done or str(c) in a.exclude.split(",") or time.time() - t0 > a.time_budget:
                continue
            case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital")
            art = case.tracks.get("SNUADC/ART")
            if art is None or not art.record_times.size:
                continue
            origin = case.origin
            origin_ns = round(Fraction(origin) * 10**9)
            # reader: physical values and file times
            rec = np.repeat(np.arange(art.record_times.size), art.record_sizes)
            t_us = (art.record_times[rec] - origin) * 1e6 + (np.arange(rec.size) - art.record_offsets[rec]) * 1e6 / art.srate
            phys = art.physical()
            edges = np.arange(0, t_us.max() + 6e7, 6e7)
            diffs_tn, diffs_adb, exact_tn, exact_adb, n = [], [], 0, 0, 0
            same, diff_between = 0, []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (t_us >= lo) & (t_us < hi)
                if m.sum() < 1000:
                    continue
                ref = float(np.mean(phys[m]))
                v1, _ = tn.window(c, "SNUADC/ART", int(lo), int(hi))
                v2, _ = adb.physical(c, "SNUADC/ART", origin_ns, int(lo), int(hi))
                n += 1
                m1 = float(np.mean(v1)) if len(v1) else float("nan")
                m2 = float(np.mean(v2)) if len(v2) else float("nan")
                same += m1 == m2
                diff_between.append(abs(m1 - m2))
                for v, diffs, key in ((v1, diffs_tn, "tn"), (v2, diffs_adb, "adb")):
                    mv = float(np.mean(v)) if len(v) else float("nan")
                    diffs.append(abs(mv - ref))
                    if mv == ref:
                        if key == "tn":
                            exact_tn += 1
                        else:
                            exact_adb += 1
            fh.write(json.dumps({"case": c, "windows": n, "timenet_exact": exact_tn, "atriumdb_exact": exact_adb,
                                 "timenet_max_abs_mmHg": max(diffs_tn, default=0), "atriumdb_max_abs_mmHg": max(diffs_adb, default=0),
                                 "tn_eq_adb": same, "tn_vs_adb_max_abs_mmHg": max(diff_between, default=0)}) + "\n")
    rows = [json.loads(x) for x in state.read_text().splitlines()]
    res = {"cases": len(rows), "windows": sum(r["windows"] for r in rows),
           "timenet_exact": sum(r["timenet_exact"] for r in rows), "atriumdb_exact": sum(r["atriumdb_exact"] for r in rows),
           "timenet_max_abs_mmHg": max((r["timenet_max_abs_mmHg"] for r in rows), default=0),
           "atriumdb_max_abs_mmHg": max((r["atriumdb_max_abs_mmHg"] for r in rows), default=0),
           "timenet_equals_atriumdb": sum(r.get("tn_eq_adb", 0) for r in rows),
           "tn_vs_adb_max_abs_mmHg": max((r.get("tn_vs_adb_max_abs_mmHg", 0) for r in rows), default=0)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("size")
    for k in ("--timenet", "--atriumdb", "--root", "--out"):
        s.add_argument(k, required=True)
    r = sub.add_parser("reads")
    r.add_argument("--db", choices=("timenet", "atriumdb"), required=True, help="atriumdb needs vitaldb-bench-atriumdb")
    for k in ("--build", "--root", "--out", "--cases"):
        r.add_argument(k, required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--n-cases", type=int, default=10)
    r.add_argument("--windows", type=int, default=10)
    r.add_argument("--order", choices=("windows-first", "full-first"), default="windows-first")
    m = sub.add_parser("map")
    for k in ("--timenet", "--atriumdb", "--root", "--out"):
        m.add_argument(k, required=True)
    m.add_argument("--time-budget", type=float, default=140)
    m.add_argument("--exclude", default="")
    a = ap.parse_args()
    {"size": cmd_size, "reads": cmd_reads, "map": cmd_map}[a.cmd](a)


if __name__ == "__main__":
    main()
