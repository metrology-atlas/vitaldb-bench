# SPDX-License-Identifier: MIT
"""Check every stored track of every built case against the shared reader (the ground truth).

    python -m vitaldb_bench.fidelity --root R --timenet DIR --out fidelity.json [--time-budget 150]
    python -m vitaldb_bench_atriumdb.fidelity --root R --timenet DIR --atriumdb DIR --out fidelity.json   (GPL package)

For each case and track it checks, in each database:

- values: the stored values decode to the file's values bit for bit (counts for integer formats, the
  stored floats otherwise), and physical values (``count * gain + offset``) match the reader's;
- times: the largest distance between where a database puts each value and where the file puts it
  (record time stamp plus sample index over the nominal rate), in nanoseconds;
- order and count: whether the file's order and every value survived (AtriumDB returns values sorted by
  time, so a record that steps back in time, or overlaps, is checked after sorting);
- gaps: how many separate pieces each database stored the track as;
- events: the number of event annotations / labels and their times.

Resumable: per-case results go to ``<out>.jsonl``; the summary aggregates by track and holds no values.
This module never imports AtriumDB code. :func:`run` takes any store with the AtriumDB store's methods;
the GPL package vitaldb-bench-atriumdb opens one and calls it.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.timing import TimingMode, segments
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import TRACK_STRING, TRACK_WAVE, VitalTrack, read_vital

from vitaldb_bench.stores import TimeNetStore, TrackData


def truth_times_ns(track: VitalTrack, origin: float) -> np.ndarray:
    """Per-value time in ns from origin, as the file implies it (float64, ~0.01 ns precision)."""
    rel = (track.record_times - origin) * 1e9
    if track.track_type != TRACK_WAVE:
        return rel
    rec = np.repeat(np.arange(track.record_times.size), track.record_sizes)
    within = np.arange(rec.size) - track.record_offsets[rec]
    return rel[rec] + within * (1e9 / float(track.srate))


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype == object or b.dtype == object:
        return list(a) == list(b)
    return a.dtype == b.dtype and a.tobytes() == b.tobytes()


def compare(truth_vals: np.ndarray, truth_t: np.ndarray, got: TrackData | None, *, sort: bool,
            bin_ns: float = 1e5, intended_t: np.ndarray | None = None) -> dict:
    if got is None:
        return {"present": False}
    tv, tt, it = truth_vals, truth_t, intended_t
    out = {"present": True, "n_truth": int(len(tt)), "n_stored": int(len(got.times_ns)), "pieces": got.pieces}
    if sort and not isinstance(tv, list):
        # A store that returns values sorted by time cannot keep the file's order where records overlap or
        # step back. Compare as sets of (time, value) instead: put both sides in a canonical order, by time
        # in bins of a quarter sample (bin_ns) and then by value, so samples a few microseconds apart pair
        # up regardless of order, while a timing error under half a bin cannot pair the wrong samples.
        def canonical(t: np.ndarray, v: np.ndarray) -> np.ndarray:
            return np.lexsort((np.asarray(v, dtype=np.float64), np.round(t / bin_ns)))

        # pair by where the importer meant to put each value (its timing model), not by the file's time:
        # under nominal timing the two differ by milliseconds, more than a bin
        o = canonical(intended_t if intended_t is not None else tt, tv)
        tv, tt = tv[o], tt[o]
        it = it[o] if it is not None else None
        out["order_kept"] = bool(np.all(np.diff(truth_t) >= 0))
        if len(got.times_ns) == len(tt):
            g = canonical(got.times_ns.astype(np.float64), got.decoded)
            got = TrackData(raw=got.raw[g], decoded=got.decoded[g], times_ns=got.times_ns[g], pieces=got.pieces)
    if len(tt) == len(got.times_ns):
        dec = got.decoded
        if not isinstance(tv, list) and dec.dtype != np.asarray(tv).dtype and dec.dtype.kind in "iu" and np.asarray(tv).dtype.kind in "iu":
            dec = dec.astype(np.asarray(tv).dtype)
        out["values_exact"] = _bits_equal(dec, np.asarray(tv, dtype=object) if isinstance(tv, list) else tv)
        if not out["values_exact"] and not isinstance(tv, list):
            out["values_mismatch"] = int(np.sum(np.asarray(dec, dtype=np.float64) != np.asarray(tv, dtype=np.float64)))
        err = np.abs(got.times_ns.astype(np.float64) - tt)
        out["time_err_ns_max"] = float(err.max())
        out["time_err_ns_p99"] = float(np.quantile(err, 0.99))
        out["time_err_ns_mean"] = float(err.mean())
        if it is not None:
            # against the importer's own timing model: what the store's time grid costs
            model = np.abs(got.times_ns.astype(np.float64) - it)
            out["model_err_ns_max"] = float(model.max())
            out["model_err_ns_p99"] = float(np.quantile(model, 0.99))
    return out


def intended_times_ns(track: VitalTrack, origin: float, timing: str) -> np.ndarray:
    """Where the importer's timing model places each sample (float64 ns from origin)."""
    out = np.empty(int(track.record_offsets[-1]), dtype=np.float64)
    for seg in segments(track, origin, TimingMode(timing)):
        k = np.arange(seg.n_samples, dtype=np.float64)
        out[seg.sample_start : seg.sample_stop] = float(seg.start_s) * 1e9 + k * float(seg.period_s) * 1e9
    return out


def check_case(caseid: int, root: Path, tn: TimeNetStore | None, adb, timing: str = "file") -> dict:
    case = read_vital(root / "vital_files" / f"{caseid:04d}.vital")
    origin = case.origin
    res: dict = {"caseid": caseid, "tracks": {}}
    for name, track in case.tracks.items():
        if not track.record_times.size:
            continue
        t_ns = truth_times_ns(track, origin)
        values = track.values if track.track_type != TRACK_STRING else list(track.values)
        r: dict = {"type": int(track.track_type), "n": int(len(t_ns))}
        if name == "EVENT":
            if tn is not None:
                ann = [a for a in tn.record(caseid).walk_annotations() if a.key == "event"]
                at = np.array([a.span.start_us * 1000 for a in ann], dtype=np.float64)
                r["timenet"] = {"present": True, "n_stored": len(ann), "n_truth": len(t_ns),
                                "labels_equal": sorted(a.value for a in ann) == sorted(values),
                                "time_err_ns_max": float(np.abs(np.sort(at) - np.sort(t_ns)).max()) if len(ann) == len(t_ns) else None}
            if adb is not None:
                labels = [lb for lb in adb.labels(caseid) if lb["label_name"] in set(values)]
                lt = np.array([lb["start_time_n"] for lb in labels], dtype=np.float64) - round(origin * 1e9)
                r["atriumdb"] = {"present": True, "n_stored": len(labels), "n_truth": len(t_ns),
                                 "time_err_ns_max": float(np.abs(np.sort(lt) - np.sort(t_ns)).max()) if len(labels) == len(t_ns) else None}
            res["tracks"][name] = r
            continue
        if tn is not None:
            intended_tn = intended_times_ns(track, origin, timing) if track.track_type == TRACK_WAVE else None
            r["timenet"] = compare(values, t_ns, tn.track(caseid, name), sort=False, intended_t=intended_tn)
        if adb is not None and track.track_type != TRACK_STRING:
            float_dtype = None if track.is_integer_format else track.dtype
            got = adb.track(caseid, name, origin, float_dtype=float_dtype)
            bin_ns, intended = 1e5, None
            if track.track_type == TRACK_WAVE:
                bin_ns = 0.25e9 / float(track.srate)
                intended = intended_times_ns(track, origin, timing)
            r["atriumdb"] = compare(values, t_ns, got, sort=True, bin_ns=bin_ns, intended_t=intended)
            if got is not None and track.is_integer_format and len(got.raw) == len(t_ns):
                _, _, phys = adb.sdk.get_data(measure_id=adb.measure(name), device_id=adb.device(caseid),
                                              start_time_n=round(origin * 1e9) - 10**9, end_time_n=2**62, time_units="ns")
                # as a multiset: the store's order differs from the file's where records overlap
                r["atriumdb"]["physical_exact"] = bool(np.array_equal(np.sort(phys), np.sort(track.physical())))
        if tn is not None and track.is_integer_format:
            sigs = tn.signals(caseid, name)
            r["timenet"]["scale_kept"] = all(s.metadata.get("scale") == track.gain and s.metadata.get("offset") == track.offset for s in sigs)
        res["tracks"][name] = r
    return res


def summarise(rows: list[dict]) -> dict:
    agg: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for row in rows:
        for name, r in row["tracks"].items():
            for db in ("timenet", "atriumdb"):
                c = r.get(db)
                if c is None:
                    continue
                g = agg[name][db]
                g["cases"] += 1
                g["present"] += bool(c.get("present"))
                g["count_match"] += c.get("n_stored") == c.get("n_truth")
                g["values_exact"] += bool(c.get("values_exact"))
                g["values_mismatch_total"] += c.get("values_mismatch", 0)
                g["missing_values_total"] += max(0, (c.get("n_truth") or 0) - (c.get("n_stored") or 0))
                g["physical_exact"] += bool(c.get("physical_exact"))
                g["scale_kept"] += bool(c.get("scale_kept"))
                g["labels_equal"] += bool(c.get("labels_equal"))
                if c.get("time_err_ns_max") is not None:
                    g["time_err_ns_max"] = max(g["time_err_ns_max"], c["time_err_ns_max"])
                    g["time_err_ns_p99_worst_case"] = max(g["time_err_ns_p99_worst_case"], c.get("time_err_ns_p99", 0))
                if c.get("model_err_ns_max") is not None:
                    g["model_err_ns_max"] = max(g["model_err_ns_max"], c["model_err_ns_max"])
                    g["model_err_ns_p99_worst_case"] = max(g["model_err_ns_p99_worst_case"], c["model_err_ns_p99"])
                g["pieces_total"] += c.get("pieces", 0)
    return {name: {db: dict(v) for db, v in dbs.items()} for name, dbs in sorted(agg.items())}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--timenet")
    ap.add_argument("--atriumdb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=150)
    ap.add_argument("--exclude", default="", help="comma list of case ids to skip (e.g. a case whose build was interrupted)")
    ap.add_argument("--timing", choices=("file", "nominal", "resync"), default="nominal", help="the builds' timing mode")
    return ap


def run(a: argparse.Namespace, adb=None) -> None:
    """Check the builds named in ``a``; ``adb`` is an opened AtriumDB store, or None."""
    t0 = time.time()
    tn = TimeNetStore(a.timenet) if a.timenet else None
    state = Path(a.out).with_suffix(".jsonl")
    done = {json.loads(x)["caseid"] for x in state.read_text().splitlines()} if state.exists() else set()
    cases = tn.caseids if tn is not None else sorted(int(e["unit"]) for e in map(json.loads, (Path(a.atriumdb) / "build_log.jsonl").read_text().splitlines()) if e.get("status") == "ok")
    slowest = 0.0
    with state.open("a") as fh:
        for caseid in cases:
            if caseid in done or str(caseid) in a.exclude.split(","):
                continue
            if time.time() - t0 + max(2 * slowest, 5) > a.time_budget:
                break
            t1 = time.time()
            fh.write(json.dumps(check_case(caseid, Path(a.root), tn, adb, a.timing)) + "\n")
            fh.flush()
            slowest = max(slowest, time.time() - t1)
    rows = [json.loads(x) for x in state.read_text().splitlines()]
    Path(a.out).write_text(json.dumps({"cases": len(rows), "of": len(cases), "excluded": a.exclude,
                                       "tracks": summarise(rows)}, indent=1))
    print(json.dumps({"checked": len(rows), "of": len(cases), "seconds": round(time.time() - t0, 1)}))



def main() -> None:
    a = parser().parse_args()
    if a.atriumdb:
        raise SystemExit("AtriumDB checks are in the GPL package: python -m vitaldb_bench_atriumdb.fidelity ...")
    run(a)


if __name__ == "__main__":
    main()
