# SPDX-License-Identifier: MIT
"""Describe the STRUCTURE of a local VitalDB copy (.vital files) without reporting data values.

    python vital_inventory.py scan  VITAL_ROOT --state inv_state.jsonl [--max-files N] [--seed 0]
                                    [--time-budget 150]
    python vital_inventory.py summary --state inv_state.jsonl --out vital_inventory.json
                                    [--event-vocab-min 20]

`scan` is resumable: it appends one JSON line per file to --state and skips files already there, so it
can be re-run until every sampled file is done (useful when each run has a time limit). The state file
holds per-file aggregates keyed by file path and stays on this computer. `summary` merges it into a
file-anonymous report meant for review before anything is shared.

Reported (aggregates only): file counts and sizes; header version, packed flag, timezone field; how case
start times are represented (relative-to-zero, shifted to year 2100, or calendar-like) as counts only;
packet and command types; device and track names, types, storage formats, units, sample rates, gain and
offset; per waveform track the records-per-track and samples-per-record distributions, the timing of each
record against the end of the previous one (contiguous, sub-sample jitter, gap, overlap, backwards), gap
sentinel runs in packed files, NaN/inf counts; per numeric track the spacing of timestamps, duplicates,
backward steps and NaN counts; per string (event) track record counts. Event labels appear in the summary
only if the same label occurs in at least --event-vocab-min files (a standard vocabulary, not free text).
Never reported in the summary: sample or numeric values, timestamps, file names, case identifiers.
Needs Python >= 3.9 and numpy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FMT = {1: ("<f4", 4), 2: ("<f8", 8), 3: ("<i1", 1), 4: ("<u1", 1), 5: ("<i2", 2), 6: ("<u2", 2), 7: ("<i4", 4), 8: ("<u4", 4)}
SENTINEL = {3: -128, 4: 255, 5: -32768, 6: 65535, 7: -2147483648, 8: 4294967295}
TYPE_NAME = {1: "wav", 2: "num", 5: "str"}

_u8 = struct.Struct("<B").unpack_from
_u16 = struct.Struct("<H").unpack_from
_i16 = struct.Struct("<h").unpack_from
_u32 = struct.Struct("<L").unpack_from
_f32 = struct.Struct("<f").unpack_from
_f64 = struct.Struct("<d").unpack_from


def _str(buf, pos):
    n = _u32(buf, pos)[0]
    pos += 4
    return buf[pos : pos + n].decode("utf-8", "replace"), pos + n


def dt_class(dt: float) -> str:
    if dt == 0:
        return "zero"
    if abs(dt) < 1e8:
        return "relative(<1e8 s)"
    year = 1970 + dt / 31556952
    if 2099 <= year < 2101:
        return "shifted~2100"
    if 1990 <= year < 2035:
        return "calendar-like"
    return "other"


def bucket_delta(d: float) -> str:
    """Bucket a record-boundary offset, in samples (next start minus previous end)."""
    if not math.isfinite(d):
        return "nonfinite"
    a = abs(d)
    if a < 1e-3:
        return "contiguous(|d|<0.001)"
    if a < 0.5:
        return "jitter(|d|<0.5)" if d > 0 else "jitter-neg(|d|<0.5)"
    if d < 0:
        return "overlap(d<=-0.5)"
    if d < 1.5:
        return "gap~1"
    if d < 100:
        return "gap(1.5-100)"
    if d < 1e4:
        return "gap(100-1e4)"
    return "gap(>=1e4)"


def bucket_interval(s: float) -> str:
    if not math.isfinite(s):
        return "nonfinite"
    if s < 0:
        return "backward"
    if s == 0:
        return "duplicate"
    for lim, name in ((0.1, "(0,0.1]"), (1.0, "(0.1,1]"), (1.5, "(1,1.5]"), (2.5, "(1.5,2.5]"), (5, "(2.5,5]"), (10, "(5,10]"), (60, "(10,60]"), (600, "(60,600]")):
        if s <= lim:
            return name
    return ">600"


def drift_bucket(us: float) -> str:
    """Bucket the largest gap, in microseconds, between a record's stored start and the start implied
    by the run's first record plus the samples before it (how far a regular axis would move it)."""
    for lim in (0.5, 1, 2, 10, 100, 1000, 10000):
        if us <= lim:
            return f"<={lim}us"
    return ">10ms"


def log2_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    lo = 1 << (n.bit_length() - 1)
    return f"[{lo},{lo * 2})"


def grid_class(dt: float) -> str:
    """Whether a timestamp sits on a microsecond / millisecond grid (meaningful for relative times)."""
    us = dt * 1e6
    if abs(us - round(us)) > 1e-3 * max(1.0, abs(us) * 2**-52 * 1e3):
        return "sub-us"
    ms = dt * 1e3
    if abs(ms - round(ms)) < 1e-6:
        return "ms-grid"
    return "us-grid"


def scan_file(path: Path) -> dict:
    out: dict = {"size": path.stat().st_size, "packets": Counter(), "cmds": Counter(), "devices": [], "tracks": {}, "errors": []}
    with gzip.open(path, "rb") as f:
        data = f.read()
    out["usize"] = len(data)
    if data[:4] != b"VITA":
        out["errors"].append("bad-signature")
        return out
    out["version"] = _u32(data, 4)[0]
    hlen = _u16(data, 8)[0]
    hdr = data[10 : 10 + hlen]
    out["headerlen"] = hlen
    out["dgmt"] = _i16(hdr, 0)[0] if hlen >= 2 else None
    h_start = _f64(hdr, 10)[0] if hlen >= 26 else 0.0
    h_end = _f64(hdr, 18)[0] if hlen >= 26 else 0.0
    out["packed"] = bool(hlen >= 27 and hdr[26] == 1)
    out["header_start_class"] = dt_class(h_start)
    pos = 10 + hlen
    n = len(data)
    devs: dict[int, str] = {}
    trk: dict[int, dict] = {}
    min_dt = math.inf
    max_end = -math.inf
    while pos + 5 <= n:
        ptype = data[pos]
        plen = _u32(data, pos + 1)[0]
        body = pos + 5
        pos = body + plen
        if pos > n:
            out["errors"].append("truncated-packet")
            break
        out["packets"][str(ptype)] += 1
        if ptype == 9:
            did = _u32(data, body)[0]
            dtype_, p = _str(data, body + 4)
            name, p = _str(data, p)
            has_port = p + 4 < body + plen
            name = name or dtype_
            devs[did] = name
            out["devices"].append({"name": name, "type": dtype_, "port_present": has_port})
        elif ptype == 0:
            b = data[body:pos]
            tid = _u16(b, 0)[0]
            ttype, fmt = b[2], b[3]
            name, p = _str(b, 4)
            fields = {"unit": "", "srate": 0.0, "gain": 1.0, "offset": 0.0, "montype": 0, "did": 0}
            try:
                fields["unit"], p = _str(b, p)
                p += 8  # mindisp, maxdisp
                p += 4  # colour
                fields["srate"] = _f32(b, p)[0]
                p += 4
                fields["gain"] = _f64(b, p)[0]
                p += 8
                fields["offset"] = _f64(b, p)[0]
                p += 8
                fields["montype"] = b[p]
                p += 1
                fields["did"] = _u32(b, p)[0]
                p += 4
            except (struct.error, IndexError):
                fields["short_trkinfo"] = True
            dname = devs.get(fields["did"], "") if fields["did"] else ""
            dtname = f"{dname}/{name}" if dname else name
            if dtname in out["tracks"]:
                out["errors"].append("duplicate-track-name")
            t = {"dtname": dtname, "type": TYPE_NAME.get(ttype, str(ttype)), "fmt": fmt, **{k: v for k, v in fields.items() if k != "did"},
                 "device_known": bool(dname), "n_recs": 0, "n_samples": 0, "rec_len": Counter(), "boundary": Counter(), "intervals": Counter(),
                 "rec_backward": 0, "nan": 0, "inf": 0, "sentinel_runs": 0, "sentinel_samples": 0, "grid": Counter(), "str_len": Counter(),
                 "labels": Counter(), "drift_us": Counter(), "overlap": Counter(), "dup_same_value": 0, "dup_diff_value": 0,
                 "_run_start": None, "_run_n": 0, "_run_maxdev": 0.0, "_prev_val": None, "_prev_end": None, "_prev_dt": None, "_first": None, "_last": None}
            trk[tid] = t
            out["tracks"][dtname] = t
        elif ptype == 1:
            if plen < 12:
                continue
            infolen = _u16(data, body)[0]
            dt = _f64(data, body + 2)[0]
            tid = _u16(data, body + 10)[0]
            p = body + 2 + infolen
            t = trk.get(tid)
            if t is None:
                out["errors"].append("rec-for-unknown-track")
                continue
            t["n_recs"] += 1
            if len(t["grid"]) < 4 or t["n_recs"] % 97 == 0:
                t["grid"][grid_class(dt)] += 1
            if t["_first"] is None:
                t["_first"] = dt
            min_dt = min(min_dt, dt)
            if t["_prev_dt"] is not None and dt < t["_prev_dt"]:
                t["rec_backward"] += 1
            if t["type"] == "wav":
                if t["fmt"] not in FMT:
                    continue
                code, w = FMT[t["fmt"]]
                ns = _u32(data, p)[0]
                arr = np.frombuffer(data, dtype=code, count=ns, offset=p + 4)
                t["n_samples"] += ns
                t["rec_len"][log2_bucket(ns)] += 1
                sr = t["srate"]
                if t["_prev_end"] is not None and sr > 0:
                    d = (dt - t["_prev_end"]) * sr
                    t["boundary"][bucket_delta(d)] += 1
                    if d <= -0.5:
                        t["overlap"][log2_bucket(int(-d))] += 1
                    if abs(d) >= 0.5:  # a new continuous run starts here
                        t["drift_us"][drift_bucket(t["_run_maxdev"])] += 1
                        t["_run_start"], t["_run_n"], t["_run_maxdev"] = dt, 0, 0.0
                if t["_run_start"] is None:
                    t["_run_start"], t["_run_n"], t["_run_maxdev"] = dt, 0, 0.0
                elif sr > 0:
                    dev = abs(dt - (t["_run_start"] + t["_run_n"] / sr)) * 1e6
                    t["_run_maxdev"] = max(t["_run_maxdev"], dev)
                t["_run_n"] += ns
                end = dt + ns / sr if sr > 0 else dt
                t["_prev_end"] = end
                max_end = max(max_end, end)
                if t["fmt"] in (1, 2):
                    fin = np.isfinite(arr)
                    t["nan"] += int(np.isnan(arr).sum())
                    t["inf"] += int((~fin).sum()) - int(np.isnan(arr).sum())
                    miss = ~fin
                else:
                    miss = arr == SENTINEL[t["fmt"]]
                if miss.any():
                    t["sentinel_samples"] += int(miss.sum()) if t["fmt"] not in (1, 2) else 0
                    edges = np.diff(np.concatenate(([0], miss.view(np.int8), [0])))
                    t["sentinel_runs"] += int((edges == 1).sum())
            elif t["type"] == "num":
                if t["fmt"] in FMT:
                    code, w = FMT[t["fmt"]]
                    v = np.frombuffer(data, dtype=code, count=1, offset=p)[0]
                    if t["fmt"] in (1, 2) and not math.isfinite(float(v)):
                        t["nan"] += 1
                    if t["_prev_dt"] is not None and dt == t["_prev_dt"]:
                        same = (v == t["_prev_val"]) or (v != v and t["_prev_val"] != t["_prev_val"])
                        t["dup_same_value" if same else "dup_diff_value"] += 1
                    t["_prev_val"] = v
                if t["_prev_dt"] is not None:
                    t["intervals"][bucket_interval(dt - t["_prev_dt"])] += 1
                max_end = max(max_end, dt)
            elif t["type"] == "str":
                try:
                    s, _ = _str(data, p + 4)
                except struct.error:
                    s = ""
                t["str_len"][log2_bucket(len(s))] += 1
                t["labels"][s] += 1
                max_end = max(max_end, dt)
            t["_prev_dt"] = dt
            t["_last"] = dt
        elif ptype == 6:
            out["cmds"][str(data[body])] += 1
    out["first_rec_class"] = dt_class(min_dt) if math.isfinite(min_dt) else "none"
    if math.isfinite(min_dt) and h_start:
        out["header_vs_first_rec_s"] = log2_bucket(int(abs(h_start - min_dt)))
    out["duration_h"] = round((max_end - min_dt) / 3600, 1) if math.isfinite(min_dt) and math.isfinite(max_end) else None
    out["header_end_matches"] = bool(h_end and math.isfinite(max_end) and abs(h_end - max_end) < 1.0)
    # track start offsets relative to the file's first record, bucketed (device start-up latency)
    for t in out["tracks"].values():
        if t["type"] == "wav" and t["_run_start"] is not None:
            t["drift_us"][drift_bucket(t["_run_maxdev"])] += 1
        if t["_first"] is not None and math.isfinite(min_dt):
            t["start_after_first_s"] = log2_bucket(int(t["_first"] - min_dt))
        for k in ("_prev_end", "_prev_dt", "_first", "_last", "_run_start", "_run_n", "_run_maxdev", "_prev_val"):
            t.pop(k, None)
    return out


def cmd_scan(a) -> None:
    root = Path(a.root)
    # skip files modified in the last few minutes: they may still be being unzipped or copied
    fresh = time.time() - a.min_age
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".vital" and p.stat().st_mtime < fresh)
    other = Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file() and p.suffix.lower() != ".vital")
    random.Random(a.seed).shuffle(files)
    if a.max_files:
        files = files[: a.max_files]
    state = Path(a.state)
    done = set()
    if state.exists():
        for line in state.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["path"])
    meta_path = state.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({"vital_files_total": len(files) if not a.max_files else None,
                                     "sampled": len(files), "other_extensions": other,
                                     "csv_schemas": table_schemas(root)}, default=list))
    t0 = time.time()
    todo = [p for p in files if str(p) not in done]
    with state.open("a") as fh:
        for i, p in enumerate(todo):
            if time.time() - t0 > a.time_budget:
                break
            try:
                rec = scan_file(p)
            except Exception as e:  # noqa: BLE001 - record and carry on
                rec = {"errors": [f"exception:{type(e).__name__}"]}
            rec["path"] = str(p)
            fh.write(json.dumps(rec, default=dict) + "\n")
            fh.flush()
    left = len(todo) - (i + 1 if todo else 0)
    print(json.dumps({"done": len(done) + (i + 1 if todo else 0), "remaining": max(left, 0), "seconds": round(time.time() - t0, 1)}))


def table_schemas(root: Path) -> list:
    """Column names and inferred types of any CSV tables (e.g. the clinical-information table)."""
    out = []
    for p in sorted(root.rglob("*.csv"))[:20]:
        with p.open(newline="", encoding="utf-8", errors="replace") as fh:
            import csv

            r = csv.reader(fh)
            head = next(r, [])
            types = defaultdict(Counter)
            for j, row in enumerate(r):
                if j >= 200:
                    break
                for c, v in zip(head, row):
                    types[c][("empty" if v == "" else "num" if re.fullmatch(r"-?\d+(\.\d+)?([eE]-?\d+)?", v) else "text")] += 1
        out.append({"name_shape": re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", p.name)), "columns": [[c, dict(types[c])] for c in head]})
    return out


def cmd_summary(a) -> None:
    recs = [json.loads(line) for line in Path(a.state).read_text().splitlines() if line.strip()]
    meta_path = Path(a.state).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    S: dict = {"files_scanned": len(recs), "meta": meta}
    c = defaultdict(Counter)
    sizes, usizes, durs = [], [], []
    tracks: dict[str, dict] = {}
    labels_files = Counter()
    for r in recs:
        for e in r.get("errors", []):
            c["errors"][e] += 1
        if "size" not in r:
            continue
        sizes.append(r["size"])
        usizes.append(r.get("usize", 0))
        if r.get("duration_h") is not None:
            durs.append(r["duration_h"])
        for k in ("version", "headerlen", "packed", "dgmt", "header_start_class", "first_rec_class", "header_vs_first_rec_s", "header_end_matches"):
            c[k][str(r.get(k))] += 1
        for k, v in r["packets"].items():
            c["packet_types_total"][k] += v
        for k, v in r["cmds"].items():
            c["cmd_codes_total"][k] += v
        c["devices_per_file"][len(r["devices"])] += 1
        for d in r["devices"]:
            c["device_names"][f"{d['name']} (type {d['type']})"] += 1
            c["device_port_present"][str(d["port_present"])] += 1
        c["tracks_per_file"][log2_bucket(len(r["tracks"]))] += 1
        file_labels = set()
        for name, t in r["tracks"].items():
            g = tracks.setdefault(name, {"files": 0, "variants": Counter(), "n_recs": Counter(), "rec_len": Counter(), "boundary": Counter(),
                                         "intervals": Counter(), "rec_backward_files": 0, "nan_files": 0, "inf_files": 0, "sentinel_runs": Counter(),
                                         "grid": Counter(), "str_len": Counter(), "start_after_first_s": Counter(), "samples_total": 0,
                                         "drift_us": Counter(), "overlap": Counter(), "dup_same_value": 0, "dup_diff_value": 0})
            g["files"] += 1
            g["variants"][f"type={t['type']} fmt={t['fmt']} unit={t['unit']!r} srate={t['srate']} gain={t['gain']!r} offset={t['offset']!r} montype={t['montype']}"] += 1
            g["n_recs"][log2_bucket(t["n_recs"])] += 1
            g["samples_total"] += t["n_samples"]
            for k in ("rec_len", "boundary", "intervals", "grid", "str_len", "drift_us", "overlap"):
                g[k].update(t[k])
            g["rec_backward_files"] += t["rec_backward"] > 0
            g["dup_same_value"] += t.get("dup_same_value", 0)
            g["dup_diff_value"] += t.get("dup_diff_value", 0)
            g["nan_files"] += t["nan"] > 0
            g["inf_files"] += t["inf"] > 0
            g["sentinel_runs"][log2_bucket(t["sentinel_runs"])] += 1
            g["start_after_first_s"][t.get("start_after_first_s", "none")] += 1
            file_labels.update(t["labels"].keys())
        labels_files.update(file_labels)

    def q(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return {p: xs[min(len(xs) - 1, int(p / 100 * len(xs)))] for p in (0, 10, 50, 90, 100)}

    S["compressed_bytes_quantiles"] = q(sizes)
    S["uncompressed_bytes_quantiles"] = q(usizes)
    S["compressed_bytes_total_scanned"] = sum(sizes)
    S["duration_hours_quantiles"] = q(durs)
    S.update({k: dict(v) for k, v in c.items()})
    S["tracks"] = {k: {kk: (dict(vv) if isinstance(vv, Counter) else vv) for kk, vv in v.items()}
                   for k, v in sorted(tracks.items(), key=lambda kv: -kv[1]["files"])}
    S["event_vocabulary"] = {k: v for k, v in labels_files.most_common() if v >= a.event_vocab_min}
    S["event_labels_below_threshold"] = sum(1 for v in labels_files.values() if v < a.event_vocab_min)
    Path(a.out).write_text(json.dumps(S, indent=1, default=str))
    print(f"wrote {a.out}: {len(recs)} files, {len(tracks)} distinct tracks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("root")
    s.add_argument("--state", required=True)
    s.add_argument("--max-files", type=int, default=0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--time-budget", type=float, default=150)
    s.add_argument("--min-age", type=float, default=300, help="skip files modified less than this many seconds ago")
    m = sub.add_parser("summary")
    m.add_argument("--state", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--event-vocab-min", type=int, default=20)
    a = ap.parse_args()
    {"scan": cmd_scan, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
