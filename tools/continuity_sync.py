# SPDX-License-Identifier: MIT
"""Gaps and overlaps between records, as recorded and after synchronisation, per signal. Per case, this machine;
aggregates leave.

    python tools/continuity_sync.py scan --root R --cases 1,2,3 --out part.jsonl
    python tools/continuity_sync.py agg part*.jsonl --out DIR

For each record junction of a track, the junction offset is the next record's first sample time minus the time the
previous record's last sample implies (its start plus its samples over the nominal rate): 0 for continuous data,
positive for a gap, negative for an overlap. It is measured twice:

  as recorded   on the record stamps, in file order
  synchronised  on the device's fitted stamp clock (physiosync.stamp_clock / corrected_times): drift removed, clock
                corrections taken out, the signal the SNUADC loses at each reset (17.7 ms) added back

A clock correction looks like a gap or an overlap on the stamps (a SNUADC reset pulls the stamps back by 0.48 s, a
BIS correction pushes them forward by about 0.17 s) and disappears on the synchronised times; a real gap stays.
Each case writes, per track and per state, the counts of gaps and of overlaps in log-spaced bins of size (1e-4 to
1e4 s), the number of junctions and the track's duration: no times, no samples. agg sums them and draws
N(gap >= x) against x for every signal, before and after.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

TRACKS = {  # track: (label, correction sign for stamp_clock, signal lost at each large event (s))
    "SNUADC/ECG_II": ("SNUADC (ECG II)", -1, 0.0177),
    "BIS/EEG1_WAV": ("BIS EEG", 1, 0.0),
    "Primus/AWP": ("Primus AWP", None, 0.0),
    "Primus/CO2": ("Primus CO2", None, 0.0),
    "CardioQ/FLOW": ("CardioQ flow", None, 0.0),
}
EDGES = np.logspace(-4, 4, 81)


def junctions(starts: np.ndarray, sizes: np.ndarray, fs: float, b: float = 0.0) -> np.ndarray:
    """Next record's start minus the end the previous record implies (s)."""
    return starts[1:] - (starts[:-1] + sizes[:-1] / fs * (1 + b))


def hist(j: np.ndarray) -> dict:
    return {"gap": np.histogram(j[j > 0], EDGES)[0].tolist(), "overlap": np.histogram(-j[j < 0], EDGES)[0].tolist()}


def scan_case(c: int, root: str) -> dict:
    from physiosync import corrected_times, stamp_clock
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names=set(TRACKS))
    out = {"case": c, "tracks": {}}
    for name, (_, sign, loss) in TRACKS.items():
        tr = v.tracks.get(name)
        if tr is None or tr.record_times.size < 3:
            continue
        n = tr.record_times.size
        rt = np.asarray(tr.record_times, float)
        st = np.asarray(tr.record_offsets[:n], np.int64)
        sz = np.asarray(tr.record_sizes[:n], np.int64)
        fs = float(tr.srate)
        before = junctions(rt, sz, fs)
        clk = stamp_clock(rt, st, fs, correction_sign=sign, event_loss_s=loss)
        ct = corrected_times(clk, st)
        after = junctions(ct, sz, fs, clk["rate_ppm"] * 1e-6)
        out["tracks"][name] = {"junctions": int(n - 1), "hours": round(float(rt[-1] - rt[0]) / 3600, 3),
                               "before": hist(before), "after": hist(after),
                               "rate_ppm": round(float(clk["rate_ppm"]), 2), "events": clk["corrections"]["events"]}
    return out


def scan(a) -> None:
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    with open(out, "a", encoding="utf-8") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            t1 = time.time()
            try:
                r = scan_case(c, a.root)
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 2)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def agg(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = [json.loads(x) for p in a.parts for x in open(p, encoding="utf-8") if x.strip()]
    S = {}
    for r in R:
        for name, t in (r.get("tracks") or {}).items():
            s = S.setdefault(name, {"recordings": 0, "junctions": 0, "hours": 0.0,
                                    **{f"{st}_{k}": np.zeros(EDGES.size - 1, np.int64)
                                       for st in ("before", "after") for k in ("gap", "overlap")}})
            s["recordings"] += 1
            s["junctions"] += t["junctions"]
            s["hours"] += t["hours"]
            for st in ("before", "after"):
                for k in ("gap", "overlap"):
                    s[f"{st}_{k}"] += np.asarray(t[st][k], np.int64)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = {"recordings_read": len(R), "errors": sum("error" in r for r in R), "edges_s": EDGES.round(6).tolist(),
            "tracks": {}}
    names = [n for n in TRACKS if n in S]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.3), sharey=True, squeeze=False)
    GREY, BLUE = "#9b9a94", "#2a78d6"
    for ax, name in zip(axes[0], names):
        s = S[name]
        row = {"recordings": s["recordings"], "junctions": s["junctions"], "hours": round(s["hours"], 1)}
        for st in ("before", "after"):
            for k in ("gap", "overlap"):
                h = s[f"{st}_{k}"]
                row[f"{st}_{k}"] = h.tolist()
                ge = np.cumsum(h[::-1])[::-1]            # N(size >= lower edge of each bin)
                for x in (0.01, 0.1, 1.0, 10.0):
                    row[f"{st}_{k}_ge_{x}s"] = int(ge[np.searchsorted(EDGES, x)]) if x < EDGES[-1] else 0
                col = GREY if st == "before" else BLUE
                ls = "-" if k == "gap" else ":"
                m = ge > 0
                ax.step(EDGES[:-1][m], ge[m], where="post", color=col, ls=ls, lw=1.8 if k == "gap" else 1.4,
                        label=f"{k}s, {'as recorded' if st == 'before' else 'synchronised'}")
        summ["tracks"][name] = row
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 1e4)
        ax.set_title(f"{TRACKS[name][0]}\n{s['recordings']:,} recordings", fontsize=9, loc="left")
        ax.set_xlabel("size (s)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(color="#e4e3df", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("number of gaps or overlaps at least this size", fontsize=8)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "continuity_sync.png", dpi=150, facecolor="white")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in summ["tracks"].items()},
                     indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("--root", required=True); s.add_argument("--cases", required=True)
    s.add_argument("--out", required=True)
    g = sp.add_parser("agg"); g.add_argument("parts", nargs="+"); g.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
