"""SNUADC record stamps in every file: the stamp arithmetic slope and the reset clusters. Per case, this machine only; aggregates and figures leave.

Two censuses from the stamps of one SNUADC waveform per file (every SNUADC channel shares its stamps):

1. **Stamp arithmetic.** Within each stretch between corrections, the stamp minus the time given by the sample
   count, e = (stamp - start) x srate - samples so far, is fitted for slope in ppm. The six-file study found
   -0.954 ppm in every stretch. Also counts where placing records by ``ceil(stamp x srate)`` (as the vitaldb
   package does) would overwrite a sample.
2. **Reset clusters.** Every place a record is stamped before the previous one ends is classified: a cluster the
   TimeNet reset-order rule reorders (standard, or with a repeated stamp it restamps), another backward step of
   about the reset size, either one clean step (a reset already written in order) or several that the rule
   leaves in file order (unmatched; its stamp pattern is kept), or a small step (frequent-correction mode).
   Files whose records step back far more often than resets do (one test file: 2,093 "clusters" in 8 h, every
   record overlapping its neighbour by 80-120 ms) are counted apart as anomalous: the rule matches there by
   coincidence and should not be applied.

Commands (resumable; skips finished cases; time-boxed):

    python tools/snuadc_stamp_census.py scan --root $R --out $W/results/clock/census/stamps.jsonl --time-budget 150
    python tools/snuadc_stamp_census.py summary --scan $W/results/clock/census/stamps.jsonl --out $W/results/clock/census

``summary`` writes ``stamp_census_summary.json`` and three figures (aggregates only):
``stamp_slope_hist.png``, ``reset_cluster_classes.png``, ``reset_unmatched_patterns.png``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.resets import _clusters
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

TRACKS = ["SNUADC/ECG_II", "SNUADC/ART", "SNUADC/PLETH", "SNUADC/CVP", "SNUADC/ECG_V5", "SNUADC/FEM"]
RESET_LO, RESET_HI = -0.6, -0.3   # summed backward step of a reset cluster, s


def one(path: Path) -> dict:
    v = read_vital(path, track_names=set(TRACKS))
    name = next((n for n in TRACKS if n in v.tracks and v.tracks[n].record_times.size > 10), None)
    out = {"case": int(path.stem), "track": name}
    if name is None:
        return out
    tr = v.tracks[name]
    t = np.asarray(tr.record_times, float)
    sz = np.asarray(tr.record_sizes[: t.size], float)
    fs = float(tr.srate)
    half = 0.5 / fs
    dev = np.diff(t) - sz[:-1] / fs                 # gap (+) or overlap (-) between consecutive records, s
    breaks = np.flatnonzero(np.abs(dev) >= half)     # corrections, gaps and resets
    out.update(srate=fs, records=int(t.size), hours=round(float(t[-1] - t[0]) / 3600, 3),
               overlap_share=round(float(np.mean(dev < -half)), 5),   # share of record junctions that step back
               median_dev_ms=round(float(np.median(dev)) * 1000, 3))
    # 1. slope of stamp minus count within each clean stretch of at least 10 minutes
    edges = np.r_[0, breaks + 1, t.size]
    slopes, weights = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 3000 / max(sz[a], 1) or t[b - 1] - t[a] < 600:
            continue
        n = np.r_[0, np.cumsum(sz[a:b - 1])]
        e = (t[a:b] - t[a]) * fs - n                # samples
        x = t[a:b] - t[a]
        slope = np.polyfit(x, e, 1)[0] / fs * 1e6   # ppm
        slopes.append(round(float(slope), 4))
        weights.append(round(float(x[-1]) / 3600, 3))
    out.update(stretch_ppm=slopes, stretch_hours=weights)
    # ceil placement (vitaldb package): a record whose first index falls before the previous record's end overwrites
    idx = np.ceil((t - t[0]) * fs - 1e-9)
    ends = idx[:-1] + sz[:-1]
    clean = np.abs(dev) < half
    out["ceil_overwrites"] = int(np.sum((idx[1:] < ends) & clean))
    # 2. backward steps and reset clusters
    clusters = _clusters(t, sz.astype(int), fs)
    restamped = 0
    for c in clusters:
        restamped += int(any(abs(t[k] - g) >= half for k, g in zip(c.order, c.times)))
    covered = set()
    for c in clusters:
        covered.update(range(c.first - 1, c.first + 5))
    back = [int(i) for i in np.flatnonzero(dev < -half)]
    unmatched, in_order, small, patterns = 0, 0, 0, Counter()
    i = 0
    while i < len(back):
        k = back[i]
        if k in covered:
            i += 1
            continue
        group = [k]
        while i + 1 < len(back) and back[i + 1] - group[-1] <= 2:
            i += 1
            group.append(back[i])
        total = float(np.sum(dev[group[0]:group[-1] + 1]))
        if RESET_LO < total < RESET_HI and len(group) == 1:
            in_order += 1   # one clean step back: a reset already written in sampling order
        elif RESET_LO < total < RESET_HI:
            unmatched += 1
            patterns[" ".join(str(int(round(x * 100))) for x in dev[group[0]:group[0] + 6])] += 1
        else:
            small += 1
        i += 1
    out.update(clusters=len(clusters), restamped=restamped, unmatched=unmatched, in_order=in_order,
               small_back_steps=small,
               unmatched_patterns=dict(patterns), standard=len(clusters) - restamped)
    return out


def scan(a) -> None:
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["case"] for line in out.open() if line.strip()}
    files = sorted(Path(a.root, "vital_files").glob("*.vital"))
    t0, n = time.time(), 0
    with out.open("a") as f:
        for p in files:
            if int(p.stem) in done:
                continue
            if time.time() - t0 > a.time_budget:
                break
            try:
                row = one(p)
            except Exception as ex:  # noqa: BLE001 - recorded, not fatal
                row = {"case": int(p.stem), "error": f"{type(ex).__name__}: {ex}"[:200]}
            f.write(json.dumps(row) + "\n")
            n += 1
    left = len(files) - len(done) - n
    print(f"scanned {n} files this call; {len(done) + n} of {len(files)} done; {left} left")


def summary(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in open(a.scan) if line.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    have = [r for r in rows if r.get("track")]
    ppm = np.array([p for r in have for p in r.get("stretch_ppm", [])])
    hrs = np.array([h for r in have for h in r.get("stretch_hours", [])])
    per_file = np.array([np.median(r["stretch_ppm"]) for r in have if r.get("stretch_ppm")])
    pats = Counter()
    for r in have:
        pats.update(r.get("unmatched_patterns", {}))
    s = {
        "files": len(rows), "files_with_snuadc": len(have), "errors": sum(1 for r in rows if "error" in r),
        "stretches": int(ppm.size), "stretch_hours": round(float(hrs.sum()), 1),
        "stretch_ppm": {"median": float(np.median(ppm)) if ppm.size else None,
                        "p1": float(np.percentile(ppm, 1)) if ppm.size else None,
                        "p99": float(np.percentile(ppm, 99)) if ppm.size else None,
                        "share_within_0.05_of_-0.954": float(np.mean(np.abs(ppm + 0.954) < 0.05)) if ppm.size else None},
        "files_median_ppm_outside_-0.954_pm_0.05": int(np.sum(np.abs(per_file + 0.954) >= 0.05)),
        "ceil_overwrites": {"total": int(sum(r.get("ceil_overwrites", 0) for r in have)),
                            "per_hour": round(sum(r.get("ceil_overwrites", 0) for r in have)
                                              / max(sum(r.get("hours", 0) for r in have), 1e-9), 4)},
        "anomalous_files": {
            "rule": "more than 2 reset clusters per recording hour, or more than 1% of record junctions stepping back",
            "count": sum(1 for r in have if r.get("clusters", 0) > 2 * max(r.get("hours", 0), 0.1)
                         or r.get("overlap_share", 0) > 0.01),
            "clusters_in_them": sum(r.get("clusters", 0) for r in have
                                    if r.get("clusters", 0) > 2 * max(r.get("hours", 0), 0.1)
                                    or r.get("overlap_share", 0) > 0.01)},
        "resets": {"files_with_any": sum(1 for r in have if r.get("clusters") or r.get("unmatched")),
                   "reordered_standard": sum(r.get("standard", 0) for r in have),
                   "reordered_restamped": sum(r.get("restamped", 0) for r in have),
                   "already_in_order": sum(r.get("in_order", 0) for r in have),
                   "unmatched": sum(r.get("unmatched", 0) for r in have),
                   "small_back_steps": sum(r.get("small_back_steps", 0) for r in have),
                   "unmatched_patterns_top": pats.most_common(15)},
    }
    (out / "stamp_census_summary.json").write_text(json.dumps(s, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    if ppm.size:
        lo, hi = np.percentile(ppm, [0.5, 99.5])
        ax.hist(np.clip(ppm, lo, hi), bins=120, weights=hrs, color="#3f51b5")
    ax.axvline(-0.954, color="#e65100", ls="--", lw=1, label="six-file value, -0.954 ppm")
    ax.set_xlabel("stamp minus sample count, slope within a stretch (ppm)")
    ax.set_ylabel("recording hours")
    ax.set_title(f"SNUADC stamp arithmetic: {ppm.size:,} stretches in {len(have):,} files")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "stamp_slope_hist.png", dpi=150)
    plt.close(fig)

    r = s["resets"]
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ["reordered\n(standard)", "reordered\n(stamp repeated)", "already\nin order", "unmatched:\nleft as written"]
    vals = [r["reordered_standard"], r["reordered_restamped"], r["already_in_order"], r["unmatched"]]
    ax.bar(labels, vals, color=["#3f51b5", "#7986cb", "#9e9e9e", "#e65100"])
    for k, v in enumerate(vals):
        ax.text(k, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("resets")
    ax.set_title(f"SNUADC resets in {r['files_with_any']:,} files")
    fig.tight_layout()
    fig.savefig(out / "reset_cluster_classes.png", dpi=150)
    plt.close(fig)

    top = r["unmatched_patterns_top"][:10]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if top:
        ax.barh([p for p, _ in top][::-1], [c for _, c in top][::-1], color="#e65100")
    ax.set_xlabel("resets")
    ax.set_title("Unmatched resets: gaps between the next six records (x10 ms)", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "reset_unmatched_patterns.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: s[k] for k in ("files", "stretches", "stretch_ppm", "resets")}, indent=1)[:1500])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--time-budget", type=float, default=150)
    p = sub.add_parser("summary")
    p.add_argument("--scan", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else summary(a)


if __name__ == "__main__":
    main()
