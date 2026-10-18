# SPDX-License-Identifier: MIT
"""Aggregation for tools/signal_survey.py: per signal, over cases. Aggregates only (no case values).

Writes DIR/summary.json, DIR/waves.csv, DIR/numerics.csv and figures DIR/survey_*.png.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from signal_survey import FLAT_EDGES, GAP_EDGES, SEG_EDGES


def q(x, ps=(10, 50, 90)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return [round(float(np.percentile(x, p)), 4) for p in ps] if x.size else None


def agg(paths, out: Path) -> None:
    W = defaultdict(lambda: defaultdict(list))
    N = defaultdict(lambda: defaultdict(list))
    Wc = defaultdict(lambda: {"gap": np.zeros(len(GAP_EDGES) - 1, int), "seg": np.zeros(len(SEG_EDGES) - 1, int),
                              "flat": np.zeros(len(FLAT_EDGES) - 1, int), "plateau_s": Counter(), "plateau_cases": Counter(),
                              "fmt": Counter(), "fs": Counter(), "min": Counter(), "max": Counter(), "grid": Counter()})
    Nc = defaultdict(lambda: {"gap": np.zeros(len(GAP_EDGES) - 1, int)})
    cases = 0
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if "wave" not in r:
                continue
            cases += 1
            for n, w in r["wave"].items():
                if not w.get("n"):
                    continue
                d, c = W[n], Wc[n]
                hours = w["n"] / w["fs"] / 3600 if w.get("fs") else None
                d["hours"].append(hours)
                for k in ("eff_bits", "bits_used", "coverage", "segments", "seg_median_s", "seg_longest_s", "gap_longest_s",
                          "at_min", "at_max", "distinct", "nonfinite", "run_max", "run_min"):
                    d[k].append(w.get(k))
                if hours:
                    d["gaps_per_h"].append(w.get("gaps", 0) / hours)
                    fl = w.get("flat") or {}
                    d["flat_s_per_h"].append(fl.get("total_s", 0) / hours)
                    d["flat_any10"].append(float((fl.get("longest_s") or 0) >= 10))
                c["fmt"][(w.get("fmt"), w.get("storage_bits"))] += 1
                c["fs"][w.get("fs")] += 1
                if w.get("min") is not None:
                    c["min"][w["min"] if isinstance(w["min"], int) else round(w["min"], 2)] += 1
                    c["max"][w["max"] if isinstance(w["max"], int) else round(w["max"], 2)] += 1
                if w.get("grid_step"):
                    c["grid"][w["grid_step"]] += 1
                c["gap"] += np.asarray(w.get("gap_hist") or 0, int) if w.get("gap_hist") else 0
                c["seg"] += np.asarray(w.get("seg_hist") or 0, int) if w.get("seg_hist") else 0
                if w.get("flat"):
                    c["flat"] += np.asarray(w["flat"]["hist"], int)
                for lv, k_, s_ in w.get("plateaus") or []:
                    key = lv if isinstance(lv, int) else round(lv, 2)
                    c["plateau_s"][key] += s_
                    c["plateau_cases"][key] += 1
                d["gain"].append(w.get("gain"))
            for n, m in r.get("num", {}).items():
                if m.get("n", 0) < 3:
                    continue
                d = N[n]
                span_h = (m.get("span_s") or 0) / 3600
                d["median_interval_s"].append(m.get("median_interval_s"))
                d["gap_longest_s"].append(m.get("gap_longest_s"))
                d["longest_same_value_s"].append(m.get("longest_same_value_s"))
                d["zeros"].append(m.get("zeros"))
                if span_h > 0:
                    d["gaps_per_h"].append(sum((m.get("gap_hist") or [])[1:]) / span_h)   # gaps of 0.1 s or more beyond 1.5x
                if m.get("gap_hist"):
                    Nc[n]["gap"] += np.asarray(m["gap_hist"], int)
    out.mkdir(parents=True, exist_ok=True)
    waves = {}
    for n, d in sorted(W.items(), key=lambda kv: -len(kv[1]["hours"])):
        c = Wc[n]
        (fmt, bits), _ = c["fmt"].most_common(1)[0]
        tot_h = float(np.nansum([h for h in d["hours"] if h]))
        waves[n] = {
            "cases": len(d["hours"]), "hours": round(tot_h, 1), "fs": c["fs"].most_common(1)[0][0], "format": fmt,
            "storage_bits": bits, "min_code_mode": c["min"].most_common(3), "max_code_mode": c["max"].most_common(3),
            "grid_step_mode": c["grid"].most_common(1)[0][0] if c["grid"] else None,
            "bits_used": q(d["bits_used"]), "eff_bits": q(d["eff_bits"]),
            "share_at_max": q(d["at_max"]), "share_at_min": q(d["at_min"]),
            "cases_at_max_over_0.1pct": round(float(np.mean([(v or 0) > 0.001 for v in d["at_max"]])), 3),
            "cases_at_min_over_0.1pct": round(float(np.mean([(v or 0) > 0.001 for v in d["at_min"]])), 3),
            "plateau_levels": [[lv, round(s / 3600, 2), c["plateau_cases"][lv]] for lv, s in c["plateau_s"].most_common(8)],
            "flat_s_per_h": q(d["flat_s_per_h"]), "cases_with_flat_10s": round(float(np.mean(d["flat_any10"])), 3),
            "flat_runs_hist": dict(zip([f"{a:g}-{b:g}s" for a, b in zip(FLAT_EDGES[:-1], FLAT_EDGES[1:])], c["flat"].tolist())),
            "coverage": q(d["coverage"], (1, 10, 50)), "segments_per_case": q(d["segments"]),
            "gaps_per_h": q(d["gaps_per_h"]), "seg_median_s": q(d["seg_median_s"]), "seg_longest_s": q(d["seg_longest_s"]),
            "gap_longest_s": q(d["gap_longest_s"]),
            "gap_hist": dict(zip([f"{a:g}-{b:g}s" for a, b in zip(GAP_EDGES[:-1], GAP_EDGES[1:])], c["gap"].tolist())),
            "segment_hist": dict(zip([f"{a:g}-{b:g}s" for a, b in zip(SEG_EDGES[:-1], SEG_EDGES[1:])], c["seg"].tolist())),
        }
    nums = {}
    for n, d in sorted(N.items(), key=lambda kv: -len(kv[1]["median_interval_s"])):
        nums[n] = {"cases": len(d["median_interval_s"]), "median_interval_s": q(d["median_interval_s"]),
                   "gaps_per_h": q(d["gaps_per_h"]), "gap_longest_s": q(d["gap_longest_s"]),
                   "longest_same_value_s": q(d["longest_same_value_s"]), "share_zero": q(d["zeros"]),
                   "gap_hist": dict(zip([f"{a:g}-{b:g}s" for a, b in zip(GAP_EDGES[:-1], GAP_EDGES[1:])], Nc[n]["gap"].tolist()))}
    (out / "summary.json").write_text(json.dumps({"cases": cases, "waves": waves, "numerics": nums}, indent=1, default=str))
    import csv
    with open(out / "waves.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["signal", "cases", "hours", "fs", "format", "storage_bits", "min_code", "max_code", "bits_used_p50",
                     "eff_bits_p10", "eff_bits_p50", "eff_bits_p90", "cases_at_max_>0.1%", "cases_at_min_>0.1%",
                     "top_plateau_level", "top_plateau_hours", "flat_s_per_h_p50", "flat_s_per_h_p90", "coverage_p10",
                     "coverage_p50", "gaps_per_h_p50", "gaps_per_h_p90", "seg_median_s_p50", "gap_longest_s_p50"])
        for n, w in waves.items():
            pl = w["plateau_levels"][0] if w["plateau_levels"] else [None, None, None]
            g = lambda k, i: (w[k][i] if w[k] else None)  # noqa: E731
            wr.writerow([n, w["cases"], w["hours"], w["fs"], w["format"], w["storage_bits"], w["min_code_mode"][0][0] if w["min_code_mode"] else None,
                         w["max_code_mode"][0][0] if w["max_code_mode"] else None, g("bits_used", 1), g("eff_bits", 0), g("eff_bits", 1),
                         g("eff_bits", 2), w["cases_at_max_over_0.1pct"], w["cases_at_min_over_0.1pct"], pl[0], pl[1],
                         g("flat_s_per_h", 1), g("flat_s_per_h", 2), g("coverage", 1), g("coverage", 2), g("gaps_per_h", 1),
                         g("gaps_per_h", 2), g("seg_median_s", 1), g("gap_longest_s", 1)])
    with open(out / "numerics.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["signal", "cases", "median_interval_s", "gaps_per_h_p50", "gaps_per_h_p90", "gap_longest_s_p50",
                     "gap_longest_s_p90", "longest_same_value_s_p50", "longest_same_value_s_p90", "share_zero_p50"])
        for n, m in nums.items():
            g = lambda k, i: (m[k][i] if m[k] else None)  # noqa: E731
            wr.writerow([n, m["cases"], g("median_interval_s", 1), g("gaps_per_h", 1), g("gaps_per_h", 2), g("gap_longest_s", 1),
                         g("gap_longest_s", 2), g("longest_same_value_s", 1), g("longest_same_value_s", 2), g("share_zero", 1)])
    print(f"{cases} cases; {len(waves)} waveform signals; {len(nums)} numeric signals")
    figures(out)


MAIN = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/PLETH", "SNUADC/ART", "SNUADC/CVP", "SNUADC/FEM", "Primus/CO2",
        "Primus/AWP", "BIS/EEG1_WAV", "BIS/EEG2_WAV", "CardioQ/FLOW", "CardioQ/ABP"]


def figures(out: Path) -> None:
    """Gap and segment durations per waveform (per 100 recording hours) and amplitude bits per converter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.loads((out / "summary.json").read_text())
    W = s["waves"]
    names = [n for n in MAIN if n in W]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    for a_, key, title in ((ax[0], "gap_hist", "Gaps between continuous segments"),
                           (ax[1], "segment_hist", "Continuous segments")):
        cols = list(W[names[0]][key].keys())
        M = np.array([[W[n][key][c] / W[n]["hours"] * 100 for c in cols] for n in names])
        im = a_.imshow(np.log10(M + 0.1), cmap="Blues", aspect="auto", vmin=-1, vmax=np.log10(M.max() + 0.1))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a_.text(j, i, f"{M[i, j]:.0f}" if M[i, j] >= 1 else (f"{M[i, j]:.1f}" if M[i, j] >= 0.05 else "·"),
                        ha="center", va="center", fontsize=7.5, color="white" if np.log10(M[i, j] + 0.1) > 0.6 * np.log10(M.max() + 0.1) else "black")
        a_.set_xticks(range(len(cols)), [c.replace("inf", "∞").replace("-", "–") for c in cols], rotation=30, fontsize=8)
        a_.set_yticks(range(len(names)), [f"{n} ({W[n]['cases']:,})" for n in names], fontsize=8)
        a_.set_title(f"{title}, per 100 recording hours", fontsize=9)
        a_.set_xlabel("duration")
    fig.tight_layout()
    fig.savefig(out / "survey_gaps_segments.png", dpi=150)
    plt.close(fig)

    fig, a_ = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(names))
    for i, n in enumerate(names):
        w = W[n]
        if w.get("storage_bits"):
            a_.plot([0, w["storage_bits"]], [i, i], color="#e0e0e0", lw=8, solid_capstyle="butt")
        if w.get("bits_used"):
            a_.plot([0, w["bits_used"][1]], [i, i], color="#9ecae1", lw=8, solid_capstyle="butt")
        if w.get("eff_bits"):
            lo, med, hi = w["eff_bits"]
            a_.plot([lo, hi], [i, i], color="#08519c", lw=2.5)
            a_.plot(med, i, "o", color="#08519c", ms=5)
    a_.set_yticks(y, names, fontsize=8)
    a_.invert_yaxis()
    a_.set_xlabel("bits")
    a_.set_title("Storage word (grey), range used (light blue, median case) and effective bits (p10–p90, dot = median)", fontsize=9)
    a_.set_xlim(0, 33)
    a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "survey_bits.png", dpi=150)
    plt.close(fig)


def duration_histograms(gap_dir: Path, out: Path) -> None:
    """Histograms of every gap and every continuous segment per waveform (from tools/gap_durations.py), log x axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G, S, cases = defaultdict(list), defaultdict(list), Counter()
    for p in sorted(Path(gap_dir).glob("part_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            for n, t in r.get("tracks", {}).items():
                G[n].extend(t["gaps"])
                S[n].extend(t["segs"])
                cases[n] += 1
    names = [n for n in MAIN if n in cases and n != "BIS/EEG2_WAV" and n != "CardioQ/ABP"]
    stats = {}
    for what, D, edges, xl in (("gaps", G, np.logspace(-3, 5, 161), "gap duration (s)"),
                               ("segments", S, np.logspace(-2, 5, 141), "continuous segment duration (s)")):
        fig, axs = plt.subplots(len(names) // 2 + len(names) % 2, 2, figsize=(11, 2.1 * (len(names) // 2 + 1)),
                                sharex=True)
        for a_, n in zip(axs.flat, names):
            x = np.asarray(D[n], float)
            x = x[x > 0]
            a_.hist(x, bins=edges, color="#08519c", histtype="stepfilled", alpha=.85)
            a_.set_xscale("log")
            a_.set_yscale("log")
            a_.set_title(f"{n}: {x.size:,} {what} in {cases[n]:,} cases", fontsize=8.5, loc="left")
            a_.set_ylabel("count per bin", fontsize=7.5)
            a_.tick_params(labelsize=7.5)
            for v, lab in ((1, "1 s"), (60, "1 min"), (3600, "1 h")):
                a_.axvline(v, color="#bbb", lw=.6, ls=":")
            a_.spines[["top", "right"]].set_visible(False)
            if x.size:
                stats.setdefault(n, {})[what] = {"n": int(x.size), "p10_p50_p90_s": [round(float(v), 3) for v in np.percentile(x, [10, 50, 90])]}
        for a_ in axs.flat[len(names):]:
            a_.axis("off")
        for a_ in axs[-1]:
            a_.set_xlabel(xl + ", log scale; 20 bins per decade", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"survey_{what}_hist.png", dpi=140)
        plt.close(fig)
    (out / "durations.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1)[:2500])
