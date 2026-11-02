# SPDX-License-Identifier: MIT
"""Aggregate tools/pat.py's per-case output into one summary and the raw PAT distribution figure. Aggregates only.

    python tools/pat.py --root $R --cases $W/pat/cases_pat300.json --out $W/pat_summary/pat.jsonl --time-budget 3600
    python tools/pat_summary.py --scan $W/pat_summary/pat.jsonl --out $W/pat_summary

Writes OUT/pat_summary.json:
- ``cases``: cases with at least one beat; ``beats``: beats paired to an R peak (steepest upstroke found);
  ``beats_by_fiducial``: beats inside the -1 to +1 s histogram range for each fiducial;
- ``edges`` and ``hist``: the summed per-beat histograms (2 ms bins) for foot, slope, peak and the pairing-free
  check (corr);
- ``case_median``: each case's median PAT per fiducial; ``rr_median_s``: each case's median R-R;
- ``medians``: the per-beat medians (from the histograms, bin centres), the median of the per-case foot medians,
  the median per-case upstroke minus foot, and the median R-R.

and OUT/pat-distribution.png: per-beat PAT to the foot, upstroke and peak; the per-case foot medians; and the
pairing-free check, each with the typical finger PAT (0.2-0.35 s) shaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIDS = ("foot", "slope", "peak")
TYPICAL = (0.2, 0.35)   # typical finger PAT in adults, from the literature graded in the atlas study ppg-offset-reference


def hist_median(h: np.ndarray, start: float, step: float) -> float:
    c = np.cumsum(h)
    return float(start + step * (np.searchsorted(c, c[-1] / 2) + 0.5))


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("beats")]
    nb = len(ok[0]["foot"]["hist"])
    edges = {"start": -1.0, "step": 0.002}
    hist = {k: np.sum([r[k]["hist"] for r in ok], axis=0) if ok else np.zeros(nb, int) for k in FIDS + ("corr",)}
    cm = {k: [r[k]["q"][2] for r in ok if r[k]["q"]] for k in FIDS}
    up = [r["slope"]["q"][2] - r["foot"]["q"][2] for r in ok if r["slope"]["q"] and r["foot"]["q"]]
    rr = [r["rr_median_s"] for r in ok if r.get("rr_median_s")]
    return {
        "cases_listed": len(rows), "cases": len(ok),
        "skipped": sum(1 for r in rows if "skip" in r), "errors": sum(1 for r in rows if "error" in r),
        "beats": int(sum(r["beats"] for r in ok)),
        "beats_by_fiducial": {k: int(hist[k].sum()) for k in FIDS},
        "edges": edges, "hist": {k: v.tolist() for k, v in hist.items()},
        "case_median": cm, "rr_median_s": rr,
        "medians": {**{f"per_beat_{k}_s": hist_median(hist[k], edges["start"], edges["step"]) for k in FIDS},
                    "per_case_foot_s": float(np.median(cm["foot"])) if cm["foot"] else None,
                    "per_case_upstroke_minus_foot_s": float(np.median(up)) if up else None,
                    "rr_s": float(np.median(rr)) if rr else None},
    }


def figure(s: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = s["edges"]; h = {k: np.array(v) for k, v in s["hist"].items()}
    x = e["start"] + e["step"] * (np.arange(h["foot"].size) + 0.5)
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    for a in ax:
        a.axvspan(*TYPICAL, color="#f3e7b3", alpha=0.6, lw=0)
        a.axvline(0, color="k", lw=0.5)
    lab = {"foot": ("pulse foot (tangent)", "#2f6db5", 1.8), "slope": ("steepest upstroke", "#9aa0aa", 1.0),
           "peak": ("systolic peak", "#cfd2d6", 1.0)}
    for k in ("peak", "slope", "foot"):
        ax[0].plot(x, h[k] / 1000, color=lab[k][1], lw=lab[k][2], label=lab[k][0])
    ax[0].set_xlim(-0.2, 1.0); ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel("beats per 2 ms bin (thousands)")
    ax[0].set_title(f"Pulse arrival time per beat: R-peak to PPG, {s['beats']:,} beats from {s['cases']} cases", fontsize=10)
    h_, l_ = ax[0].get_legend_handles_labels(); ax[0].legend(h_[::-1], l_[::-1], fontsize=8, loc="upper left")
    cm = np.array(s["case_median"]["foot"])
    ax[1].hist(cm, bins=np.arange(-0.2, 1.0001, 0.01), color="#2f6db5")
    med = float(np.median(cm)); ax[1].axvline(med, color="k", lw=1.2)
    ax[1].text(med + 0.005, ax[1].get_ylim()[1] * 0.8, f"median {med:.3f} s", fontsize=8)
    ax[1].set_xlim(-0.2, 1.0); ax[1].set_ylabel("cases")
    ax[1].set_title("Median pulse-foot PAT per case (10 ms bins)", fontsize=10)
    ax[2].plot(x, h["corr"] / 1000, color="#2f6db5", lw=1)
    ax[2].set_xlim(-1, 1); ax[2].set_ylim(bottom=0)
    ax[2].set_ylabel("pulse feet per 2 ms bin (thousands)"); ax[2].set_xlabel("time after R-peak, s")
    rr = s["medians"]["rr_s"]
    ax[2].set_title("Pairing-free check: every PPG pulse foot relative to every R-peak within ±1 s"
                    + (f" (median R-R {rr:.2f} s)" if rr else ""), fontsize=10)
    for a in ax:
        a.text(np.mean(TYPICAL), a.get_ylim()[1] * 0.85, "typical finger PAT\n(~0.2–0.35 s)", ha="center",
               fontsize=8, color="#7a6a1e")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True, help="tools/pat.py output (one JSON line per case)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.scan).read_text().splitlines() if x.strip()]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s = summarise(rows)
    (out / "pat_summary.json").write_text(json.dumps(s, indent=1) + "\n")
    figure(s, out / "pat-distribution.png")
    print(json.dumps({k: s[k] for k in ("cases_listed", "cases", "skipped", "errors", "beats", "beats_by_fiducial",
                                        "medians")}, indent=1))


if __name__ == "__main__":
    main()
