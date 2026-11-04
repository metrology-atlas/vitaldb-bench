# SPDX-License-Identifier: MIT
"""Sawtooth rates and periods over the whole release, by SNUADC port (metrology-atlas studies ppg-sawtooth-period,
ppg-sawtooth-slips), from the full-release slips run of tools/sawtooth_char.py.

    python tools/sawtooth_stats.py --slips $W/sawtooth_char/full/slips.jsonl \
        --summary $W/sawtooth_char/full/summary.json --modes $W/results/sync_full/snuadc_mode_label.json \
        --out $W/sawtooth_stats

Per recording with a period (the slips run's constant-period grid fit, ``period.constant`` ok, at least 3 slips):
the tooth length L (time between slips) and the creep rate frame / L (ppm; frame = 1/30 s), signed: positive for the
usual sawtooth (PAT shrinks, steps up), negative for the reversed one (the summary's ``reversed`` rows: their tooth
and rate); how well the slips sit on the grid (the fit's largest violation, s) and how much the period changes over
the recording (the quadratic fit's q, as ppm per hour of the rate where it fitted). By SNUADC port and correction
mode: rate quartiles, the spread between recordings, and the oximeter frame clock's rate (the run's frame_clock ppm).

Writes recordings.csv, summary.json and sawtooth-stats.png.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FRAME = 1 / 30.0


def q(x, ps=(5, 25, 50, 75, 95)) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return {"n": int(x.size), **({f"p{p}": round(float(np.percentile(x, p)), 3) for p in ps} if x.size else {})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slips", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    modes = json.loads(Path(a.modes).read_text())
    rev = {r["case"]: r for r in json.loads(Path(a.summary).read_text())["reversed"]["rows"]}
    rows = []
    for line in open(a.slips):
        r = json.loads(line)
        c = r["case"]
        re_ = r.get("real") or {}
        fc = r.get("frame_clock") if isinstance(r.get("frame_clock"), dict) else {}
        base = {"case": c, "port": r.get("port") or "?", "mode": modes.get(str(c), "?"),
                "minutes": re_.get("minutes"), "frame_ppm": fc.get("ppm"), "slips": len(re_.get("slips") or [])}
        per = (re_.get("period") or {})
        con = per.get("constant") or {}
        if c in rev and rev[c].get("on_grid", 0) >= 3 and rev[c]["up"] <= 1:
            rows.append({**base, "direction": "down", "L_s": rev[c]["tooth_s"], "rate_ppm": -rev[c]["rate_ppm"],
                         "violation_s": None, "q": None})
        elif con.get("ok") and con.get("L") and base["slips"] >= 3:
            L = float(con["L"])
            quad = per.get("quadratic") or {}
            rows.append({**base, "direction": "up", "L_s": L, "rate_ppm": FRAME / L * 1e6,
                         "violation_s": con.get("violation_s"), "q": quad.get("q") if quad.get("ok") else None})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "recordings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    up = [r for r in rows if r["direction"] == "up"]
    down = [r for r in rows if r["direction"] == "down"]
    S = {"slips_run": a.slips, "recordings_with_period": len(rows), "up": len(up), "down": len(down),
         "rate_ppm": {"up": q([r["rate_ppm"] for r in up]), "down": q([-r["rate_ppm"] for r in down])},
         "tooth_min": {"up": q([r["L_s"] / 60 for r in up]), "down": q([r["L_s"] / 60 for r in down])},
         "grid_violation_s": q([r["violation_s"] for r in up]),
         "q_quadratic": q([r["q"] for r in up])}
    grp = defaultdict(list)
    for r in rows:
        grp[(r["port"], r["mode"], r["direction"])].append(r)
    S["by_port"] = {}
    for (port, mode, d), rs in sorted(grp.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < 10:
            continue
        rate = np.array([abs(r["rate_ppm"]) for r in rs])
        S["by_port"][f"{port} {mode} {d}"] = {"recordings": len(rs), "rate_ppm": q(rate, (10, 25, 50, 75, 90)),
                                              "within_5pct_of_median": round(float(np.mean(np.abs(rate / np.median(rate) - 1) < 0.05)), 3),
                                              "frame_ppm": q([r["frame_ppm"] for r in rs], (25, 50, 75))}
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: v for k, v in S.items() if k != "by_port"}, indent=1))
    for k, v in S["by_port"].items():
        print(k, v["recordings"], v["rate_ppm"].get("p10"), v["rate_ppm"].get("p50"), v["rate_ppm"].get("p90"),
              v["within_5pct_of_median"], v["frame_ppm"].get("p50"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    rate = np.array([r["rate_ppm"] for r in rows])
    ax[0, 0].hist(np.clip(rate, -120, 120), bins=np.arange(-120, 121, 1), color="C0")
    ax[0, 0].set_xlabel("sawtooth creep rate (ppm; negative = reversed sawtooth; beyond +/-120 in the end bins)")
    ax[0, 0].set_ylabel("recordings")
    ax[0, 0].set_title(f"Sawtooth rate per recording ({len(up)} usual, {len(down)} reversed)")
    keys = [k for k in S["by_port"]][:16]
    data = [[abs(r["rate_ppm"]) for r in grp[tuple(k.rsplit(" ", 2))]] for k in keys]
    ax[0, 1].boxplot(data, vert=False, showfliers=False)
    for i, d_ in enumerate(data):
        ax[0, 1].scatter(np.clip(d_, 0, 120), np.full(len(d_), i + 1) + np.random.default_rng(i).uniform(-0.25, 0.25, len(d_)),
                         s=3, alpha=0.3, color="C0")
    ax[0, 1].set_yticks(range(1, len(keys) + 1))
    ax[0, 1].set_yticklabels([f"{k} ({len(d_)})" for k, d_ in zip(keys, data)], fontsize=7)
    ax[0, 1].set_xlim(0, 120)
    ax[0, 1].set_xlabel("|rate| (ppm)")
    ax[0, 1].set_title("By SNUADC port, correction mode and direction (10 or more recordings)")
    L = np.array([r["L_s"] / 60 for r in rows])
    ax[1, 0].hist(np.clip(L, 0, 40), bins=np.arange(0, 40.5, 0.25), color="C2")
    ax[1, 0].set_xlabel("tooth length: time between slips (min; over 40 in the last bin)")
    ax[1, 0].set_ylabel("recordings")
    ax[1, 0].set_title("Period between slips")
    fr = np.array([r["frame_ppm"] if r["frame_ppm"] is not None else np.nan for r in rows], float)
    ok = np.isfinite(fr)
    ax[1, 1].scatter(fr[ok], np.clip(rate[ok], -120, 120), s=4, alpha=0.3)
    ax[1, 1].set_xlabel("oximeter frame clock against the SNUADC clock (ppm)")
    ax[1, 1].set_ylabel("sawtooth rate (ppm)")
    ax[1, 1].set_title("Sawtooth rate against the frame clock, per recording")
    fig.tight_layout()
    fig.savefig(out / "sawtooth-stats.png", dpi=90)


if __name__ == "__main__":
    main()
