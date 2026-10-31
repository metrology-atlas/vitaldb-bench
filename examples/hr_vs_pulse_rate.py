# SPDX-License-Identifier: MIT
"""Method comparison: heart rate from the ECG against pulse rate from the plethysmogram, paired at the moment each
describes.

    python examples/hr_vs_pulse_rate.py scan --root R --cases 1,2,3 --out part.jsonl [--time-budget 170]
    python examples/hr_vs_pulse_rate.py agg part*.jsonl --out DIR --labels RESULTS [--pr-delay 2.79]

The Solar8000 reports heart rate (HR, from its ECG) and pulse rate (PLETH_HR, from its oximeter) every 2 s. They are
related measurands, not one: they agree only while every beat makes a pulse at the finger. Each is stamped some
seconds after the interval it describes ends: HR by the recording's own delay (its numeric_delay label, in one of two
groups about 2 s apart), PLETH_HR by about 2.8 s (the atlas's solar8000-pleth-hr form). So the moment matters.

scan, per recording: both numerics held between their records and read on HR's stamps. For each lag L (-20 to +20 s,
0.5 s steps) PLETH_HR is read at HR's stamp + L, and the scan keeps sufficient statistics only:
  changes      over consecutive 10 s blocks, sums for the correlation of HR changes with PLETH_HR changes;
  differences  PLETH_HR minus HR: count, sum, sum of squares, and the count beyond 10 /min;
  by level     for lags -6 to +6 s, the differences binned by the mean of the two (10 /min bins), as in a
               Bland-Altman plot.
It writes one JSON line per recording (aggregates of that recording only; stays on this machine).

agg: the lag at which changes agree best, pooled within recordings (bootstrap over recordings), against the lag the
atlas predicts for each recording (pair the two where their windows end together: L = d_PR - d_HR, with d_HR the
recording's HR delay label and d_PR the PLETH_HR delay), or where their windows' centres meet (adding half the
difference of their window lengths); agreement at lag 0 (as stamped), at the atlas's lag and at
the best lag: the cohort's mean difference and limits of agreement, the SD between recordings of each recording's
mean difference, the SD within a recording, and the share of time the two differ by more than 10 /min (pulse deficit,
artefact or a lost signal). The best lag is also compared between the HR delay groups. Writes DIR/summary.json
(aggregates only) and DIR/hr_vs_pulse_rate.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
LAGS = np.round(np.arange(-20, 20.01, 0.5), 1)
LEVEL_LAGS = [L for L in LAGS if -6 <= L <= 6]
BLOCK = 10.0
LEVELS = np.arange(30, 181, 10)
BIG = 10.0


def _held(t, x, at):
    """The value in force at each time in `at` (the last record at or before it); NaN before the first."""
    i = np.searchsorted(t, at, side="right") - 1
    v = np.full(at.size, np.nan)
    ok = (i >= 0) & (at - t[np.clip(i, 0, None)] <= 6.0)    # a record older than 6 s is a gap, not a value
    v[ok] = x[i[ok]]
    return v


def scan_case(c, root):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names={"Solar8000/HR", "Solar8000/PLETH_HR"})
    hr, pr = v.tracks.get("Solar8000/HR"), v.tracks.get("Solar8000/PLETH_HR")
    if hr is None or pr is None or hr.record_times.size < 300 or pr.record_times.size < 300:
        return {"case": c, "skip": "no HR and PLETH_HR together"}
    th, xh = hr.record_times.astype(float), hr.physical().astype(float)
    tp, xp = pr.record_times.astype(float), pr.physical().astype(float)
    kh = np.isfinite(xh) & (xh >= 30) & (xh <= 200)
    kp = np.isfinite(xp) & (xp >= 30) & (xp <= 200)
    th, xh, tp, xp = th[kh], xh[kh], tp[kp], xp[kp]
    if th.size < 300 or tp.size < 300:
        return {"case": c, "skip": "too few plausible values"}
    t0 = th[0]
    blk = np.floor((th - t0) / BLOCK).astype(int)
    out = {"case": c, "hr_records": int(th.size), "lags": {}}
    for L in LAGS:
        p = _held(tp, xp, th + L)
        ok = np.isfinite(p)
        if ok.sum() < 100:
            continue
        d = p[ok] - xh[ok]
        # changes over consecutive 10 s blocks (both series' block means)
        b = blk[ok]
        nb = np.bincount(b)
        mh = np.bincount(b, xh[ok]) / np.where(nb, nb, 1)
        mp = np.bincount(b, p[ok]) / np.where(nb, nb, 1)
        full = nb >= 3
        k = full[1:] & full[:-1]
        dh, dp = np.diff(mh)[k], np.diff(mp)[k]
        row = {"n": int(d.size), "s": round(float(d.sum()), 3), "ss": round(float((d * d).sum()), 3),
               "big": int((np.abs(d) > BIG).sum()),
               "ch": [int(dh.size), round(float(dh.sum()), 3), round(float(dp.sum()), 3), round(float((dh * dh).sum()), 3),
                      round(float((dp * dp).sum()), 3), round(float((dh * dp).sum()), 3)]}
        if L in LEVEL_LAGS:
            avg = (p[ok] + xh[ok]) / 2
            lv = np.clip(np.searchsorted(LEVELS, avg, side="right") - 1, 0, LEVELS.size - 2)
            row["lv"] = [[int(np.sum(lv == i)), round(float(d[lv == i].sum()), 2), round(float((d[lv == i] ** 2).sum()), 2)]
                         for i in range(LEVELS.size - 1)]
        out["lags"][str(L)] = row
    return out


def scan(a) -> None:
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t0 = time.time()
    with open(out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            if a.time_budget and time.time() - t0 > a.time_budget:
                break
            t1 = time.time()
            try:
                r = scan_case(c, a.root)
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def _corr(ch):
    n, sx, sy, sxx, syy, sxy = ch
    if n < 3:
        return np.nan
    cov, vx, vy = sxy - sx * sy / n, sxx - sx * sx / n, syy - sy * sy / n
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


def _labels(folder):
    """HR delay (s) and group per recording, from numeric_delay_labels.csv and hr_delay_group_labels.csv."""
    dh, grp = {}, {}
    p = Path(folder) / "numeric_delay_labels.csv"
    if p.exists():
        for r in csv.DictReader(p.open()):
            if r["track"] == "Solar8000/HR" and r["quality"] == "good":
                dh[int(r["case"])] = (float(r["delay_from_window_end_s"]), float(r["window_s"]))
    p = Path(folder) / "hr_delay_group_labels.csv"
    if p.exists():
        for r in csv.DictReader(p.open()):
            grp[int(r["case"])] = r["group"]
    return dh, grp


def _nearest(L):
    return str(LAGS[int(np.argmin(np.abs(LAGS - L)))])


def _agreement(rows, lag_of):
    """Cohort mean and limits of agreement, between- and within-recording SD, share beyond 10 /min, at each
    recording's lag (a function of the recording)."""
    n = s = ss = big = 0
    means, within_ss, within_n = [], 0.0, 0
    for r in rows:
        x = r["lags"].get(lag_of(r))
        if not x or x["n"] < 100:
            continue
        n += x["n"]; s += x["s"]; ss += x["ss"]; big += x["big"]
        m = x["s"] / x["n"]
        means.append(m)
        within_ss += x["ss"] - x["n"] * m * m; within_n += x["n"] - 1
    mean = s / n
    sd = np.sqrt(ss / n - mean * mean)
    return {"recordings": len(means), "values": int(n), "mean_diff": round(mean, 2), "sd_diff": round(float(sd), 2),
            "loa": [round(mean - 1.96 * sd, 1), round(mean + 1.96 * sd, 1)],
            "between_recording_sd": round(float(np.std(means)), 2),
            "within_recording_sd": round(float(np.sqrt(within_ss / within_n)), 2),
            "share_beyond_10": round(big / n, 4)}


def agg(a) -> None:
    rows = [json.loads(l) for p in a.parts for l in Path(p).read_text().splitlines() if l.strip()]
    rows = list({r["case"]: r for r in rows}.values())
    ok = [r for r in rows if r.get("lags")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dh, grp = _labels(a.labels) if a.labels else ({}, {})
    keys = [str(L) for L in LAGS]

    def pooled(sub, idx=None):
        cur = []
        for k in keys:
            tot = np.zeros(6)
            for i in (idx if idx is not None else range(len(sub))):
                x = sub[i]["lags"].get(k)
                if x:
                    tot += np.array(x["ch"], float)
            cur.append(_corr(tot))
        return np.array(cur)

    cor = pooled(ok)
    best = int(np.nanargmax(cor))
    rng = np.random.default_rng(0)
    boot = [LAGS[int(np.nanargmax(pooled(ok, rng.integers(0, len(ok), len(ok)))))] for _ in range(100)]
    # the lag the atlas predicts per recording: windows ending together
    atlas_lag = {r["case"]: a.pr_delay - dh[r["case"]][0] for r in ok if r["case"] in dh}
    # or with their windows' centres together, since the two windows differ in length
    centre_lag = {c: L + (a.pr_window - dh[c][1]) / 2 for c, L in atlas_lag.items()}
    labelled = [r for r in ok if r["case"] in atlas_lag]
    by_group = {}
    for g in ("early", "late"):
        sub = [r for r in ok if grp.get(r["case"]) == g]
        if len(sub) >= 50:
            cg = pooled(sub)
            pred = np.median([atlas_lag[r["case"]] for r in sub if r["case"] in atlas_lag])
            by_group[g] = {"recordings": len(sub), "best_lag_s": float(LAGS[int(np.nanargmax(cg))]),
                           "atlas_lag_median_s": round(float(pred), 2)}
    summ = {
        "recordings_read": len(rows), "with_both": len(ok), "with_hr_delay_label": len(labelled),
        "pr_delay_s": a.pr_delay,
        "timing": {"best_lag_s": float(LAGS[best]),
                   "best_lag_boot_p2.5_p97.5": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                   "corr_at_best": round(float(cor[best]), 3), "corr_at_0": round(float(cor[keys.index("0.0")]), 3),
                   "atlas_lag_median_s": round(float(np.median(list(atlas_lag.values()))), 2) if atlas_lag else None,
                   "atlas_centre_lag_median_s": round(float(np.median(list(centre_lag.values()))), 2) if centre_lag else None,
                   "by_hr_group": by_group,
                   "lag_curve": [[float(L), round(float(x), 4)] for L, x in zip(LAGS, cor)]},
        "agreement": {"as_stamped": _agreement(ok, lambda r: "0.0"),
                      "at_atlas_lag": _agreement(labelled, lambda r: _nearest(atlas_lag[r["case"]])),
                      "at_atlas_centre_lag": _agreement(labelled, lambda r: _nearest(centre_lag[r["case"]])),
                      "at_best_lag": _agreement(ok, lambda r: keys[best])},
    }
    # Bland-Altman by level, at the cohort's best lag if it is within -6..6 s
    kb = keys[best] if LAGS[best] in LEVEL_LAGS else "0.0"
    tot = np.zeros((LEVELS.size - 1, 3))
    for r in ok:
        x = r["lags"].get(kb)
        if x and "lv" in x:
            tot += np.array(x["lv"], float)
    summ["agreement"]["by_level"] = [{"from": int(LEVELS[i]), "values": int(n_), "mean_diff": round(s_ / n_, 2),
                                      "sd_diff": round(float(np.sqrt(max(ss_ / n_ - (s_ / n_) ** 2, 0))), 2)}
                                     for i, (n_, s_, ss_) in enumerate(tot) if n_ >= 500]
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "timing"} | {"timing": {k: v for k, v in summ["timing"].items() if k != "lag_curve"}}, indent=1))
    figure(cor, best, summ, atlas_lag, out)


def figure(cor, best, summ, atlas_lag, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    ax[0].plot(LAGS, cor, color="#3f51b5")
    ax[0].axvline(LAGS[best], color="#e65100", lw=0.8, ls=":", label=f"best {LAGS[best]:+.1f} s")
    if summ["timing"]["atlas_lag_median_s"] is not None:
        ax[0].axvline(summ["timing"]["atlas_lag_median_s"], color="0.3", lw=0.8, ls="--",
                      label=f"atlas {summ['timing']['atlas_lag_median_s']:+.1f} s")
    ax[0].axvline(0, color="0.7", lw=0.6)
    ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_xlabel("pulse rate read this long after the HR stamp (s)")
    ax[0].set_ylabel("correlation of 10 s changes")
    ax[0].set_title("A. When the two agree", fontsize=9, loc="left")
    lv = summ["agreement"]["by_level"]
    ax[1].errorbar([x["from"] + 5 for x in lv], [x["mean_diff"] for x in lv], yerr=[1.96 * x["sd_diff"] for x in lv],
                   fmt="o-", color="#e65100", ms=3, lw=1, capsize=2)
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].set_xlabel("mean of pulse rate and HR (/min)"); ax[1].set_ylabel("pulse rate minus HR (/min)")
    ax[1].set_title("B. Mean difference and 95% limits, by rate", fontsize=9, loc="left")
    names = ["as_stamped", "at_atlas_lag", "at_best_lag"]
    lab = ["as stamped", "atlas lag", "best lag"]
    A = summ["agreement"]
    x = np.arange(3)
    ax[2].bar(x - 0.2, [A[k]["within_recording_sd"] for k in names], 0.4, color="#3f51b5", label="within a recording")
    ax[2].bar(x + 0.2, [A[k]["between_recording_sd"] for k in names], 0.4, color="#9fa8da", label="between recordings")
    ax[2].set_xticks(x, lab, fontsize=8); ax[2].set_ylabel("SD of the difference (/min)")
    ax[2].legend(frameon=False, fontsize=7)
    ax[2].set_title("C. Where the difference lies", fontsize=9, loc="left")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Pulse rate against heart rate: {summ['with_both']:,} recordings", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "hr_vs_pulse_rate.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    for k in ("--root", "--cases", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--time-budget", type=float, default=0)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--out", required=True)
    g.add_argument("--labels", help="folder with numeric_delay_labels.csv and hr_delay_group_labels.csv")
    g.add_argument("--pr-delay", type=float, default=2.79, help="PLETH_HR delay from its window's end (s)")
    g.add_argument("--pr-window", type=float, default=10.0, help="PLETH_HR window length (s; the atlas gives 8-12)")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
