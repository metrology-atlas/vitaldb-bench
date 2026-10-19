# SPDX-License-Identifier: MIT
"""Worked example: cuff blood pressure against the arterial line, at the cuff's measurement moment.

    python examples/cuff_vs_arterial.py scan --root R --cases 1,2,3 --out part.jsonl [--time-budget 170]
    python examples/cuff_vs_arterial.py agg part*.jsonl --out DIR

The Solar8000 reports each cuff (NIBP) reading as a held value, repeated every 2 s; a new reading is a change of
value or a value after a gap. Its stamp is when the monitor reported it, near the end of the inflation-deflation
cycle. Where the same patient has an arterial line, the monitor's ART_MBP numeric gives the invasive mean pressure.

Per cuff reading: the arterial mean over a 10 s window ending L seconds before the cuff stamp, for L from -60 to
+60 s (2 s steps). agg finds the lag at which changes between consecutive cuff readings best follow changes in the
arterial mean (removing each patient's level difference), then, at that lag, compares the two:

  population   mean cuff-minus-arterial difference, and how it depends on the pressure (10 mmHg bins of the mean of
               the two, as in a Bland-Altman plot);
  individual   each patient's mean difference (between-patient spread) and the spread within a patient;
  hypotension  readings with arterial MAP < 65: the share the cuff also puts below 65, and per patient, whether any
               reading is below 65 by cuff and by arterial line.

scan writes one JSON line per recording, with that recording's readings (stays on this machine); agg writes
DIR/summary.json (aggregates only) and DIR/cuff_vs_arterial.png.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
LAGS = np.arange(-60, 61, 2)
WIN = 10.0
LIM = 65.0


def scan_case(c, root):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names={"Solar8000/NIBP_MBP", "Solar8000/ART_MBP"})
    nb, ar = v.tracks.get("Solar8000/NIBP_MBP"), v.tracks.get("Solar8000/ART_MBP")
    if nb is None or ar is None or nb.record_times.size < 2 or ar.record_times.size < 100:
        return {"case": c, "skip": "no cuff and arterial numerics together"}
    tn, xn = nb.record_times.astype(float), nb.physical().astype(float)
    ta, xa = ar.record_times.astype(float), ar.physical().astype(float)
    ok = np.isfinite(xa) & (xa > 20) & (xa < 200)
    ta, xa = ta[ok], xa[ok]
    new = np.r_[True, (np.diff(xn) != 0) | (np.diff(tn) > 10)]
    et, ex = tn[new], xn[new]
    k = np.isfinite(ex) & (ex > 20) & (ex < 200) & (et > ta[0] + 70) & (et < ta[-1] + 50)
    et, ex = et[k], ex[k]
    if et.size < 1:
        return {"case": c, "skip": "no cuff reading while the arterial line ran"}
    cs = np.r_[0, np.cumsum(xa)]
    art = []
    for L in LAGS:
        e, s = np.searchsorted(ta, et - L), np.searchsorted(ta, et - L - WIN)
        m = (e - s) >= 3
        a = np.full(et.size, np.nan)
        a[m] = (cs[e[m]] - cs[s[m]]) / (e[m] - s[m])
        art.append(np.round(a, 2))
    # is the arterial line plausible around each reading (no flush or zeroing within 60 s)?
    clean = np.array([np.all((xa[(ta > t - 60) & (ta < t + 10)] > 25)) and np.ptp(xa[(ta > t - 60) & (ta < t + 10)]) < 60
                      if ((ta > t - 60) & (ta < t + 10)).sum() >= 10 else False for t in et])
    return {"case": c, "t": np.round(et - et[0], 1).tolist(), "cuff": ex.tolist(), "clean": clean.tolist(),
            "art": [[None if not np.isfinite(x) else float(x) for x in row] for row in art]}


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


def agg(a) -> None:
    rows = [json.loads(l) for p in a.parts for l in Path(p).read_text().splitlines() if l.strip()]
    rows = list({r["case"]: r for r in rows}.values())   # one line per recording (a resumed run may repeat one)
    ok = [r for r in rows if "cuff" in r]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A = {r["case"]: np.array([[np.nan if x is None else x for x in row] for row in r["art"]], float) for r in ok}
    # 1. timing: changes between consecutive readings (< 10 min apart, clean), cuff against arterial, per lag
    cor, sdd = [], []
    for li in range(LAGS.size):
        dn, da = [], []
        for r in ok:
            t, n, cl, ar = np.array(r["t"]), np.array(r["cuff"]), np.array(r["clean"]), A[r["case"]][li]
            k = (np.diff(t) < 600) & cl[1:] & cl[:-1] & np.isfinite(ar[1:]) & np.isfinite(ar[:-1])
            dn.append(np.diff(n)[k]); da.append(np.diff(ar)[k])
        dn, da = np.concatenate(dn), np.concatenate(da)
        cor.append(float(np.corrcoef(dn, da)[0, 1])); sdd.append(float(np.std(dn - da)))
    cor = np.array(cor)
    best = int(np.argmax(cor))
    # bootstrap the best lag over recordings
    rng = np.random.default_rng(0)
    cases = [r for r in ok]
    boot = []
    pairs_by_case = []
    for r in cases:
        t, n, cl = np.array(r["t"]), np.array(r["cuff"]), np.array(r["clean"])
        k = (np.diff(t) < 600) & cl[1:] & cl[:-1]
        pairs_by_case.append((np.diff(n)[k], np.diff(A[r["case"]], axis=1)[:, k]))
    for _ in range(200):
        idx = rng.integers(0, len(cases), len(cases))
        dn = np.concatenate([pairs_by_case[i][0] for i in idx])
        da = np.concatenate([pairs_by_case[i][1] for i in idx], axis=1)
        cc = [np.corrcoef(dn[np.isfinite(da[li])], da[li][np.isfinite(da[li])])[0, 1] if np.isfinite(da[li]).sum() > 10 else np.nan for li in range(LAGS.size)]
        boot.append(LAGS[int(np.nanargmax(cc))])
    # 2. agreement at the best lag (clean readings)
    case_id, cuff, art = [], [], []
    for r in ok:
        cl, ar = np.array(r["clean"]), A[r["case"]][best]
        k = cl & np.isfinite(ar)
        case_id += [r["case"]] * int(k.sum()); cuff += list(np.array(r["cuff"])[k]); art += list(ar[k])
    case_id, cuff, art = np.array(case_id), np.array(cuff), np.array(art)
    d = cuff - art
    bins = np.arange(40, 131, 10)
    by_level = []
    avg = (cuff + art) / 2      # binned by the mean of the two (Bland-Altman), not by either one: binning by one
    for lo in bins[:-1]:        # noisy measure alone manufactures a level-dependent bias (regression to the mean)
        m = (avg >= lo) & (avg < lo + 10)
        if m.sum() >= 30:
            by_level.append({"art_from": int(lo), "readings": int(m.sum()), "mean_diff": round(float(d[m].mean()), 2),
                             "sd_diff": round(float(d[m].std()), 2)})
    uc = np.unique(case_id)
    per_mean = np.array([d[case_id == c].mean() for c in uc if (case_id == c).sum() >= 3])
    within = np.concatenate([d[case_id == c] - d[case_id == c].mean() for c in uc if (case_id == c).sum() >= 3])
    hyp = art < LIM
    per_case_any = [((art[case_id == c] < LIM).any(), (cuff[case_id == c] < LIM).any()) for c in uc]
    pa = np.array(per_case_any)
    summ = {"recordings_read": len(rows), "with_readings": len(ok), "readings_clean_at_lag": int(d.size),
            "patients_in_agreement": int(uc.size),
            "timing": {"best_lag_s": int(LAGS[best]), "best_lag_boot_p2.5_p97.5": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                       "corr_at_best": round(float(cor[best]), 3), "corr_at_0": round(float(cor[list(LAGS).index(0)]), 3),
                       "corr_at_60_before": round(float(cor[list(LAGS).index(60)]), 3),
                       "consecutive_pairs": int(dn.size), "lag_curve": [[int(L), round(float(x), 3)] for L, x in zip(LAGS, cor)]},
            "population": {"mean_diff": round(float(d.mean()), 2), "sd_diff": round(float(d.std()), 2),
                           "loa": [round(float(d.mean() - 1.96 * d.std()), 1), round(float(d.mean() + 1.96 * d.std()), 1)],
                           "by_arterial_level": by_level},
            "individual": {"patient_mean_diff_p10_50_90": [round(float(np.percentile(per_mean, p)), 2) for p in (10, 50, 90)],
                           "between_patient_sd": round(float(per_mean.std()), 2), "within_patient_sd": round(float(within.std()), 2),
                           "patients_with_3_plus": int(per_mean.size)},
            "hypotension": {"readings_art_below_65": int(hyp.sum()),
                            "cuff_also_below_65": round(float(np.mean(cuff[hyp] < LIM)), 3) if hyp.any() else None,
                            "cuff_below_65_when_art_not": round(float(np.mean(cuff[~hyp] < LIM)), 4),
                            "patients_art_any_below": int(pa[:, 0].sum()), "patients_cuff_any_below": int(pa[:, 1].sum()),
                            "patients_art_below_missed_by_cuff": int((pa[:, 0] & ~pa[:, 1]).sum())}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    figure(cor, best, art, d, per_mean, summ, out)


def figure(cor, best, art, d, per_mean, summ, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    ax[0].plot(LAGS, cor, color="#3f51b5")
    ax[0].axvline(LAGS[best], color="#e65100", lw=0.8, ls=":")
    ax[0].set_xlabel("arterial window ends this long before the cuff stamp (s)")
    ax[0].set_ylabel("correlation of changes")
    ax[0].set_title(f"A. The cuff value belongs to its stamp ({-LAGS[best]:+d} s)", fontsize=9, loc="left")
    ax[1].scatter((art + art + d) / 2, d, s=2, alpha=0.15, color="#3f51b5", lw=0)
    lv = summ["population"]["by_arterial_level"]
    ax[1].plot([x["art_from"] + 5 for x in lv], [x["mean_diff"] for x in lv], "o-", color="#e65100", ms=3)
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].axvline(65, color="0.4", lw=0.8, ls=":")
    ax[1].set_xlim(30, 140); ax[1].set_ylim(-50, 50)
    ax[1].set_xlabel("mean of cuff and arterial MAP (mmHg)"); ax[1].set_ylabel("cuff minus arterial (mmHg)")
    ax[1].set_title("B. Population: almost no bias at any pressure", fontsize=9, loc="left")
    ax[2].hist(np.clip(per_mean, -30, 30), bins=np.arange(-30, 31, 2), color="#3f51b5", alpha=0.8)
    ax[2].axvline(0, color="0.4", lw=0.8)
    ax[2].set_xlabel("a patient's mean cuff minus arterial (mmHg)")
    ax[2].set_yticks([])
    ax[2].set_title(f"C. Individual: between-patient SD {summ['individual']['between_patient_sd']:.1f} mmHg", fontsize=9, loc="left")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Cuff against arterial line: {summ['readings_clean_at_lag']:,} readings, {summ['patients_in_agreement']:,} patients",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "cuff_vs_arterial.png", dpi=140)
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
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
