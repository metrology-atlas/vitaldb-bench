"""Worked example: arterial-to-monitor gradients (PaCO2 - EtCO2, SaO2 - SpO2) as probability distributions.

    python examples/blood_gas_gradients.py run --resolution <vitaldb-resolution checkout> --root R
        --draws-csv lab_draw_labels.csv --out OUT --cases A:B          (repeat over chunks; resumable)
    python examples/blood_gas_gradients.py summarise --out OUT [--figure CASE ...] --root R
    python examples/blood_gas_gradients.py lagscan --root R --draws-csv lab_draw_labels.csv --out OUT --cases A:B
    python examples/blood_gas_gradients.py lagsummary --out OUT      (timing scan and within-patient change figures)

Each blood gas is paired with the monitor at the moment the blood was drawn. The draw time is estimated from the
arterial-line access that precedes the lab stamp (tools/lab_draw_labels.py; only gases with a single candidate
access, `lab_draw:single`; the lab stamp itself is 3-12 min after the draw). For every gas the example draws
--k Monte Carlo samples of

- the draw time: uniform over the access (its start to its end), plus the monitor's stamping delay (a numeric
  describes a window that ended 1-9 s before its stamp; uniform);
- the monitor value: the median of Primus/ETCO2 or Solar8000/PLETH_SPO2 over +-60 s of that time (the headline),
  or the single reading nearest to it (the window comparison); each reading is a whole number, so the true value
  is uniform over +-0.5 of it, and an SpO2 of 100 is censored: the true value is uniform over 99.5-100;
- the lab value: from the vitaldb-resolution library (lab_distributions), with SaO2 capped at 100.

and summarises each gradient by its median and 95% interval. It also gives the plain point gradient at the draw
and at the lab stamp (the usual pairing in the release), and the widths with the draw time fixed at the start of
the access (rounding only), to show how much each source adds.

Writes OUT/parts/*.csv (per gas; stays on this machine), OUT/summary.json (aggregates only), OUT/bg_cohort.png,
OUT/bg_scatter.png
and OUT/bg_examples.png (the gases given with --figure). A case id appears in a figure only because the example
discusses that case.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRACKS = {"etco2": "Primus/ETCO2", "spo2": "Solar8000/PLETH_SPO2"}
LAB = {"etco2": "pco2", "spo2": "sao2"}
WIN = 60.0
STAMP_DELAY = (1.0, 9.0)
NICE = {"etco2": "EtCO₂", "spo2": "SpO₂", "pco2": "PaCO₂", "sao2": "SaO₂"}


def load_tracks(path: Path) -> dict[str, np.ndarray]:
    import vitaldb

    vf = vitaldb.VitalFile(str(path), list(TRACKS.values()))
    out = {}
    for key, nm in TRACKS.items():
        t = vf.trks.get(nm)
        if t is None or not t.recs:
            continue
        a = np.array([(r["dt"], r["val"]) for r in t.recs if r["val"] is not None], float)
        a = a[np.isfinite(a[:, 1]) & (a[:, 1] > 0)]
        if len(a):
            out[key] = a[np.argsort(a[:, 0])]
    return out


def window_median(rec: np.ndarray, t: np.ndarray, half: float = WIN) -> np.ndarray:
    lo = np.searchsorted(rec[:, 0], t - half)
    hi = np.searchsorted(rec[:, 0], t + half, side="right")
    return np.array([np.median(rec[a:b, 1]) if b > a else np.nan for a, b in zip(lo, hi)])


def nearest(rec: np.ndarray, t: np.ndarray, maxgap: float = WIN) -> np.ndarray:
    i = np.clip(np.searchsorted(rec[:, 0], t), 1, len(rec) - 1)
    j = np.where(np.abs(rec[i - 1, 0] - t) <= np.abs(rec[i, 0] - t), i - 1, i)
    v = rec[j, 1].copy()
    v[np.abs(rec[j, 0] - t) > maxgap] = np.nan
    return v


def true_monitor(key: str, reading: np.ndarray, rng) -> np.ndarray:
    """Whole-number reading -> true value: uniform +-0.5; SpO2 capped at 100 (a reading of 100 is 99.5-100)."""
    x = reading + rng.uniform(-0.5, 0.5, reading.shape)
    if key == "spo2":
        top = reading >= 100
        x[top] = rng.uniform(99.5, 100.0, top.sum())
        x = np.minimum(x, 100.0)
    return x


def q(x: np.ndarray) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return (np.nan,) * 3
    lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
    return lo, med, hi


def run(a) -> None:
    sys.path.insert(0, str(a.resolution))
    import vitaldb_resolution as vr

    reg = vr.load()
    rng = np.random.default_rng(0)
    d = pd.read_csv(a.draws_csv)
    d = d[(d.panel == "blood_gas") & (d.label == "lab_draw:single")]
    lab = pd.read_csv(a.root / "lab_data.csv")
    lab = lab[lab.name.isin(LAB.values())].reset_index(drop=True)
    dists = reg.lab_distributions(lab)
    lab["dist"] = dists
    labd = {(c, t, n): dd for c, t, n, dd in zip(lab.caseid, lab.dt, lab.name, lab.dist)}
    labv = {(c, t, n): v for c, t, n, v in zip(lab.caseid, lab.dt, lab.name, lab.result)}
    A, B = map(int, a.cases.split(":"))
    cases = sorted(d.case.unique())[A:B]
    part = a.out / "parts" / f"{A:05d}_{B:05d}.csv"
    if part.exists():
        print(f"{part.name} exists; skipped")
        return
    rows = []
    for c in cases:
        try:
            tr = load_tracks(a.root / "vital_files" / f"{c:04d}.vital")
        except Exception:  # noqa: BLE001
            continue
        for g in d[d.case == c].itertuples():
            k = a.k
            t_draw = rng.uniform(g.start_s, max(g.end_s, g.start_s + 1e-3), k) + rng.uniform(*STAMP_DELAY, k)
            t_fixed = np.full(k, g.start_s + np.mean(STAMP_DELAY))
            for key, rec in tr.items():
                ln = LAB[key]
                dd = labd.get((c, g.lab_rel_s, ln))
                if dd is None or getattr(dd, "status", None) != "ok":
                    continue
                lv = dd.sample(k, rng)
                if ln == "sao2":
                    lv = np.where(lv > 100, rng.uniform(99.5, 100.0, k), lv) if labv[(c, g.lab_rel_s, ln)] >= 100 else np.minimum(lv, 100)
                m_head = window_median(rec, t_draw)
                m_fix = window_median(rec, t_fixed)
                m_near = nearest(rec, t_draw)
                head = lv - true_monitor(key, m_head, rng)
                fix = lv - true_monitor(key, m_fix, rng)
                near = lv - true_monitor(key, m_near, rng)
                p_draw = window_median(rec, np.array([g.start_s]))[0]
                p_stamp = window_median(rec, np.array([g.lab_time_s]))[0]
                lab_rec = labv[(c, g.lab_rel_s, ln)]
                row = {"case": c, "gas": g.draw_id, "kind": key, "lab": lab_rec, "monitor_draw": p_draw,
                       "monitor_stamp": p_stamp, "point_draw": lab_rec - p_draw, "point_stamp": lab_rec - p_stamp,
                       "access_s": g.end_s - g.start_s, "lag_s": g.lag_s}
                for name, x in (("head", head), ("fix", fix), ("near", near)):
                    lo, med, hi = q(x)
                    row.update({f"{name}_lo": lo, f"{name}_med": med, f"{name}_hi": hi})
                rows.append(row)
    part.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    print(f"cases {A}:{B} ({len(cases)}), {len(rows)} gradients -> {part.name}")


def agg(x) -> dict:
    x = pd.Series(x).dropna()
    if not len(x):
        return {}
    return {"n": int(len(x)), "median": round(float(x.median()), 3), "p25": round(float(x.quantile(.25)), 3),
            "p75": round(float(x.quantile(.75)), 3), "p5": round(float(x.quantile(.05)), 3),
            "p95": round(float(x.quantile(.95)), 3)}


def summarise(a) -> None:
    df = pd.concat([pd.read_csv(p) for p in sorted((a.out / "parts").glob("*.csv"))], ignore_index=True)
    s = {"gases": int(df.gas.nunique()), "cases": int(df.case.nunique())}
    for key in TRACKS:
        x = df[df.kind == key].dropna(subset=["head_med"])
        w = x.head_hi - x.head_lo
        s[key] = {
            "n": int(len(x)),
            "point_at_draw": agg(x.point_draw), "point_at_stamp": agg(x.point_stamp),
            "draw_minus_stamp_abs_ge_3": round(float(((x.point_draw - x.point_stamp).abs() >= 3).mean()), 3),
            "median_headline": agg(x.head_med),
            "width95_headline": agg(w), "width95_rounding_only": agg(x.fix_hi - x.fix_lo),
            "width95_nearest_reading": agg(x.near_hi - x.near_lo),
            "nearest_minus_window_median": agg(x.near_med - x.head_med),
            "interval_contains_0": round(float(((x.head_lo <= 0) & (x.head_hi >= 0)).mean()), 3),
            "monitor_at_ceiling": round(float((x.monitor_draw >= 100).mean()), 3) if key == "spo2" else None,
        }
    (a.out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    cohort_figure(df, a.out / "bg_cohort.png")
    scatter_figure(df, a.out / "bg_scatter.png")
    if a.figure:
        examples_figure(df, a.figure, a.root, a.out / "bg_examples.png")


def cohort_figure(df: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    x = df[df.kind == "etco2"]
    b = np.arange(-15, 30.5, 1) - 0.5
    ax[0].hist(x.point_stamp.dropna(), bins=b, histtype="step", color="grey", lw=1.2, label="monitor at the lab stamp")
    ax[0].hist(x.point_draw.dropna(), bins=b, histtype="step", color="#08519c", lw=1.5, label="monitor at the estimated draw")
    ax[0].set(xlabel="PaCO₂ − EtCO₂ (mmHg)", ylabel="blood gases", title="Point gradient, by pairing time")
    ax[0].legend(fontsize=8, frameon=False)
    wb = np.linspace(0, 6, 49)
    for col, lab, c in (("fix", "rounding only", "#9ecae1"), ("head", "+ draw time", "#08519c"),
                        ("near", "nearest reading, + draw time", "#e6550d")):
        ax[1].hist((x[f"{col}_hi"] - x[f"{col}_lo"]).dropna(), bins=wb, histtype="step", lw=1.4, color=c, label=lab)
    ax[1].set(xlabel="95% interval width of PaCO₂ − EtCO₂ (mmHg)", ylabel="blood gases", title="Per-gas uncertainty")
    ax[1].legend(fontsize=8, frameon=False)
    y = df[df.kind == "spo2"].dropna(subset=["head_med"])
    top = y.monitor_draw >= 100
    for m, c, lab in ((~top, "#08519c", f"SpO₂ below 100 ({(~top).sum():,})"), (top, "#d62728", f"SpO₂ = 100, censored ({top.sum():,})")):
        yy = y[m].sort_values("head_med").reset_index(drop=True)
        ax[2].vlines(np.arange(len(yy)) / max(len(yy) - 1, 1), yy.head_lo, yy.head_hi, color=c, lw=.3, alpha=.4)
        ax[2].plot(np.arange(len(yy)) / max(len(yy) - 1, 1), yy.head_med, ".", ms=1.5, color=c, label=lab)
    ax[2].axhline(0, color="k", lw=.6)
    ax[2].set(xlabel="blood gases, ranked by the median gradient (share)", ylabel="SaO₂ − SpO₂ (%)",
              title="SaO₂ − SpO₂: median and 95% interval", ylim=(-10, 8))
    ax[2].legend(fontsize=8, frameon=False, loc="lower right", markerscale=6)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def scatter_figure(df: pd.DataFrame, out: Path) -> None:
    """Lab value against the monitor at the estimated draw, one point per gas. Every value is a whole number, so
    each point is drawn at a random place within its rounding interval (the true value is somewhere in it; values
    of 100 within 99.5-100), which shows the density that a grid of stacked points would hide."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.4))

    def jit(v, cap=None):
        v = np.asarray(v, float)
        x = v + rng.uniform(-0.5, 0.5, v.shape)
        if cap is not None:
            top = v >= cap
            x[top] = rng.uniform(cap - 0.5, cap, top.sum())
        return x

    c = df[df.kind == "etco2"].dropna(subset=["lab", "monitor_draw"])
    ax[0].scatter(jit(c.monitor_draw), jit(c.lab), s=3, alpha=.25, lw=0, color="#08519c",
                  label=f"{len(c):,} blood gases")
    lim = max(80, float(np.ceil(max(c.monitor_draw.max(), c.lab.quantile(.999)) / 10) * 10))
    ax[0].plot([0, lim], [0, lim], color="#555", ls="--", lw=.8, label="equal")
    med = float(np.median(c.lab - c.monitor_draw))
    ax[0].plot([0, lim], [med, lim + med], color="#d62728", lw=1, label=f"equal + {med:.0f} mmHg (median gradient)")
    ax[0].set(xlim=(0, lim), ylim=(0, lim), xlabel="EtCO₂ at the draw (mmHg)", ylabel="PaCO₂ (mmHg)",
              title="Arterial against end-tidal CO₂")
    ax[0].set_aspect("equal")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left", markerscale=4)

    o = df[df.kind == "spo2"].dropna(subset=["lab", "monitor_draw"])
    top = o.monitor_draw >= 100
    for m, col, lab in ((~top, "#08519c", f"SpO₂ below 100 ({(~top).sum():,})"),
                        (top, "#d62728", f"SpO₂ = 100: true value 99.5–100 ({top.sum():,})")):
        oo = o[m]
        ax[1].scatter(jit(oo.monitor_draw, 100), jit(oo.lab, 100), s=3, alpha=.25, lw=0, color=col, label=lab)
    ax[1].plot([80, 100], [80, 100], color="#555", ls="--", lw=.8, label="equal")
    ax[1].set(xlim=(80, 100.5), ylim=(80, 100.5), xlabel="SpO₂ at the draw (%)", ylabel="SaO₂ (%)",
              title="Arterial against pulse oximetry (axes from 80%)")
    ax[1].set_aspect("equal")
    ax[1].legend(frameon=False, fontsize=8, loc="upper left", markerscale=4)
    below = int(((o.monitor_draw < 80) | (o.lab < 80)).sum())
    if below:
        ax[1].text(100.3, 80.3, f"{below} gases below 80% not shown", fontsize=7.5, color="#555", ha="right")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def examples_figure(df: pd.DataFrame, gases: list[str], root: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(gases), figsize=(5.5 * len(gases), 5.6), squeeze=False, sharex="col")
    for j, gid in enumerate(gases):
        c = int(gid.split("-")[0])
        tr = load_tracks(root / "vital_files" / f"{c:04d}.vital")
        g = df[(df.gas == gid) & (df.kind == "etco2")].iloc[0]
        meta = EX_META.get((c, g.gas))
        for i, key in enumerate(("etco2", "spo2")):
            ax = axes[i, j]
            rec = tr.get(key)
            r = df[(df.case == c) & (df.gas == g.gas) & (df.kind == key)]
            if rec is None or r.empty or meta is None:
                continue
            r = r.iloc[0]
            t0 = meta["start_s"]
            m = (rec[:, 0] > t0 - 900) & (rec[:, 0] < t0 + 900)
            ax.plot((rec[m, 0] - t0) / 60, rec[m, 1], color="#4292c6", lw=.8, label=TRACKS[key])
            ax.axvspan(0, (meta["end_s"] - t0) / 60, color="#fdae6b", alpha=.5, label="arterial-line access (draw)")
            ax.axvline((meta["lab_time_s"] - t0) / 60, color="grey", ls="--", lw=.8, label="lab stamp")
            ax.plot([0], [r.lab], "D", color="#d62728", ms=6, label=f"{NICE[LAB[key]]} (drawn here)")
            ax.set_ylabel("mmHg" if key == "etco2" else "%")
            ax.set_title(f"case {c}: {NICE[LAB[key]]} − {NICE[key]} = {r.head_med:.1f} [{r.head_lo:.1f}, {r.head_hi:.1f}]", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.legend(fontsize=7, frameon=False, loc="best")
        axes[1, j].set_xlabel("minutes from the start of the access")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


EX_META: dict = {}


LAGS_MIN = np.arange(-10, 15.5, 1.0)   # minutes from the start of the matched access


def lagscan(a) -> None:
    """Timing scan: the monitor (median over +-60 s) at each offset in LAGS_MIN from the start of the access, per gas.
    If the estimated draw is right, agreement with the lab value should peak near offset 0, not at the lab stamp."""
    d = pd.read_csv(a.draws_csv)
    d = d[(d.panel == "blood_gas") & (d.label == "lab_draw:single")]
    lab = pd.read_csv(a.root / "lab_data.csv")
    lab = lab[lab.name.isin(LAB.values())]
    labv = {(c, t, n): v for c, t, n, v in zip(lab.caseid, lab.dt, lab.name, lab.result)}
    A, B = map(int, a.cases.split(":"))
    cases = sorted(d.case.unique())[A:B]
    part = a.out / "lagparts" / f"{A:05d}_{B:05d}.csv"
    if part.exists():
        print(f"{part.name} exists; skipped")
        return
    rows = []
    for c in cases:
        try:
            tr = load_tracks(a.root / "vital_files" / f"{c:04d}.vital")
        except Exception:  # noqa: BLE001
            continue
        for g in d[d.case == c].itertuples():
            for key, rec in tr.items():
                lv = labv.get((c, g.lab_rel_s, LAB[key]))
                if lv is None:
                    continue
                m = window_median(rec, g.start_s + 60 * LAGS_MIN)
                row = {"case": c, "gas": g.draw_id, "kind": key, "lab": lv, "stamp_min": (g.lab_time_s - g.start_s) / 60}
                row.update({f"m{int(l):+d}": v for l, v in zip(LAGS_MIN, m)})
                row["m_stamp"] = window_median(rec, np.array([g.lab_time_s]))[0]
                rows.append(row)
    part.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    print(f"cases {A}:{B} ({len(cases)}), {len(rows)} rows -> {part.name}")


def _change_r(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Changes between consecutive gases of a case: (change in lab value, change in monitor column)."""
    x = df.sort_values(["case", "gi"])
    g = x.groupby("case")
    return g.lab.diff(), g[col].diff()


def _boot_r(dl, dm, cases, n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"case": cases, "dl": dl, "dm": dm}).dropna()
    gx = [v for _, v in X.groupby("case")]
    out = []
    for _ in range(n):
        s = pd.concat([gx[i] for i in rng.integers(0, len(gx), len(gx))])
        out.append(np.corrcoef(s.dl, s.dm)[0, 1])
    return np.percentile(out, [2.5, 97.5])


def lag_summary(a) -> None:
    """Agreement against offset (from lagparts/) and the within-case change figure (from parts/)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = pd.concat([pd.read_csv(p) for p in sorted((a.out / "lagparts").glob("*.csv"))], ignore_index=True)
    L["gi"] = L.gas.str.split("-").str[1].astype(int)
    res = {}
    # CO2 only: for SpO2 the informative gases are those below the 100% ceiling, and selecting them at any one
    # offset would favour that offset; the oximetry timing question is left open
    fig, axs = plt.subplots(1, 1, figsize=(7.5, 4.4))
    ax = [axs]
    for j, key in enumerate(("etco2",)):
        x = L[L.kind == key]
        rs, lo, hi, pooled = [], [], [], []
        for l in LAGS_MIN:
            col = f"m{int(l):+d}"
            dl, dm = _change_r(x, col)
            ok = dl.notna() & dm.notna()
            rs.append(np.corrcoef(dl[ok], dm[ok])[0, 1])
            b = _boot_r(dl[ok], dm[ok], x.sort_values(["case", "gi"]).case[ok], n=200)
            lo.append(b[0]); hi.append(b[1])
            xx = x.dropna(subset=[col])
            pooled.append(np.corrcoef(xx.lab, xx[col])[0, 1])
        dl, dm = _change_r(x, "m_stamp")
        ok = dl.notna() & dm.notna()
        r_stamp = np.corrcoef(dl[ok], dm[ok])[0, 1]
        ax[j].fill_between(LAGS_MIN, lo, hi, color="#9ecae1", alpha=.6, lw=0)
        ax[j].plot(LAGS_MIN, rs, "o-", color="#08519c", ms=4, label="within-patient: change vs change (95% CI)")
        ax[j].plot(LAGS_MIN, pooled, "s--", color="#636363", ms=3, lw=1, label="all gases pooled")
        st = float(x.stamp_min.median())
        ax[j].axvline(0, color="#e6550d", lw=1)
        ax[j].text(0.2, 0.95, "estimated draw", color="#e6550d", fontsize=8, transform=ax[j].get_xaxis_transform())
        ax[j].axvline(st, color="#555", ls=":", lw=1)
        ax[j].text(st + .2, 0.95, f"lab stamp (median {st:.1f} min)", color="#555", fontsize=8,
                   transform=ax[j].get_xaxis_transform())
        name = "PaCO₂ against EtCO₂: agreement by when the monitor is read"
        ax[j].set(xlabel="monitor read at minutes from the start of the access", ylabel="correlation",
                  title=name, ylim=(0, 1))
        ax[j].title.set_fontsize(9)
        ax[j].legend(frameon=False, fontsize=8, loc="lower left")
        ax[j].spines[["top", "right"]].set_visible(False)
        k = int(np.argmax(rs))
        res[key] = {"gases": int(len(x)), "change_pairs_at_0": int(ok.sum()),
                    "best_offset_min": float(LAGS_MIN[k]), "r_best": round(float(rs[k]), 3),
                    "r_at_draw": round(float(rs[list(LAGS_MIN).index(0)]), 3), "r_at_stamp": round(float(r_stamp), 3),
                    "r_by_offset": {int(l): round(float(r), 3) for l, r in zip(LAGS_MIN, rs)},
                    "pooled_by_offset": {int(l): round(float(r), 3) for l, r in zip(LAGS_MIN, pooled)},
                    "stamp_median_min": round(st, 1)}
    fig.tight_layout()
    fig.savefig(a.out / "bg_lagscan.png", dpi=150)
    plt.close(fig)

    # oximetry: gases chosen by the lab value (SaO2 <= 98, below the analyser's ceiling), never by the monitor, so no
    # offset is favoured; Spearman rank correlation of changes, which tolerates SpO2 = 100 as a tie
    from scipy.stats import spearmanr  # noqa: PLC0415

    def rank_r(dl, dm):
        return spearmanr(dl, dm).correlation

    def boot_rank(dl, dm, cases, n=200, seed=0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({"case": cases, "dl": dl, "dm": dm}).dropna()
        gx = [v for _, v in X.groupby("case")]
        out = []
        for _ in range(n):
            q = pd.concat([gx[i] for i in rng.integers(0, len(gx), len(gx))])
            out.append(rank_r(q.dl, q.dm))
        return np.percentile(out, [2.5, 97.5])

    O = L[L.kind == "spo2"].sort_values(["case", "gi"])
    fig, axs = plt.subplots(1, 1, figsize=(7.5, 4.4))
    for sel, col, lab, ci_on in ((O.lab <= 98, "#08519c", "gases with SaO₂ ≤ 98% (95% CI)", True),
                                 (O.lab > 0, "#636363", "all gases", False)):
        x = O[sel]
        g = x.groupby("case")
        dl = g.lab.diff()
        rs, lo, hi = [], [], []
        for l in LAGS_MIN:
            dm = g[f"m{int(l):+d}"].diff()
            ok = dl.notna() & dm.notna()
            rs.append(rank_r(dl[ok], dm[ok]))
            if ci_on:
                b = boot_rank(dl[ok], dm[ok], x.case[ok])
                lo.append(b[0]); hi.append(b[1])
        dm = g["m_stamp"].diff()
        ok = dl.notna() & dm.notna()
        key = "spo2_sao2_le98" if ci_on else "spo2_all"
        res[key] = {"gases": int(len(x)), "pairs_at_0": int((dl.notna() & g["m+0"].diff().notna()).sum()),
                    "best_offset_min": float(LAGS_MIN[int(np.nanargmax(rs))]),
                    "r_at_draw": round(float(rs[list(LAGS_MIN).index(0)]), 3),
                    "r_at_stamp": round(float(rank_r(dl[ok], dm[ok])), 3),
                    "r_by_offset": {int(l): round(float(r), 3) for l, r in zip(LAGS_MIN, rs)}}
        if ci_on:
            axs.fill_between(LAGS_MIN, lo, hi, color="#9ecae1", alpha=.6, lw=0)
        axs.plot(LAGS_MIN, rs, "o-" if ci_on else "s--", color=col, ms=4 if ci_on else 3, lw=1.5 if ci_on else 1,
                 label=f"{lab}: change vs change")
    st = float(O.stamp_min.median())
    axs.axvline(0, color="#e6550d", lw=1)
    axs.text(0.2, 0.95, "estimated draw", color="#e6550d", fontsize=8, transform=axs.get_xaxis_transform())
    axs.axvline(st, color="#555", ls=":", lw=1)
    axs.text(st + .2, 0.95, f"lab stamp (median {st:.1f} min)", color="#555", fontsize=8, transform=axs.get_xaxis_transform())
    axs.set(xlabel="monitor read at minutes from the start of the access", ylabel="rank correlation (within patients)",
            title="SaO₂ against SpO₂: agreement by when the oximeter is read", ylim=(0, 1))
    axs.title.set_fontsize(9)
    axs.legend(frameon=False, fontsize=8, loc="lower left")
    axs.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out / "bg_lagscan_spo2.png", dpi=150)
    plt.close(fig)

    # within-case change figure: draw vs stamp, CO2
    P = pd.concat([pd.read_csv(p) for p in sorted((a.out / "parts").glob("*.csv"))], ignore_index=True)
    c = P[P.kind == "etco2"].dropna(subset=["lab", "monitor_draw", "monitor_stamp"]).copy()
    c["gi"] = c.gas.str.split("-").str[1].astype(int)
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
    for j, (col, title) in enumerate((("monitor_draw", "Monitor read at the estimated draw"),
                                       ("monitor_stamp", "Monitor read at the lab stamp"))):
        dl, dm = _change_r(c, col)
        ok = dl.notna() & dm.notna()
        r = np.corrcoef(dl[ok], dm[ok])[0, 1]
        ci = _boot_r(dl[ok], dm[ok], c.sort_values(["case", "gi"]).case[ok])
        jx = dm[ok] + rng.uniform(-.5, .5, ok.sum())   # whole numbers: spread within the rounding interval
        jy = dl[ok] + rng.uniform(-.5, .5, ok.sum())
        ax[j].scatter(jx, jy, s=4, alpha=.3, lw=0, color="#08519c")
        ax[j].axline((0, 0), slope=1, color="#555", ls="--", lw=.8, label="equal change")
        ax[j].axhline(0, color="#bbb", lw=.6); ax[j].axvline(0, color="#bbb", lw=.6)
        ax[j].set(title=f"{title}: r = {r:.2f} (95% CI {ci[0]:.2f}–{ci[1]:.2f})", xlim=(-20, 20), ylim=(-20, 20),
                  xlabel="change in EtCO₂ between consecutive gases (mmHg)")
        ax[j].title.set_fontsize(9)
        ax[j].set_aspect("equal")
        ax[j].spines[["top", "right"]].set_visible(False)
        res.setdefault("change_figure", {})[col] = {"pairs": int(ok.sum()), "r": round(float(r), 3),
                                                     "ci95": [round(float(v), 3) for v in ci]}
    ax[0].set_ylabel("change in PaCO₂ between consecutive gases (mmHg)")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.out / "bg_changes.png", dpi=150)
    plt.close(fig)
    (a.out / "lag_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if "by_offset" not in kk} for k, v in res.items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--resolution", required=True, type=Path)
    r.add_argument("--root", required=True, type=Path)
    r.add_argument("--draws-csv", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--cases", required=True, help="A:B, a slice of the sorted case list")
    r.add_argument("--k", type=int, default=400)
    lg = sub.add_parser("lagscan")
    lg.add_argument("--root", required=True, type=Path)
    lg.add_argument("--draws-csv", required=True, type=Path)
    lg.add_argument("--out", required=True, type=Path)
    lg.add_argument("--cases", required=True)
    ls = sub.add_parser("lagsummary")
    ls.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("summarise")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--root", type=Path)
    s.add_argument("--draws-csv", type=Path)
    s.add_argument("--figure", nargs="*", default=[], help="gas ids (draw_id, CASE-N) to plot")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "lagscan":
        lagscan(a)
    elif a.cmd == "lagsummary":
        lag_summary(a)
    else:
        if a.figure and a.draws_csv is not None:
            d = pd.read_csv(a.draws_csv)
            for g in d[d.draw_id.isin(a.figure)].itertuples():
                EX_META[(g.case, g.draw_id)] = {"start_s": g.start_s, "end_s": g.end_s, "lab_time_s": g.lab_time_s}
        summarise(a)


if __name__ == "__main__":
    main()
