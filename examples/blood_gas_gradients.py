"""Worked example: arterial-to-monitor gradients (PaCO2 - EtCO2, SaO2 - SpO2) as probability distributions.

    python examples/blood_gas_gradients.py run --resolution <vitaldb-resolution checkout> --root R
        --draws-csv lab_draw_labels.csv --out OUT --cases A:B          (repeat over chunks; resumable)
    python examples/blood_gas_gradients.py summarise --out OUT [--figure CASE ...] --root R

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

Writes OUT/parts/*.csv (per gas; stays on this machine), OUT/summary.json (aggregates only), OUT/bg_cohort.png
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
    ax[0].set(xlabel="PaCO2 − EtCO2 (mmHg)", ylabel="blood gases", title="Point gradient, by pairing time")
    ax[0].legend(fontsize=8, frameon=False)
    wb = np.linspace(0, 6, 49)
    for col, lab, c in (("fix", "rounding only", "#9ecae1"), ("head", "+ draw time", "#08519c"),
                        ("near", "nearest reading, + draw time", "#e6550d")):
        ax[1].hist((x[f"{col}_hi"] - x[f"{col}_lo"]).dropna(), bins=wb, histtype="step", lw=1.4, color=c, label=lab)
    ax[1].set(xlabel="95% interval width of PaCO2 − EtCO2 (mmHg)", ylabel="blood gases", title="Per-gas uncertainty")
    ax[1].legend(fontsize=8, frameon=False)
    y = df[df.kind == "spo2"].dropna(subset=["head_med"])
    top = y.monitor_draw >= 100
    for m, c, lab in ((~top, "#08519c", f"SpO2 below 100 ({(~top).sum():,})"), (top, "#d62728", f"SpO2 = 100, censored ({top.sum():,})")):
        yy = y[m].sort_values("head_med").reset_index(drop=True)
        ax[2].vlines(np.arange(len(yy)) / max(len(yy) - 1, 1), yy.head_lo, yy.head_hi, color=c, lw=.3, alpha=.4)
        ax[2].plot(np.arange(len(yy)) / max(len(yy) - 1, 1), yy.head_med, ".", ms=1.5, color=c, label=lab)
    ax[2].axhline(0, color="k", lw=.6)
    ax[2].set(xlabel="blood gases, ranked by the median gradient (share)", ylabel="SaO2 − SpO2 (%)",
              title="SaO2 − SpO2: median and 95% interval", ylim=(-10, 8))
    ax[2].legend(fontsize=8, frameon=False, loc="lower right", markerscale=6)
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
            ax.plot([0], [r.lab], "D", color="#d62728", ms=6, label=f"{LAB[key]} (drawn here)")
            ax.set_ylabel("mmHg" if key == "etco2" else "%")
            ax.set_title(f"case {c}: {LAB[key]} − {key} = {r.head_med:.1f} [{r.head_lo:.1f}, {r.head_hi:.1f}]", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.legend(fontsize=7, frameon=False, loc="best")
        axes[1, j].set_xlabel("minutes from the start of the access")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


EX_META: dict = {}


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
    s = sub.add_parser("summarise")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--root", type=Path)
    s.add_argument("--draws-csv", type=Path)
    s.add_argument("--figure", nargs="*", default=[], help="gas ids (draw_id, CASE-N) to plot")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        if a.figure and a.draws_csv is not None:
            d = pd.read_csv(a.draws_csv)
            for g in d[d.draw_id.isin(a.figure)].itertuples():
                EX_META[(g.case, g.draw_id)] = {"start_s": g.start_s, "end_s": g.end_s, "lab_time_s": g.lab_time_s}
        summarise(a)


if __name__ == "__main__":
    main()
