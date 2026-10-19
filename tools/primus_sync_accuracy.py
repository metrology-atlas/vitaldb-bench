# SPDX-License-Identifier: MIT
"""How well breath matching places the Primus on the SNUADC timeline, over the whole release.

    python tools/primus_sync_accuracy.py scan --root R --cases 1,2,3 --modes M --ports P --step-rates S --out part.jsonl
    python tools/primus_sync_accuracy.py agg part*.jsonl [--out DIR]

For every recording with Primus airway pressure (AWP) and SNUADC central venous (CVP) or arterial pressure (ART), the
lag of the AWP breath cycle against each reference is measured in 10-minute windows (examples/ppv_by_breath.cvp_lag:
breath-triggered averages, split-half agreement within 0.1 s), with the SNUADC on its stamp model, so on the
recorder PC's timeline. Two arms: CVP, and the arterial mean pressure (ART low-passed over 2 s), whose breath
modulation gives a second, physiologically different reference. The Primus's AWP pauses (stamp gaps over 5 s)
split each recording into stretches.

Accuracy, three ways (agg):
  1. Leave one out. Fit one slope and one offset per stretch with each window left out, predict it; the error
     distribution is the accuracy of a Primus time placed by the fit (per arm, over all windows).
  2. Two arms. In recordings with both, the slopes from CVP and from ART, and the CVP arm's prediction against the
     ART arm's windows after removing one constant (the physiology between the two references).
  3. Injection. In one recording in ten the AWP times are also shifted by +0.25 s and warped by +100 ppm before
     the CVP (else ART) arm is measured again; the recovered change in offset and slope is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppv_by_breath as P  # noqa: E402
from pat import sample_times  # noqa: E402

INJ_SHIFT_S, INJ_PPM = 0.25, 100.0


def lowpass_mean(x, fs, win_s=2.0):
    k = max(1, int(win_s * fs))
    c = np.cumsum(np.r_[0.0, np.nan_to_num(x)])
    y = (c[k:] - c[:-k]) / k
    return np.r_[np.full(k // 2, y[0]), y, np.full(x.size - y.size - k // 2, y[-1])]


def scan_case(c, a, modes, ports, step_rates):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={"Primus/AWP", "SNUADC/CVP", "SNUADC/ART"})
    aw = v.tracks.get("Primus/AWP")
    arms = [x for x in a.arms.split(",") if x]
    refs = {k: v.tracks.get(f"SNUADC/{k.split('_')[0].upper()}") for k in arms}
    refs = {k: tr for k, tr in refs.items() if tr is not None and tr.record_times.size}
    if aw is None or not aw.record_times.size or not refs:
        return {"case": c, "skip": "no AWP, or neither CVP nor ART"}
    ta = sample_times(aw)
    xa = np.nan_to_num(aw.physical().astype(np.float64))[: ta.size]
    on = P.breath_onsets(ta, xa, float(aw.srate))
    if on.size < 200:
        return {"case": c, "skip": "fewer than 200 mechanical breaths"}
    z = float(ta[0])
    g = np.diff(ta)
    res = {"case": c, "z": z, "pauses": (ta[1:][g > 5] - z).round(1).tolist(), "arms": {}}
    ref_series = {}
    for k, tr in refs.items():
        # "<ref>_raw": the reference on its recorded stamps (no stamp model), for the as-recorded comparison
        tc = sample_times(tr) if k.endswith("_raw") else P.model_times(tr, ports.get(c), modes.get(c), step_rates)
        xc = tr.physical().astype(np.float64)[: tc.size]
        if k.startswith("art"):
            xc = lowpass_mean(np.nan_to_num(xc, nan=float(np.nanmedian(xc))), float(tr.srate))
        ref_series[k] = (tc, xc)
        tl, ll = P.cvp_lag(ta, xa, tc, xc, on)
        res["arms"][k] = {"t": (tl - z).round(1).tolist(), "lag": ll.round(4).tolist()}
    if c % 10 == 0 and a.inject and {"cvp", "art"} & set(ref_series):
        k = "cvp" if "cvp" in ref_series else "art"
        ta2 = ta + INJ_SHIFT_S + INJ_PPM * 1e-6 * (ta - z)
        on2 = on + INJ_SHIFT_S + INJ_PPM * 1e-6 * (on - z)
        tl, ll = P.cvp_lag(ta2, xa, *ref_series[k], on2)
        res["injected"] = {"arm": k, "shift_s": INJ_SHIFT_S, "ppm": INJ_PPM,
                           "t": (tl - z).round(1).tolist(), "lag": ll.round(4).tolist()}
    return res


def scan(a):
    modes = {int(k): v for k, v in json.loads(Path(a.modes).read_text()).items()}
    ports = {int(k): (v or {}).get("port") for k, v in json.loads(Path(a.ports).read_text()).items()}
    step_rates = json.loads(Path(a.step_rates).read_text())
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t_start = time.time()
    with open(out, "a", encoding="utf-8") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            if a.time_budget and time.time() - t_start > a.time_budget:
                break
            t1 = time.time()
            try:
                r = scan_case(c, a, modes, ports, step_rates)
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def design(t, pauses):
    st = np.searchsorted(np.asarray(pauses, float), t)
    u = np.unique(st)
    return np.column_stack([t / 3600] + [(st == s).astype(float) for s in u]), st


def fit(t, y, pauses):
    X, st = design(t, pauses)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta, X, st


def loo_errors(t, y, pauses):
    """Leave-one-out prediction errors (s) of the common-slope, per-stretch-offset fit."""
    X, st = design(t, pauses)
    err = []
    for i in range(t.size):
        m = np.ones(t.size, bool); m[i] = False
        if (st[m] == st[i]).sum() < 2 or m.sum() <= X.shape[1]:
            continue          # the left-out window's stretch needs at least two other windows
        cols = np.any(X[m] != 0, axis=0)
        beta = np.linalg.lstsq(X[m][:, cols], y[m], rcond=None)[0]
        err.append(float(X[i, cols] @ beta - y[i]))
    return err


def agg(a):
    R = [json.loads(l) for p in a.parts for l in open(p, encoding="utf-8") if l.strip()]
    pct = lambda x, ps=(10, 25, 50, 75, 90): ([round(float(v), 4) for v in np.percentile(x, ps)] + [len(x)]) if len(x) else None
    out = {"recordings_read": len(R), "errors": sum(1 for r in R if "error" in r),
           "skipped": {k: sum(1 for r in R if r.get("skip") == k) for k in {r.get("skip") for r in R if r.get("skip")}}}
    slopes = {"cvp": {}, "art": {}}
    for arm in ("cvp", "art"):
        errs, n_rec = [], 0
        for r in R:
            A = (r.get("arms") or {}).get(arm)
            if not A or len(A["t"]) < 6:
                continue
            t, y = np.array(A["t"]), np.array(A["lag"])
            e = loo_errors(t, y, r["pauses"])
            if e:
                errs += e; n_rec += 1
            beta, _, _ = fit(t, y, r["pauses"])
            slopes[arm][r["case"]] = beta[0] / 3600 * 1e6          # ppm
        ae = np.abs(errs)
        out[f"loo_{arm}"] = {"recordings": n_rec, "abs_error_s_p50_p90_p95": [round(float(np.percentile(ae, p)), 4) for p in (50, 90, 95)] if len(ae) else None,
                             "slope_ppm": pct(list(slopes[arm].values()))}
    both = [c for c in slopes["cvp"] if c in slopes["art"]]
    if both:
        d = np.array([slopes["cvp"][c] - slopes["art"][c] for c in both])
        cross = []
        for r in R:
            if r.get("case") not in both:
                continue
            Ac, Aa = r["arms"]["cvp"], r["arms"]["art"]
            beta, _, _ = fit(np.array(Ac["t"]), np.array(Ac["lag"]), r["pauses"])
            ta, ya = np.array(Aa["t"]), np.array(Aa["lag"])
            X, st = design(ta, r["pauses"])
            stc = np.unique(np.searchsorted(np.asarray(r["pauses"], float), np.array(Ac["t"])))
            ok = np.isin(st, stc)
            if ok.sum() < 3:
                continue
            Xc, _ = design(np.r_[np.array(Ac["t"]), ta[ok]], r["pauses"])
            pred = Xc[-ok.sum():, : beta.size] @ beta if Xc.shape[1] == beta.size else None
            if pred is None:
                continue
            res = ya[ok] - pred
            cross += list(res - np.median(res))
        out["two_arms"] = {"recordings": len(both), "slope_cvp_minus_art_ppm": pct(d),
                           "abs_slope_diff_median_ppm": round(float(np.median(np.abs(d))), 1),
                           "cvp_fit_vs_art_windows_abs_s_p50_p90": [round(float(np.percentile(np.abs(cross), p)), 4) for p in (50, 90)] if cross else None}
    inj = []
    for r in R:
        I = r.get("injected")
        A = (r.get("arms") or {}).get(I["arm"]) if I else None
        if not I or not A or len(I["t"]) < 6 or len(A["t"]) < 6:
            continue
        b0, _, _ = fit(np.array(A["t"]), np.array(A["lag"]), r["pauses"])
        b1, _, _ = fit(np.array(I["t"]), np.array(I["lag"]), r["pauses"])
        # AWP moved later by shift + ppm*t, so the AWP-to-reference lag falls by the same amount
        # first stretch's offset is the lag at the recording's start (t = 0), where the injected shift is 0.25 s
        inj.append([-(b1[0] - b0[0]) / 3600 * 1e6, -float(b1[1] - b0[1])])
    if inj:
        J = np.array(inj)
        out["injection"] = {"recordings": len(J), "injected_ppm": INJ_PPM, "recovered_ppm": pct(J[:, 0]),
                            "injected_shift_s": INJ_SHIFT_S, "recovered_shift_s": pct(J[:, 1])}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    for k in ("root", "cases", "modes", "ports", "step_rates", "out"):
        s.add_argument("--" + k.replace("_", "-"), required=True)
    s.add_argument("--time-budget", type=float, default=0)
    s.add_argument("--arms", default="cvp,art", help="references: cvp, art, and cvp_raw / art_raw for recorded stamps")
    s.add_argument("--no-inject", dest="inject", action="store_false", help="skip the injection test")
    g = sp.add_parser("agg"); g.add_argument("parts", nargs="+"); g.add_argument("--out")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
