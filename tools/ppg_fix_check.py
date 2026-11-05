# SPDX-License-Identifier: MIT
"""Check physiosync's ECG-PPG delay correction on VitalDB cases. Per-case numbers only.

    python tools/ppg_fix_check.py --root R --cases cases.json --out out.jsonl --adopted-pat 0.27 [--part 0 --of 4]

PAT = filtered steepest PPG upstroke minus sub-sample R-peak (as ``pat_res``), on the longest continuous
ECG+PPG stretch (up to ``--minutes``). Fits ``fit_ecg_ppg_delay`` (frame 1/30 s) and reports slips and
rates (and each tooth's confidence tier; --apply-min-tier sets which tiers are corrected), then the same slip
detector run on the corrected PAT (should find only false detections), and the
5-95% spread of 1-minute medians before and after. 15-beat medians of both series are kept for plots.

    python tools/ppg_fix_check.py --method rebuilt --scan $W/sawtooth_char/scan_full --quality $W/slip_rebuild/quality \
        --rates $W/sawtooth_stats/recordings.csv --modes $W/results/sync_full/snuadc_mode_label.json \
        --rule $W/label_quality_final/summary.json --cases cases.json --out out.jsonl --adopted-pat 0.27

--method rebuilt fits ``fit_ecg_ppg_delay(method="rebuilt")`` (physiosync 0.3.1) on the beats, fiducials, arterial
upstrokes and beat quality of the full-release scan (tools/sawtooth_char.py, tools/slip_rebuild.py quality), the
setup's tooth as the prior, and reports the same measures: PAT is the scan's steepest upstroke minus its R peak on the
scan's stretch (the longest, up to 90 minutes); the steps are those the model corrects (confirmed slips and one-frame
steps the shape cannot judge, never slips placed only within an interval); the creep is corrected only where the model
corrects it (rate_source "teeth"; "reported" for a tooth from two slips, "none" without a sawtooth). --root is not
read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pat import ECG, PPG, r_peaks, sample_times
from pat_res import ecg_subsample, ppg_onsets
from physiosync import detect_slips, fit_ecg_ppg_delay, match_beats
from physiosync.ecg_ppg import _p2p
from vitaldb_reader.vital_file import (
    read_vital,
)

warnings.filterwarnings("ignore")


def rebuilt_context(a) -> dict:
    import slip_rebuild as proto
    modes = json.loads(Path(a.modes).read_text())
    return {"proto": proto, "modes": modes, "rates": proto.load_rates(a.rates, modes),
            "rule": json.loads(Path(a.rule).read_text())["rule"]["params"]}


def tooth_slopes(tr, y, edges):
    """Slope inside each tooth (15-beat medians, 20 s clear of its ends), ppm."""
    out = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        sel = np.flatnonzero((tr > e0 + 20) & (tr < e1 - 20))
        n15 = sel.size // 15
        if n15 < 8:
            continue
        tt = tr[sel[: 15 * n15]].reshape(n15, 15).mean(1)
        yy = np.median(y[sel[: 15 * n15]].reshape(n15, 15), 1)
        out.append(round(float(np.polyfit(tt - tt[0], yy, 1)[0]) * 1e6, 2))
    return out


def rebuilt_case(c: int, a, ctx: dict) -> dict:
    from physiosync import SlipOptions
    from physiosync.ecg_ppg import _local_steps
    zp, qp = Path(a.scan) / f"{c:04d}.npz", Path(a.quality) / f"{c:04d}.npz"
    if not zp.exists() or not qp.exists():
        return {"skip": "not in the full-release scan"}
    z, Qz = np.load(zp), np.load(qp)
    Q = {k: Qz[k] for k in Qz.files if not k.startswith("inj_")}
    F = {j: z[j] for j in ("foot", "up", "peak", "down50")}
    port = str(z["port"]) or "?"
    pr = ctx["proto"].prior_range(ctx["rates"], port, ctx["modes"].get(str(c), "?"), c)
    m = fit_ecg_ppg_delay(z["r_t"], z["up"], adopted_pat_s=a.adopted_pat, matched=True, method="rebuilt", fiducials=F,
                          beat_quality=Q, ref_t=z["art_t"], slip_options=SlipOptions(rule_params=ctx["rule"], prior_by_sign=pr))
    ok = np.isfinite(z["up"])
    tr, tq = z["r_t"][ok], z["up"][ok]
    raw = tq - tr
    cor = m.correct(tq) - tr
    again = detect_slips(tr, cor, frame_s=1 / 30)
    before = detect_slips(tr, raw, frame_s=1 / 30)
    steps_t = [e for e, s in zip(m.tooth_edges, m.tooth_steps) if s]
    s_raw, s_cor = _local_steps(tr, raw, 15), _local_steps(tr, cor, 15)
    bnd = np.r_[tr[0], 0.5 * (tr[1:] + tr[:-1])]
    idx = [int(np.argmin(np.abs(bnd - x))) for x in steps_t]
    # signed the sawtooth's way, so a reversed sawtooth's steps read positive like an upward one's
    sg = -1.0 if m.direction == "down" else 1.0
    st_raw = [round(sg * float(s_raw[k]) * 1000, 2) for k in idx if np.isfinite(s_raw[k])]
    st_cor = [round(sg * float(s_cor[k]) * 1000, 2) for k in idx if np.isfinite(s_cor[k])]
    # how each corrected step was found (the slip's kind and the arterial verdict), in the same order
    kinds = []
    for x, k in zip(steps_t, idx):
        if not np.isfinite(s_raw[k]):
            continue
        sl = min(m.slips, key=lambda q: abs(q["t"] - x)) if m.slips else {}
        kinds.append(sl.get("kind", "?") if abs(sl.get("t", 1e18) - x) < 1 else "?")
    # control: the same 15-beat step half way between consecutive corrected steps (no slip there), corrected series
    mids = [0.5 * (x + y) for x, y in zip(steps_t[:-1], steps_t[1:])]
    st_mid = [round(sg * float(s_cor[int(np.argmin(np.abs(bnd - x)))]) * 1000, 2) for x in mids]
    st_mid = [x for x in st_mid if np.isfinite(x)]
    creep = m.qc.get("creep")
    corrected = bool(m.qc.get("sawtooth_fitted")) and creep == "corrected"
    slip_edges = [m.t_start] + steps_t + [m.t_end]
    n = raw.size // 15
    t0 = tr[0]
    return {
        "method": "rebuilt", "port": port, "prior_ppm": None,
        "tooth_slope_raw_ppm": [sg * x for x in tooth_slopes(tr, raw, slip_edges)] if corrected else [],
        "tooth_slope_corr_ppm": [sg * x for x in tooth_slopes(tr, cor, slip_edges)] if corrected else [],
        "step_at_slips_raw_ms": st_raw, "step_at_slips_corr_ms": st_cor, "step_kinds": kinds, "step_mid_corr_ms": st_mid,
        "rate_source": "teeth" if corrected else ("reported" if m.qc.get("sawtooth_fitted") else "none"),
        "direction": m.direction, "tooth_s": m.qc.get("tooth_s"), "drift": m.drift,
        "n_kept": len(steps_t), "n_rejected": 0, "n_inserted": 0, "n_uncertain": len(m.uncertain),
        "minutes": round((m.t_end - m.t_start) / 60, 1), "beats": int(raw.size),
        "n_slips": len(m.slips_t),
        "rates_ppm": [round(sg * x * 1e6, 2) for x in m.tooth_rate],
        "sources": ["tooth" if x else "none" for x in m.tooth_rate],
        "tiers": list(m.tooth_tier), "tier_reasons": list(m.tooth_reason), "apply_min_tier": None,
        "offset_ms": round(m.offset_s * 1000, 2), "pat_median_raw_ms": round(m.qc["pat_median_raw_s"] * 1000, 2),
        "pat_median_corr_ms": round(m.qc["pat_median_corrected_s"] * 1000, 3),
        "p2p_raw_ms": round(_p2p(tr, raw), 2), "p2p_corr_ms": round(_p2p(tr, cor), 2),
        "slips_before": len(before["t"]), "slips_after": len(again["t"]),
        "slip_steps_after_ms": [round(x * 1000, 2) for x in again["step_s"]],
        "series": [np.round(tr[: 15 * n].reshape(n, 15).mean(1) - t0, 1).tolist(),
                   np.round(np.median(raw[: 15 * n].reshape(n, 15), 1) * 1000, 2).tolist(),
                   np.round(np.median(cor[: 15 * n].reshape(n, 15), 1) * 1000, 2).tolist()],
        "slips_t": [round(x - t0, 1) for x in m.slips_t], "inserted_t": [],
        "uncertain_t": [round(u["q50"] - t0, 1) for u in m.uncertain if u.get("q50") is not None],
        "tooth_edges_t": [round(x - t0, 1) for x in m.tooth_edges],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="", help="method detect: the VitalDB root")
    ap.add_argument("--method", choices=("detect", "rebuilt"), default="detect")
    for k in ("scan", "quality", "rates", "modes", "rule"):
        ap.add_argument("--" + k, help="method rebuilt")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adopted-pat", type=float, required=True)
    ap.add_argument("--minutes", type=float, default=90.0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=150)
    ap.add_argument("--priors", help="JSON {case: rate_ppm}: the creep rate known for the case's recording setup")
    ap.add_argument("--apply-min-tier", default="B", choices=["A", "B", "C"],
                    help="lowest tooth tier the correction is applied to (physiosync default B; C applies every "
                         "fitted tooth, as before the tiers existed)")
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    cases = [c for i, c in enumerate(json.loads(Path(a.cases).read_text())["cases"]) if i % a.of == a.part]
    priors = {int(k): v for k, v in json.loads(Path(a.priors).read_text()).items()} if a.priors else {}
    ctx = rebuilt_context(a) if a.method == "rebuilt" else None
    t_start, slowest = time.time(), 0.0
    with out.open("a") as fh:
        for c in cases:
            if c in done:
                continue
            if time.time() - t_start + max(1.5 * slowest, 5) > a.time_budget:
                break
            t1 = time.time()
            res: dict = {"case": c}
            if ctx is not None:
                try:
                    res.update(rebuilt_case(c, a, ctx))
                except Exception as ex:  # noqa: BLE001
                    res["error"] = f"{type(ex).__name__}: {ex}"
                res["seconds"] = round(time.time() - t1, 2)
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                slowest = max(slowest, res["seconds"])
                continue
            try:
                case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG})
                e, p = case.tracks.get(ECG), case.tracks.get(PPG)
                if e is None or p is None or not e.record_times.size or not p.record_times.size or e.srate != 500:
                    res["skip"] = "no 500 Hz ECG_II and PLETH"
                else:
                    fs = 500.0
                    te, tp = sample_times(e), sample_times(p)
                    gap = np.flatnonzero(np.abs(np.diff(te) - 1 / fs) > 0.01 / fs)
                    be = np.r_[0, gap + 1, te.size]
                    k = int(np.argmax(np.diff(be)))
                    ea, eb = be[k], min(be[k + 1], be[k] + int(a.minutes * 60 * fs))
                    pa, pb = np.searchsorted(tp, te[ea]), np.searchsorted(tp, te[eb - 1])
                    if eb - ea < 1200 * fs or pb - pa < 1200 * fs or np.max(np.abs(np.diff(tp[pa:pb]) - 1 / fs)) > 0.01 / fs:
                        res["skip"] = "no 20 min continuous stretch"
                    else:
                        xe = e.physical()[ea:eb].astype(np.float64)
                        rp = r_peaks(xe, fs)
                        _, up_f = ppg_onsets(np.asarray(p.values[pa:pb]), fs)
                        rt = te[ea] + ecg_subsample(xe, rp, fs) / fs
                        pt = tp[pa] + up_f / fs
                        i, j, _ = match_beats(rt, pt)
                        tr, tq = rt[i], pt[j]
                        prior = priors.get(c)
                        m = fit_ecg_ppg_delay(tr, tq, adopted_pat_s=a.adopted_pat, frame_s=1 / 30, matched=True,
                                              rate_prior_ppm=prior, apply_min_tier=a.apply_min_tier)
                        raw = tq - tr
                        cor = m.correct(tq) - tr
                        again = detect_slips(tr, cor, frame_s=1 / 30)
                        before = detect_slips(tr, raw, frame_s=1 / 30)
                        # the step left at each modelled slip (15 beats after minus before): ~1 frame raw, ~0 corrected
                        from physiosync.ecg_ppg import _local_steps
                        s_raw, s_cor = _local_steps(tr, raw, 15), _local_steps(tr, cor, 15)
                        bnd = np.r_[tr[0], 0.5 * (tr[1:] + tr[:-1])]
                        idx = [int(np.argmin(np.abs(bnd - x))) for x in m.tooth_edges[1:-1]]
                        st_raw = [round(float(s_raw[k]) * 1000, 2) for k in idx if np.isfinite(s_raw[k])]
                        st_cor = [round(float(s_cor[k]) * 1000, 2) for k in idx if np.isfinite(s_cor[k])]
                        # split-half: grid period fitted on each half separately
                        mid = tr[tr.size // 2]
                        halves = []
                        for sel in (tr < mid, tr >= mid):
                            try:
                                mh = fit_ecg_ppg_delay(tr[sel], tq[sel], adopted_pat_s=a.adopted_pat, frame_s=1 / 30,
                                                       matched=True, rate_prior_ppm=prior, apply_min_tier=a.apply_min_tier)
                                halves.append(mh.qc.get("grid_period_s"))
                            except ValueError:
                                halves.append(None)
                        # slope inside each tooth (15-beat medians, 20 s clear of its ends): ~ -rate raw, ~0 corrected
                        def tooth_slopes(y):
                            out = []
                            for e0, e1 in zip(m.tooth_edges[:-1], m.tooth_edges[1:]):
                                sel = np.flatnonzero((tr > e0 + 20) & (tr < e1 - 20))
                                n15 = sel.size // 15
                                if n15 < 8:
                                    continue
                                tt = tr[sel[: 15 * n15]].reshape(n15, 15).mean(1)
                                yy = np.median(y[sel[: 15 * n15]].reshape(n15, 15), 1)
                                out.append(round(float(np.polyfit(tt - tt[0], yy, 1)[0]) * 1e6, 2))
                            return out
                        res.update({"prior_ppm": prior, "tooth_slope_raw_ppm": tooth_slopes(raw),
                                    "tooth_slope_corr_ppm": tooth_slopes(cor), "unresolved": bool(m.qc.get("unresolved_slips_t"))})
                        res.update({"step_at_slips_raw_ms": st_raw, "step_at_slips_corr_ms": st_cor,
                                    "grid_period_s": m.qc.get("grid_period_s"), "rate_source": m.qc.get("rate_source"),
                                    "n_kept": m.qc.get("n_kept"), "n_rejected": m.qc.get("n_rejected"), "half_periods_s": halves})
                        n = raw.size // 15
                        t0 = tr[0]
                        res.update({
                            "minutes": round((eb - ea) / fs / 60, 1), "beats": int(raw.size),
                            "n_slips": m.qc["n_slips"], "n_inserted": m.qc["n_inserted"],
                            "rates_ppm": [round(x * 1e6, 2) for x in m.tooth_rate], "sources": m.tooth_source,
                            "tiers": list(m.tooth_tier), "tier_reasons": list(m.tooth_reason),
                            "apply_min_tier": a.apply_min_tier,
                            "offset_ms": round(m.offset_s * 1000, 2), "pat_median_raw_ms": round(m.qc["pat_median_raw_s"] * 1000, 2),
                            "pat_median_corr_ms": round(m.qc["pat_median_corrected_s"] * 1000, 3),
                            "p2p_raw_ms": round(_p2p(tr, raw), 2), "p2p_corr_ms": round(_p2p(tr, cor), 2),
                            "slips_before": len(before["t"]), "slips_after": len(again["t"]),
                            "slip_steps_after_ms": [round(x * 1000, 2) for x in again["step_s"]],
                            "series": [np.round(tr[: 15 * n].reshape(n, 15).mean(1) - t0, 1).tolist(),
                                       np.round(np.median(raw[: 15 * n].reshape(n, 15), 1) * 1000, 2).tolist(),
                                       np.round(np.median(cor[: 15 * n].reshape(n, 15), 1) * 1000, 2).tolist()],
                            "slips_t": [round(x - t0, 1) for x in m.slips_t], "inserted_t": [round(x - t0, 1) for x in m.inserted_t],
                            "tooth_edges_t": [round(x - t0, 1) for x in m.tooth_edges],
                        })
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"
            res["seconds"] = round(time.time() - t1, 2)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            slowest = max(slowest, res["seconds"])
    print(json.dumps({"part": a.part, "done": sum(1 for _ in out.open()), "seconds": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
