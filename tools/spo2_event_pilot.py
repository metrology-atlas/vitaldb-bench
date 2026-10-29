# SPDX-License-Identifier: MIT
"""SpO2 / PLETH_HR latency from step events in the PLETH waveform. Pilot, per case, this machine only.

    python tools/spo2_event_pilot.py --root R --cases 1,2 --units units.csv --step-rates rates.json --out out.jsonl

SNUADC samples are on the stamp-model line (recorder time); Solar8000 numerics carry recorder stamps (2 s polls).
The PLETH is the oximeter's analogue output, ~0.37-0.40 s behind the finger (hardware delay, PAT work): all
times here are relative to the recorded PLETH. Event types, analysed separately:

1. Pulse loss / return. PLETH pulse amplitude (max-min per 0.5 s, running max over 4 s) against the case's
   median: pulsatile >= 30%, 'flat' < 5% (probe off or no signal), else 'low' (low perfusion). An episode is a
   non-pulsatile run >= 3 s with >= 30 s pulsatile before and >= 20 s after. For PLETH_SPO2 and PLETH_HR:
   whether the numeric dropped out (a record gap > 5 s inside the episode), the hold time (pulse loss to the
   last record before the dropout; that record is the last value known, the dropout is known one poll later),
   and the resume time (pulse return to the first record after). Episodes where Solar8000/HR also stops are
   recorder gaps and are flagged, not used.
2. Electrocautery bursts: SNUADC ECG 30-60 Hz log power envelope (tools/shared_artifacts.py) at least 10x its
   10 s running median for >= 0.3 s, gaps < 0.5 s merged; isolated (no other burst 30 s before or after);
   whether the PLETH 30-240 Hz envelope also rose.
3. PLETH-only artefacts: PLETH 30-240 Hz envelope burst, or a pulse amplitude > 3x the case median for >= 1 s,
   with no ECG burst within 2 s; isolated as above.
4. Control: random times with the same exclusions (pulsatile, no burst within 30 s).

For 2-4 per event: SpO2 and PLETH_HR on a 1 s grid from -20 to +60 s around the onset (the value of the last
record at or before each instant, stamped at its record; NaN where the last record is > 5 s old), less the
median over -20..-2 s; the burst duration; dropouts.
5. PLETH_HR algorithm: pulse intervals from PLETH peaks, fitted with tools/numeric_derivation_pilot.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

sys.path.insert(0, str(Path(__file__).parent))
from numeric_derivation_pilot import VALID, analyse_track, candidates, numeric_obs  # noqa: E402
from vent_sync_pilot import snu_times  # noqa: E402
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: E402

warnings.filterwarnings("ignore")
DT = 0.05
GRID = np.arange(-20, 61, 1.0)
STALE = 5.0


def envelope(t, x, fs, band, t0, n):
    ok = np.isfinite(x)
    x = np.where(ok, x, 0.0)
    band = (band[0], min(band[1], 0.45 * fs))  # 100 Hz files: 30-240 Hz becomes 30-45 Hz (below Nyquist)
    sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x) ** 2
    b = np.floor((t - t0) / DT).astype(np.int64)
    k = (b >= 0) & (b < n) & ok
    s = np.bincount(b[k], y[k], minlength=n)
    c = np.bincount(b[k], minlength=n)
    m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    e = np.log10(m + 1e-12)
    okm = np.isfinite(m)
    e[~okm] = np.nanmedian(e)
    e = ndimage.median_filter(e, 11)
    e = e - ndimage.median_filter(e, 201)
    e[~okm] = np.nan
    return e


def runs(mask, min_len, merge_gap):
    m = np.r_[False, mask, False]
    d = np.diff(m.astype(int))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = []
    for s, e in zip(st, en):
        if out and s - out[-1][1] <= merge_gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e - s >= min_len]


def track_tv(v, name):
    tr = v.tracks.get(name)
    if tr is None or not tr.record_times.size:
        return None
    t = np.asarray(tr.record_times, float)
    x = np.asarray(tr.physical(), float)[: t.size]
    o = np.argsort(t, kind="stable")
    t, x = t[o], x[o]
    ok = np.isfinite(x) & (x > 0)
    return t[ok], x[ok]


def on_grid(tv, times):
    t, x = tv
    i = np.searchsorted(t, times, side="right") - 1
    ic = np.clip(i, 0, t.size - 1)
    return np.where((i >= 0) & (times - t[ic] <= STALE), x[ic], np.nan)


def gaps_in(tv, a, b):
    """Record gaps > STALE s overlapping [a, b]: list of (last record before, first record after)."""
    t = tv[0]
    i0, i1 = np.searchsorted(t, a - 120), np.searchsorted(t, b + 600)
    tt = t[max(i0 - 1, 0) : i1 + 1]
    d = np.diff(tt)
    g = np.flatnonzero(d > STALE)
    return [(float(tt[j]), float(tt[j + 1])) for j in g if tt[j + 1] >= a and tt[j] <= b + 300]


def pulse_peaks(t, x, fs):
    ok = np.isfinite(x)
    xf = np.where(ok, x, np.nanmedian(x))
    sos = signal.butter(2, (0.5, 8), btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, xf)
    w = int(10 * fs)
    amp = ndimage.maximum_filter1d(y, w) - ndimage.minimum_filter1d(y, w)
    pk, _ = signal.find_peaks(y, distance=int(0.33 * fs))
    keep = (y[pk] - ndimage.minimum_filter1d(y, int(1.5 * fs))[pk]) > 0.3 * amp[pk]
    pk = pk[keep & ok[pk]]
    tp = t[pk]
    iv = np.diff(tp)
    g = (iv >= 0.25) & (iv <= 2.5)
    return tp[1:][g], iv[g]


def run_case(root, c, u, step_rates, seed):
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital",
                   track_names={"SNUADC/ECG_II", "SNUADC/PLETH", "Solar8000/PLETH_SPO2", "Solar8000/PLETH_HR",
                                "Solar8000/HR"})
    res = {"case": c}
    if "SNUADC/PLETH" not in v.tracks or "Solar8000/PLETH_SPO2" not in v.tracks:
        res["skip"] = "no PLETH or SpO2"
        return res
    sp, ph, hr = (track_tv(v, n) for n in ("Solar8000/PLETH_SPO2", "Solar8000/PLETH_HR", "Solar8000/HR"))
    P = v.tracks["SNUADC/PLETH"]
    tp, clk = snu_times(P, u, step_rates)
    xp = P.physical().astype(np.float64)[: tp.size]
    fs = float(P.srate)
    t0 = float(np.floor(tp[0]))
    n = int((tp[-1] - t0) / DT) + 1
    e_pl = envelope(tp, xp, fs, (30, 240), t0, n)
    e_ec = None
    if "SNUADC/ECG_II" in v.tracks:
        E = v.tracks["SNUADC/ECG_II"]
        te, _ = snu_times(E, u, step_rates)
        e_ec = envelope(te, E.physical().astype(np.float64)[: te.size], float(E.srate), (30, 60), t0, n)
        del E, te
    # pulse amplitude, 0.5 s bins -> 4 s running max
    sos = signal.butter(2, (0.5, 8), btype="bandpass", fs=fs, output="sos")
    okp = np.isfinite(xp)
    yb = signal.sosfiltfilt(sos, np.where(okp, xp, np.nanmedian(xp)))
    b = np.floor((tp - t0) / 0.5).astype(np.int64)
    nb = int(b[-1]) + 1
    mx = np.full(nb, -np.inf)
    mn = np.full(nb, np.inf)
    np.maximum.at(mx, b[okp], yb[okp])
    np.minimum.at(mn, b[okp], yb[okp])
    amp = np.where(np.isfinite(mx - mn), mx - mn, 0.0)
    ramp = ndimage.maximum_filter1d(amp, 8, origin=3)
    med = float(np.median(ramp[ramp > 0])) if np.any(ramp > 0) else 0.0
    if med <= 0:
        res["skip"] = "flat PLETH"
        return res
    puls = ramp >= 0.3 * med
    flat = ramp < 0.05 * med
    tb = t0 + 0.5 * np.arange(nb)
    res["pleth_clock"] = {"rate_ppm": round(clk["rate_ppm"], 1), "source": clk["rate_source"]}

    # ---- 1. pulse loss / return
    eps = []
    for s, e in runs(~puls, 6, 0):
        if s < 60 or e + 40 > nb or not puls[s - 60 : s].all() or not puls[e : e + 40].all():
            continue
        # sharp edges from the unsmoothed 0.5 s amplitude: last pulsatile bin before, first after
        pb = np.flatnonzero(amp[max(s - 12, 0) : e] >= 0.3 * med)
        pa = np.flatnonzero(amp[s:e + 12] >= 0.3 * med)
        s_ = max(s - 12, 0) + (pb[-1] + 1 if pb.size else 12)
        pa = pa[pa >= (s_ - s)]
        e_ = s + (pa[0] if pa.size else e - s)
        loss, ret = float(tb[s_]), float(tb[e_])
        if ret - loss < 2.0:
            continue
        typ = "flat" if flat[s:e].mean() > 0.5 else "low"
        ep = {"loss": round(loss, 2), "dur": round(ret - loss, 1), "type": typ}
        rec_gap = hr is not None and any(g[1] - g[0] > STALE for g in gaps_in(hr, loss, ret))
        ep["recorder_gap"] = bool(rec_gap)
        for key, tv in (("spo2", sp), ("phr", ph)):
            if tv is None:
                continue
            # dropout: a record gap that starts between 10 s before the loss and the return
            gs = [g for g in gaps_in(tv, loss, ret) if loss - 10 < g[0] < ret]
            if gs:
                g0 = gs[0]
                # the first record after the return that is followed by >= 20 s without gaps
                t_ = tv[0]
                i = np.searchsorted(t_, max(ret, gs[-1][1]) - 0.01)
                res_t = None
                while i < t_.size - 1:
                    j = np.searchsorted(t_, t_[i] + 20)
                    if j < t_.size and np.all(np.diff(t_[i : j + 1]) <= STALE):
                        res_t = float(t_[i])
                        break
                    i += 1
                ep[key] = {"dropout": True, "hold": round(g0[0] - loss, 2), "gap_end": round(gs[-1][1] - ret, 2),
                           "resume": None if res_t is None else round(res_t - ret, 2), "n_gaps": len(gs)}
            else:
                seg = on_grid(tv, np.arange(loss, ret + 2, 2.0))
                ep[key] = {"dropout": False, "held_const": bool(np.nanstd(seg) == 0) if np.isfinite(seg).any() else None}
        eps.append(ep)
    res["episodes"] = eps

    # ---- 2/3. bursts
    def bursts(env, thr=1.0, min_s=0.3):
        if env is None:
            return []
        r = runs(np.nan_to_num(env, nan=0) > thr, int(min_s / DT), int(0.5 / DT))
        return [(t0 + s * DT, t0 + e * DT) for s, e in r]

    ec = bursts(e_ec)
    pl_hf = bursts(e_pl)
    big = [(float(tb[s]), float(tb[e - 1]) + 0.5) for s, e in runs(amp > 3 * med, 2, 1)]  # e is exclusive; tb is 0.5 s bins
    kind_of = {**{x: "hf" for x in pl_hf}, **{x: "big_amp" for x in big}}
    ec_on = np.array([a for a, _ in ec]) if ec else np.zeros(0)
    all_on = np.sort(np.r_[ec_on, [a for a, _ in pl_hf], [a for a, _ in big]])
    all_end = np.sort(np.r_[[b_ for _, b_ in ec], [b_ for _, b_ in pl_hf], [b_ for _, b_ in big]])

    def isolated(a, b_):
        before = all_end[(all_end < a - 0.01) & (all_end > a - 30)]
        after = all_on[(all_on > b_ + 0.01) & (all_on < b_ + 30)]
        return before.size == 0 and after.size == 0

    def pulsatile_at(a, b_):
        i0, i1 = int((a - 20 - t0) / 0.5), int((b_ + 30 - t0) / 0.5)
        if i0 < 0 or i1 >= nb:
            return False
        w = puls[i0:i1].copy()
        k0, k1 = int((a - t0) / 0.5) - i0, int((b_ - t0) / 0.5) - i0 + 1
        w[max(k0, 0) : k1] = True        # the burst itself may disturb the pulse
        return bool(w.all())

    def response(a, b_):
        out = {"on": round(a, 2), "dur": round(b_ - a, 2)}
        for key, tv in (("spo2", sp), ("phr", ph)):
            if tv is None:
                continue
            y = on_grid(tv, a + GRID)
            base = np.nanmedian(y[(GRID >= -20) & (GRID <= -2)])
            out[key] = None if not np.isfinite(base) else [None if not np.isfinite(z) else round(float(z - base), 1) for z in y]
        return out

    def ev_list(lst, need_pl=None):
        out = []
        for a, b_ in lst:
            if not isolated(a, b_) or not pulsatile_at(a, b_):
                continue
            r = response(a, b_)
            if need_pl is not None:
                i0, i1 = int((a - t0) / DT), int((b_ - t0) / DT) + 1
                r["pleth_hf_rise"] = bool(np.nanmax(e_pl[i0:i1], initial=-9) > 0.5)
            out.append(r)
        return out

    res["cautery"] = ev_list(ec, need_pl=True)
    pl_only = []
    for a, b_ in sorted(pl_hf + big):
        if ec_on.size and np.any(np.abs(ec_on - a) < 2.0):
            continue
        if any(not (b_ < x - 2 or a > y + 2) for x, y in ec):
            continue
        pl_only.append((a, b_))
    res["pleth_only"] = ev_list(pl_only)
    for r_ in res["pleth_only"]:
        r_["kind"] = next((kind_of[x] for x in kind_of if abs(x[0] - r_["on"]) < 0.01), None)
    rng = np.random.default_rng(seed + c)
    ctrl = []
    for a in rng.uniform(tb[0] + 60, tb[-1] - 90, 400):
        if len(ctrl) >= 60:
            break
        if all_on.size and np.min(np.abs(all_on - a)) < 30:
            continue
        if all_end.size and np.min(np.abs(all_end - a)) < 30:
            continue
        if pulsatile_at(a, a + 0.5):
            ctrl.append(response(a, a + 0.5))
    res["control"] = ctrl
    res["counts"] = {"ecg_bursts": len(ec), "pleth_hf_bursts": len(pl_hf), "big_amp": len(big)}

    # ---- 5. PLETH_HR algorithm
    if ph is not None:
        tpk, iv = pulse_peaks(tp, xp, fs)
        tr = v.tracks.get("Solar8000/PLETH_HR")
        if tpk.size > 500:
            ot, ov, og = numeric_obs(tr, *VALID["HR"])
            m = (ot > tpk[0] + 60) & (ot < tpk[-1])
            if m.sum() > 300:
                cands = candidates("rate_beat", tpk, iv)
                r = analyse_track("HR", ot[m], ov[m], og[m], cands, solar=True, span=VALID["HR"][1] - VALID["HR"][0],
                                  ev_int=(tpk, iv))
                if r:
                    res["pleth_hr_fit"] = r
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--step-rates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    step_rates = json.loads(Path(a.step_rates).read_text())
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in Path(a.out).read_text().splitlines() if l.strip()}
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        t1 = time.time()
        try:
            res = run_case(a.root, c, U.get(c, {}), step_rates, a.seed)
        except Exception as e:  # noqa: BLE001
            res = {"case": c, "error": repr(e)[:300]}
        res["secs"] = round(time.time() - t1, 1)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
