# SPDX-License-Identifier: MIT
"""Rebuilt sawtooth slip finding, prototype before it goes into physiosync (metrology-atlas studies ppg-sawtooth,
ppg-sawtooth-slips). Works from the full-release scan of tools/sawtooth_char.py (beats and PPG fiducials) and
re-reads the waveforms for beat quality and pulse shape.

    python tools/slip_rebuild.py select --key $W/sawtooth_char/labels/key.json --slips $W/sawtooth_char/full/slips.jsonl \
        --hosts 150 --out $W/slip_rebuild/cases.json
    python tools/slip_rebuild.py quality --root $R --scan $W/sawtooth_char/scan_full --cases $W/slip_rebuild/cases.json \
        --out $W/slip_rebuild/quality [--part 0 --of 8] [--time-budget S]
    python tools/slip_rebuild.py fit --scan $W/sawtooth_char/scan_full --quality $W/slip_rebuild/quality \
        --rates $W/sawtooth_stats/recordings.csv --modes $W/results/sync_full/snuadc_mode_label.json \
        --rule $W/label_quality_final/summary.json --out $W/slip_rebuild/fit.jsonl [--r peak|template] [--resp on|off]
        [--gate on|off] [--prior setup|none]
    python tools/slip_rebuild.py eval --fit $W/slip_rebuild/fit.jsonl --key $W/sawtooth_char/labels/key.json \
        --verdicts $W/label_eval_final/label_verdicts.json --slips $W/sawtooth_char/full/slips.jsonl \
        --scan $W/sawtooth_char/scan_full --out $W/slip_rebuild/eval

**quality** (per recording, the scan's stretch): for every matched beat, the ECG II QRS (+/- 0.12 s, 250 Hz) and the
R-aligned PLETH pulse (0.1-1.0 s after R, 0.5-10 Hz band, 125 Hz), as label_quality.py measures them; each beat's
correlation with the local median beat (blocks of 30 beats, median over +/- 60 beats; PLETH at its best shift within
+/- 40 ms, so a slip does not count against the shape); irregular R-R (more than 20% from the 31-beat median);
clipping (samples at the stretch's lowest or highest raw value); and the R peak refined against the local QRS
template (cross-correlation, +/- 10 ms, two passes), so an R peak that moves within a malformed complex does not move
the delay. Where the scan injected a sawtooth into the PLETH waveform, the same injection is made again (same seed,
checked against the scan's truth) and the injected pulses measured too.

**fit** (per recording, real beats and the injected ones):

1. **Gating.** Beats whose QRS or pulse correlates under 0.9 with the local median beat, with an irregular R-R, or
   clipped, are left out of the delay series (--gate off keeps them).
2. **Delay series** of every foot, upstroke and peak against R (refined with --r template), as sawtooth_char's
   detector; the step test (one frame against none, sharpness, persistence) also fits the breathing: a sinusoid at the
   local respiratory frequency (the strongest peak 0.1-0.7 Hz of the delay series in 64 s windows, where it stands
   out) with free amplitude and phase in each window (--resp off leaves it out).
3. **Candidates** both ways: strong one-frame steps up and down.
4. **Setup prior.** The tooth length range of the recording's SNUADC port, correction mode and direction (5th-95th
   centile of the other recordings' rates, from sawtooth_stats; at least 10 recordings), else 100-4,000 s. The prior
   only sets where the fit looks first: a fit over the broad range replaces it when that confirms more slips.
5. **Phase-locked fit** over the whole stretch: the tooth length and phase whose predicted slips collect the most
   evidence (the one-frame log likelihood ratio within +/- max(10 s, 2% of a tooth) of each prediction, capped at 30,
   at least -15, and 0 where no good beats; minus half a point per prediction, so a fraction of the true tooth, whose
   extra predictions fall where there is clearly no step, loses). The score of half and twice the tooth is reported.
6. **Search at each predicted time**: the best one-frame boundary within the window, then the shape test there.
7. **Shape test** (label_quality.py's measures and its rule, fitted on the even-numbered labels only): the median
   pulse 3-45 s after against 3-45 s before; a slip moves it 20-45 ms the sawtooth's way with the shape kept (match at
   the best shift at least the rule's); the fiducials step alike. "Cannot judge" where too many beats are unlike the
   median beat or R-R is irregular.
8. **Outcome per predicted slip**: slip (step and shape agree), step without judgement (one-frame step, shape cannot
   be judged), no slip (the shape says no step: physiology, the slip is not there), unresolved (no step, cannot judge:
   a slip is inserted at the prediction for the correction). Strong steps off the grid are kept only when the shape
   confirms them; runs on a shifted grid (phase jumps) are fitted run by run.

Writes one JSON row per recording.

**eval**: against the manual labels (the odd-numbered events are held out from the rule's fit; both halves
reported), the old detectors at the same events (physiosync's slips moved to PPG time, and sawtooth_char's), and the
waveform injections (found, false, placement error, period error).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

sys.path.insert(0, str(Path(__file__).parent))
import sawtooth_char as stc  # noqa: E402
from label_quality import best_shift, call  # noqa: E402
from pat import ECG, PPG, sample_times  # noqa: E402
from sawtooth_char import FIDS, FRAME, FS, PPB, band, grid_runs, inject, long_step, noise_sd, place  # noqa: E402
from vitaldb_reader.vital_file import read_vital  # noqa: E402

warnings.filterwarnings("ignore")

PPG_WIN, PPG_FS = (0.1, 1.0), 125.0
ECG_WIN, ECG_FS = (-0.12, 0.12), 250.0
REF_WIN, REF_FS, REF_LAG = 0.06, 1000.0, 10      # R refinement: +/- 60 ms at 1 kHz, lags +/- 10 ms
GOOD_CORR = 0.9
NAMED = (3685, 6385, 5937, 743, 2336, 5339)      # recordings discussed with the labeller


# ------------------------------------------------------------------------------------------------ select

def select_cmd(a) -> None:
    K = json.loads(Path(a.key).read_text())
    lab = sorted({int(e["case"]) for e in K["events"].values()})
    hosts = []
    for line in open(a.slips):
        r = json.loads(line)
        if r.get("inj_score") and r["case"] not in lab:
            hosts.append(int(r["case"]))
    rng = np.random.default_rng(a.seed)
    hosts = sorted(rng.choice(hosts, size=min(a.hosts, len(hosts)), replace=False).tolist())
    cases = sorted(set(lab) | set(hosts) | set(NAMED))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"labelled": lab, "injection_hosts": hosts, "named": list(NAMED), "cases": cases}) + "\n")
    print(json.dumps({"labelled": len(lab), "injection_hosts": len(hosts), "cases": len(cases)}))


# ------------------------------------------------------------------------------------------------ quality

def snippets(t0: float, fs_src: float, x: np.ndarray, centres: np.ndarray, a: float, b: float, fs_out: float) -> np.ndarray:
    """x (sampled from t0 at fs_src) at centres + a .. b, fs_out grid, by linear interpolation; NaN rows out of range."""
    g = a + np.arange(int(round((b - a) * fs_out))) / fs_out
    out = np.full((centres.size, g.size), np.nan, np.float32)
    pos = (centres[:, None] + g[None, :] - t0) * fs_src
    ok = (pos[:, 0] >= 0) & (pos[:, -1] <= x.size - 1)
    if ok.any():
        out[ok] = np.interp(pos[ok].ravel(), np.arange(x.size), x).reshape(-1, g.size)
    return out


def block_templates(S: np.ndarray, block: int = 30, half: int = 60) -> np.ndarray:
    T = np.full_like(S, np.nan)
    for i in range(0, S.shape[0], block):
        lo, hi = max(0, i - half), min(S.shape[0], i + block + half)
        X = S[lo:hi]
        X = X[np.isfinite(X).all(1)]
        if len(X) >= 5:
            T[i:i + block] = np.median(X, axis=0)
    return T


def rowcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    d = np.sqrt((A * A).sum(1) * (B * B).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, (A * B).sum(1) / d, np.nan)


def shifted_corr(S: np.ndarray, T: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the best correlation of S against T over integer lags +/- max_lag (and that lag)."""
    n = S.shape[1]
    best = np.full(S.shape[0], -2.0)
    lag = np.zeros(S.shape[0])
    for L in range(-max_lag, max_lag + 1):
        c = rowcorr(S[:, max_lag + L:n - max_lag + L], T[:, max_lag:n - max_lag])
        c = np.where(np.isfinite(c), c, -2.0)
        up = c > best
        best[up], lag[up] = c[up], L
    best[best <= -2.0] = np.nan
    return best, lag


def refine_r(r: np.ndarray, te0: float, xe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R peaks refined against the local QRS template (two passes); returns the refined times and the lag (s)."""
    rr = r.copy()
    pad = REF_LAG / 1000.0
    for _ in range(2):
        S = snippets(te0, FS, xe, rr, -REF_WIN - pad, REF_WIN + pad, REF_FS)
        T = block_templates(S)
        best, lag = shifted_corr(S, T, REF_LAG)
        # sub-sample: parabola through the correlation at lag-1, lag, lag+1
        n = S.shape[1]
        cm, c0, cp = (np.full(r.size, np.nan) for _ in range(3))
        for i_, L in enumerate((-1, 0, 1)):
            Ls = np.clip(lag + L, -REF_LAG, REF_LAG).astype(int)
            vals = np.full(r.size, np.nan)
            for Lv in np.unique(Ls):
                m = Ls == Lv
                vals[m] = rowcorr(S[m][:, REF_LAG + Lv:n - REF_LAG + Lv], T[m][:, REF_LAG:n - REF_LAG])
            (cm, c0, cp)[i_][:] = vals
        den = cm - 2 * c0 + cp
        with np.errstate(invalid="ignore", divide="ignore"):
            sub = np.where(np.abs(den) > 1e-12, 0.5 * (cm - cp) / den, 0.0)
        sub = np.clip(np.nan_to_num(sub), -0.5, 0.5)
        shift = (lag + sub) / REF_FS
        shift = np.where(np.isfinite(best) & (best > 0.8), shift, 0.0)
        rr = rr + shift
    return rr, rr - r


def rr_irregular(r: np.ndarray) -> np.ndarray:
    from scipy.ndimage import median_filter
    rr = np.diff(r, prepend=np.nan)
    med = median_filter(np.where(np.isfinite(rr), rr, np.nanmedian(rr)), size=31, mode="nearest")
    return np.abs(rr / med - 1) > 0.2


def clip_share(t0: float, x: np.ndarray, centres: np.ndarray, a: float, b: float) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    rail = ((x == lo) | (x == hi)).astype(np.float32)
    S = snippets(t0, FS, rail, centres, a, b, FS)
    return np.nanmean(S, axis=1)


def beat_quality(r: np.ndarray, te0: float, xe_band: np.ndarray, xe_raw: np.ndarray, tp0: float, xp_band: np.ndarray,
                 xp_raw: np.ndarray) -> dict:
    r_ref, r_lag = refine_r(r, te0, xe_band)
    Se = snippets(te0, FS, xe_band, r, ECG_WIN[0], ECG_WIN[1], ECG_FS)
    Sp = snippets(tp0, FS, xp_band, r, PPG_WIN[0], PPG_WIN[1], PPG_FS)
    ecg_corr = rowcorr(Se, block_templates(Se))
    ppg_corr, _ = shifted_corr(Sp, block_templates(Sp), 5)
    return {"r_ref": r_ref, "r_lag": r_lag.astype(np.float32), "ecg_corr": ecg_corr.astype(np.float32),
            "ppg_corr": ppg_corr.astype(np.float32), "rr_irr": rr_irregular(r),
            "ppg_clip": clip_share(tp0, xp_raw, r, *PPG_WIN).astype(np.float32),
            "ecg_clip": clip_share(te0, xe_raw, r, *ECG_WIN).astype(np.float32),
            "ppg_snip": Sp.astype(np.float16), "ecg_snip": Se.astype(np.float16)}


def stretch(e, p, minutes: float = 90.0):
    """The scan's stretch (sawtooth_char.scan): the longest gapless ECG block, at most ``minutes``."""
    te, tp = sample_times(e), sample_times(p)
    gap = np.flatnonzero(np.abs(np.diff(te) - 1 / FS) > 0.01 / FS)
    be = np.r_[0, gap + 1, te.size]
    k = int(np.argmax(np.diff(be)))
    ea, eb = be[k], min(be[k + 1], be[k] + int(minutes * 60 * FS))
    pa, pb = np.searchsorted(tp, te[ea]), np.searchsorted(tp, te[eb - 1])
    return te, tp, ea, eb, pa, pb


def quality_cmd(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for i, c in enumerate(json.loads(Path(a.cases).read_text())["cases"]) if i % a.of == a.part]
    t_start, n, slowest = time.time(), 0, 0.0
    for c in cases:
        f = out / f"{c:04d}.npz"
        zf = Path(a.scan) / f"{c:04d}.npz"
        if f.exists() or (out / f"{c:04d}.skip").exists():
            continue
        if not zf.exists():
            (out / f"{c:04d}.skip").write_text("not in the scan")
            continue
        if time.time() - t_start + max(1.5 * slowest, 5) > a.time_budget:
            break
        t1 = time.time()
        try:
            z = np.load(zf)
            case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG})
            e, p = case.tracks[ECG], case.tracks[PPG]
            te, tp, ea, eb, pa, pb = stretch(e, p)
            if pa != int(z["ppg_first_index"]):
                raise ValueError(f"stretch differs from the scan's ({pa} against {int(z['ppg_first_index'])})")
            xe_raw = e.physical()[ea:eb].astype(np.float64)
            sos = signal.butter(2, (1.0, 40.0), btype="bandpass", fs=FS, output="sos")
            xe = signal.sosfiltfilt(sos, xe_raw)
            raw = np.asarray(p.values[pa:pb], dtype=np.float64)
            res = beat_quality(z["r_t"], te[ea], xe, xe_raw, tp[pa], band(raw), raw)
            if "inj_truth" in z.files:
                truth = json.loads(str(z["inj_truth"]))
                xw, tr2 = inject(raw, tp[pa], np.random.default_rng(10_000 + c))
                if abs(tr2["rate_ppm"] - truth["rate_ppm"]) > 1e-6:
                    raise ValueError("injection differs from the scan's")
                qi = beat_quality(z["inj_r_t"], te[ea], xe, xe_raw, tp[pa], band(xw), xw)
                res.update({"inj_" + k: v for k, v in qi.items()})
            np.savez_compressed(f, **res)
        except Exception as ex:  # noqa: BLE001
            (out / f"{c:04d}.skip").write_text(f"error {type(ex).__name__}: {ex}"[:300])
        slowest = max(slowest, time.time() - t1)
        n += 1
    print(json.dumps({"part": a.part, "new": n, "seconds": round(time.time() - t_start, 1),
                      "done": len(list(out.glob("*.npz"))), "skipped": len(list(out.glob("*.skip")))}))


# ------------------------------------------------------------------------------------------------ fit: series

def resp_phase(t: np.ndarray, y: np.ndarray, fs: float = 4.0, win: float = 64.0, step: float = 16.0) -> np.ndarray:
    """Respiratory phase (rad) at each time t from the delay series y: the strongest peak 0.1-0.7 Hz in 64 s windows
    where it stands out (peak power at least 6 times the band's median), the frequency interpolated and integrated;
    NaN where no window has a clear peak."""
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size < 200:
        return np.full(ok.size, np.nan)
    g = np.arange(t[0], t[-1], 1 / fs)
    u = np.interp(g, t, y)
    from scipy.ndimage import median_filter
    u = u - median_filter(u, size=int(30 * fs), mode="nearest")
    nw = int(win * fs)
    fr = np.fft.rfftfreq(nw, 1 / fs)
    band_ = (fr >= 0.1) & (fr <= 0.7)
    cen, fq = [], []
    for i in range(0, g.size - nw, int(step * fs)):
        seg = u[i:i + nw] * np.hanning(nw)
        P = np.abs(np.fft.rfft(seg)) ** 2
        pb = P[band_]
        k = int(np.argmax(pb))
        if pb[k] >= 6 * np.median(pb):
            cen.append(g[i + nw // 2])
            fq.append(fr[band_][k])
    if len(cen) < 2:
        return np.full(ok.size, np.nan)
    f_g = np.interp(g, cen, fq)
    phi_g = 2 * np.pi * np.cumsum(f_g) / fs
    # no phase further than one window from a window with a clear peak
    far = np.min(np.abs(g[:, None] - np.array(cen)[None, :]), axis=1) > win if len(cen) < 2000 else np.zeros(g.size, bool)
    phi_g[far] = np.nan
    out = np.full(ok.size, np.nan)
    out[ok] = np.interp(t, g, phi_g)
    return out


def step_scores(t: np.ndarray, y: np.ndarray, sd: float, sign: int, phi: np.ndarray | None) -> dict:
    """sawtooth_char.step_scores with the breathing fitted too: cos and sin of the respiratory phase as two more
    columns in every model (no step, one-frame step, ramp, free step), where the phase is known."""
    n = y.size
    W, RH = stc.GLR_W, stc.RAMP_HALF
    out = {k: np.full(n, np.nan) for k in ("llr", "fhat", "sharp", "llr_free")}
    if n < 2 * W + 2:
        return out
    ys = sign * y
    b = np.arange(W, n - W + 1)
    off = np.arange(-W, W)
    Y = ys[b[:, None] + off[None, :]]
    T = t[b[:, None] + off[None, :]] - t[b][:, None]
    T = T / max(np.median(np.abs(T)), 1e-9)
    H = np.broadcast_to((off >= 0).astype(float), Y.shape)
    ramp = np.broadcast_to(np.clip((off + RH + 0.5) / (2 * RH + 1), 0, 1), Y.shape)
    cols = [np.ones_like(Y), T]
    if phi is not None and np.isfinite(phi).any():
        P = phi[b[:, None] + off[None, :]]
        have = np.isfinite(P).all(1)[:, None]
        cols += [np.where(have, np.cos(np.nan_to_num(P)), 0.0), np.where(have, np.sin(np.nan_to_num(P)), 0.0)]

    def rss(X: np.ndarray, Yr: np.ndarray):
        XtX = np.einsum("nki,nkj->nij", X, X)
        XtY = np.einsum("nki,nk->ni", X, Yr)
        beta = np.linalg.solve(XtX + 1e-9 * np.eye(X.shape[2])[None], XtY[..., None])[..., 0]
        r = Yr - np.einsum("nki,ni->nk", X, beta)
        return np.einsum("nk,nk->n", r, r), beta

    X0 = np.stack(cols, axis=2)
    rss0, _ = rss(X0, Y)
    rss1, _ = rss(X0, Y - FRAME * H)
    rssr, _ = rss(X0, Y - FRAME * ramp)
    rss2, beta2 = rss(np.concatenate([X0, H[..., None]], axis=2), Y)
    s2 = 2 * sd ** 2
    out["llr"][b] = (rss0 - rss1) / s2
    out["sharp"][b] = (rssr - rss1) / s2
    out["fhat"][b] = beta2[:, -1]
    out["llr_free"][b] = (rss0 - rss2) / s2
    return out


def strong_steps(t: np.ndarray, y: np.ndarray, sd: float, sign: int, sc: dict) -> list[int]:
    """sawtooth_char.detect's strong-slip rule on these scores."""
    llr, f, sh = sc["llr"], sc["fhat"], sc["sharp"]
    quiet = stc.local_noise(y) <= stc.NOISE_GATE * sd
    ok = np.zeros(y.size, bool)
    for b in np.flatnonzero((llr > stc.LLR_STRONG) & (f > 0.6 * FRAME) & (f < 1.4 * FRAME) & (sh > stc.SHARP_MIN) & quiet):
        pe = long_step(y, int(b), sign)
        pe2 = long_step(y, int(b), sign, stc.PERSIST_SHORT)
        ok[b] = (0.5 * FRAME < pe < 1.5 * FRAME) or (0.5 * FRAME < pe2 < 1.5 * FRAME)
    out: list[int] = []
    for b in np.flatnonzero(ok):
        lo, hi = max(b - 30 * PPB, 0), min(b + 30 * PPB + 1, llr.size)
        if llr[b] < np.nanmax(llr[lo:hi]):
            continue
        if out and t[b] - t[out[-1]] < 60:
            if llr[b] > llr[out[-1]]:
                out[-1] = int(b)
            continue
        out.append(int(b))
    return out


# ------------------------------------------------------------------------------------------------ fit: shape test

def shape_measures(r: np.ndarray, Sp: np.ndarray, Se: np.ndarray, F: dict, tau: float, half: float = 45.0,
                   guard: float = 3.0) -> dict:
    """label_quality.event_measures on the stored pulses: the keys its rule reads."""
    out: dict = {}
    tpl = {}
    bad, irr = [], []
    steps = {}
    for side, lo, hi in (("before", tau - half, tau - guard), ("after", tau + guard, tau + half)):
        i = np.flatnonzero((r >= lo) & (r <= hi))
        out[side + "_beats"] = int(i.size)
        if i.size >= 3:
            rr = np.diff(r[i])
            irr.append(float(np.mean(np.abs(rr / np.median(rr) - 1) > 0.2)))
        for name, S in (("ppg", Sp), ("ecg", Se)):
            X = np.asarray(S[i], float)
            X = X[np.isfinite(X).all(1)] if X.size else X
            if len(X) >= 5:
                m = np.median(X, axis=0)
                cc = rowcorr(X, np.broadcast_to(m, X.shape))
                bad.append(float(np.mean(cc < GOOD_CORR)))
                tpl[(side, name)] = m
        for fid in ("foot", "up", "peak"):
            v = (F[fid][i] - r[i]) if i.size else np.empty(0)
            v = v[np.isfinite(v)]
            steps.setdefault(fid, {})[side] = float(np.median(v) * 1000) if v.size >= 3 else None
    st = [steps[f]["after"] - steps[f]["before"] for f in steps
          if steps[f].get("after") is not None and steps[f].get("before") is not None]
    out["fid_step_ms"] = float(np.median(st)) if st else None
    out["fid_spread_ms"] = float(max(st) - min(st)) if len(st) >= 2 else None
    if ("before", "ppg") in tpl and ("after", "ppg") in tpl:
        s, c, c0 = best_shift(tpl[("before", "ppg")], tpl[("after", "ppg")], PPG_FS)
        out.update(ppg_shift_ms=s, ppg_corr_best=c, ppg_corr_zero=c0)
    out["worst_bad"] = max(bad) if bad else None
    out["worst_rr_irregular"] = max(irr) if irr else None
    return out


def shape_call(m: dict, sign: int, P: dict) -> str:
    """label_quality.call (the rule fitted on the even-numbered labels), and the shift must go the sawtooth's way."""
    c = call(m, P["bad_beats_max"], P["rr_irregular_max"], P["match_min"], P["fiducial_spread_max_ms"] or 1e9)
    if c == "slip" and np.sign(m.get("ppg_shift_ms") or 0) != sign:
        return "no slip"
    return c


# ------------------------------------------------------------------------------------------------ fit: period

def evidence(t: np.ndarray, sc: dict, y: np.ndarray, sign: int, g0: float, g1: float) -> np.ndarray:
    """Per second of the stretch, the best one-frame score of any boundary in it: the log likelihood ratio, at least
    -15, capped at 30 where the step looks like a slip (free step 0.5-1.6 frames, sharper than a ramp, the new level
    holding over 15-40 or 15-75 beats) and at 8 where it does not (at most 0 where the free step is not 0.4-1.7
    frames); -inf where no boundary was scored."""
    e = np.full(int(np.ceil(g1 - g0)) + 1, -np.inf)
    llr, f, sh = sc["llr"], sc["fhat"], sc["sharp"]
    ok = np.isfinite(llr)
    s = np.clip(llr, -15, 8)
    s = np.where((f > 0.4 * FRAME) & (f < 1.7 * FRAME), s, np.minimum(s, 0))
    for b in np.flatnonzero(ok & (llr > 8) & (f > 0.5 * FRAME) & (f < 1.6 * FRAME) & (sh > 0)):
        pl = long_step(y, int(b), sign)
        ps = long_step(y, int(b), sign, stc.PERSIST_SHORT)
        if (0.4 * FRAME < pl < 1.6 * FRAME) or (0.4 * FRAME < ps < 1.6 * FRAME):
            s[b] = min(llr[b], 30.0)
    k = ((t - g0)).astype(int)
    np.maximum.at(e, np.clip(k[ok], 0, e.size - 1), s[ok])
    return e


def window_max(e: np.ndarray, w: int) -> np.ndarray:
    m = maximum_filter1d(e, size=2 * w + 1, mode="constant", cval=-np.inf)
    return np.where(np.isfinite(m), m, 0.0)


def win_for(L: float) -> int:
    return int(np.clip(0.02 * L, 10, 30))


JUMP = 20.0          # cost of a phase jump between two slips
M_MAX = 8            # teeth one grid step may span (missed slips in between)


def chain_nodes(e: np.ndarray, w: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Seconds with positive evidence that are the best within +/- w s (one per plateau, its middle): the places a
    slip may sit."""
    m = maximum_filter1d(e, size=2 * w + 1, mode="constant", cval=-np.inf)
    k = np.flatnonzero((e > 0) & (e >= m))
    keep, grp = [], []
    for i in k:
        if grp and i - grp[-1] > w:
            keep.append(grp[len(grp) // 2])
            grp = []
        grp.append(i)
    if grp:
        keep.append(grp[len(grp) // 2])
    keep = np.array(keep, int)
    return keep.astype(float), e[keep]


def chain_fit(T: np.ndarray, v: np.ndarray, Ew: np.ndarray, L: float) -> tuple[float, list[int], list[str]]:
    """The best chain of slips for tooth L: each slip follows the previous one a whole number of teeth later (within
    +/- w x sqrt(teeth)), the slips the grid predicts in between scored on their evidence (clearly no step there
    costs up to 15 each), half a point per tooth; or after a phase jump, at a cost of JUMP. Returns the score, the
    chain (node indices) and each link's kind."""
    N = T.size
    if N == 0:
        return 0.0, [], []
    w = win_for(L)
    G = T[None, :] - T[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.round(G / L)
    grid = (m >= 1) & (m <= M_MAX) & (np.abs(G - m * L) <= w * np.sqrt(np.maximum(m, 1)))
    inter = np.zeros((N, N))
    span = Ew.size - 1
    for j in range(1, M_MAX):
        pos = T[:, None] + j * np.where(m > 0, G / np.maximum(m, 1), 0)
        inter += np.where(grid & (j < m), Ew[np.clip(pos.astype(int), 0, span)], 0.0)
    tr = np.where(grid, np.maximum(inter - 0.5 * m, -JUMP), -JUMP)
    tr = np.where(G > 0.25 * L, tr, -np.inf)          # two slips closer than a quarter tooth cannot both be on it
    best = np.full(N, -np.inf)
    prev = np.full(N, -1)
    for k in range(N):
        cand = best[:k] + tr[:k, k] if k else np.empty(0)
        if cand.size and np.max(cand) > 0:
            j = int(np.argmax(cand))
            best[k], prev[k] = v[k] + cand[j], j
        else:
            best[k] = v[k]
    k = int(np.argmax(best))
    S = float(best[k])
    chain = []
    while k >= 0:
        chain.append(k)
        k = int(prev[k])
    chain = chain[::-1]
    links = ["grid" if grid[a_, b_] and tr[a_, b_] > -JUMP else "jump" for a_, b_ in zip(chain[:-1], chain[1:])]
    return S, chain, links


def period_fit(e: np.ndarray, L_lo: float, L_hi: float) -> dict:
    """The tooth length whose best chain scores highest, over L_lo-L_hi (log steps of 0.3%, then 0.05% about the
    best); the chain's score at half and twice that tooth is reported."""
    cache: dict = {}

    def ew(L):
        w = win_for(L)
        if w not in cache:
            cache[w] = window_max(e, w)
        return cache[w]
    span = e.size - 1
    L_hi = min(L_hi, 0.9 * span)
    T, v = chain_nodes(e)
    if L_hi <= L_lo or T.size < 2:
        return {}
    Ls = np.exp(np.arange(np.log(L_lo), np.log(L_hi), 0.003))
    sc_ = [chain_fit(T, v, ew(L), L)[0] for L in Ls]
    k = int(np.argmax(sc_))
    Lf = np.exp(np.arange(np.log(Ls[k]) - 0.003, np.log(Ls[k]) + 0.003, 0.0005))
    scf = [chain_fit(T, v, ew(L), L)[0] for L in Lf]
    L = float(Lf[int(np.argmax(scf))])
    S, chain, links = chain_fit(T, v, ew(L), L)
    return {"L": L, "score": S, "chain_t": [float(T[i]) for i in chain], "links": links,
            "score_half": chain_fit(T, v, ew(L / 2), L / 2)[0] if L / 2 >= 30 else None,
            "score_double": chain_fit(T, v, ew(2 * L), 2 * L)[0]}


# ------------------------------------------------------------------------------------------------ fit: per recording

def prior_range(rates: dict, port: str, mode: str, case: int) -> tuple[tuple[float, float] | None, int | None]:
    """5th-95th centile tooth range of the setup's other recordings, by direction: {+1: (lo, hi), -1: ...}."""
    out = {}
    for d, key in ((1, "up"), (-1, "down")):
        v = [abs(x) for c, x in rates.get((port, mode, key), []) if c != case]
        if len(v) >= 10:
            lo, hi = np.percentile(v, [5, 95])
            out[d] = (FRAME / (hi * 1e-6) / 1.03, FRAME / (lo * 1e-6) * 1.03, len(v))
    return out


def predicted(L: float, t_ref: float, g0: float, g1: float) -> np.ndarray:
    k0 = int(np.ceil((g0 - t_ref) / L))
    k1 = int(np.floor((g1 - t_ref) / L))
    return t_ref + L * np.arange(k0, k1 + 1)


def judge_at(p: float, w: float, sign: int, ctx: dict) -> dict:
    """At a predicted time: the best one-frame boundary within w and its tests; the shape test there, and, unless
    that confirms a slip, at the five boundaries of the window with the largest free step the sawtooth's way (0.4-1.7
    frames) and at the prediction itself, keeping the first that confirms one (a boundary that failed the step tests
    can still be where the pulse moved)."""
    t, sc, y = ctx["t"], ctx["sc"], ctx["y"]
    sel = np.flatnonzero((np.abs(t - p) <= w) & np.isfinite(sc["llr"]))
    row = {"t_pred": round(p, 2)}
    tau = p
    step = False
    bp = None
    if sel.size:
        bp = int(sel[np.argmax(sc["llr"][sel])])
        f, llr, sh = sc["fhat"][bp], sc["llr"][bp], sc["sharp"][bp]
        pl = long_step(y, bp, sign)
        ps = long_step(y, bp, sign, stc.PERSIST_SHORT)
        held = (0.4 * FRAME < pl < 1.6 * FRAME) or (0.4 * FRAME < ps < 1.6 * FRAME)
        edge = not np.isfinite(pl) and not np.isfinite(ps)
        step = bool(llr > stc.LLR_GRID and sh > -10 and ((held and 0.5 * FRAME < f < 1.6 * FRAME) or
                                                          (edge and 0.5 * FRAME < f < 1.8 * FRAME)))
        row.update(llr=round(float(llr), 1), step_ms=round(float(f) * 1000, 1), sharp=round(float(sh), 1), step=step)
        tau = float(0.5 * (t[bp - 1] + t[bp]))

    def judged(tau_):
        m = shape_measures(ctx["r"], ctx["Sp"], ctx["Se"], ctx["F"], tau_)
        return m, shape_call(m, sign, ctx["rule"])
    m, c = judged(tau)
    row["bp"] = bp if step else None
    if c != "slip" and sel.size:
        ff = sc["fhat"][sel] * sign
        cand = sel[(ff > 0.4 * FRAME) & (ff < 1.7 * FRAME)]
        cand = cand[np.argsort(-sc["llr_free"][cand])]
        tried = [tau]
        for b_ in list(cand[:20]) + [None]:
            tb = p if b_ is None else float(0.5 * (t[b_ - 1] + t[b_]))
            if min(abs(tb - x) for x in tried) < 3:
                continue
            tried.append(tb)
            m2, c2 = judged(tb)
            if c2 == "slip":
                m, c, tau, row["bp"] = m2, c2, tb, b_
                row["moved_s"] = round(tb - (float(0.5 * (t[bp - 1] + t[bp])) if bp is not None else p), 1)
                break
            if len(tried) > 6:
                break
    row["shape"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
                    if k in ("ppg_shift_ms", "ppg_corr_best", "fid_step_ms", "fid_spread_ms", "worst_bad", "worst_rr_irregular")}
    row["call"] = c
    row["outcome"] = ("slip" if c == "slip" else
                      "step, cannot judge" if step and c == "cannot judge" else
                      "no slip" if c == "no slip" else "unresolved")
    row["t"] = round(tau, 3)
    return row


def place_row(row: dict, sign: int, ctx: dict) -> None:
    if row.get("bp") is None:
        return
    b = int(ctx["beat_pt"][row["bp"]])
    p = place(b, ctx["F"], ctx["r"], ctx["sd"], sign)
    if p:
        row.update(t=round(p["t"], 3), lo=round(p["lo"], 3), hi=round(p["hi"], 3))


def runs_of(chain_t: list[float], links: list[str]) -> list[list[float]]:
    runs, cur = [], [chain_t[0]] if chain_t else []
    for t_, l_ in zip(chain_t[1:], links):
        if l_ == "grid":
            cur.append(t_)
        else:
            runs.append(cur)
            cur = [t_]
    if cur:
        runs.append(cur)
    return runs


def run_slots(runs: list[list[float]], L: float, g0: float, g1: float) -> list[tuple[int, float]]:
    """Predicted slips of every run of two or more slips, over its own stretch: out to halfway to the neighbouring
    runs' slips (or the ends of the stretch). Each run's phase and tooth by least squares on its slips."""
    out = []
    for k, ref in enumerate(runs):
        if len(ref) < 2:
            continue
        ref = np.array(ref)
        n_ = np.r_[0, np.cumsum(np.maximum(np.round(np.diff(ref) / L), 1))]
        A = np.column_stack([np.ones_like(n_), n_])
        t0_, L_ = np.linalg.lstsq(A, ref, rcond=None)[0] if len(ref) >= 3 else (ref[0], L)
        if not 0.9 * L < L_ < 1.1 * L:
            t0_, L_ = float(np.median(ref - n_ * L)), L
        before = [x for j, r_ in enumerate(runs) if j < k for x in r_]
        after = [x for j, r_ in enumerate(runs) if j > k for x in r_]
        lo_ = 0.5 * (max(before) + ref[0]) if before else g0
        hi_ = 0.5 * (min(after) + ref[-1]) if after else g1
        out += [(k, float(p)) for p in predicted(L_, t0_, lo_, hi_)]
    return out


def fit_direction(sign: int, ctx: dict, prior: tuple | None, g0: float, g1: float) -> dict:
    t, sc = ctx["t"], ctx["sc_by_sign"][sign]
    ctx = {**ctx, "sc": sc}
    e = evidence(t, sc, ctx["y"], sign, g0, g1)
    best = None
    for name, rng_ in ((("prior", prior[:2]),) if prior else ()) + (("broad", (100.0, 4000.0)),):
        pf = period_fit(e, *rng_)
        if not pf:
            continue
        chain_abs = [g0 + x for x in pf["chain_t"]]
        runs = runs_of(chain_abs, pf["links"])
        rows = [{**judge_at(p, win_for(pf["L"]), sign, ctx), "run": k} for k, p in run_slots(runs, pf["L"], g0, g1)]
        pf.update(source=name, runs=sum(len(r_) >= 2 for r_ in runs), rows=rows,
                  confirmed=sum(r_["outcome"] == "slip" for r_ in rows), chain_abs=chain_abs, runs_t=runs)
        if best is None or pf["confirmed"] > best["confirmed"]:
            best = pf
    if best is None:
        return {"sign": sign, "confirmed": 0, "rows": [], "e": e}
    # again from the accepted slips: each run's phase and tooth from where its slips were found
    for _ in range(2):
        runs = []
        for k in sorted({r_["run"] for r_ in best["rows"]}):
            acc = [r_["t"] for r_ in best["rows"] if r_["run"] == k and r_["outcome"] in ("slip", "step, cannot judge")]
            if len(acc) >= 2:
                runs.append(acc)
        if not runs:
            break
        rows = [{**judge_at(p, win_for(best["L"]), sign, ctx), "run": k} for k, p in run_slots(runs, best["L"], g0, g1)]
        best["rows"], best["confirmed"] = rows, sum(r_["outcome"] == "slip" for r_ in rows)
    for r_ in best["rows"]:
        if r_["outcome"] in ("slip", "step, cannot judge"):
            place_row(r_, sign, ctx)
    # the tooth: from the accepted slips of each run (whole numbers of teeth apart), pooled
    num, den = 0.0, 0.0
    for k in sorted({r_["run"] for r_ in best["rows"]}):
        acc = np.array([r_["t"] for r_ in best["rows"] if r_["run"] == k and r_["outcome"] == "slip"])
        if acc.size >= 2:
            n_ = np.round(np.diff(acc) / best["L"])
            num, den = num + np.sum(np.diff(acc)[n_ >= 1]), den + np.sum(n_[n_ >= 1])
    best["L_search"] = best["L"]
    if den:
        best["L"] = float(num / den)
    best["sign"] = sign
    best["rate_ppm"] = FRAME / best["L"] * 1e6
    best["e"] = e
    return best


def find_slips(r_scan: np.ndarray, F: dict, Q: dict, pref: str, opts) -> dict:
    """Everything for one set of beats (real or injected); Q holds that set's quality arrays (no prefix)."""
    r = Q["r_ref"] if opts.r == "template" else r_scan
    good = np.ones(r.size, bool)
    if opts.gate == "on":
        good = (np.nan_to_num(Q["ecg_corr"]) >= GOOD_CORR) & (np.nan_to_num(Q["ppg_corr"]) >= GOOD_CORR) & ~Q["rr_irr"] & \
            (np.nan_to_num(Q["ppg_clip"]) < 0.02) & (np.nan_to_num(Q["ecg_clip"]) < 0.02)
    Fg = {j: np.where(good, F[j], np.nan) for j in FIDS}
    t_pt, y, beat_pt = stc.point_series(Fg, r)
    if t_pt.size < 600:
        raise ValueError(f"too few good beats ({int(good.sum())} of {good.size})")
    sd = noise_sd(y)
    y = stc.clean(y, sd)
    phi = resp_phase(t_pt, y) if opts.resp == "on" else None
    ctx = {"t": t_pt, "y": y, "beat_pt": beat_pt, "r": r_scan, "F": F, "Sp": Q["ppg_snip"], "Se": Q["ecg_snip"],
           "rule": opts.rule_params, "sd": {j: noise_sd(F[j] - r) for j in FIDS}}
    ctx["sc_by_sign"] = {s: step_scores(t_pt, y, sd, s, phi) for s in (1, -1)}
    ctx["strong"] = {s: strong_steps(t_pt, y, sd, s, ctx["sc_by_sign"][s]) for s in (1, -1)}
    g0, g1 = float(t_pt[0]), float(t_pt[-1])
    out = {"beats": int(r.size), "good_share": round(float(good.mean()), 3), "noise_ms": round(sd * 1000, 2),
           "resp_share": round(float(np.isfinite(phi).mean()), 3) if phi is not None else None,
           "strong_up": len(ctx["strong"][1]), "strong_down": len(ctx["strong"][-1]), "start": g0, "end": g1}
    fits = {s: fit_direction(s, ctx, opts.prior_by_sign.get(s) if opts.prior == "setup" else None, g0, g1) for s in (1, -1)}
    best = max(fits.values(), key=lambda f: (f.get("confirmed", 0), f["sign"]))
    other = fits[-best["sign"]]
    out["direction"] = "up" if best["sign"] > 0 else "down"
    out["fit"] = {k: best.get(k) for k in ("L", "L_search", "rate_ppm", "score", "score_half", "score_double", "source", "confirmed",
                                         "runs", "chain_abs", "links")}
    out["other_direction_confirmed"] = other.get("confirmed", 0)
    rows = best.get("rows", [])
    out["predicted"] = rows
    # a sawtooth: two predicted slips confirmed by the shape, or one and two more one-frame steps on the grid whose
    # shape cannot be judged
    unj = sum(r_["outcome"] == "step, cannot judge" for r_ in rows)
    out["sawtooth"] = bool(best.get("confirmed", 0) >= 2 or (best.get("confirmed", 0) >= 1 and unj >= 2))
    # every other place with slip-like evidence, either way (strong steps, and the seconds whose one-frame step passes
    # the slip tests), not at a predicted slip of the fit: kept when the shape test confirms it
    w = win_for(best["L"]) if best.get("L") else 10
    pred_t = np.array([r_["t_pred"] for r_ in rows]) if out["sawtooth"] else np.empty(0)
    extra = []
    for s_ in (1, -1):
        sc_s = ctx["sc_by_sign"][s_]
        T, v = chain_nodes(fits[s_]["e"])
        cands = sorted({float(g0 + x) for x, vv in zip(T, v) if vv > 8} |
                       {float(0.5 * (t_pt[bp - 1] + t_pt[bp])) for bp in ctx["strong"][s_]})
        for tb in cands:
            if s_ == best["sign"] and pred_t.size and np.min(np.abs(pred_t - tb)) <= w:
                continue
            sel = np.flatnonzero((np.abs(t_pt - tb) <= 10) & np.isfinite(sc_s["llr"]))
            bp = int(sel[np.argmax(sc_s["llr"][sel])]) if sel.size else None
            tb_ = float(0.5 * (t_pt[bp - 1] + t_pt[bp])) if bp else tb
            if any(abs(x["t"] - tb_) < 30 and x["sign"] == s_ for x in extra):
                continue
            m = shape_measures(r_scan, Q["ppg_snip"], Q["ecg_snip"], F, tb_)
            o = {"t": round(tb_, 3), "sign": s_, "bp": bp, "llr": round(float(sc_s["llr"][bp]), 1) if bp else None,
                 "call": shape_call(m, s_, opts.rule_params),
                 "shape": {k: (round(vv, 3) if isinstance(vv, float) else vv) for k, vv in m.items()
                           if k in ("ppg_shift_ms", "ppg_corr_best", "worst_bad", "worst_rr_irregular")}}
            if o["call"] == "slip":
                place_row(o, s_, {**ctx, "sc": sc_s})
            extra.append(o)
    out["off_grid"] = extra
    # the slips: confirmed at a prediction, or one-frame steps there whose shape cannot be judged (sawtooth only),
    # and shape-confirmed steps elsewhere (off the grid, or the other way)
    sl = [{"t": r_["t"], "lo": r_.get("lo"), "hi": r_.get("hi"), "kind": r_["outcome"], "t_pred": r_["t_pred"],
           "sign": best["sign"]} for r_ in rows
          if r_["outcome"] == "slip" or (out["sawtooth"] and r_["outcome"] == "step, cannot judge")]
    for o in extra:
        if o["call"] != "slip" or any(abs(x["t"] - o["t"]) < 30 for x in sl):
            continue
        sl.append({"t": o["t"], "lo": o.get("lo"), "hi": o.get("hi"), "sign": o["sign"],
                   "kind": "off grid" if o["sign"] == best["sign"] else "other way"})
    out["inserted"] = [r_["t_pred"] for r_ in rows if r_["outcome"] == "unresolved"] if out["sawtooth"] else []
    out["slips"] = sorted(sl, key=lambda x: x["t"])
    out["sign"] = best["sign"]
    return out


def load_rates(path: str, modes: dict) -> dict:
    rates = defaultdict(list)
    for row in csv.DictReader(open(path)):
        rates[(row["port"], row["mode"], row["direction"])].append((int(row["case"]), float(row["rate_ppm"])))
    return rates


def fit_cmd(a) -> None:
    S = json.loads(Path(a.rule).read_text())
    a.rule_params = S["rule"]["params"]
    modes = json.loads(Path(a.modes).read_text())
    rates = load_rates(a.rates, modes)
    files = [f for i, f in enumerate(sorted(Path(a.quality).glob("*.npz"))) if i % a.of == a.part]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t0 = time.time()
    with out.open("a") as fh:
        for f in files:
            c = int(f.stem)
            if c in done:
                continue
            if time.time() - t0 > a.time_budget:
                break
            z = np.load(Path(a.scan) / f.name)
            Qz = np.load(f)
            port = str(z["port"]) or "?"
            mode = modes.get(str(c), "?")
            a.prior_by_sign = prior_range(rates, port, mode, c)
            row = {"case": c, "port": port, "mode": mode,
                   "prior": {("up" if k > 0 else "down"): [round(v[0], 1), round(v[1], 1), v[2]] for k, v in a.prior_by_sign.items()},
                   "opts": {"r": a.r, "resp": a.resp, "gate": a.gate, "prior": a.prior}}
            try:
                Q = {k: Qz[k] for k in Qz.files if not k.startswith("inj_")}
                row["real"] = find_slips(z["r_t"], {j: z[j] for j in FIDS}, Q, "", a)
                if "inj_truth" in z.files and "inj_r_ref" in Qz.files:
                    Qi = {k[4:]: Qz[k] for k in Qz.files if k.startswith("inj_")}
                    row["inj"] = find_slips(z["inj_r_t"], {j: z["inj_" + j] for j in FIDS}, Qi, "inj_", a)
                    row["inj_truth"] = json.loads(str(z["inj_truth"]))
            except Exception as ex:  # noqa: BLE001
                import traceback
                row["error"] = f"{type(ex).__name__}: {ex}"[:300]
                row["trace"] = traceback.format_exc()[-1500:]
            fh.write(json.dumps(row, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
            fh.flush()
    print(json.dumps({"part": a.part, "seconds": round(time.time() - t0, 1),
                      "done": sum(1 for _ in out.open()) if out.exists() else 0}))


# ------------------------------------------------------------------------------------------------ eval

REAL = ("ppg_slip_up", "ppg_step_up_unconfirmed", "ppg_slip_down", "ppg_step_down_unconfirmed")
NEAR_S = 15.0


def group_of(v: str) -> str:
    return "real" if v in REAL else "none" if v == "none" else "unclear" if v in ("unclear", "artefact") else "other"


def mark_offset(v: dict) -> float | None:
    c = v.get("clicks") or {}
    xs = []
    for k in ("ecg_ppg", "art_ppg"):
        x = c.get(k)
        if isinstance(x, list) and x:
            xs.append(float(np.mean(x)))
        elif isinstance(x, (int, float)):
            xs.append(float(x))
    return float(np.mean(xs)) if xs else None


def eval_cmd(a) -> None:
    K = json.loads(Path(a.key).read_text())["events"]
    V = json.loads(Path(a.verdicts).read_text())
    fits = {}
    for line in open(a.fit):
        r = json.loads(line)
        fits[r["case"]] = r
    old = {}
    for line in open(a.slips):
        r = json.loads(line)
        if r["case"] in fits:
            old[r["case"]] = r
    ids = sorted(V)
    held = {e for i, e in enumerate(ids) if i % 2 == 1}
    rows = []
    for e in ids:
        ev, v = K[e], V[e]
        c = int(ev["case"])
        if c not in fits or "real" not in fits[c]:
            continue
        R = fits[c]["real"]
        off = mark_offset(v)
        t = float(ev["t"]) + (off if off is not None and abs(off) <= 60 else 0.0)   # where the labeller marked the step
        near = [s for s in R["slips"] if abs(s["t"] - t) <= NEAR_S]
        # the call at the event: a slip found near it; else the shape test there (cannot judge or no slip)
        if near:
            new = "slip"
        else:
            pr = [p for p in R.get("predicted", []) if abs(p["t_pred"] - t) <= NEAR_S]
            new = "cannot judge" if any(p["outcome"] in ("unresolved", "step, cannot judge") for p in pr) else "no slip"
        o = old.get(c, {})
        dl = float(o.get("delay_mode_s") or 0.0)
        ps = [x + dl for x in (o.get("fit") or {}).get("slips_t", [])]
        oldnew = [s["t"] for s in (o.get("real") or {}).get("slips", [])] + [s["t"] for s in (o.get("real") or {}).get("control", [])]
        off = 0.0 if off is not None else None
        err_new = min((s["t"] - (t + off) for s in near), key=abs) if near and off is not None else None
        olds = [x for x in oldnew if abs(x - t) <= NEAR_S]
        err_old = min((x - (t + off) for x in olds), key=abs) if olds and off is not None else None
        pss = [x for x in ps if abs(x - t) <= NEAR_S]
        err_ps = min((x - (t + off) for x in pss), key=abs) if pss and off is not None else None
        rows.append({"event": e, "case": c, "stratum": ev["stratum"], "verdict": v["verdict"], "group": group_of(v["verdict"]),
                     "held_out": e in held, "new": new, "sawtooth_char": "slip" if olds else "no slip",
                     "physiosync": "slip" if pss else "no slip", "err_new_s": err_new, "err_old_s": err_old,
                     "err_physiosync_s": err_ps})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def table(sel, key):
        c = Counter((r["group"], r[key]) for r in sel)
        return {g: {k: c.get((g, k), 0) for k in ("slip", "no slip", "cannot judge")} for g in ("real", "none", "unclear")}

    def prec_rec(sel, key):
        tp = sum(1 for r in sel if r["group"] == "real" and r[key] == "slip")
        fp = sum(1 for r in sel if r["group"] == "none" and r[key] == "slip")
        fn = sum(1 for r in sel if r["group"] == "real" and r[key] != "slip")
        return {"real_found": tp, "real_missed": fn, "none_called_slip": fp}

    def errs(sel, key):
        v = np.array([abs(r[key]) for r in sel if r[key] is not None])
        return {"n": int(v.size), **({"median_s": round(float(np.median(v)), 2), "p90_s": round(float(np.percentile(v, 90)), 2)}
                                     if v.size else {})}
    S = {"fit": a.fit, "opts": next(iter(fits.values())).get("opts"), "labels": {}}
    for name, sel in (("held_out", [r for r in rows if r["held_out"]]), ("all", rows)):
        S["labels"][name] = {"events": len(sel), "by_method": {k: table(sel, k) for k in ("new", "sawtooth_char", "physiosync")},
                             "counts": {k: prec_rec(sel, k) for k in ("new", "sawtooth_char", "physiosync")},
                             "placement_vs_mark": {k: errs(sel, f"err_{k}_s") for k in ("new", "old", "physiosync")},
                             "by_stratum": {st: table([r for r in sel if r["stratum"] == st], "new")
                                            for st in sorted({r["stratum"] for r in sel})}}
    # injections: upward slips only (the injected direction; sawtooth_char's injection score counts only those), and
    # not where the host's own beats have a slip of their own (some hosts carry a reversed sawtooth, or isolated slips)
    inj = {"hosts": 0, "true": 0, "found": 0, "false": 0, "down_found": 0, "own_skipped": 0, "err_s": [], "L_err_s": [],
           "old_found": 0, "old_false": 0, "old_err_s": [], "physiosync_err_s": []}
    for c, r in fits.items():
        if "inj" not in r:
            continue
        ts = np.array(r["inj_truth"]["slips_t"])
        own = np.array([s["t"] for s in (r.get("real") or {}).get("slips", [])])
        found = []
        for s in r["inj"]["slips"]:
            if s.get("sign", 1) < 0:
                inj["down_found"] += 1
            elif own.size and np.min(np.abs(own - s["t"])) < 30 and (not ts.size or np.min(np.abs(ts - s["t"])) >= 60):
                inj["own_skipped"] += 1
            else:
                found.append(s["t"])
        inj["hosts"] += 1
        inj["true"] += ts.size
        for s in ts:
            d = [f - s for f in found if abs(f - s) < 60]
            if d:
                inj["found"] += 1
                inj["err_s"].append(min(d, key=abs))
        inj["false"] += sum(1 for f in found if not ts.size or np.min(np.abs(ts - f)) >= 60)
        if r["inj"].get("sawtooth") and r["inj"]["fit"].get("L") and r["inj"].get("sign", 1) > 0:
            inj["L_err_s"].append(r["inj"]["fit"]["L"] - r["inj_truth"]["period_s"])
        o = old.get(c, {}).get("inj_score")
        if o:
            inj["old_found"] += o["found"]
            inj["old_false"] += o["false"]
            inj["old_err_s"] += [x["err_s"] for x in o["rows"] if x]
            inj["physiosync_err_s"] += [x for x in o.get("physiosync_err_s", []) if x is not None]

    def qe(v):
        v = np.abs(np.array(v, float))
        return {"n": int(v.size), **({"median_s": round(float(np.median(v)), 2), "p95_s": round(float(np.percentile(v, 95)), 2)}
                                     if v.size else {})}
    S["injections"] = {"hosts": inj["hosts"], "true": inj["true"],
                       "new": {"found": inj["found"], "false": inj["false"], "down_found": inj["down_found"],
                               "own_slips_skipped": inj["own_skipped"], "placement": qe(inj["err_s"]),
                               "period_err": qe(inj["L_err_s"])},
                       "sawtooth_char": {"found": inj["old_found"], "false": inj["old_false"], "placement": qe(inj["old_err_s"])},
                       "physiosync": {"found": len(inj["physiosync_err_s"]), "placement": qe(inj["physiosync_err_s"])}}
    # per recording, real beats: direction, rate, how the predicted slips came out
    oc = Counter()
    for r in fits.values():
        for p in (r.get("real") or {}).get("predicted", []) if (r.get("real") or {}).get("sawtooth") else []:
            oc[p["outcome"]] += 1
    S["recordings"] = {"fitted": len(fits), "errors": sum(1 for r in fits.values() if "error" in r),
                       "sawtooth": sum(1 for r in fits.values() if (r.get("real") or {}).get("sawtooth")),
                       "down": sum(1 for r in fits.values() if (r.get("real") or {}).get("sawtooth") and r["real"]["sign"] < 0),
                       "predicted_outcomes": dict(oc)}
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


# ------------------------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--key", required=True)
    s.add_argument("--slips", required=True)
    s.add_argument("--hosts", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s = sub.add_parser("quality")
    s.add_argument("--root", required=True)
    s.add_argument("--scan", required=True)
    s.add_argument("--cases", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--part", type=int, default=0)
    s.add_argument("--of", type=int, default=1)
    s.add_argument("--time-budget", type=float, default=1e9)
    s = sub.add_parser("fit")
    for k in ("scan", "quality", "rates", "modes", "rule", "out"):
        s.add_argument("--" + k, required=True)
    s.add_argument("--r", choices=("peak", "template"), default="template")
    s.add_argument("--resp", choices=("on", "off"), default="on")
    s.add_argument("--gate", choices=("on", "off"), default="on")
    s.add_argument("--prior", choices=("setup", "none"), default="setup")
    s.add_argument("--part", type=int, default=0)
    s.add_argument("--of", type=int, default=1)
    s.add_argument("--time-budget", type=float, default=1e9)
    s = sub.add_parser("eval")
    for k in ("fit", "key", "verdicts", "slips", "out"):
        s.add_argument("--" + k, required=True)
    a = ap.parse_args()
    {"select": select_cmd, "quality": quality_cmd, "fit": fit_cmd, "eval": eval_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
