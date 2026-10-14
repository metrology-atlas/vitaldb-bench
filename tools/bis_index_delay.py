# SPDX-License-Identifier: MIT
"""Link S of the timing graph: how the BIS index responds in time to its own EEG. This machine only; aggregates leave.

    python tools/bis_index_delay.py scan --root R --cases 1,2,3 --out out.jsonl
    python tools/bis_index_delay.py selftest --root R --case 1
    python tools/bis_index_delay.py agg out*.jsonl --summary summary.json

Method (no trained model, so the reference has no hidden lag of its own):
  * EEG features at 1 Hz from BIS/EEG1_WAV, each from the 2 s Blackman window that ends at its stamp (the
    availability rule: a feature exists only once its window has ended; the device's own EEG stamps; the index comes from the same device, so their clock corrections are shared): log band powers
    (0.5-4, 4-8, 8-13, 13-30, 30-47 Hz), log total power, SEF95, beta ratio log10(P30-47 / P11-20) and a
    suppression flag (|x - mean| < 5 uV over the window).
  * Rank-one distributed-lag model, BIS(t) = c + sum_k h_k g(t - k) with g = F a, lags k = -10..120 s, fitted by
    alternating least squares (a by OLS given h; h by ridge with a second-difference penalty given a), h scaled to
    sum 1. SQI >= 50 only. Negative lags are allowed so a misaligned or acausal fit shows itself.
  * From h: the best exponential-average kernel is the primary summary (the self-test recovers W and d; the raw
    centroid is biased late by the ridge tail). Also the centroid, lags at 10/50/90% of the positive mass, the share of mass at negative lags, and the best
    exponential-average kernel (W, d: alpha = 2 / (W + 1) starting at lag d) and best trailing boxcar (L, d).
  * Within-case stability: h re-fitted per 30-min block with a held at the whole-case value.
  * Check of the method: BIS/SR (suppression ratio, by definition a trailing ~63 s share) against the suppression
    flag alone, where suppression occurs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
from pat import sample_times  # noqa: E402
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: E402

FS = 128
LAGS = np.arange(-10, 121)
BLOCK = 1800
NAMES = ["delta", "theta", "alpha", "beta", "gamma", "total", "sef95", "beta_ratio", "supp"]


def load(root, c):
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital",
                   track_names={"BIS/EEG1_WAV", "BIS/BIS", "BIS/SQI", "BIS/SR"})
    e = v.tracks.get("BIS/EEG1_WAV")
    b = v.tracks.get("BIS/BIS")
    if e is None or b is None or not e.record_times.size or not b.record_times.size:
        return None
    t = sample_times(e)
    x = np.asarray(e.physical(), float)[: t.size]
    T0 = float(np.floor(t.min()))
    n = int(np.ceil(t.max() - T0)) + 2
    grid = np.full(n * FS, np.nan)
    i = np.round((t - T0) * FS).astype(np.int64)
    ok = (i >= 0) & (i < grid.size) & np.isfinite(x)
    grid[i[ok]] = x[ok]

    def per_s(tr):
        out = np.full(n, np.nan)
        if tr is None or not tr.record_times.size:
            return out
        k = np.round(tr.record_times - T0).astype(np.int64)
        val = np.asarray(tr.physical(), float)[: k.size]
        m = (k >= 0) & (k < n)
        out[k[m]] = val[m]
        return out

    phase = float(np.median((b.record_times - T0) % 1.0))
    return grid, per_s(b), per_s(v.tracks.get("BIS/SQI")), per_s(v.tracks.get("BIS/SR")), n, phase


def features(grid, n):
    """1 Hz features; second s uses samples [(s-2)*FS, s*FS), the window ending at s (stamped at its end)."""
    F = np.full((n, len(NAMES)), np.nan)
    pad = np.concatenate([np.full(2 * FS, np.nan), grid])
    W = np.lib.stride_tricks.sliding_window_view(pad, 2 * FS)[::FS][:n]
    good = np.isfinite(W).all(axis=1)
    X = signal.detrend(W[good], axis=1)
    P = np.abs(np.fft.rfft(X * np.blackman(2 * FS), axis=1)) ** 2
    f = np.fft.rfftfreq(2 * FS, 1 / FS)

    def band(lo, hi):
        return P[:, (f >= lo) & (f < hi)].sum(axis=1) + 1e-12
    cum = np.cumsum(P[:, (f >= 0.5) & (f < 30)], axis=1)
    fb = f[(f >= 0.5) & (f < 30)]
    sef = fb[np.argmax(cum >= 0.95 * cum[:, -1:], axis=1)]
    supp = (np.abs(W[good] - W[good].mean(axis=1, keepdims=True)).max(axis=1) < 5).astype(float)
    F[good] = np.column_stack([np.log10(band(0.5, 4)), np.log10(band(4, 8)), np.log10(band(8, 13)),
                               np.log10(band(13, 30)), np.log10(band(30, 47)), np.log10(band(0.5, 47)), sef,
                               np.log10(band(30, 47) / band(11, 20)), supp])
    return F


def lagmat(g, lags):
    n = g.size
    G = np.zeros((n, lags.size))
    for j, k in enumerate(lags):
        if k >= 0:
            G[k:, j] = g[: n - k]
        else:
            G[: n + k, j] = g[-k:]
    return G


def lagconv(z, h):
    """Same as lagmat(z, LAGS) @ h, by convolution."""
    return np.convolve(z, h)[-LAGS[0]: -LAGS[0] + z.size]


def ridge_h(G, y, lam):
    K = G.shape[1]
    D = np.diff(np.eye(K), 2, axis=0)
    Gc = np.column_stack([G, np.ones(len(y))])
    A = Gc.T @ Gc
    A[:K, :K] += lam * D.T @ D
    sol = np.linalg.solve(A + 1e-9 * np.eye(K + 1), Gc.T @ y)
    return sol[:K], sol[K]


def ema_kernel(W, d, lags):
    a = 2.0 / (W + 1.0)
    k = lags - d
    h = np.where(k >= 0, a * (1 - a) ** np.maximum(k, 0), 0.0)
    return h / h.sum()


def box_kernel(L, d, lags):
    h = ((lags >= d) & (lags < d + L)).astype(float)
    return h / h.sum()


def fast_part(h, lags, d_fixed=None):
    """Non-negative fit h ~ w1 box(L, d) + w2 slow EMA(Ws, d) + w3 flat floor; the box is the fast part."""
    hn = h / h.sum()
    flat = np.ones(lags.size) / lags.size
    best = None
    for d in ([d_fixed] if d_fixed is not None else range(-2, 11)):
        for Ws in (60, 120, 240):
            slow = ema_kernel(Ws, d, lags)
            for L in range(3, 61):
                w, res = nnls(np.column_stack([box_kernel(L, d, lags), slow, flat]), hn)
                if best is None or res < best[0]:
                    best = (res, L, d, Ws, w)
    res, L, d, Ws, w = best
    tot = w.sum() if w.sum() > 0 else 1.0
    return {"fast_L": L, "fast_d": d, "slow_W": Ws, "fast_share": round(float(w[0] / tot), 3),
            "slow_share": round(float(w[1] / tot), 3), "fast_res": round(float(res), 5)}


def describe(h, lags):
    hp = np.clip(h, 0, None)
    s = hp.sum()
    if s <= 0:
        return {}
    c = np.cumsum(hp) / s
    q = lambda p: float(lags[np.searchsorted(c, p)])  # noqa: E731
    hn = h / h.sum() if h.sum() > 0 else hp / s
    best_e = min(((float(np.sum((hn - ema_kernel(W, d, lags)) ** 2)), W, d)
                  for W in range(1, 91) for d in range(-5, 31)))
    best_b = min(((float(np.sum((hn - box_kernel(L, d, lags)) ** 2)), L, d)
                  for L in range(1, 91) for d in range(-5, 31)))
    return {"centroid": round(float((lags * hp).sum() / s), 2), "q10": q(0.1), "q50": q(0.5), "q90": q(0.9),
            "peak": float(lags[np.argmax(h)]), "neg_mass": round(float(hp[lags < 0].sum() / s), 3),
            "ema_W": best_e[1], "ema_d": best_e[2], "ema_sse": round(best_e[0], 5),
            "box_L": best_b[1], "box_d": best_b[2], "box_sse": round(best_b[0], 5)}


def fit_case(F, y, valid, lam=50.0, iters=8, h0=None):
    mu = np.nanmean(F[valid], axis=0)
    sd = np.nanstd(F[valid], axis=0) + 1e-9
    Z = np.nan_to_num((F - mu) / sd)
    Z = np.clip(Z, -5, 5)
    h = ema_kernel(15, 0, LAGS) if h0 is None else h0
    for _ in range(iters):
        Zh = np.column_stack([lagconv(Z[:, j], h) for j in range(Z.shape[1])])
        X = np.column_stack([Zh[valid], np.ones(valid.sum())])
        a = np.linalg.lstsq(X, y[valid], rcond=None)[0][:-1]
        g = Z @ a
        G = lagmat(g, LAGS)
        h, _ = ridge_h(G[valid], y[valid], lam)
        sc = h.sum()
        if abs(sc) < 1e-9:
            break
        h, a = h / sc, a * sc
    g = Z @ a
    G = lagmat(g, LAGS)
    pred = G @ h
    c0 = float(np.mean(y[valid] - pred[valid]))
    r2 = 1 - np.var(y[valid] - pred[valid] - c0) / np.var(y[valid])
    return h, a, g, float(r2)


def valid_mask(y, sqi, F):
    finite_rows = np.isfinite(F).all(axis=1).astype(float)
    cover = np.convolve(finite_rows, np.ones(LAGS.size), mode="full")[LAGS[-1]: LAGS[-1] + len(y)] / LAGS.size
    cover = np.concatenate([cover, np.zeros(max(0, len(y) - cover.size))])[: len(y)]
    return np.isfinite(y) & (y > 0) & (y <= 100) & np.isfinite(sqi) & (sqi >= 50) & (cover >= 0.8)


def scan_case(root, c):
    L = load(root, c)
    if L is None:
        return {"case": c, "skip": "no BIS EEG or index"}
    grid, y, sqi, sr, n, phase = L
    F = features(grid, n)
    valid = valid_mask(y, sqi, F)
    if valid.sum() < 1800:
        return {"case": c, "skip": "under 30 min valid", "valid_s": int(valid.sum())}
    h, a, g, r2 = fit_case(F, y, valid)
    out = {"case": c, "feature_anchor": "window_end", "valid_s": int(valid.sum()), "r2": round(r2, 3), "index_phase_s": round(phase, 3),
           "a": dict(zip(NAMES, np.round(a, 3).tolist())), "h": np.round(h, 4).tolist(), **describe(h, LAGS)}
    if h.sum() > 0:
        out.update(fast_part(h, LAGS))
    G = lagmat(g, LAGS)
    blocks = []
    for s0 in range(0, n, BLOCK):
        m = valid.copy()
        m[:s0] = False
        m[s0 + BLOCK:] = False
        if m.sum() < 900 or np.std(y[m]) < 3:
            continue
        hb, _ = ridge_h(G[m], y[m], 50.0)
        if hb.sum() <= 0:
            continue
        d = describe(hb / hb.sum(), LAGS)
        fp = fast_part(hb, LAGS, d_fixed=out.get("fast_d"))
        blocks.append({"h0": s0 / 3600, "n": int(m.sum()), **{k: d.get(k) for k in ("centroid", "q50", "ema_W", "ema_d")},
                       "fast_L": fp["fast_L"], "fast_share": fp["fast_share"]})
    out["blocks"] = blocks
    supp = np.nan_to_num(F[:, NAMES.index("supp")])
    msr = np.isfinite(sr) & np.isfinite(F).all(axis=1)
    if msr.sum() >= 1800 and np.sum(sr[msr] > 5) >= 120:
        hs, _ = ridge_h(lagmat(supp, LAGS)[msr], sr[msr], 50.0)
        if hs.sum() > 0:
            out["sr_check"] = {k: v for k, v in describe(hs / hs.sum(), LAGS).items() if k in ("centroid", "q10", "q50", "q90", "box_L", "box_d", "ema_W", "ema_d")}
    return out


def selftest(root, c):
    """Recover a known kernel from synthetic index values built from this case's real EEG features."""
    grid, y, sqi, sr, n, _ = load(root, c)
    F = features(grid, n)
    valid = valid_mask(y, sqi, F)
    rng = np.random.default_rng(0)
    res = []
    for W, d in ((15, 3), (30, 8), (45, 0)):
        Z = np.nan_to_num((F - np.nanmean(F[valid], 0)) / (np.nanstd(F[valid], 0) + 1e-9))
        a = rng.normal(size=Z.shape[1])
        g = Z @ a
        ysyn = 50 + 4 * (lagmat(g, LAGS) @ ema_kernel(W, d, LAGS)) + rng.normal(0, 2, n)
        h, _, _, r2 = fit_case(F, ysyn, valid)
        dsc = describe(h, LAGS)
        res.append({"true_W": W, "true_d": d, "fit_W": dsc["ema_W"], "fit_d": dsc["ema_d"], "centroid": dsc["centroid"],
                    "true_centroid": round(float((LAGS * ema_kernel(W, d, LAGS)).sum()), 2), "r2": round(r2, 3)})
    print(json.dumps(res, indent=1))


def agg(paths, out):
    R = [json.loads(l) for p in paths for l in open(p)]
    ok = [r for r in R if "centroid" in r]
    pick = lambda k, rs=ok: np.array([r[k] for r in rs if r.get(k) is not None], float)  # noqa: E731
    pct = lambda v: np.percentile(v, [10, 25, 50, 75, 90]).round(2).tolist() if v.size else []  # noqa: E731
    good = [r for r in ok if r["r2"] >= 0.5 and r["neg_mass"] <= 0.1]
    S = {"cases": len(R), "fitted": len(ok), "good_fit": len(good),
         "skipped": {k: sum(1 for r in R if r.get("skip") == k) for k in {r.get("skip") for r in R if r.get("skip")}},
         "errors": sum(1 for r in R if "error" in r),
         "r2_p10_25_50_75_90": pct(pick("r2")), "neg_mass_p50": float(np.median(pick("neg_mass"))) if ok else None}
    for k in ("centroid", "q10", "q50", "q90", "peak", "ema_W", "ema_d", "box_L", "box_d", "fast_L", "fast_d",
              "fast_share", "slow_share"):
        S[f"good_{k}_p10_25_50_75_90"] = pct(pick(k, good))
    fl = pick("fast_L", good)
    S["good_fast_L_hist"] = np.histogram(fl, bins=[0, 12, 14, 17, 20, 25, 28, 33, 40, 50, 61])[0].tolist()
    S["good_fast_L_bins"] = [0, 12, 14, 17, 20, 25, 28, 33, 40, 50, 61]
    S["good_fast_L_eq15"] = int(np.sum(fl == 15))
    stab = []
    for r in good:
        bl = [b["fast_L"] for b in r["blocks"] if b.get("fast_L") is not None]
        if len(bl) >= 3:
            stab.append(float(np.mean(np.abs(np.array(bl) - r["fast_L"]) <= 3)))
    S["blocks_within_3s_of_case_fast_L_p25_50_75"] = np.percentile(stab, [25, 50, 75]).round(2).tolist() if stab else []
    S["good_ema_beats_box"] = round(float(np.mean([r["ema_sse"] < r["box_sse"] for r in good])), 3) if good else None
    S["good_ema_W_hist"] = np.histogram(pick("ema_W", good), bins=[0, 10, 20, 25, 35, 40, 50, 91])[0].tolist()
    S["good_ema_W_bins"] = [0, 10, 20, 25, 35, 40, 50, 91]
    within = [np.std([b["centroid"] for b in r["blocks"] if b.get("centroid") is not None]) for r in good
              if sum(b.get("centroid") is not None for b in r["blocks"]) >= 3]
    S["within_case_sd_centroid_p50_p90"] = np.percentile(within, [50, 90]).round(2).tolist() if within else []
    between = pick("centroid", good)
    S["between_case_sd_centroid"] = round(float(np.std(between)), 2) if between.size else None
    srs = [r["sr_check"] for r in ok if "sr_check" in r]
    S["sr_check_cases"] = len(srs)
    for k in ("centroid", "q90", "box_L", "box_d"):
        S[f"sr_{k}_p25_50_75"] = np.percentile([s[k] for s in srs], [25, 50, 75]).round(1).tolist() if srs else []
    S["index_phase_s_p50"] = float(np.median(pick("index_phase_s"))) if ok else None
    S["mean_kernel_good"] = np.round(np.mean([r["h"] for r in good], axis=0), 4).tolist() if good else []
    S["lags"] = LAGS.tolist()
    Path(out).write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v for k, v in S.items() if k not in ("mean_kernel_good", "lags")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "selftest", "agg"])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--case", type=int)
    ap.add_argument("--out")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.mode == "agg":
        agg(a.inputs, a.summary)
        return
    if a.mode == "selftest":
        selftest(a.root, a.case)
        return
    done = set()
    if Path(a.out).exists():
        for line in open(a.out):
            try:
                done.add(json.loads(line)["case"])
            except Exception:  # noqa: BLE001, S112
                continue
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        try:
            r = scan_case(a.root, c)
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:300]}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
