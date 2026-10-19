# SPDX-License-Identifier: MIT
"""Is the Primus drift a straight line between its AWP pauses? Compares models of the breath-matching lag.

    python tools/primus_stretch_models.py PARTS... [--out DIR]

Reads the window lags of tools/primus_sync_accuracy.py (arms cvp and art; times from the recording's first AWP
stamp, pauses in the same frame). For each recording with the CVP arm, four models of the lag against time, each
judged by leave-one-out error (every window predicted from all the others):

  line_common   one slope for the recording, one offset per stretch between AWP pauses (the current model)
  line_each     a slope and an offset per stretch
  quad_each     a quadratic per stretch (a line where a stretch has fewer than 5 windows)
  local         local-linear regression within each stretch (tricube weights, half-width 45 min)

Windows that straddle a pause (centre within 5 min of it) are scored separately: they mix two lags.

A better fit by a more flexible model does not prove the clock is curved: the airway-to-CVP physiology can drift
too. The two references separate them. A bend in the Primus clock moves the CVP and the arterial lag together;
a physiological change need not. So for recordings with both arms, the residuals of line_common in the two arms
are compared at matching windows (within 2.5 min), for every model: their correlation, and the share of their
variance that is common, is the part of the misfit that belongs to the Primus clock. A model that leaves little
common residual has captured the clock; what it leaves is physiology and noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

H = 45 * 60.0
STRADDLE_S = 300.0


def stretch_index(t, pauses):
    return np.searchsorted(np.asarray(pauses, float), t)


def predict(model, t, y, st, i):
    """Predict window i from all the other windows, or nan if the model cannot."""
    m = np.ones(t.size, bool)
    m[i] = False
    same = m & (st == st[i])
    if same.sum() < 2:
        return np.nan
    if model == "line_common":
        u = np.unique(st[m])
        X = np.column_stack([t[m] / 3600] + [(st[m] == s).astype(float) for s in u])
        b = np.linalg.lstsq(X, y[m], rcond=None)[0]
        k = list(u).index(st[i]) if st[i] in u else None
        return float(b[0] * t[i] / 3600 + b[1 + k]) if k is not None else np.nan
    tt, yy = t[same] / 3600, y[same]
    if model == "line_each" or (model == "quad_each" and same.sum() < 5):
        b = np.polyfit(tt, yy, 1)
        return float(np.polyval(b, t[i] / 3600))
    if model == "quad_each":
        b = np.polyfit(tt, yy, 2)
        return float(np.polyval(b, t[i] / 3600))
    if model == "local":
        d = np.abs(t[same] - t[i]) / H
        w = np.where(d < 1, (1 - d ** 3) ** 3, 0.0)
        if (w > 0).sum() < 3:
            return np.nan
        X = np.column_stack([np.ones(tt.size), tt - t[i] / 3600])
        Wh = np.sqrt(w)
        b = np.linalg.lstsq(X * Wh[:, None], yy * Wh, rcond=None)[0]
        return float(b[0])
    raise ValueError(model)


MODELS = ("line_common", "line_each", "quad_each", "local")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    errs = {m: [] for m in MODELS}
    strad = {m: [] for m in MODELS}
    resid_pairs = {m: [] for m in MODELS}
    n_rec = 0
    for p in a.parts:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            A = (r.get("arms") or {}).get("cvp")
            if not A or len(A["t"]) < 6:
                continue
            n_rec += 1
            t, y, pa = np.array(A["t"], float), np.array(A["lag"], float), np.array(r["pauses"], float)
            st = stretch_index(t, pa)
            near = np.array([np.min(np.abs(x - pa)) < STRADDLE_S if pa.size else False for x in t])
            for m in MODELS:
                e = np.array([predict(m, t, y, st, i) - y[i] for i in range(t.size)])
                errs[m] += list(np.abs(e[~near & np.isfinite(e)]))
                strad[m] += list(np.abs(e[near & np.isfinite(e)]))
            B = (r.get("arms") or {}).get("art")
            if B and len(B["t"]) >= 6:
                ta, yb = np.array(B["t"], float), np.array(B["lag"], float)
                sa = stretch_index(ta, pa)
                for m in MODELS:
                    ec = np.array([predict(m, t, y, st, i) - y[i] for i in range(t.size)])
                    ea = np.array([predict(m, ta, yb, sa, i) - yb[i] for i in range(ta.size)])
                    for i in range(t.size):
                        j = np.argmin(np.abs(ta - t[i]))
                        if abs(ta[j] - t[i]) <= 150 and np.isfinite(ec[i]) and np.isfinite(ea[j]) and not near[i]:
                            if abs(ec[i]) < 1.5 and abs(ea[j]) < 1.5:      # leave out breath-identity slips
                                resid_pairs[m].append((ec[i], ea[j], r["case"]))
    out = {"recordings": n_rec, "models": {}}
    for m in MODELS:
        e = np.array(errs[m]); s = np.array(strad[m])
        out["models"][m] = {"windows": int(e.size), "abs_error_ms_p50_p90": [round(float(np.percentile(e, q)) * 1000, 1) for q in (50, 90)],
                            "share_over_200ms": round(float(np.mean(e > 0.2)), 3),
                            "straddling_abs_error_ms_p50": round(float(np.median(s)) * 1000, 1) if s.size else None}
    out["two_references"] = {}
    for m, pairs in resid_pairs.items():
        if not pairs:
            continue
        R = np.array([p[:2] for p in pairs], float)
        c = float(np.corrcoef(R[:, 0], R[:, 1])[0, 1])
        cov = float(np.cov(R[:, 0], R[:, 1])[0, 1])
        out["two_references"][m] = {"matched_windows": int(len(R)), "recordings": len({p[2] for p in pairs}),
                                    "residual_correlation_cvp_art": round(c, 3),
                                    "common_sd_ms": round(float(np.sqrt(max(cov, 0))) * 1000, 1),
                                    "cvp_residual_sd_ms": round(float(np.std(R[:, 0])) * 1000, 1),
                                    "share_of_cvp_residual_variance_common": round(max(cov, 0) / float(np.var(R[:, 0])), 3)}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
