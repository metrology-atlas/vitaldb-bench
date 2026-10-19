# SPDX-License-Identifier: MIT
"""Pause-aware Primus synchronisation: where to let the Primus offset and slope start afresh. Aggregates only.

    python tools/primus_pause_aware.py agg --sync DIR --breaks DIR [--out DIR]

--sync holds tools/primus_sync_accuracy.py parts (the CVP arm: AWP-to-CVP breath lag in 10-minute windows on the SNUADC
stamp model); --breaks holds tools/primus_co2_gaps.py parts (which AWP pauses are connection breaks, CO2 stopping too).
Each recording's lags are fitted four ways, and every window is predicted from all the others (leave one out):

  awp_offset   one slope per recording, a new offset at every AWP pause (the census model until now)
  awp_slope    a new slope and offset at every AWP pause
  brk_offset   one slope per recording, a new offset only at breaks
  brk_slope    a new slope and offset only at breaks
  kink         a new slope at every AWP pause, joined (no jump) at regular pauses, a new offset only at breaks

At the regular pauses, where CO2 carries on, the stamps keep their timing (median step -1 ms), so they need no new
offset; at breaks they step by up to seconds. Errors are compared on the windows every model can predict, and the
errors are written per window for the census (windows.npz).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

MODELS = ("awp_offset", "awp_slope", "brk_offset", "brk_slope", "kink")


def loo(t: np.ndarray, y: np.ndarray, cuts: np.ndarray, per_stretch_slope: bool) -> np.ndarray:
    """Leave-one-out errors (s); NaN where the left-out window's stretch has fewer than 2 other windows."""
    st = np.searchsorted(cuts, t)
    u = np.unique(st)
    cols = [(st == s).astype(float) for s in u]
    cols += [t * (st == s) for s in u] if per_stretch_slope else [t]
    X = np.column_stack(cols)
    err = np.full(t.size, np.nan)
    for i in range(t.size):
        m = np.ones(t.size, bool)
        m[i] = False
        if (st[m] == st[i]).sum() < 2:
            continue
        keep = np.any(X[m] != 0, axis=0)
        b = np.linalg.lstsq(X[m][:, keep], y[m], rcond=None)[0]
        err[i] = X[i, keep] @ b - y[i]
    return err


def loo_kink(t: np.ndarray, y: np.ndarray, pauses: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Leave-one-out errors of the joined model; NaN where the window's AWP stretch has fewer than 2 other windows."""
    st = np.searchsorted(pauses, t)
    sb = np.searchsorted(breaks, t)
    cols = [(sb == s).astype(float) for s in np.unique(sb)] + [t] + [np.clip(t - p, 0, None) for p in pauses]
    X = np.column_stack(cols)
    err = np.full(t.size, np.nan)
    for i in range(t.size):
        m = np.ones(t.size, bool)
        m[i] = False
        if (st[m] == st[i]).sum() < 2:
            continue
        keep = np.any(X[m] != 0, axis=0) & (np.ptp(X[m], axis=0) > 0) | (np.arange(X.shape[1]) < np.unique(sb).size)
        keep &= np.any(X[m] != 0, axis=0)
        b = np.linalg.lstsq(X[m][:, keep], y[m], rcond=None)[0]
        err[i] = X[i, keep] @ b - y[i]
    return err


def read(d: str) -> dict:
    return {r["case"]: r for p in sorted(glob.glob(str(Path(d) / "part_*.jsonl")))
            for r in map(json.loads, filter(str.strip, open(p, encoding="utf-8")))}


def pct(v):
    v = np.abs(np.asarray(v))
    return {"windows": int(v.size), "median_ms": round(float(np.median(v)) * 1000, 1),
            "p90_ms": round(float(np.percentile(v, 90)) * 1000, 1)} if v.size else None


def agg(a) -> None:
    S, B = read(a.sync), read(a.breaks)
    E = {m: [] for m in MODELS}
    near = {m: [] for m in MODELS}          # windows within 30 min after a regular (non-break) pause
    cases, mismatch, n_p, n_b = 0, 0, 0, 0
    for c, r in S.items():
        A = (r.get("arms") or {}).get("cvp")
        b = B.get(c)
        if not A or len(A["t"]) < 6 or not b or "pauses" not in b:
            continue
        pz = np.asarray(r["pauses"], float)
        if pz.size != len(b["pauses"]) or (pz.size and np.max(np.abs(pz - np.asarray(b["pauses"]))) > 0.2):
            mismatch += 1
            continue
        brk = pz[np.asarray(b["co2_break"], bool)] if pz.size else pz
        t, y = np.asarray(A["t"], float) / 3600, np.asarray(A["lag"], float)
        P, Bk = pz / 3600, brk / 3600
        errs = {"awp_offset": loo(t, y, P, False), "awp_slope": loo(t, y, P, True),
                "brk_offset": loo(t, y, Bk, False), "brk_slope": loo(t, y, Bk, True),
                "kink": loo_kink(t, y, P, Bk)}
        ok = np.all([np.isfinite(e) for e in errs.values()], axis=0)
        reg = P[~np.isin(P, Bk)]
        after_reg = np.array([np.any((tt - reg > 0) & (tt - reg < 0.5)) for tt in t]) if reg.size else np.zeros(t.size, bool)
        for m in MODELS:
            E[m].append(errs[m][ok])
            near[m].append(errs[m][ok & after_reg])
        cases += 1
        n_p += pz.size
        n_b += brk.size
    out = {"recordings": cases, "pause_lists_mismatched": mismatch, "pauses": n_p, "breaks": n_b,
           "common_windows": {m: pct(np.concatenate(E[m])) for m in MODELS},
           "windows_within_30min_after_a_regular_pause": {m: pct(np.concatenate(near[m])) for m in MODELS}}
    print(json.dumps(out, indent=1))
    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "summary.json").write_text(json.dumps(out, indent=1))
        np.savez_compressed(o / "windows.npz", **{m: np.concatenate(E[m]) for m in MODELS})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("agg")
    g.add_argument("--sync", required=True)
    g.add_argument("--breaks", required=True)
    g.add_argument("--out")
    agg(ap.parse_args())


if __name__ == "__main__":
    main()
