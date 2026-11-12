# SPDX-License-Identifier: MIT
"""Is the recompute test's ~1 s HR lag a delay, or a sign that our recomputation and the monitor disagree? Each
recording's lag against how exactly the atlas's HR rule reproduces that recording's monitor values.

    python examples/recompute_lag_vs_exact.py --lags LAG_PARTS... --derivation DERIV_PARTS... --out DIR

LAG_PARTS: tools/numeric_recompute_lag.py per-recording output (each recording's lag read alone with its peak()).
DERIV_PARTS: tools/hr_derivation.py per-recording output; its candidate rmean|8|15 gives the share of monitor values
the rule reproduces exactly at the recording's best delay. Writes OUT/summary.json: the Spearman correlation, and the
median lag by tertile of the exact share. Aggregates only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from numeric_recompute_lag import peak  # noqa: E402


def rows(paths):
    out = {}
    for p in paths:
        for l in Path(p).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                out[r["case"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lags", nargs="+", required=True)
    ap.add_argument("--derivation", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    L, Dv = rows(a.lags), rows(a.derivation)
    lag, ex = [], []
    for c, r in Dv.items():
        e = (r.get("cands") or {}).get("rmean|8|15")
        if not e or c not in L or not L[c].get("hr"):
            continue
        try:
            x = peak([L[c]], "hr")
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if np.isfinite(x):
            lag.append(x)
            ex.append(e[2])
    lag, ex = np.array(lag), np.array(ex)
    rk = lambda v: np.argsort(np.argsort(v))
    rho = float(np.corrcoef(rk(lag), rk(ex))[0, 1])
    q = np.quantile(ex, [1 / 3, 2 / 3])
    S = {"recordings": int(lag.size), "spearman_lag_vs_exact": round(rho, 3),
         "exact_tertile_edges": [round(float(v), 4) for v in q], "by_exact_tertile": {}}
    for name, k in (("low", ex < q[0]), ("mid", (ex >= q[0]) & (ex < q[1])), ("high", ex >= q[1])):
        S["by_exact_tertile"][name] = {"n": int(k.sum()), "median_exact": round(float(np.median(ex[k])), 4),
                                       "median_lag_s": round(float(np.median(lag[k])), 3)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
