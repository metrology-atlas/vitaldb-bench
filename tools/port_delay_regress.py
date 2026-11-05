# SPDX-License-Identifier: MIT
"""Regression of physiosync 0.3.0's fit_ecg_ppg_delay against what it replaces, per recording (real beats):

    python tools/port_delay_regress.py --old $W/slip_rebuild/dev/ecg_ppg_0_2_0.py --scan $W/sawtooth_char/scan_full \
        --quality $W/slip_rebuild/quality --cases $W/slip_rebuild/cases.json --rates $W/sawtooth_stats/recordings.csv \
        --modes $W/results/sync_full/snuadc_mode_label.json --rule $W/label_quality_final/summary.json \
        --out $W/slip_rebuild/port_delay_regress.jsonl [--part 0 --of 2 --time-budget 150]
    python tools/port_delay_regress.py ... --out $W/slip_rebuild/port_delay_regress.jsonl --summary

1. method="detect" (the default) against physiosync 0.2.0's ecg_ppg.py (--old, saved with
   `git show d2fc986:physiosync/ecg_ppg.py`): the same call (R peaks and upstrokes paired, frame 1/30 s, the arterial
   upstroke as reference) must give the same model, every field of 0.2.0 equal.
2. method="rebuilt" against tools/slip_tiers.py: its sawtooth part D(t) minus its median must equal
   slip_tiers.correction minus its median (within 1 microsecond, on a 1 s grid over the stretch), and the tier at each
   grid time must equal slip_tiers.teeth's.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import slip_rebuild as proto  # noqa: E402
import slip_tiers  # noqa: E402
from physiosync import SlipOptions, fit_ecg_ppg_delay, find_slips  # noqa: E402

FIDS = ("foot", "up", "peak", "down50")
NEW = ("method", "tooth_steps", "direction", "slips", "uncertain", "drift")


def same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def tier_at(x: np.ndarray, rows: list[dict], k0: str = "t0", k1: str = "t1") -> list[str]:
    return [next((r["tier"] for r in rows if r[k0] <= v <= r[k1]), "?") for v in x]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("old", "scan", "quality", "cases", "rates", "modes", "rule", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=1e9)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.summary:
        rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        print(json.dumps({"recordings": len(rows), "errors": sum(1 for r in rows if "error" in r),
                          "detect_equal": sum(1 for r in rows if r.get("detect") == []),
                          "rebuilt_equal": sum(1 for r in rows if r.get("rebuilt") == []),
                          "differences": dict(Counter(d for r in rows for k in ("detect", "rebuilt") for d in r.get(k, []))),
                          "max_abs_diff_us": max((r.get("max_abs_diff_us", 0) for r in rows), default=None),
                          "sawtooth_fitted": sum(1 for r in rows if r.get("sawtooth")),
                          "tiers_s": dict(sum((Counter(r.get("tier_time_s", {})) for r in rows), Counter()))}, indent=1))
        return
    spec = importlib.util.spec_from_file_location("ecg_ppg_old", a.old)
    old = importlib.util.module_from_spec(spec)
    sys.modules["ecg_ppg_old"] = old  # dataclasses look the module up
    spec.loader.exec_module(old)
    rule = json.loads(Path(a.rule).read_text())["rule"]["params"]
    modes = json.loads(Path(a.modes).read_text())
    rates = proto.load_rates(a.rates, modes)
    cases = [c for i, c in enumerate(json.loads(Path(a.cases).read_text())["cases"]) if i % a.of == a.part]
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t0 = time.time()
    with out.open("a") as fh:
        for c in cases:
            if c in done or time.time() - t0 > a.time_budget:
                continue
            row = {"case": c}
            try:
                z = np.load(Path(a.scan) / f"{c:04d}.npz")
                Qz = np.load(Path(a.quality) / f"{c:04d}.npz")
                Q = {k: Qz[k] for k in Qz.files if not k.startswith("inj_")}
                F = {j: z[j] for j in FIDS}
                r, up, art = z["r_t"], z["up"], z["art_t"]
                # 1. detect, against 0.2.0
                ok = np.isfinite(up)
                kw = dict(adopted_pat_s=0.25, frame_s=1 / 30, matched=True, ref_t=art[ok])
                d = []
                try:
                    mo = asdict(old.fit_ecg_ppg_delay(r[ok], up[ok], **kw))
                except ValueError as ex:
                    mo = f"ValueError: {ex}"
                try:
                    mn = asdict(fit_ecg_ppg_delay(r[ok], up[ok], **kw))
                    mn = {k: v for k, v in mn.items() if k not in NEW}
                except ValueError as ex:
                    mn = f"ValueError: {ex}"
                if isinstance(mo, dict) and isinstance(mn, dict):
                    d += [f"detect field {k}" for k in mo if not same(mo[k], mn.get(k))]
                    d += [f"detect new field {k}" for k in mn if k not in mo]
                elif mo != mn:
                    d.append("detect error differs")
                row["detect"] = d
                # 2. rebuilt, against slip_tiers
                so = SlipOptions(rule_params=rule, prior_by_sign=proto.prior_range(rates, str(z["port"]) or "?",
                                                                                   modes.get(str(c), "?"), c))
                m = fit_ecg_ppg_delay(r, up, adopted_pat_s=0.25, matched=True, ref_t=art, method="rebuilt",
                                      fiducials=F, beat_quality=Q, slip_options=so)
                R = find_slips(r, F, Q, art_t=art, opts=so)
                g = np.arange(m.t_start, m.t_end, 1.0)
                Dn = m._saw(g)
                Do = slip_tiers.correction(g, R)
                diff = np.abs((Dn - np.median(Dn)) - (Do - np.median(Do)))
                e = []
                row["max_abs_diff_us"] = round(float(diff.max()) * 1e6, 4) if g.size else 0.0
                if g.size and diff.max() > 1e-6:
                    e.append("delay")
                if g.size and tier_at(g, m.quality_intervals(), "start", "end") != tier_at(g, slip_tiers.teeth(R)):
                    e.append("tiers")
                if bool(R.get("sawtooth")) != m.qc["sawtooth_fitted"]:
                    e.append("sawtooth")
                row["rebuilt"] = e
                row["sawtooth"] = m.qc["sawtooth_fitted"]
                row["tier_time_s"] = {k: round(v, 1) for k, v in m.qc["tier_time_s"].items() if v}
            except Exception as ex:  # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    print(json.dumps({"part": a.part, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
