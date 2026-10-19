# SPDX-License-Identifier: MIT
"""Are the Primus numerics stamped on the Primus waveform's clock, or by the PC? Per recording, this machine only.

    python tools/primus_numeric_clock.py scan --root R --cases 1,2,3 --out part.jsonl [--time-budget 150]
    python tools/primus_numeric_clock.py agg part*.jsonl [--probe probe.jsonl] [--out DIR]

Breath matching puts the Primus waveform stamps on a drift of about +104 ppm against the recorder PC
(tools/primus_drift_probe.py). If the Primus numerics (PIP_MBAR) were stamped by the PC on arrival, the delay from
the breath a PIP value describes (its AWP peak, on the waveform's stamps) to the PIP stamp would change by about
0.37 s per hour; if both are stamped on the Primus's clock, it would not change.

Per recording: breath onsets and the AWP peak of each breath (on the waveform's own stamps); for each PIP record
at stamp t and a candidate delay d (0-12 s, 0.05 s steps), the prediction is the peak of the last breath that ended
by t - d; per 30-minute block the delay that matches most records (|PIP - peak| <= 0.6 mbar, PIP is whole mbar).
The slope of the block delays against time, within the stretches between AWP pauses (one offset per stretch) and
over the whole recording, is the numeric clock against the waveform clock (s/h; ppm = s/h / 0.0036).
agg: the slopes' distribution, and with --probe their per-recording comparison with the breath-matching drift.
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
from vent_sync_pilot import breath_onsets  # noqa: E402
from pat import sample_times  # noqa: E402
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: E402

D = np.arange(0.0, 12.0001, 0.05)
BLOCK_S, MIN_REC, TOL = 1800.0, 60, 0.6


def block_delay(pt, pv, end, peak):
    """Delay d maximising the share of PIP records equal (within TOL) to the last breath peak ended by t - d."""
    best, share = np.nan, -1.0
    for d in D:
        k = np.searchsorted(end, pt - d, side="right") - 1
        ok = k >= 0
        s = float(np.mean(np.abs(peak[k[ok]] - pv[ok]) <= TOL)) if ok.sum() >= MIN_REC else -1.0
        if s > share:
            best, share = float(d), s
    return best, share


def scan_case(root, c):
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names={"Primus/AWP", "Primus/PIP_MBAR"})
    aw, pip = v.tracks.get("Primus/AWP"), v.tracks.get("Primus/PIP_MBAR")
    if aw is None or pip is None or not aw.record_times.size or not pip.record_times.size:
        return {"case": c, "skip": "no AWP or PIP"}
    ta = sample_times(aw)
    xa = np.nan_to_num(aw.physical().astype(np.float64)[: ta.size])
    on = breath_onsets(ta, xa, float(aw.srate))
    if on.size < 200:
        return {"case": c, "skip": "too few breaths"}
    # breath k: from onset k to onset k+1; its peak and its end (the next onset)
    idx = np.searchsorted(ta, on)
    peak = np.array([xa[a:b].max() if b > a else np.nan for a, b in zip(idx[:-1], idx[1:])])
    end = on[1:]
    ok = np.isfinite(peak) & (np.diff(on) < 12)
    peak, end = peak[ok], end[ok]
    pt = np.asarray(pip.record_times, float)
    pv = np.asarray(pip.values if hasattr(pip, "values") else pip.physical(), float)[: pt.size]
    m = np.isfinite(pv) & (pv >= 3) & (pv <= 80) & (pt > end[0] + 15) & (pt < end[-1])
    pt, pv = pt[m], pv[m]
    if pt.size < 2 * MIN_REC:
        return {"case": c, "skip": "too few PIP records"}
    z = float(pt[0])
    rows = []
    for b0 in np.arange(pt[0], pt[-1], BLOCK_S):
        s = (pt >= b0) & (pt < b0 + BLOCK_S)
        if s.sum() < MIN_REC:
            continue
        d, sh = block_delay(pt[s], pv[s], end, peak)
        rows.append([round(float(np.median(pt[s]) - z), 1), d, round(sh, 3), int(s.sum())])
    g = np.diff(ta)
    pauses = (ta[1:][g > 5] - z).round(1).tolist()
    return {"case": c, "blocks": rows, "pauses": pauses, "pip_interval_s": round(float(np.median(np.diff(pt))), 2)}


def slopes(r, min_share=0.5):
    B = np.array([b for b in r.get("blocks", []) if b[2] >= min_share], float)
    if len(B) < 4:
        return None
    t, d = B[:, 0], B[:, 1]
    whole = np.polyfit(t / 3600, d, 1)[0]
    st = np.searchsorted(np.array(r["pauses"] or [], float), t)
    u = np.unique(st)
    X = np.column_stack([t / 3600] + [(st == s).astype(float) for s in u])
    beta = np.linalg.lstsq(X, d, rcond=None)[0]
    within = beta[0] if len(t) > len(u) + 1 else np.nan
    return {"whole_s_h": float(whole), "within_s_h": float(within), "blocks": int(len(t)),
            "median_share": float(np.median(B[:, 2])), "delay_median_s": float(np.median(d))}


def scan(a):
    cases = [int(x) for x in a.cases.split(",") if x]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in open(a.out, encoding="utf-8") if l.strip()}
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as f:
        for c in cases:
            if c in done:
                continue
            if a.time_budget and time.time() - t0 > a.time_budget:
                break
            try:
                r = scan_case(a.root, c)
            except Exception as e:  # keep going; record the failure
                r = {"case": c, "error": repr(e)[:200]}
            f.write(json.dumps(r) + "\n")
            f.flush()


def agg(a):
    R = [json.loads(l) for p in a.parts for l in open(p, encoding="utf-8") if l.strip()]
    S = {r["case"]: s for r in R if (s := slopes(r))}
    w = np.array([s["whole_s_h"] for s in S.values()])
    wi = np.array([s["within_s_h"] for s in S.values() if np.isfinite(s["within_s_h"])])
    pct = lambda x: np.percentile(x, [10, 25, 50, 75, 90]).round(3).tolist() if len(x) else None
    out = {"recordings_read": len(R), "with_slope": len(S),
           "skipped": {k: sum(1 for r in R if r.get("skip") == k) for k in {r.get("skip") for r in R if r.get("skip")}},
           "errors": sum(1 for r in R if "error" in r),
           "slope_whole_s_per_h_p10_25_50_75_90": pct(w), "slope_within_s_per_h_p10_25_50_75_90": pct(wi),
           "slope_within_ppm_median": round(float(np.median(wi)) / 0.0036, 1) if len(wi) else None,
           "pip_delay_median_s": round(float(np.median([s["delay_median_s"] for s in S.values()])), 2) if S else None,
           "match_share_median": round(float(np.median([s["median_share"] for s in S.values()])), 3) if S else None,
           "expected_if_pc_stamped_s_per_h": round(104e-6 * 3600, 3)}
    if a.probe:
        P = {json.loads(l)["case"]: json.loads(l) for l in open(a.probe, encoding="utf-8")}
        pairs = [(S[c]["within_s_h"], P[c]["slope_common_ppm"] * 3600e-6) for c in S if c in P and np.isfinite(S[c]["within_s_h"])]
        if len(pairs) >= 5:
            x, y = np.array(pairs).T
            out["vs_breath_matching"] = {"recordings": len(pairs), "r": round(float(np.corrcoef(x, y)[0, 1]), 3),
                                         "numeric_slope_s_h_median": round(float(np.median(x)), 3),
                                         "breath_drift_s_h_median": round(float(np.median(y)), 3)}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("--root", required=True); s.add_argument("--cases", required=True)
    s.add_argument("--out", required=True); s.add_argument("--time-budget", type=float, default=0)
    g = sp.add_parser("agg"); g.add_argument("parts", nargs="+"); g.add_argument("--probe"); g.add_argument("--out")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
