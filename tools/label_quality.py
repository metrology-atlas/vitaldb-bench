# SPDX-License-Identifier: MIT
"""Signal quality and a shape test at each labelled sawtooth event, against the manual labels (metrology-atlas study
ppg-sawtooth-slips). Uses the waveforms the labelling page shows (tools/sawtooth_char.py label-waves: ECG II at
250 Hz, PLETH and ART at 125 Hz, +/- 62 s, with R peaks and per-beat fiducials) and the verdicts of label-eval.

    python tools/label_quality.py --waves $W/sawtooth_char/labels/waves --verdicts $W/label_eval_X/label_verdicts.json \
        --out $W/label_quality [--half 45 --guard 3]

Per event, the beats from --guard to --half seconds before the event ("before") and after it ("after"):

* **PPG shape shift.** The median R-aligned PLETH pulse (0.1-1.0 s after R) on each side, normalised; the shift of
  the "after" pulse against the "before" one that matches best (-80 to +80 ms, 1 ms steps), the correlation there and
  at no shift. A frame slip moves the pulse by about one frame (33 ms) without changing its shape: best shift near
  +/-33 ms, correlation there near 1. A change of shape matches poorly at every shift.
* **Fiducial agreement.** The step (median after minus median before) in R-to-foot, R-to-upstroke and R-to-peak; their
  spread (largest minus smallest). A slip moves all three alike.
* **ECG shape.** The median R-aligned QRS (+/- 0.12 s) on each side: correlation at no shift and the best shift (an R
  peak that moved within the complex shows as a shift).
* **Beat quality**, per side: each beat's correlation with that side's median beat (ECG and PLETH; the share under
  0.9), the share of irregular R-R intervals (more than 20% from the side's median), the share of beats without a
  PPG foot, and the share of samples at the channel's lowest or highest value (clipping).

A rule, fitted on half the events and scored on the other half (events alternately assigned by id): "can't
judge" when the share of unlike beats on either side exceeds Q or irregular R-R exceeds I; otherwise a slip when the
pulse shifts by 20-45 ms with a match of at least C after the shift and the fiducial steps spread by at most F.
Q, I, C and F are chosen on the first half to maximise slips called right minus five times the false ones (no-step
events called slips); the held-out half gives the scores.

Writes events.csv (one row per labelled event, with its stratum and verdict) and summary.json (each measure's
quartiles by verdict group: real slip, no step, can't tell; and the rule's fit and held-out scores).
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path

import numpy as np

REAL = ("ppg_slip_up", "ppg_step_up_unconfirmed", "ppg_slip_down", "ppg_step_down_unconfirmed")


def channel(d: dict | None):
    if not d:
        return None, None
    x = np.frombuffer(base64.b64decode(d["b64"]), dtype="<i2").astype(float) * d["scale"] + d["offset"]
    t = d["t0"] + np.arange(x.size) / d["fs"]
    return t, x


def segments(t, x, centres, a, b, fs):
    n = int(round((b - a) * fs))
    out = []
    for c in centres:
        i0 = int(round((c + a - t[0]) * fs))
        if 0 <= i0 and i0 + n <= x.size:
            out.append(x[i0:i0 + n])
    return np.array(out)


def z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0


def best_shift(before, after, fs, max_ms=80):
    """Shift (ms) of `after` against `before` that maximises the correlation (positive: after is later)."""
    tt = np.arange(before.size) / fs
    fine = np.arange(0, tt[-1], 0.001)
    bb = z(np.interp(fine, tt, before))
    aa = np.interp(fine, tt, after)
    best, bc, c0 = 0, -2.0, None
    for s in range(-max_ms, max_ms + 1):
        # after(t) ~ before(t - s): compare after shifted back by s
        sh = np.interp(fine + s / 1000.0, fine, aa, left=np.nan, right=np.nan)
        ok = np.isfinite(sh)
        if ok.sum() < 0.6 * fine.size:
            continue
        c = float(np.corrcoef(bb[ok], sh[ok])[0, 1])
        if s == 0:
            c0 = c
        if c > bc:
            best, bc = s, c
    return best, bc, c0


def side_quality(t, x, centres, a, b, fs):
    seg = segments(t, x, centres, a, b, fs)
    if len(seg) < 5:
        return None, None
    tpl = np.median(seg, axis=0)
    cc = np.array([np.corrcoef(s, tpl)[0, 1] if s.std() > 0 else 0.0 for s in seg])
    return float(np.mean(cc < 0.9)), tpl


def clipping(x):
    if x is None or not x.size:
        return None
    lo, hi = x.min(), x.max()
    return float(np.mean((x == lo) | (x == hi)))


def event_measures(w: dict, half: float, guard: float) -> dict:
    te, xe = channel(w.get("ecg"))
    tp, xp = channel(w.get("ppg"))
    beats = w.get("beats") or []
    out = {}
    for side, lo, hi in (("before", -half, -guard), ("after", guard, half)):
        bs = [b for b in beats if b.get("r") is not None and lo <= b["r"] <= hi]
        r = np.array([b["r"] for b in bs])
        out[f"{side}_beats"] = int(r.size)
        if r.size >= 3:
            rr = np.diff(r)
            out[f"{side}_rr_irregular"] = float(np.mean(np.abs(rr / np.median(rr) - 1) > 0.2))
        out[f"{side}_no_foot"] = float(np.mean([b.get("foot") is None for b in bs])) if bs else None
        for fid in ("foot", "up", "peak"):
            v = [b[fid] - b["r"] for b in bs if b.get(fid) is not None]
            out[f"{side}_{fid}_ms"] = float(np.median(v) * 1000) if len(v) >= 3 else None
        if te is not None and r.size >= 5:
            q, tpl = side_quality(te, xe, r, -0.12, 0.12, 250.0)
            out[f"{side}_ecg_bad"], out[f"_{side}_ecg_tpl"] = q, tpl
            sel = (te >= lo) & (te <= hi)
            out[f"{side}_ecg_clip"] = clipping(xe[sel])
        if tp is not None and r.size >= 5:
            q, tpl = side_quality(tp, xp, r, 0.1, 1.0, 125.0)
            out[f"{side}_ppg_bad"], out[f"_{side}_ppg_tpl"] = q, tpl
            sel = (tp >= lo) & (tp <= hi)
            out[f"{side}_ppg_clip"] = clipping(xp[sel])
    steps = [out[f"after_{f}_ms"] - out[f"before_{f}_ms"] for f in ("foot", "up", "peak")
             if out.get(f"after_{f}_ms") is not None and out.get(f"before_{f}_ms") is not None]
    out["fid_step_ms"] = float(np.median(steps)) if steps else None
    out["fid_spread_ms"] = float(max(steps) - min(steps)) if len(steps) >= 2 else None
    if out.get("_before_ppg_tpl") is not None and out.get("_after_ppg_tpl") is not None:
        s, c, c0 = best_shift(out["_before_ppg_tpl"], out["_after_ppg_tpl"], 125.0)
        out.update(ppg_shift_ms=s, ppg_corr_best=c, ppg_corr_zero=c0)
    if out.get("_before_ecg_tpl") is not None and out.get("_after_ecg_tpl") is not None:
        s, c, c0 = best_shift(out["_before_ecg_tpl"], out["_after_ecg_tpl"], 250.0, max_ms=40)
        out.update(ecg_shift_ms=s, ecg_corr_best=c, ecg_corr_zero=c0)
    for k in [k for k in out if k.startswith("_")]:
        del out[k]
    bad = [out.get(k) for k in ("before_ecg_bad", "after_ecg_bad", "before_ppg_bad", "after_ppg_bad")]
    out["worst_bad"] = max([b for b in bad if b is not None], default=None)
    irr = [out.get(k) for k in ("before_rr_irregular", "after_rr_irregular")]
    out["worst_rr_irregular"] = max([b for b in irr if b is not None], default=None)
    clip = [out.get(k) for k in ("before_ecg_clip", "after_ecg_clip", "before_ppg_clip", "after_ppg_clip")]
    out["worst_clip"] = max([b for b in clip if b is not None], default=None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waves", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--half", type=float, default=45.0)
    ap.add_argument("--guard", type=float, default=3.0)
    a = ap.parse_args()
    V = json.loads(Path(a.verdicts).read_text())
    rows = []
    for e, v in sorted(V.items()):
        f = Path(a.waves) / f"{e}.json"
        if not f.exists():
            continue
        m = event_measures(json.loads(f.read_text()), a.half, a.guard)
        grp = "real" if v["verdict"] in REAL else ("none" if v["verdict"] == "none" else
                                                     ("unclear" if v["verdict"] in ("unclear", "artefact") else "other"))
        rows.append({"event": e, "stratum": v["stratum"], "verdict": v["verdict"], "group": grp, **m})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    measures = ["ppg_shift_ms", "ppg_corr_best", "ppg_corr_zero", "fid_step_ms", "fid_spread_ms", "ecg_shift_ms",
                "ecg_corr_zero", "worst_bad", "worst_rr_irregular", "worst_clip"]
    S = {"events": len(rows), "groups": {}}
    for g in ("real", "none", "unclear", "other"):
        sel = [r for r in rows if r["group"] == g]
        if not sel:
            continue
        S["groups"][g] = {"n": len(sel)}
        for m in measures:
            vals = np.array([abs(r[m]) if m in ("ppg_shift_ms", "fid_step_ms", "ecg_shift_ms") else r[m]
                             for r in sel if r.get(m) is not None], float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                S["groups"][g][m] = [round(float(x), 3) for x in np.percentile(vals, [25, 50, 75])]
    S["rule"] = fit_rule(rows)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


def call(r: dict, Q: float, I: float, C: float, F: float) -> str:
    g = lambda k: r.get(k) if r.get(k) is not None and np.isfinite(r.get(k)) else None
    wb, ri = g("worst_bad"), g("worst_rr_irregular")
    if wb is None or wb > Q or (ri is not None and ri > I):
        return "cannot judge"
    sh, cb, sp = g("ppg_shift_ms"), g("ppg_corr_best"), g("fid_spread_ms")
    if sh is not None and 20 <= abs(sh) <= 45 and cb is not None and cb >= C and (sp is None or sp <= F):
        return "slip"
    return "no slip"


def table(rows, p):
    from collections import Counter
    c = Counter((r["group"], call(r, *p)) for r in rows)
    return {g: {k: c.get((g, k), 0) for k in ("slip", "no slip", "cannot judge")} for g in ("real", "none", "unclear")}


def fit_rule(rows) -> dict:
    ids = sorted(r["event"] for r in rows)
    first = {e for i, e in enumerate(ids) if i % 2 == 0}
    fit = [r for r in rows if r["event"] in first]
    held = [r for r in rows if r["event"] not in first]
    best, score = None, -1e9
    for Q in (0.2, 0.3, 0.4, 0.5):
        for I in (0.1, 0.15, 0.2, 0.3):
            for C in (0.95, 0.97, 0.98, 0.99):
                for F in (5.0, 8.0, 15.0, 1e9):
                    t = table(fit, (Q, I, C, F))
                    sc = t["real"]["slip"] - 5 * t["none"]["slip"]
                    if sc > score:
                        best, score = (Q, I, C, F), sc
    Q, I, C, F = best
    return {"params": {"bad_beats_max": Q, "rr_irregular_max": I, "match_min": C, "fiducial_spread_max_ms": F if F < 1e8 else None},
            "fit_half": {"events": len(fit), **table(fit, best)}, "held_out_half": {"events": len(held), **table(held, best)},
            "all": table(rows, best)}


if __name__ == "__main__":
    main()
