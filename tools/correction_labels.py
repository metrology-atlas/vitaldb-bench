# SPDX-License-Identifier: MIT
"""Yes/no review of the rebuilt sawtooth correction (metrology-atlas study ppg-sawtooth): a blinded, stratified sample
of teeth from a tools/slip_tiers.py run, each shown as the measured ECG-PPG delay before and after the correction,
and the reviewer's verdicts scored by tier.

    python tools/correction_labels.py sample --fit $W/slip_rebuild/fit_v19.jsonl --teeth $W/slip_tiers_v19/teeth.csv \
        --scan $W/sawtooth_char/scan_full --out $W/correction_labels [--seed 1]
    python tools/correction_labels.py eval --key $W/correction_labels/key.json --answers ANSWERS.json \
        --out $W/correction_labels/eval

**sample**: per tier, at random and at most one tooth per recording: A 30, B 20, C all (up to 10), U 20, I 25, N 8
(no correction: the control). Each item is the tooth (shaded) with up to 10 min either side, as 15-beat medians of the
measured delay and of the corrected delay (shifted to the same median over the window), and the times where the
correction steps (a band for a slip placed only within an interval). The order is shuffled and the items carry no
tier. Writes items.json (what the review page shows) and key.json (item -> recording, tooth, tier, reason).

**check**: every step the reviewer drew as missed, against the beats: the step in ECG-PPG, ECG-ART and ART-PPG (median
of the good beats 3-30 s after minus 3-30 s before the drawn time, the boundary searched within +/- 20 s for the
largest ECG-PPG step), the share of good beats, the steps in systolic pressure and heart rate; the arterial verdict
(slip: ART-PPG steps 0.6-1.4 frames and ECG-ART under 0.3 frame; the circulation: ECG-ART steps 0.3 frame or more;
none without a clean arterial line); and what the fit did there (the nearest predicted slip and its outcome, steps
off the grid and their shape call, transients).

    python tools/correction_labels.py check --key ... --answers ... --fit $W/slip_rebuild/fit_v19.jsonl \
        --scan $W/sawtooth_char/scan_full --quality $W/slip_rebuild/quality --out $W/correction_labels/check

**eval**: the answers against the key. The first form: item -> yes / no / unsure (agreement per tier and per reason, the
share of yes among the decided). The second: a verdict per correction found (right / wrong / can't tell, by tier),
and the steps the reviewer drew as missed (time, size, by tier; with the absolute time for checking against the fit).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from slip_tiers import correction, events, series  # noqa: E402

PER_TIER = {"A": 30, "B": 20, "C": 10, "U": 20, "I": 25, "N": 8}
PAD_S = 600.0


def sample_cmd(a) -> None:
    fits = {}
    for line in open(a.fit):
        r = json.loads(line)
        if "real" in r:
            fits[r["case"]] = r
    teeth = list(csv.DictReader(open(a.teeth)))
    rng = np.random.default_rng(a.seed)
    chosen = []
    for tier, n in PER_TIER.items():
        pool = [x for x in teeth if x["tier"] == tier]
        seen = set()
        for i in rng.permutation(len(pool)):
            x = pool[i]
            if x["case"] in seen:
                continue
            seen.add(x["case"])
            chosen.append(x)
            if sum(1 for y in chosen if y["tier"] == tier) == n:
                break
    order = rng.permutation(len(chosen))
    items, key = [], {}
    for k, i in enumerate(order):
        x = chosen[i]
        c = int(x["case"])
        R = fits[c]["real"]
        z = np.load(Path(a.scan) / f"{c:04d}.npz")
        tt, yy = series(z)
        cor = yy - correction(tt, R)
        t0, t1 = float(x["t0"]), float(x["t1"])
        sel = (tt >= t0 - PAD_S) & (tt <= t1 + PAD_S)
        cor = cor - np.median(cor[sel]) + np.median(yy[sel])
        g = float(tt[sel][0])
        ev = []
        for e in events(R):
            if not (tt[sel][0] <= e["t"] <= tt[sel][-1]):
                continue
            if e["kind"] == "uncertain":
                if e["p"] >= 0.5:
                    ev.append({"lo": round((e["q05"] - g) / 60, 3), "hi": round((e["q95"] - g) / 60, 3)})
            else:
                ev.append({"t": round((e["t"] - g) / 60, 3)})
        iid = f"R{k + 1:03d}"
        items.append({"id": iid, "m": [round((v - g) / 60, 3) for v in tt[sel]],
                      "raw": [round(v * 1000, 1) for v in yy[sel]], "cor": [round(v * 1000, 1) for v in cor[sel]],
                      "tooth": [round((t0 - g) / 60, 3), round((t1 - g) / 60, 3)], "steps": ev})
        key[iid] = {"case": c, "k": int(x["k"]), "tier": x["tier"], "reason": x["reason"], "edges": x["edges"],
                    "direction": x["direction"], "port": x["port"], "t0": t0, "t1": t1, "window_start": g}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(items, separators=(",", ":")) + "\n")
    (out / "key.json").write_text(json.dumps({"fit": a.fit, "teeth": a.teeth, "seed": a.seed, "per_tier": PER_TIER,
                                              "items": key}, indent=1) + "\n")
    print(json.dumps({"items": len(items), "by_tier": dict(Counter(v["tier"] for v in key.values())),
                      "bytes": (out / "items.json").stat().st_size}))


def eval_cmd(a) -> None:
    K = json.loads(Path(a.key).read_text())["items"]
    A = json.loads(Path(a.answers).read_text())
    rows, found, missed = [], [], []
    for iid, k in K.items():
        v = A.get(iid) or {}
        rows.append({"item": iid, **k, "verdict": v.get("verdict"), "note": v.get("note", ""), "done": bool(v.get("done")),
                     "found_verdicts": json.dumps(v.get("found") or {}), "missed": len(v.get("missed") or [])})
        for idx, fv in (v.get("found") or {}).items():
            found.append({"item": iid, "tier": k["tier"], "step": int(idx), "verdict": fv})
        for s in v.get("missed") or []:
            missed.append({"item": iid, "case": k["case"], "tier": k["tier"], "t_min": s["t"], "size_ms": round(s["y1"] - s["y0"], 1),
                           "t_abs": k["window_start"] + 60 * s["t"]})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def tab(sel):
        c = Counter(r["verdict"] for r in sel)
        n = c.get("yes", 0) + c.get("no", 0)
        return {"items": len(sel), "yes": c.get("yes", 0), "no": c.get("no", 0), "unsure": c.get("unsure", 0),
                "unanswered": len(sel) - n - c.get("unsure", 0),
                "agree_share_of_decided": round(c.get("yes", 0) / n, 3) if n else None}
    by_reason = defaultdict(list)
    for r in rows:
        by_reason[(r["tier"], r["reason"].split(" (")[0])].append(r)
    S = {"answers": a.answers, "all": tab(rows), "by_tier": {t: tab([r for r in rows if r["tier"] == t]) for t in PER_TIER},
         "by_reason": {f"{t}: {why}": tab(v) for (t, why), v in sorted(by_reason.items())},
         "by_direction": {d: tab([r for r in rows if r["direction"] == d]) for d in ("up", "down")},
         "disagreed": [{k: r[k] for k in ("item", "case", "tier", "reason", "note")} for r in rows if r["verdict"] == "no"],
         "unsure": [{k: r[k] for k in ("item", "case", "tier", "reason", "note")} for r in rows if r["verdict"] == "unsure"],
         "notes": [{k: r[k] for k in ("item", "case", "tier", "verdict", "note")} for r in rows if r["note"]]}
    # the newer review form: a verdict per correction found, and the steps the reviewer drew as missed
    S["per_correction"] = {t: dict(Counter(f["verdict"] for f in found if f["tier"] == t)) for t in PER_TIER}
    S["missed_drawn"] = {"steps": len(missed), "items": len({m["item"] for m in missed}),
                         "size_ms": {"one_frame_like (20-45)": sum(1 for m in missed if 20 <= abs(m["size_ms"]) <= 45),
                                     "smaller": sum(1 for m in missed if abs(m["size_ms"]) < 20),
                                     "larger": sum(1 for m in missed if abs(m["size_ms"]) > 45)},
                         "by_tier": dict(Counter(m["tier"] for m in missed)), "list": missed}
    S["done_items"] = sum(1 for r in rows if r["done"])
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: S[k] for k in ("all", "by_tier", "by_direction")}, indent=1))


FRAME_MS = 1000 / 30


def _side(t, y, ok, a, b):
    v = y[(t >= a) & (t <= b) & ok & np.isfinite(y)]
    return float(np.median(v)) if v.size >= 5 else None


def check_cmd(a) -> None:
    K = json.loads(Path(a.key).read_text())["items"]
    A = json.loads(Path(a.answers).read_text())
    fits = {}
    want = {K[i]["case"] for i, v in A.items() if v.get("missed")}
    for line in open(a.fit):
        r = json.loads(line)
        if r["case"] in want:
            fits[r["case"]] = r
    rows = []
    for iid, v in sorted(A.items()):
        for s in v.get("missed") or []:
            k = K[iid]
            c = k["case"]
            R = fits[c]["real"]
            z = np.load(Path(a.scan) / f"{c:04d}.npz")
            q = np.load(Path(a.quality) / f"{c:04d}.npz")
            r = z["r_t"]
            good = (np.nan_to_num(q["ecg_corr"]) >= 0.9) & (np.nan_to_num(q["ppg_corr"]) >= 0.9) & ~q["rr_irr"]
            ep = (z["up"] - r) * 1000
            art = z["art_t"]
            ea, ap = (art - r) * 1000, (z["up"] - art) * 1000
            hr = 60 / np.diff(r, prepend=np.nan)
            t_draw = k["window_start"] + 60 * s["t"]
            # the boundary: the largest ECG-PPG step (good beats) within +/- 20 s of the drawn time
            best = None
            for tc in np.arange(t_draw - 20, t_draw + 20.1, 1.0):
                b_, a_ = _side(r, ep, good, tc - 30, tc - 3), _side(r, ep, good, tc + 3, tc + 30)
                if b_ is not None and a_ is not None and (best is None or abs(a_ - b_) > abs(best[1])):
                    best = (tc, a_ - b_)
            tc = best[0] if best else t_draw

            def step(y, ok=good):
                b_, a_ = _side(r, y, ok, tc - 30, tc - 3), _side(r, y, ok, tc + 3, tc + 30)
                return round(a_ - b_, 1) if b_ is not None and a_ is not None else None
            near = (r >= tc - 30) & (r <= tc + 30)
            row = {"item": iid, "case": c, "tier": k["tier"], "t_min": s["t"], "drawn_ms": round(s["y1"] - s["y0"], 1),
                   "t_abs": round(tc, 2), "ecg_ppg_ms": step(ep), "ecg_art_ms": step(ea), "art_ppg_ms": step(ap),
                   "hr_step": step(hr), "good_share": round(float(good[near].mean()), 2) if near.any() else None}
            sg = np.sign(row["ecg_ppg_ms"] or 0)
            if row["art_ppg_ms"] is None or row["ecg_art_ms"] is None:
                row["art_verdict"] = "no clean arterial line"
            elif 0.6 * FRAME_MS <= sg * row["art_ppg_ms"] <= 1.4 * FRAME_MS and abs(row["ecg_art_ms"]) < 0.3 * FRAME_MS:
                row["art_verdict"] = "slip"
            elif abs(row["ecg_art_ms"]) >= 0.3 * FRAME_MS:
                row["art_verdict"] = "circulation (ECG-ART moves)"
            else:
                row["art_verdict"] = "neither"
            pr = [p for p in R.get("predicted", []) if abs(p["t_pred"] - tc) <= 120]
            p = min(pr, key=lambda p: abs(p["t_pred"] - tc)) if pr else None
            row["fit_slot"] = f"{p['outcome']} ({round(p['t_pred'] - tc)} s)" if p else None
            og = [o for o in R.get("off_grid", []) if abs(o["t"] - tc) <= 30]
            row["fit_offgrid"] = "; ".join(f"{o['call']} {'+' if o['sign'] > 0 else '-'}" for o in og) or None
            row["fit_transient"] = any(abs(x["t"] - tc) <= 30 for x in R.get("transient", []))
            row["fit_slip_within_30s"] = any(abs(x["t"] - tc) <= 30 for x in R.get("slips", []))
            rows.append(row)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "missed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    frame_like = [x for x in rows if x["ecg_ppg_ms"] is not None and 20 <= abs(x["ecg_ppg_ms"]) <= 45]
    S = {"drawn": len(rows), "measured_frame_like": len(frame_like),
         "by_art_verdict": dict(Counter(x["art_verdict"] for x in rows)),
         "frame_like_by_art_verdict": dict(Counter(x["art_verdict"] for x in frame_like)),
         "by_fit": dict(Counter((x["fit_slot"] or "no prediction").split(" (")[0] + " / " + (x["fit_offgrid"] or "no candidate")
                                for x in rows))}
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))
    for x in rows:
        print(x["item"], x["case"], x["tier"], x["t_min"], "drawn", x["drawn_ms"], "| EP", x["ecg_ppg_ms"], "EA", x["ecg_art_ms"],
              "AP", x["art_ppg_ms"], "HR", x["hr_step"], "good", x["good_share"], "|", x["art_verdict"], "|", x["fit_slot"],
              "|", x["fit_offgrid"], "| transient" if x["fit_transient"] else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    for k in ("fit", "teeth", "scan", "out"):
        s.add_argument("--" + k, required=True)
    s.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("check")
    for k in ("key", "answers", "fit", "scan", "quality", "out"):
        s.add_argument("--" + k, required=True)
    s = sub.add_parser("eval")
    for k in ("key", "answers", "out"):
        s.add_argument("--" + k, required=True)
    a = ap.parse_args()
    {"sample": sample_cmd, "eval": eval_cmd, "check": check_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
