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

**eval**: the answers (item -> yes / no / unsure, with an optional note) against the key: agreement per tier and per
reason (the share of yes among the decided).
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
    rows = []
    for iid, k in K.items():
        v = A.get(iid) or {}
        rows.append({"item": iid, **k, "verdict": v.get("verdict"), "note": v.get("note", "")})
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
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: S[k] for k in ("all", "by_tier", "by_direction")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    for k in ("fit", "teeth", "scan", "out"):
        s.add_argument("--" + k, required=True)
    s.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("eval")
    for k in ("key", "answers", "out"):
        s.add_argument("--" + k, required=True)
    a = ap.parse_args()
    {"sample": sample_cmd, "eval": eval_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
