# SPDX-License-Identifier: MIT
"""Which breath each Primus stretch is on: a rule from pauses in ventilation, checked by leaving stretches out.

    python tools/primus_breath_rule.py --work $W --out $W/primus_breath_rule

Inputs (all saved outputs, nothing is read from the recordings):
  $W/primus_awp_anchor/breaths.json        per recording: the matched-lag line of each stretch between AWP pauses
                                           (tools/primus_sync_accuracy.py windows; CVP arm, else ART) and its
                                           breath period T (tools/primus_awp_anchor.py breaths)
  $W/primus_awp_anchor/records/part_*.jsonl  each AWP gap and whether it is a whole number of records (the 0.48 s
                                           grid carries on, and so does the offset) (primus_awp_anchor.py records)
  $W/primus_vent_events/part_*.jsonl       pauses in ventilation, each edge's event lag and its resolution
                                           (tools/primus_vent_events.py)

The matched lag of a stretch is right up to a whole number of breaths k: corrected = matched - k T. The rule:
  1. own event: a stretch with resolved CVP pause edges takes k = round((line(t) - event lag) / T), the median over
     its edges (resolved: peak r >= 0.5, 0.15 above the correlation one breath either side, not at the edge of the
     lag range; the CVP event method has no bias against the matching, primus-awp-anchoring);
  2. carry: across a gap that keeps the 0.48 s grid the offset carries on (median step +0.01 s), so a stretch next to
     one with k takes the k that makes its corrected start continuous with the neighbour's corrected line;
     repeated until nothing changes; a gap that re-anchors stops it;
  3. main mode: any stretch still without k takes the k whose corrected start offset is nearest the main mode
     (median corrected start offset of the stretches fixed by step 1, CVP);
  4. ART-only recordings: steps 1-3 with ART pause edges, their event lag corrected by the ART bias (median ART
     event minus matched lag where they agree, from primus_vent_events summary.json bias_check).

Checks. Leave one out: for every stretch fixed by its own CVP events, hide its events, apply steps 2 and 3 (with the
recording's other events), and compare with its own k: the share right, per step, and the share of stretches each
step reaches. Baseline: k = 0 (trust the matching). Writes stretch_k.json (per stretch: k, the step that set it,
corrected start offset) and summary.json.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

LAG_RANGE = (-3.0, 12.0)


def resolved(x):
    nb = [y for y in x.get("r_one_breath_either_side") or [] if y is not None]
    return x["r"] >= 0.5 and bool(nb) and x["r"] - max(nb) >= 0.15 and LAG_RANGE[0] < x["lag"] < LAG_RANGE[1]


def load(W: Path):
    B = json.loads((W / "primus_awp_anchor" / "breaths.json").read_text())["stretches"]
    gaps = {}
    for p in glob.glob(str(W / "primus_awp_anchor" / "records" / "part_*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("awp"):
                gaps[r["case"]] = r["awp"]["gaps"]
    ev, pauses = {}, {}
    for p in glob.glob(str(W / "primus_vent_events" / "part_*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            pauses[r["case"]] = r.get("pauses", [])
            for e in r.get("events", []):
                for arm in ("cvp", "art"):
                    for edge, x in (e.get(arm) or {}).items():
                        if resolved(x):
                            t = e[edge]
                            ev.setdefault((r["case"], arm), []).append((float(t), float(x["lag"]), float(e["T"])))
    s = json.loads((W / "primus_vent_events" / "summary.json").read_text())
    art_bias = s["bias_check"]["art"]["event_minus_matched_s_p10_25_50_75_90"][2]
    return B, gaps, ev, pauses, art_bias


def line(st, t):
    return st["at_start"] + st["slope_ppm"] * 1e-6 * (t - st["start"])


def own_k(st, i, edges, P):
    """k from the events inside stretch i (index by the AWP pauses), median over its edges."""
    ks = []
    for t, lag, T in edges:
        if int(np.searchsorted(P, t)) == st["i"]:
            Ts = st["T"] or T
            ks.append((line(st, t) - lag) / Ts)
    return int(np.round(np.median(ks))) if ks else None


def gap_keeps_grid(gaps, start):
    g = [q for q in gaps if abs(q["after"] - start) <= 2.0]
    return bool(g) and abs(g[0]["len_records"] - round(g[0]["len_records"])) < 0.02


def assign(rec, gaps, edges, P, m0, use_own=True, hide=None):
    """k and the step that set it, per stretch of one recording."""
    st = [x for x in rec["stretches"] if x.get("T")]
    K = {}
    if use_own:
        for i, x in enumerate(st):
            if i == hide:
                continue
            k = own_k(x, i, edges, P)
            if k is not None:
                K[i] = (k, 1)
    changed = True
    while changed:
        changed = False
        for i in range(len(st)):
            if i in K:
                continue
            for j in (i - 1, i + 1):
                if j in K and 0 <= j < len(st):
                    later = st[max(i, j)]
                    if not gap_keeps_grid(gaps, later["start"]):
                        continue
                    kj = K[j][0]
                    target = line(st[j], st[i]["start"]) - kj * st[j]["T"]
                    k = int(np.round((st[i]["at_start"] - target) / st[i]["T"]))
                    K[i] = (k, 2)
                    changed = True
                    break
    for i, x in enumerate(st):
        if i not in K:
            K[i] = (int(np.round((x["at_start"] - m0) / x["T"])), 3)
    return st, K


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W = Path(a.work)
    B, gaps, ev, pauses, art_bias = load(W)
    # main mode: corrected start offsets of stretches fixed by their own CVP events
    own = []
    for rec in B:
        if rec["arm"] != "cvp":
            continue
        E = ev.get((rec["case"], "cvp"), [])
        P = np.array(pauses.get(rec["case"], []), float)
        for x in rec["stretches"]:
            if x.get("T"):
                k = own_k(x, None, E, P)
                if k is not None:
                    own.append(x["at_start"] - k * x["T"])
    m0 = float(np.median(own))
    out, checks = [], {"own": 0, "hidden_right_by_step": {2: [0, 0], 3: [0, 0]}, "baseline_k0_right": 0}
    steps_count = {"cvp": {1: 0, 2: 0, 3: 0}, "art": {1: 0, 2: 0, 3: 0}}
    for rec in B:
        arm = rec["arm"]
        E = ev.get((rec["case"], arm), [])
        if arm == "art":
            E = [(t, lag - art_bias, T) for t, lag, T in E]
        P = np.array(pauses.get(rec["case"], []), float)
        G = gaps.get(rec["case"], [])
        st, K = assign(rec, G, E, P, m0)
        for i, x in enumerate(st):
            k, step = K[i]
            steps_count[arm][step] += 1
            out.append({"case": rec["case"], "arm": arm, "stretch": x["i"], "k": k, "step": step,
                        "corrected_start_s": round(x["at_start"] - k * x["T"], 3), "T": x["T"]})
        if arm != "cvp":
            continue
        for i, x in enumerate(st):  # leave one out
            k_own = own_k(x, i, E, P)
            if k_own is None:
                continue
            checks["own"] += 1
            checks["baseline_k0_right"] += int(k_own == 0)
            _, K2 = assign(rec, G, E, P, m0, hide=i)
            k2, step2 = K2[i]
            checks["hidden_right_by_step"][step2][0] += int(k2 == k_own)
            checks["hidden_right_by_step"][step2][1] += 1
    cs = np.array([x["corrected_start_s"] for x in out if x["arm"] == "cvp" and x["step"] == 1])
    summ = {"main_mode_s": round(m0, 3), "stretches": len(out),
            "stretches_by_step": steps_count,
            "share_k_nonzero": {arm: round(float(np.mean([x["k"] != 0 for x in out if x["arm"] == arm])), 3) for arm in ("cvp", "art")},
            "corrected_start_own_events_cvp_s_p5_10_25_50_75_90_95": [round(float(v), 3) for v in np.percentile(cs, [5, 10, 25, 50, 75, 90, 95])] if cs.size else None,
            "leave_one_out": {"stretches_with_own_cvp_events": checks["own"],
                              "baseline_k0_right_share": round(checks["baseline_k0_right"] / max(checks["own"], 1), 3),
                              **{f"step{s}_reached": v[1] for s, v in checks["hidden_right_by_step"].items()},
                              **{f"step{s}_right_share": round(v[0] / v[1], 3) if v[1] else None
                                 for s, v in checks["hidden_right_by_step"].items()}},
            "art_bias_used_s": art_bias}
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "stretch_k.json").write_text(json.dumps(out))
    (o / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
