# SPDX-License-Identifier: MIT
"""End-to-end check of PPG sawtooth corrections against the arterial line (metrology-atlas study
ppg-sawtooth-correction): finger PAT minus arterial PAT has no sawtooth of its own, so after a good correction it is
flat apart from slow physiology between the two sites, the same under every correction.

    python tools/correction_vs_art.py scan --scan $W/sawtooth_char/scan_full --fit $W/slip_rebuild/full3/fit.jsonl \
        --out $W/correction_vs_art/part_K.jsonl --part K --of N [--time-budget S]
    python tools/correction_vs_art.py agg $W/correction_vs_art/part_*.jsonl --out $W/correction_vs_art/agg

Per recording of the full-release scan with an arterial upstroke on at least half its beats: d = PLETH steepest
upstroke minus arterial upstroke (per beat; the R peak cancels), corrected by each policy, and its spread: the robust
SD (IQR / 1.349) of 30 s window medians, in ms. Policies (the delay model D(t) is subtracted from d):

  none      no correction;
  detect    physiosync fit_ecg_ppg_delay, method "detect" (frame 1/30 s, the arterial line as reference, default
            tiers: B and above corrected);
  v26       the rebuilt model as first released (physiosync 0.3.1; one creep rate per recording);
  final     the correction chosen by this check (physiosync 0.3.2 delay_at, tools/slip_tiers.py v6), which must equal
            v26+nostep+close+unc+mask (checked per recording, final_check_us);
  and v26 with options, joined by "+":
    gap     no creep across a gap of more than 1.5 teeth between slips (tier U long teeth);
    nostep  no correction where the creep is not corrected (no sawtooth, or a tooth from two slips): steps without
            their creep turn a sawtooth into a ramp;
    grid    off-grid slips and slips the other way left out;
    unc     slips placed only within an interval (p >= 0.5) corrected as a ramp across it, weighted by p;
    close   each stretch between two slips the sawtooth's way creeps at frame / its own length, so it closes exactly
            (as the detect method does), however long; the ends, and stretches next to other steps, at frame / the
            fitted tooth. With one creep rate for the whole recording, a tooth length a few percent off adds up over
            the teeth; and a gap holding missed slips, with no creep or with the fitted creep and no steps, leaves a
            frame or more per gap. With "gap" too, "gap" wins over "close" in gaps over 1.5 teeth;
    const   one creep rate for the recording, set by its slips so that creep and steps balance from the first slip to
            the last: frame x (net steps between them, uncertain ones by p) / their time apart (else the fitted tooth);
            from a tooth before the first slip to a tooth after the last;
    mask    teeth of tier U (unresolved, tools/slip_tiers.py) are not corrected: the model is held at its median there,
            as physiosync's detect method does for teeth below its threshold.

Recordings are grouped by the rebuilt fit: "sawtooth" (creep corrected), "two-slip tooth" (creep reported), "slips,
no sawtooth", "no slip". agg gives each policy's spread (median, IQR, 90th centile) overall and by group, and per
policy the share of recordings it leaves worse than no correction by more than 2 ms. The arterial line is also
evidence inside both fits (it confirms or vetoes steps), so this check is not wholly independent of them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from physiosync.slips import delay_at  # noqa: E402
from slip_tiers import events, teeth  # noqa: E402

FRAME = 1 / 30
WIN = 30.0
LONG_TOOTH = 1.5
OPTIONS = ("gap", "nostep", "grid", "unc", "close")
POLICIES = ("none", "detect", "v26", "v26+gap", "v26+nostep", "v26+gap+nostep", "v26+gap+nostep+grid",
            "v26+gap+nostep+close", "v26+gap+nostep+grid+close", "v26+nostep+close", "v26+nostep+grid+close",
            "v26+nostep+grid+close+unc", "v26+nostep+close+unc", "v26+nostep+close+unc+mask",
            "v26+nostep+grid+close+unc+mask", "v26+nostep+const+unc+mask", "v26+nostep+const+unc", "final")
FINAL = "v26+nostep+close+unc+mask"   # the policy chosen for v2 (5 November): tools/slip_tiers.py v6, physiosync 0.3.2


def model(t: np.ndarray, R: dict, opts: set[str]) -> np.ndarray:
    """The rebuilt delay model with options (see the module docstring); no options = physiosync 0.3.1 (v26)."""
    sign = R.get("sign", 1)
    saw = bool(R.get("sawtooth"))
    reported = saw and (R.get("fit") or {}).get("creep") == "reported"
    if "nostep" in opts and (not saw or reported):
        return np.zeros(t.size)
    ev = events(R)
    if "grid" in opts:
        ev = [e for e in ev if e.get("what") not in ("off grid", "other way")]
    N = np.zeros(t.size)
    for e in ev:
        if e["kind"] == "uncertain":
            if e["p"] < 0.5 or "unc" not in opts:
                continue
            N += e["p"] * np.clip((t - e["q05"]) / max(e["q95"] - e["q05"], 1e-6), 0, 1)
        else:
            N += (t >= e["t"]) * (e["sign"] * sign)
    D = sign * FRAME * N
    L = (R.get("fit") or {}).get("L")
    real = [e for e in ev if e["kind"] != "uncertain" or e["p"] >= 0.5]
    if saw and L and real and not reported:
        if "const" in opts:
            # net steps from the first slip to the last, the sawtooth's way (uncertain ones by p, if counted)
            net = sum((e["p"] if e["kind"] == "uncertain" else e["sign"] * sign) for e in real[1:]
                      if e["kind"] != "uncertain" or "unc" in opts)
            span = real[-1]["t"] - real[0]["t"]
            Lc = span / net if net >= 1 and span > 0 else L
            lo, hi = real[0]["t"] - Lc, real[-1]["t"] + Lc
            D -= sign * FRAME * (np.clip(t, lo, hi) - lo) / Lc
        lo, hi = real[0]["t"] - L, real[-1]["t"] + L
        # creep in teeth: (start, end, teeth crept per second)
        segs = [(lo, real[0]["t"], 1 / L)]
        for a_, b_ in zip(real[:-1], real[1:]):
            g = b_["t"] - a_["t"]
            if "gap" in opts and g > LONG_TOOTH * L:
                rate = 0.0
            elif ("close" in opts and g > 0 and b_["kind"] != "uncertain" and b_["sign"] == sign
                  and (a_["kind"] == "uncertain" or a_["sign"] == sign)):
                rate = 1 / g
            else:
                rate = 1 / L
            segs.append((a_["t"], b_["t"], rate))
        segs.append((real[-1]["t"], hi, 1 / L))
        run = np.zeros(t.size)
        for s0, s1, rate in segs:
            run += rate * (np.clip(t, s0, s1) - s0)
        if "const" not in opts:
            D -= sign * FRAME * run
    if "mask" in opts and saw and not reported:
        grid = np.arange(R["start"], R["end"] + 1.0, 1.0)
        med = float(np.median(model(grid, R, opts - {"mask"})))
        for x in teeth(R):
            if x["tier"] == "U":
                D = np.where((t >= x["t0"]) & (t < x["t1"]), med, D)
    return D


def spread(t: np.ndarray, d: np.ndarray) -> float | None:
    ok = np.isfinite(t) & np.isfinite(d)
    if ok.sum() < 300:
        return None
    k = np.floor((t[ok] - t[ok][0]) / WIN).astype(int)
    med = np.array([np.median(d[ok][k == w]) for w in np.unique(k) if np.sum(k == w) >= 15])
    if med.size < 20:
        return None
    return round(float(np.subtract(*np.percentile(med, [75, 25])) / 1.349 * 1000), 2)


def group_of(R: dict) -> str:
    if R.get("sawtooth"):
        return "two-slip tooth" if (R.get("fit") or {}).get("creep") == "reported" else "sawtooth"
    return "slips, no sawtooth" if R.get("slips") else "no slip"


def scan_case(z, R: dict) -> dict:
    from physiosync import fit_ecg_ppg_delay
    r, up, art = z["r_t"], z["up"], z["art_t"]
    ok = np.isfinite(up) & np.isfinite(art)
    if ok.mean() < 0.5 or ok.sum() < 600:
        return {"skip": "arterial upstroke on less than half the beats"}
    t, d = up[ok], (up - art)[ok]
    res = {"group": group_of(R), "beats": int(ok.sum()), "spread_ms": {}}
    res["spread_ms"]["none"] = spread(t, d)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = fit_ecg_ppg_delay(r[ok], up[ok], adopted_pat_s=0.27, frame_s=FRAME, matched=True, ref_t=art[ok])
        res["spread_ms"]["detect"] = spread(t, d - m.shift(t))
    except ValueError as ex:
        res["detect_error"] = str(ex)[:100]
    res["spread_ms"]["v26"] = spread(t, d - model(t, R, set()))
    for p in POLICIES[3:-1]:
        res["spread_ms"][p] = spread(t, d - model(t, R, set(p.split("+")[1:])))
    final = delay_at(t, R, FRAME)
    res["spread_ms"]["final"] = spread(t, d - final)
    # the released correction (physiosync delay_at, slip_tiers v6) must equal this tool's model of the chosen policy
    res["final_check_us"] = round(float(np.max(np.abs(model(t, R, set(FINAL.split("+")[1:])) - final))) * 1e6, 3)
    return res


def scan(a) -> None:
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.scan).glob("*.npz"))
    files = [f for i, f in enumerate(files) if i % a.of == a.part]
    want = {int(f.stem) for f in files}
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    fits = {}
    with open(a.fit) as fh:
        for line in fh:
            c = int(line[9:line.index(",")]) if line.startswith('{"case": ') else None
            if c in want and c not in done:
                r = json.loads(line)
                if r.get("real"):
                    fits[c] = r["real"]
    t0 = time.time()
    with out.open("a") as fh:
        for f in files:
            c = int(f.stem)
            if c in done or c not in fits:
                continue
            if time.time() - t0 > a.time_budget:
                break
            row = {"case": c}
            try:
                row.update(scan_case(np.load(f), fits[c]))
            except Exception as ex:  # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    print(json.dumps({"part": a.part, "seconds": round(time.time() - t0, 1)}))


def agg(a) -> None:
    rows = list({r["case"]: r for p in a.parts for r in map(json.loads, Path(p).read_text().splitlines())}.values())
    ok = [r for r in rows if "spread_ms" in r and all(r["spread_ms"].get(p) is not None for p in POLICIES)]

    def q(v):
        v = np.array(v, float)
        return {"n": int(v.size), "median": round(float(np.median(v)), 2),
                "iqr": [round(float(x), 2) for x in np.percentile(v, [25, 75])],
                "p90": round(float(np.percentile(v, 90)), 2)} if v.size else None
    S = {"recordings": len(rows), "compared": len(ok), "skipped": dict(Counter(r.get("skip") for r in rows if r.get("skip"))),
         "errors": sum(1 for r in rows if "error" in r),
         "final_check_max_us": max((r.get("final_check_us", 0) for r in ok), default=None),
         "groups": dict(Counter(r["group"] for r in ok)), "policies": {}, "by_group": {}}
    for p in POLICIES:
        v = [r["spread_ms"][p] for r in ok]
        worse = [r["spread_ms"][p] - r["spread_ms"]["none"] > 2 for r in ok]
        better = [r["spread_ms"]["none"] - r["spread_ms"][p] > 2 for r in ok]
        S["policies"][p] = {**q(v), "worse_than_none_by_2ms": round(float(np.mean(worse)), 3),
                            "better_than_none_by_2ms": round(float(np.mean(better)), 3)}
    for g in sorted(S["groups"]):
        sel = [r for r in ok if r["group"] == g]
        S["by_group"][g] = {p: {**q([r["spread_ms"][p] for r in sel]),
                                "worse_than_none_by_2ms": round(float(np.mean([r["spread_ms"][p] - r["spread_ms"]["none"] > 2
                                                                               for r in sel])), 3)} for p in POLICIES}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: S[k] for k in ("recordings", "compared", "errors", "final_check_max_us", "groups")}))
    for p, v in S["policies"].items():
        print(f"{p:28s} {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    for k in ("scan", "fit", "out"):
        s.add_argument("--" + k, required=True)
    s.add_argument("--part", type=int, default=0)
    s.add_argument("--of", type=int, default=1)
    s.add_argument("--time-budget", type=float, default=1e9)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
