# SPDX-License-Identifier: MIT
"""Assign a recording setup (PC key) to recordings whose four core devices' ports are incomplete.

Per case, this machine only; aggregates leave.

    python tools/setup_assign.py --ports ports.jsonl --drift full.drift.csv --out DIR

A setup is identified by its PC key: the serial ports of the SNUADC, Solar8000, Primus and BIS. Where one of them
is missing (mostly: no BIS was used), the candidates are the complete keys that agree on every port the recording
does have. They are scored by Bayes' rule from:

- how many recordings each candidate key has (the prior);
- the ports of the other devices (Orchestra, EV1000, Vigileo, ...), each a categorical feature with add-one
  smoothing, given the key;
- the SNUADC's and the BIS's drift where they were measured (recordings with clock corrections): normal around
  the key's median drift for that device (SD 1.5 ppm), with a 5% floor;
- with ``--frame``, the oximeter's frame clock where ``tools/frame_an.py`` found it (the raw PPG peak locking to a
  frame period: resultant at least 5 times the scan's median, and the two halves of the recording within 3 ppm):
  normal around the key's median (SD 2 ppm), with a 5% floor.

Validation: complete recordings with one core port hidden in turn (each recording scored without itself), giving
top-1 accuracy and calibration of the posterior. Writes DIR/assigned.csv (per case: key, posterior, candidates;
stays local) and DIR/summary.json (aggregates only).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CORE = ("SNUADC", "Solar8000", "Primus", "BIS")
SD_PPM = 1.5
SD_FRAME_PPM = 2.0


def load(ports_path: str, drift_path: str):
    """Ports per case, and per case the drift of each device type that had clock corrections (the all-file scan's
    ``full.drift.csv``: case, device_type, port, drift_ppm, ...)."""
    rows = {}
    for line in open(ports_path, encoding="utf-8"):
        r = json.loads(line)
        rows[r["case"]] = {k: v for k, v in (r.get("ports") or {}).items() if v}
    d = pd.read_csv(drift_path)
    drift = defaultdict(dict)
    for c, dev, v in zip(d["case"], d["device_type"], d["drift_ppm"]):
        if dev in ("SNUADC", "BIS") and pd.notna(v):
            drift[int(c)][dev] = float(v)
    return rows, drift


def load_frame(paths: list[str]) -> dict[int, float]:
    """Frame-clock rate (ppm against the SNUADC) per case, where the raw PPG peak locks to it."""
    out = {}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            ph = (r.get("phase") or {}).get("peak") or {}
            a, h1, h2 = ph.get("all") or {}, ph.get("half1") or {}, ph.get("half2") or {}
            if (a.get("r_best") and a.get("r_median") and a["r_best"] >= 5 * a["r_median"]
                    and h1.get("best_ppm") is not None and h2.get("best_ppm") is not None
                    and abs(h1["best_ppm"] - h2["best_ppm"]) <= 3):
                out[int(r["case"])] = float(a["best_ppm"])
    return out


def key_of(p: dict):
    return tuple(p.get(d) for d in CORE)


class Model:
    def __init__(self, rows: dict, drift: dict, frame: dict | None = None):
        self.full = {c: p for c, p in rows.items() if all(p.get(d) for d in CORE)}
        self.count = Counter(key_of(p) for p in self.full.values())
        self.extra = defaultdict(Counter)        # (key, device) -> Counter(port)
        self.levels = defaultdict(set)           # device -> ports seen
        self.drifts = defaultdict(list)          # (key, device) -> drifts
        for c, p in self.full.items():
            k = key_of(p)
            for dev, port in p.items():
                if dev not in CORE:
                    self.extra[(k, dev)][port] += 1
                    self.levels[dev].add(port)
            for dev, v in drift.get(c, {}).items():
                self.drifts[(k, dev)].append(v)
        self.drift_med = {kd: sorted(v)[len(v) // 2] for kd, v in self.drifts.items() if len(v) >= 3}
        self.drift = drift
        self.frame = frame or {}
        fr = defaultdict(list)
        for c, p in self.full.items():
            if c in self.frame:
                fr[key_of(p)].append(self.frame[c])
        self.frame_med = {k: sorted(v)[len(v) // 2] for k, v in fr.items() if len(v) >= 3}

    def score(self, case: int, p: dict, own: tuple | None = None):
        """Posterior over complete keys for a recording with ports ``p``; ``own`` (its true key) is left out of the
        counts, for validation."""
        known = {d: p.get(d) for d in CORE if p.get(d)}
        cands = [k for k in self.count if all(k[CORE.index(d)] == v for d, v in known.items())]
        logp = {}
        for k in cands:
            n = self.count[k] - (1 if k == own else 0)
            if n <= 0:
                continue
            lp = math.log(n)
            for dev, port in p.items():
                if dev in CORE:
                    continue
                cnt = self.extra[(k, dev)]
                tot = sum(cnt.values()) - (1 if k == own else 0)
                c_ = cnt[port] - (1 if k == own else 0)
                lp += math.log((c_ + 1) / (tot + len(self.levels[dev]) + 1))
            for dev, v in self.drift.get(case, {}).items():
                if (k, dev) in self.drift_med:
                    z = (v - self.drift_med[(k, dev)]) / SD_PPM
                    lp += math.log(0.95 * math.exp(-0.5 * z * z) / (SD_PPM * math.sqrt(2 * math.pi)) + 0.05 / 100)
            if case in self.frame and k in self.frame_med:
                z = (self.frame[case] - self.frame_med[k]) / SD_FRAME_PPM
                lp += math.log(0.95 * math.exp(-0.5 * z * z) / (SD_FRAME_PPM * math.sqrt(2 * math.pi)) + 0.05 / 1000)
            logp[k] = lp
        if not logp:
            return []
        m = max(logp.values())
        z = sum(math.exp(v - m) for v in logp.values())
        return sorted(((k, math.exp(v - m) / z) for k, v in logp.items()), key=lambda x: -x[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", required=True)
    ap.add_argument("--drift", required=True, help="full.drift.csv from the all-file clock correction scan")
    ap.add_argument("--frame", nargs="*", default=[], help="frame_an.py outputs (frame clock per case)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, drift = load(a.ports, a.drift)
    frame = load_frame(a.frame) if a.frame else {}
    m = Model(rows, drift, frame)
    # validation: hide each core port in turn on complete recordings
    val = {}
    for hide in CORE:
        hits, n, conf_bins = 0, 0, defaultdict(lambda: [0, 0])
        for c, p in m.full.items():
            sc = m.score(c, {k: v for k, v in p.items() if k != hide}, own=key_of(p))
            if not sc:
                continue
            n += 1
            ok = sc[0][0] == key_of(p)
            hits += ok
            b = min(int(sc[0][1] * 10), 9)
            conf_bins[b][0] += ok
            conf_bins[b][1] += 1
        val[hide] = {"recordings": n, "top1": round(hits / n, 3) if n else None,
                     "calibration": {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": [v[0], v[1]] for b, v in sorted(conf_bins.items())}}
    # assignment of the incomplete recordings
    inc = {c: p for c, p in rows.items() if not all(p.get(d) for d in CORE)}
    rec, post = [], []
    for c, p in sorted(inc.items()):
        sc = m.score(c, p)
        missing = "+".join(d for d in CORE if not p.get(d))
        if sc:
            rec.append({"case": c, "missing": missing, "key": " · ".join(sc[0][0]), "posterior": round(sc[0][1], 3),
                        "candidates": len(sc), "drift_used": bool(drift.get(c)), "frame_used": c in frame})
            post.append(sc[0][1])
        else:
            rec.append({"case": c, "missing": missing, "key": "", "posterior": 0.0, "candidates": 0, "drift_used": bool(drift.get(c)), "frame_used": c in frame})
    pd.DataFrame(rec).to_csv(out / "assigned.csv", index=False)
    df = pd.DataFrame(rec)
    summ = {"recordings": len(rows), "complete_keys": len(m.full), "distinct_keys": len(m.count),
            "frame_clock": {"cases_with": len(frame), "keys_with_median": len(m.frame_med),
                            "incomplete_with": int(sum(c in frame for c in rows if not all(rows[c].get(d) for d in CORE)))},
            "incomplete": len(inc), "missing": df.missing.value_counts().to_dict(),
            "assigned": {"no_candidate": int((df.candidates == 0).sum()),
                         "single_candidate": int((df.candidates == 1).sum()),
                         "posterior_ge_0.9": int((df.posterior >= 0.9).sum()),
                         "posterior_0.5_0.9": int(((df.posterior >= 0.5) & (df.posterior < 0.9)).sum()),
                         "posterior_lt_0.5": int(((df.posterior > 0) & (df.posterior < 0.5)).sum())},
            "validation": val}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
