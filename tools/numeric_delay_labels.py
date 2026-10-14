# SPDX-License-Identifier: MIT
"""Per-case numeric delays and the HR delay group, as label CSVs for the AtriumDB label pass. This machine only.

    python tools/numeric_delay_labels.py --scan part*.jsonl --root R --devices va/p?.jsonl --out RESULTS \
        --summary summary.json

Input: the per-case scan of tools/numeric_delay_variability.py (delay per case and per 30-min block, the
algorithm held fixed). Output, in RESULTS:

numeric_delay_labels.csv
    One row per case and track: the case's own delay from the window's end to the stamp
    (``delay_from_window_end_s``, the whole-case fit), its standard error from the scatter of its 30-min
    blocks (block SD / sqrt(blocks), at least 0.05 s; with fewer than two blocks the population within-case
    SD), the track's window length from the timing model (``window_s``), where the case sits in the
    population (percentile), and a quality: ``good`` (two or more blocks, SE <= 0.5 s), ``weak`` (otherwise).
    Cases whose fit hit the edge of the delay grid get no row. The label spans the track's records.
    ``delay_check`` is ``negative`` where the case's delay is below zero: a value cannot be stamped before its
    window ends (the availability rule), so such an estimate is wrong (usually noise around a small true delay)
    and the row's quality is ``weak`` whatever its standard error.
hr_delay_group_labels.csv
    One row per case with an HR delay: ``early`` or ``late`` when the posterior of a two-component Gaussian
    mixture (fitted to the good HR delays, each case's own SE added to the component variance) is >= 0.9,
    else ``uncertain``. The summary compares the groups with the Solar8000 serial port.

Availability: both are fitted over the whole case (the HR mixture over all cases), so they are known only at
the case's end (``available_at`` = ``case_end``); they must not be model input earlier in the case.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from timenet_connectors.datasets.vitaldb.vitaldb.timing_model import MODEL_VERSION, NUMERIC_LAG  # noqa: E402

SOLAR = {"HR", "ART_SBP", "ART_MBP", "ART_DBP", "CVP"}
TRACK = {"HR": "Solar8000/HR", "ART_SBP": "Solar8000/ART_SBP", "ART_MBP": "Solar8000/ART_MBP",
         "ART_DBP": "Solar8000/ART_DBP", "CVP": "Solar8000/CVP", "ETCO2": "Primus/ETCO2", "PIP": "Primus/PIP_MBAR",
         "RR_CO2": "Primus/RR_CO2"}
ALGO_VERSION = "numeric_delay_variability fixed-algorithm fit, 30-min blocks"
SE_FLOOR = 0.05
SE_GOOD = 0.5
POSTERIOR = 0.9


def se_of(t: dict, sigma_w: float) -> tuple[float, int]:
    b = [x["d"] for x in t["blocks"] if not x["edge"]]
    if len(b) >= 2:
        return max(SE_FLOOR, float(np.std(b, ddof=1) / np.sqrt(len(b)))), len(b)
    return sigma_w, len(b)


def em_two(x: np.ndarray, se: np.ndarray, iters: int = 200) -> dict:
    """Two-component Gaussian mixture with known per-point extra variance se**2."""
    mu = np.percentile(x, [25, 75]).astype(float)
    sd = np.array([x.std() / 2, x.std() / 2])
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        var = sd[None, :] ** 2 + se[:, None] ** 2
        lik = w[None, :] * np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2 / var) / np.sqrt(2 * np.pi * var)
        r = lik / lik.sum(axis=1, keepdims=True)
        w = r.mean(axis=0)
        mu = (r * x[:, None]).sum(axis=0) / r.sum(axis=0)
        # moment update of the component SD, net of the measurement variance
        resid = (r * ((x[:, None] - mu[None, :]) ** 2 - se[:, None] ** 2)).sum(axis=0) / r.sum(axis=0)
        sd = np.sqrt(np.maximum(resid, 0.05 ** 2))
    o = np.argsort(mu)
    return {"mu": mu[o].tolist(), "sd": sd[o].tolist(), "w": w[o].tolist()}


def posterior_late(x: float, se: float, m: dict) -> float:
    mu, sd, w = (np.array(m[k]) for k in ("mu", "sd", "w"))
    var = sd ** 2 + se ** 2
    lik = w * np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
    return float(lik[1] / lik.sum())


def spans(root: str, case: int, tracks: set[str]) -> dict[str, tuple[float, float]]:
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: PLC0415

    v = read_vital(Path(root) / "vital_files" / f"{case:04d}.vital", track_names=tracks)
    out = {}
    for n in tracks:
        tr = v.tracks.get(n)
        if tr is not None and tr.record_times.size:
            out[n] = (float(tr.record_times.min()), float(tr.record_times.max()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", nargs="+", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--devices", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--spans-cache", help="jsonl of track spans for scans made before spans were recorded")
    ap.add_argument("--time-budget", type=float, default=150.0, help="seconds for filling the spans cache")
    a = ap.parse_args()
    R = {}
    # later files override earlier ones track by track (a rescan of some tracks updates just those)
    for p in a.scan:
        for line in open(p):
            r = json.loads(line)
            if "tracks" not in r:
                continue
            cur = R.setdefault(r["case"], {"case": r["case"], "tracks": {}})
            cur["tracks"].update(r["tracks"])
            if "spans" in r:
                cur.setdefault("spans", {}).update(r["spans"])
    port = {}
    for p in a.devices or []:
        for line in open(p):
            r = json.loads(line)
            port[r["case"]] = (r.get("devices", {}).get("Solar8000") or [None, None])[1]

    # population: within-case SD per track (median block SD) and the case delays for percentiles
    pop_d, pop_w = defaultdict(list), defaultdict(list)
    for r in R.values():
        for k, t in r["tracks"].items():
            if t["edge"]:
                continue
            pop_d[k].append(t["d"])
            b = [x["d"] for x in t["blocks"] if not x["edge"]]
            if len(b) >= 2:
                pop_w[k].append(float(np.std(b, ddof=1)))
    sigma_w = {k: float(np.median(v)) if v else 1.0 for k, v in pop_w.items()}
    pop_sorted = {k: np.sort(v) for k, v in pop_d.items()}

    cache = {}
    if a.spans_cache and Path(a.spans_cache).exists():
        for line in open(a.spans_cache):
            e = json.loads(line)
            cache[e["case"]] = e["spans"]
    need = [c for c in sorted(R) if "spans" not in R[c] and c not in cache and R[c]["tracks"]]
    if need:
        if not a.spans_cache:
            raise SystemExit(f"{len(need)} scanned cases have no spans: pass --spans-cache")
        t0 = time.monotonic()
        with open(a.spans_cache, "a") as fh:
            for c in need:
                if time.monotonic() - t0 > a.time_budget:
                    print(json.dumps({"spans_left": len(need) - need.index(c)}))
                    return
                sp = spans(a.root, c, {TRACK[k] for k in R[c]["tracks"] if k in TRACK})
                fh.write(json.dumps({"case": c, "spans": sp}) + "\n")
                cache[c] = sp

    rows, hr = [], []
    for c in sorted(R):
        r = R[c]
        sp = r.get("spans") or cache.get(c, {})
        for k, t in r["tracks"].items():
            if t["edge"] or k not in TRACK or TRACK[k] not in sp:
                continue
            se, nb = se_of(t, sigma_w.get(k, 1.0))
            lag = NUMERIC_LAG.get(TRACK[k])
            pct = float(np.searchsorted(pop_sorted[k], t["d"]) / pop_sorted[k].size * 100)
            neg = t["d"] < 0
            q = "good" if nb >= 2 and se <= SE_GOOD and not neg else "weak"
            row = {"case": c, "track": TRACK[k], "label": f"numeric_delay:{q}", "start_s": repr(sp[TRACK[k]][0]),
                   "end_s": repr(sp[TRACK[k]][1]), "delay_from_window_end_s": t["d"],
                   "delay_check": "negative" if neg else "ok", "se_s": round(se, 3),
                   "blocks": nb, "window_s": None if lag is None else lag.window_s,
                   "population_percentile": round(pct, 1),
                   "algorithm": lag.algorithm if lag else "", "quality": q, "available_at": "case_end",
                   "model_version": MODEL_VERSION,
                   "method": ALGO_VERSION}
            rows.append(row)
            if k == "HR":
                hr.append(row)

    good = [h for h in hr if h["quality"] == "good"]
    model = em_two(np.array([h["delay_from_window_end_s"] for h in good]), np.array([h["se_s"] for h in good]))
    hr_rows = []
    for h in hr:
        p = posterior_late(h["delay_from_window_end_s"], h["se_s"], model)
        g = "late" if p >= POSTERIOR else "early" if p <= 1 - POSTERIOR else "uncertain"
        hr_rows.append({"case": h["case"], "label": f"hr_delay_group:{g}", "start_s": h["start_s"], "end_s": h["end_s"],
                        "group": g, "p_late": round(p, 3), "delay_from_window_end_s": h["delay_from_window_end_s"],
                        "se_s": h["se_s"], "available_at": "case_end",
                        "mixture": json.dumps({k: [round(x, 3) for x in v] for k, v in model.items()}),
                        "solar_port": port.get(h["case"]) or "", "model_version": MODEL_VERSION})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rs in (("numeric_delay_labels.csv", rows), ("hr_delay_group_labels.csv", hr_rows)):
        cols = list(dict.fromkeys(k for r in rs for k in r))
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rs)

    # aggregate summary only
    by_port = defaultdict(Counter)
    for h in hr_rows:
        by_port[h["solar_port"]][h["group"]] += 1
    S = {"cases": len(R), "rows": len(rows), "model_version": MODEL_VERSION,
         "quality_by_track": {TRACK[k]: dict(Counter(r["quality"] for r in rows if r["track"] == TRACK[k])) for k in TRACK},
         "se_p50_by_track": {TRACK[k]: float(np.median([r["se_s"] for r in rows if r["track"] == TRACK[k]]))
                             for k in TRACK if any(r["track"] == TRACK[k] for r in rows)},
         "hr_mixture": model, "hr_groups": dict(Counter(h["group"] for h in hr_rows)),
         "hr_groups_by_port": {p: dict(c) for p, c in sorted(by_port.items(), key=lambda kv: -sum(kv[1].values()))
                               if sum(c.values()) >= 20}}
    Path(a.summary).write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
