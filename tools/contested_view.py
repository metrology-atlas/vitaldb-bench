# SPDX-License-Identifier: MIT
"""Beat-level evidence for the contested items of the sawtooth correction review (metrology-atlas study ppg-sawtooth):
for every item the reviewer disagreed with, was unsure of, or annotated, everything in the recording that bears on
whether a step in the ECG-PPG delay is a frame slip or a real change.

    python tools/contested_view.py --key $W/correction_labels/key.json --answers ANSWERS.json \
        --fit $W/slip_rebuild/fit_v19.jsonl --scan $W/sawtooth_char/scan_full --quality $W/slip_rebuild/quality \
        --root $R --out $W/correction_labels/contested.json [--items R006,R013]

Per item, over the review window (the tooth and up to 10 min either side), per matched beat:

* the ECG-PPG delay at the foot, upstroke and peak (s), and whether the beat passed the quality gate;
* where there is an arterial line, ECG-ART (R to the arterial upstroke) and ART-PPG (arterial to PPG upstroke). A
  frame slip moves ECG-PPG and ART-PPG by 33 ms and leaves ECG-ART flat; a change of the circulation moves ECG-ART;
* systolic and diastolic pressure from the arterial waveform (the highest and lowest sample of each R-R interval),
  and the heart rate from R-R;

and, from the monitor's other tracks: the cuff pressures (NIBP; a reading marks a cuff cycle, which can disturb a PPG
on the same arm), the propofol and remifentanil pump rates (Orchestra), ventilator setting changes (Primus SET_*),
and the case's event marks. Bolus doses are not timed in the release (clinical_data.csv gives only totals: those of
ephedrine, phenylephrine, epinephrine and calcium are listed per item). The correction's steps are those of the fit.
Writes one JSON file for the review page.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pat import sample_times  # noqa: E402
from slip_tiers import events  # noqa: E402
from vitaldb_reader.vital_file import read_vital  # noqa: E402

PAD_S = 600.0
NUMERIC = {"nibp_sbp": "Solar8000/NIBP_SBP", "nibp_dbp": "Solar8000/NIBP_DBP", "ppf_rate": "Orchestra/PPF20_RATE",
           "rftn_rate": "Orchestra/RFTN20_RATE"}
VENT = ("Primus/SET_TV_L", "Primus/SET_RR_IPPV", "Primus/SET_INTER_PEEP", "Primus/SET_PIP", "Primus/SET_FIO2",
        "Primus/SET_INSP_TM", "Primus/SET_FRESH_FLOW")
DRUGS = ("intraop_eph", "intraop_phe", "intraop_epi", "intraop_ca")


def numeric(tr, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(tr.record_times, float)
    x = np.asarray(tr.physical(), float)
    n = min(t.size, x.size)
    t, x = t[:n], x[:n]
    sel = (t >= lo) & (t <= hi) & np.isfinite(x)
    return t[sel], x[sel]


def changes(t: np.ndarray, x: np.ndarray) -> list[tuple[float, float]]:
    """The points where a step-like track changes value (and its first value)."""
    if not t.size:
        return []
    keep = np.r_[True, np.abs(np.diff(x)) > 1e-9]
    return list(zip(t[keep].tolist(), x[keep].tolist()))


def r3(v, k=1):
    return None if v is None or not np.isfinite(v) else round(float(v), k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("key", "answers", "fit", "scan", "quality", "root", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--items", default="", help="comma-separated item ids (default: every contested item)")
    a = ap.parse_args()
    K = json.loads(Path(a.key).read_text())["items"]
    A = json.loads(Path(a.answers).read_text())
    if a.items:
        ids = a.items.split(",")
    else:
        ids = sorted(i for i, v in A.items() if v.get("verdict") in ("no", "unsure") or v.get("note"))
    fits = {}
    want = {K[i]["case"] for i in ids}
    for line in open(a.fit):
        r = json.loads(line)
        if r["case"] in want:
            fits[r["case"]] = r
    clin = {}
    cpath = Path(a.root) / "clinical_data.csv"
    if cpath.exists():
        for row in csv.DictReader(open(cpath, encoding="utf-8-sig")):
            if int(row["caseid"]) in want:
                clin[int(row["caseid"])] = {d: row.get(d) for d in DRUGS}
    out = []
    for iid in ids:
        k = K[iid]
        c = k["case"]
        R = fits[c]["real"]
        g = float(k["window_start"])
        lo, hi = g, float(k["t1"]) + PAD_S
        z = np.load(Path(a.scan) / f"{c:04d}.npz")
        q = np.load(Path(a.quality) / f"{c:04d}.npz")
        r = z["r_t"]
        sel = (r >= lo) & (r <= hi)
        idx = np.flatnonzero(sel)
        good = (np.nan_to_num(q["ecg_corr"]) >= 0.9) & (np.nan_to_num(q["ppg_corr"]) >= 0.9) & ~q["rr_irr"]
        art = z["art_t"]
        has_art = bool(np.isfinite(art[idx]).mean() > 0.3) if idx.size else False
        case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital",
                          track_names={"SNUADC/ART", "EVENT", *NUMERIC.values(), *VENT})
        sbp = np.full(idx.size, np.nan)
        dbp = np.full(idx.size, np.nan)
        tr = case.tracks.get("SNUADC/ART")
        if tr is not None and tr.record_times.size:
            ta = sample_times(tr)
            xa = np.asarray(tr.physical(), float)
            for j, i in enumerate(idx):
                if i + 1 >= r.size:
                    continue
                a0, a1 = np.searchsorted(ta, r[i]), np.searchsorted(ta, r[i + 1])
                if 10 < a1 - a0 < 1500:
                    seg = xa[a0:a1]
                    seg = seg[np.isfinite(seg)]
                    if seg.size > 10 and 20 < np.median(seg) < 250:
                        sbp[j], dbp[j] = seg.max(), seg.min()
        rr = np.diff(r, prepend=np.nan)
        m = (r[idx] - g) / 60
        beats = {"m": [round(float(v), 4) for v in m],
                 "foot": [r3((z["foot"][i] - r[i]) * 1000) for i in idx],
                 "up": [r3((z["up"][i] - r[i]) * 1000) for i in idx],
                 "peak": [r3((z["peak"][i] - r[i]) * 1000) for i in idx],
                 "good": [int(bool(good[i])) for i in idx],
                 "hr": [r3(60.0 / rr[i] if rr[i] > 0.25 else np.nan, 0) for i in idx],
                 "sbp": [r3(v, 0) for v in sbp], "dbp": [r3(v, 0) for v in dbp]}
        if has_art:
            beats["ea"] = [r3((art[i] - r[i]) * 1000) for i in idx]
            beats["ap"] = [r3((z["up"][i] - art[i]) * 1000) for i in idx]
        ctx = {}
        for name, tn in NUMERIC.items():
            tr_ = case.tracks.get(tn)
            if tr_ is not None and tr_.record_times.size:
                t_, x_ = numeric(tr_, lo, hi)
                pts = changes(t_, x_) if name.endswith("rate") else list(zip(t_.tolist(), x_.tolist()))
                if name.startswith("nibp"):     # a reading repeats every 2 s until the next: keep the changes
                    pts = changes(t_, x_)
                ctx[name] = [[round((t - g) / 60, 3), round(v, 1)] for t, v in pts]
        vent = []
        for tn in VENT:
            tr_ = case.tracks.get(tn)
            if tr_ is not None and tr_.record_times.size:
                t_, x_ = numeric(tr_, lo - 3600, hi)
                ch = changes(t_, x_)
                for t, v in ch[1:]:
                    if lo <= t <= hi:
                        vent.append([round((t - g) / 60, 3), tn.split("/")[1], round(v, 2)])
        ctx["vent"] = vent
        ev = case.tracks.get("EVENT")
        ctx["events"] = [[round((float(t) - g) / 60, 3), str(x)] for t, x in zip(ev.record_times, ev.values)
                         if lo <= float(t) <= hi] if ev is not None else []
        steps = []
        for e in events(R):
            if lo <= e["t"] <= hi:
                if e["kind"] == "uncertain":
                    steps.append({"lo": round((e["q05"] - g) / 60, 3), "hi": round((e["q95"] - g) / 60, 3), "p": e["p"]})
                else:
                    steps.append({"t": round((e["t"] - g) / 60, 3), "kind": e["kind"], "sign": e.get("sign", 1)})
        out.append({"id": iid, "verdict": A.get(iid, {}).get("verdict"), "note": A.get(iid, {}).get("note", ""),
                    "tooth": [round((float(k["t0"]) - g) / 60, 3), round((float(k["t1"]) - g) / 60, 3)],
                    "has_art": has_art, "steps": steps, "beats": beats, "ctx": ctx,
                    "drug_totals": clin.get(c, {})})
        print(iid, c, "beats", idx.size, "art", has_art, "nibp", len(ctx.get("nibp_sbp", [])), "vent", len(vent),
              "events", len(ctx["events"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps({"items": len(out), "bytes": Path(a.out).stat().st_size}))


if __name__ == "__main__":
    main()
