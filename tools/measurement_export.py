# SPDX-License-Identifier: MIT
"""Per-case PAT measurement records (physiosync.pat_measurement_model) and timing-quality labels.

    python tools/measurement_export.py --beats DIR --art DIR2 --priors P --out-dir OUT --labels L.csv
        [--adopted 0.27] [--min-tier B] [--part 0 --of 1]

With --saw-labels, also one row per tooth: sawtooth phase, slope, duration, the fiducials' resolution
(ECG one sample; PPG per recording: frame rounding if it locks to the 30 Hz grid, else jitter) and the confidence (physiosync.sawtooth_labels).

Writes OUT/<case>.json (one record per case: segments, sync-error models, slips, offset, fiducials) and
appends the quality intervals to L.csv (case, label, start_s, end_s, tier, applied, reason). Stays on this
machine; only aggregates are reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "physiosync"))
from physiosync import (
    CHAIN_FIELDS,
    LABEL_FIELDS,
    OFFSET_FIELDS,
    chain_labels,
    fiducial_error_model,
    fit_ecg_ppg_delay,
    offset_label,
    pat_measurement_model,
    sawtooth_labels,
)

F = 1 / 30
# VitalDB-specific (from tools/abp_validate.py, 201 cases): mid-tooth drift left after correction, median
# per tier, measured against arterial pressure (an upper bound: physiology included). Used as the error SD.
VITALDB_RESIDUAL_SD_S = {"A": 0.0019, "B": 0.0036, "C": 0.0050}
# chained teeth (tools/grid_mode.py, 150 cases): mid-tooth drift left, median and 90th percentile
VITALDB_CHAIN_ERROR_S = (0.0032, 0.0196)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--saw-labels", default=None, help="CSV of per-tooth sawtooth labels (parameters, resolution, confidence)")
    ap.add_argument("--chain-labels", default=None, help="CSV of rate-consistent chained teeth (separate track)")
    ap.add_argument("--fiducial", choices=("upstroke", "foot"), default="foot")
    ap.add_argument("--foot", default=None, help="DIR of foot_export results (needed for --fiducial foot)")
    ap.add_argument("--setups", default=None, help="units.csv: SNUADC port and clock mode per case (setup names)")
    ap.add_argument("--setup-offsets", default=None, help="JSON from setup_offsets: one hardware delay per setup")
    ap.add_argument("--offset-labels", default=None, help="CSV of the offset applied to each recording")
    ap.add_argument("--adopted", type=float, default=0.27)
    ap.add_argument("--min-tier", default="B")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    pri = {int(k): v for k, v in json.loads(Path(a.priors).read_text()).items()}
    setup_of: dict[int, str] = {}
    if a.setups:
        for r_ in csv.DictReader(open(a.setups)):
            if r_["snuadc_drift_ppm"]:
                setup_of[int(r_["case"])] = r_["snuadc_port"] + (" step" if float(r_["snuadc_drift_ppm"]) < -20 else " small")
    so = json.loads(Path(a.setup_offsets).read_text()) if a.setup_offsets else None
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [f for i, f in enumerate(sorted(Path(a.beats).glob("*.npz"))) if i % a.of == a.part]
    new = not Path(a.labels).exists()
    n_ok = 0
    with open(a.labels, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["case", "label", "start_s", "end_s", "tier", "applied", "reason"])
        for f in files:
            c = int(f.stem)
            try:
                z = np.load(f)
                r_t, p_t = z["r_t"], z["p_t"]
                af = Path(a.art) / f.name
                a_t = np.load(af)["a_t"] if af.exists() else None
                fid_index = z["up_index"] if "up_index" in z.files else None
                if a.fiducial == "foot":
                    p_t = np.load(Path(a.foot) / f.name)["f_t"]
                    ok = np.isfinite(p_t)
                    r_t, p_t = r_t[ok], p_t[ok]
                    a_t = a_t[ok] if a_t is not None else None
                    fid_index = (int(z["ppg_first_index"]) + np.round((p_t - float(z["ppg_t0"])) * 500.0)).astype(np.int64)
                rec_off = None
                if so is not None:
                    rec_off = so["by_recording"].get(str(c)) or so["all"]
                if a_t is not None and np.isfinite(a_t).mean() < 0.3:
                    a_t = None
                m = fit_ecg_ppg_delay(r_t, p_t, adopted_pat_s=a.adopted, frame_s=F, matched=True,
                                      rate_prior_ppm=pri.get(c), apply_min_tier=a.min_tier, ref_t=a_t,
                                      offset_s=None if rec_off is None else rec_off["offset_s"])
                kw = {"ecg_fiducial": "R peak (SNUADC ECG_II, interpolated)",
                          "ppg_fiducial": ("foot (intersecting tangent) of 0.5-10 Hz band-passed PLETH (SNUADC)"
                                           if a.fiducial == "foot" else
                                           "steepest upstroke of 0.5-10 Hz band-passed PLETH (SNUADC, interpolated)"),
                          "adopted_pat_basis": "placeholder pending choice (see physiosync.pat_reference)",
                          "r_t": r_t, "ppg_t": p_t, "residual_sd_s": VITALDB_RESIDUAL_SD_S,
                          "extra": {"case": c, "reference": "ART upstroke" if a_t is not None else None,
                                 "rate_prior_ppm": pri.get(c), "setup": setup_of.get(c)}}
                rec = pat_measurement_model(m, ecg_fs=500.0, ppg_fs=500.0, **kw)
                # this recording's PPG fiducial error: frame rounding if it locks to the 30 Hz grid, else
                # jitter from the beat-to-beat PAT noise of its A/B teeth (upper bound), less the ECG's part
                noise = [s_["beat_noise_s"] for s_ in rec["segments"] if s_["tier"] in ("A", "B") and s_.get("beat_noise_s")]
                pat_noise = float(np.median(noise)) if noise else (rec["beat_noise"] or {}).get("sd_s")
                ppg_err = fiducial_error_model(fid_index, 500.0, frame_samples=500.0 / 30, pat_noise_s=pat_noise,
                                               other_sd_s=0.002 / np.sqrt(12), ibi_range=(150, 1000))
                rec = pat_measurement_model(m, ecg_fs=500.0, ppg_fs=500.0, ppg_error=ppg_err, **kw)
                (out / f"{c}.json").write_text(json.dumps(rec, allow_nan=False))
                for q in m.quality_intervals():
                    w.writerow([c, f"ppg_timing:{q['tier']}", round(q["start"], 3), round(q["end"], 3), q["tier"],
                                q["applied"], q["reason"]])
                if a.saw_labels:
                    p_ = Path(a.saw_labels)
                    first = not p_.exists()
                    with p_.open("a", newline="") as sf:
                        sw = csv.DictWriter(sf, fieldnames=["case", *LABEL_FIELDS])
                        if first:
                            sw.writeheader()
                        for d in sawtooth_labels(rec):
                            sw.writerow({"case": c, **d})
                if a.offset_labels:
                    p_ = Path(a.offset_labels)
                    first = not p_.exists()
                    with p_.open("a", newline="") as of_:
                        ow = csv.DictWriter(of_, fieldnames=["case", "setup", *OFFSET_FIELDS])
                        if first:
                            ow.writeheader()
                        d = offset_label(rec, basis=(rec_off or {}).get("basis", "per recording (snapped)"),
                                         n_recordings=(rec_off or {}).get("n"), setup_iqr_s=(rec_off or {}).get("iqr_s"))
                        ow.writerow({"case": c, "setup": setup_of.get(c), **d})
                if a.chain_labels:
                    rate = m.qc.get("recording_rate_ppm") or (pri.get(c) if not m.qc.get("rate_conflict") else None)
                    p_ = Path(a.chain_labels)
                    first = not p_.exists()
                    with p_.open("a", newline="") as cf:
                        cw = csv.DictWriter(cf, fieldnames=["case", *CHAIN_FIELDS])
                        if first:
                            cw.writeheader()
                        for d in chain_labels(m, r_t, p_t, rate_ppm=rate, error_s=VITALDB_CHAIN_ERROR_S):
                            cw.writerow({"case": c, **d})
                n_ok += 1
            except Exception as ex:  # noqa: BLE001
                print(json.dumps({"case": c, "error": f"{type(ex).__name__}: {ex}"[:300]}))
    print(json.dumps({"part": a.part, "records": n_ok}))


if __name__ == "__main__":
    main()
