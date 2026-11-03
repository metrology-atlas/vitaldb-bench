# SPDX-License-Identifier: MIT
"""Figure pat-resolution (metrology-atlas study ppg-frame-clock): the time resolution of SNUADC ECG and PLETH and
the PAT resets, summed from the per-case output of tools/pat_res.py (cases
with both tracks at 500 Hz).

    python tools/pat_res.py --root $R --cases $W/pat/cases_pat300.json --out $W/pat_resolution/pat300_r6.jsonl \
        --reset-ms 6                                   (rerun until all 300 cases are done; about 15 minutes)
    python tools/pat_res_figure.py --scan $W/pat_resolution/pat300_r6.jsonl --example 2 --out $W/pat_resolution

Panels:

a. the change between consecutive inter-beat intervals (whole-sample fiducials: ECG R peak, PPG steepest
   upstroke), summed over cases, as a share of beats;
b. the median over cases of each track's spectrum (dB relative to its peak, 2 Hz bins);
c. per-case beat-to-beat PAT noise (sub-sample fiducials), with the floor that 2 ms quantisation of both ends alone
   would give (2 ms / sqrt(12) x sqrt(2)); a 15-beat median's noise is about SD x 1.2533 / sqrt(15);
d. the upward PAT step at each detected reset (median of 15 beats after minus 15 before) against the same tracker
   on the sign-flipped series, which has no real upward resets (the false-detection control);
e. the steps between 24 and 42 ms; the precision of one step is about the 15-beat median noise x sqrt(2);
f. the example case's 15-beat PAT medians with its detected resets (red: steps of 24-42 ms; grey dashed: others).

Writes summary.json and pat-resolution.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BIG = (24.0, 42.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--example", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in Path(a.scan).read_text().splitlines() if x.strip()]
    ok = [r for r in rows if "sub" in r and r["srate"]["ecg"] == 500.0 and r["srate"]["ppg"] == 500.0]   # 500 Hz tracks only
    lim = (len(ok[0]["ibi_dd_ecg"]) - 1) // 2
    dd = {k: np.sum([r[f"ibi_dd_{k}"] for r in ok], axis=0).astype(float) for k in ("ecg", "ppg")}
    hz = np.array(ok[0]["psd_ecg"]["hz"])
    psd = {k: np.nanmedian(np.array([r[f"psd_{k}"]["db"] for r in ok], float), axis=0) for k in ("ecg", "ppg")}
    sd = np.array([r["sub"]["sd_beat_ms"] for r in ok])
    quant = 2.0 / np.sqrt(12) * np.sqrt(2)
    steps = np.array([s for r in ok for s in r["sub"]["sawtooth"]["steps_ms"]])
    null = np.array([s for r in ok for s in r["sub"]["null"]["steps_ms"]])
    cases_with = sum(1 for r in ok if r["sub"]["sawtooth"]["steps_ms"])
    big = steps[(steps >= BIG[0]) & (steps <= BIG[1])]
    big_null = null[(null >= BIG[0]) & (null <= BIG[1])]
    med15 = float(np.median(sd)) * 1.2533 / np.sqrt(15)
    ex = next(r for r in rows if r["case"] == a.example)
    exs = ex["sub"]["sawtooth"]
    # the example's period: between consecutive large resets (steps of 24-42 ms; reset_t_s and steps_ms align
    # when the first reset is far enough from the start for its step to be measured, as in the example)
    tb = [t for t, s in zip(exs["reset_t_s"], exs["steps_ms"]) if BIG[0] <= s <= BIG[1]]
    tooth = [(p, None) for p in np.diff(tb)]
    S = {"scan": a.scan, "cases": len(rows), "cases_measured": len(ok),
         "not_500hz": sum(1 for r in rows if "sub" in r and r["srate"]["ecg"] != 500.0),
         "ibi_dd_beats": {k: int(v.sum()) for k, v in dd.items()},
         "even_share": {k: round(float(v[::2].sum() / v.sum()) if lim % 2 == 0 else float(v[1::2].sum() / v.sum()), 4) for k, v in dd.items()},
         "sd_beat_ms_q10_50_90": [round(float(x), 2) for x in np.quantile(sd, [0.1, 0.5, 0.9])],
         "quantisation_floor_ms": round(quant, 2), "median15_noise_ms": round(med15, 2),
         "step_precision_ms": round(med15 * np.sqrt(2), 2),
         "detections": int(steps.size), "null_detections": int(null.size), "cases_with_detections": cases_with,
         "near_32ms": int(big.size), "near_32ms_null": int(big_null.size),
         "big_median_iqr_ms": [round(float(x), 1) for x in np.quantile(big, [0.5, 0.25, 0.75])] if big.size else None,
         "example": {"case": a.example, "resets": exs["n_resets"], "large_reset_period_s_median": round(float(np.median([p for p, _ in tooth])), 1) if tooth else None,
                     "step_ms_median_big": round(float(np.median([s for s in exs["steps_ms"] if BIG[0] <= s <= BIG[1]])), 1),
                     "slope_ppm_median": round(float(np.median(exs["slopes_ppm"])), 1)}}
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(18.5, 10.5))
    ms = np.arange(-lim, lim + 1) * 2.0
    for k, col, lab in (("ecg", "C3", "ECG R-peak"), ("ppg", "C0", "PPG steepest upstroke")):
        ax[0, 0].step(ms, dd[k] / dd[k].sum(), where="mid", color=col, label=lab)
    ax[0, 0].set_xlim(-40, 40)
    ax[0, 0].set_xlabel("change between consecutive inter-beat intervals (ms; 1 sample = 2 ms)")
    ax[0, 0].set_ylabel("share of beats")
    ax[0, 0].set_title("a. Fiducials on whole samples: no coarser grid\n(every 2 ms step is populated)")
    ax[0, 0].legend()
    ax[0, 1].plot(hz, psd["ecg"], color="C3", label="ECG_II")
    ax[0, 1].plot(hz, psd["ppg"], color="C0", label="PLETH")
    ax[0, 1].set_xlim(0, 250)
    ax[0, 1].set_xlabel("Hz")
    ax[0, 1].set_ylabel("dB re. peak (median over cases)")
    ax[0, 1].set_title("b. Spectrum of the stored 500 Hz tracks\n(no images of a lower upstream rate)")
    ax[0, 1].legend()
    ax[0, 2].hist(np.clip(sd, 0, 12), bins=np.arange(0, 12.5, 0.5), color="C2")
    ax[0, 2].axvline(quant, color="k", ls="--")
    ax[0, 2].text(quant + 0.1, ax[0, 2].get_ylim()[1] * 0.9, f"quantisation alone\n(2 ms grid, both ends):\n{quant:.2f} ms", fontsize=9)
    ax[0, 2].set_xlim(0, 12)
    ax[0, 2].set_xlabel("beat-to-beat PAT noise SD (ms, sub-sample fiducials; over 12 in the last bin)")
    ax[0, 2].set_ylabel("cases")
    ax[0, 2].set_title(f"c. Per-beat PAT noise: median {np.median(sd):.1f} ms\n(15-beat median: about {med15:.1f} ms)")
    b = np.arange(-10, 62, 2)
    ax[1, 0].hist(steps, bins=b, color="C1", alpha=0.65, label=f"PAT ({steps.size} detections)")
    ax[1, 0].hist(null, bins=b, histtype="step", color="k", lw=2, label=f"sign-flipped control ({null.size})")
    ax[1, 0].axvline(32, color="C3", ls=":")
    ax[1, 0].text(33, ax[1, 0].get_ylim()[1] * 0.8, "32 ms\n= 16 samples", color="C3")
    ax[1, 0].set_xlabel("upward step in PAT at a detected reset (ms; 15 beats after minus 15 before)")
    ax[1, 0].set_ylabel("resets")
    ax[1, 0].set_title(f"d. Reset sizes, {cases_with} cases (6 ms threshold)\nonly the ~32 ms step exceeds the control")
    ax[1, 0].legend()
    b2 = np.arange(BIG[0], BIG[1] + 1, 1)
    ax[1, 1].hist(big, bins=b2, color="C1", alpha=0.65, label="PAT")
    ax[1, 1].hist(big_null, bins=b2, histtype="step", color="k", lw=2, label="control")
    ax[1, 1].axvline(32, color="C3", ls=":")
    q = S["big_median_iqr_ms"]
    ax[1, 1].set_xlabel("step size (ms)")
    ax[1, 1].set_ylabel("resets")
    ax[1, 1].set_title(f"e. The large resets: median {q[0]} ms, IQR {q[1]}-{q[2]}\n(measurement precision about {S['step_precision_ms']:.1f} ms)")
    ax[1, 1].legend()
    t, v = ex["sub"]["series"]
    ax[1, 2].scatter(np.array(t) / 60, v, s=4, color="C0")
    for tr in exs["reset_t_s"]:
        big_r = tr in tb
        ax[1, 2].axvline(tr / 60, color="C3" if big_r else "0.6", alpha=0.6, ls="-" if big_r else "--")
    e = S["example"]
    ax[1, 2].set_xlabel("minutes")
    ax[1, 2].set_ylabel("PAT, 15-beat median (ms)")
    ax[1, 2].set_title(f"f. Case {a.example}: {e['resets']} resets detected; between large ones {e['large_reset_period_s_median']:.0f} s"
                       f" ({e['large_reset_period_s_median'] / 60:.1f} min),\nlarge steps {e['step_ms_median_big']:.0f} ms; slope ~{e['slope_ppm_median']:.0f} ppm")
    fig.tight_layout()
    fig.savefig(out / "pat-resolution.png", dpi=100)


if __name__ == "__main__":
    main()
