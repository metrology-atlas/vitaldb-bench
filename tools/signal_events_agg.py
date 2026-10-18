# SPDX-License-Identifier: MIT
"""Aggregations over the tools/signal_events.py cache. Aggregates only.

    python tools/signal_events_agg.py values DIR --part I --of N --out OUTDIR     (value histograms, resumable by part)
    python tools/signal_events_agg.py resolution OUTDIR                          (merge parts; converter resolution)

values: per waveform track, the summed count of every value over the cases of one part (cases with case % N == I).
resolution: merges the parts and, per track, reverse-engineers the converter: the codes used, their step (greatest
common divisor of the differences between used codes for integer tracks; the commonest difference for float
tracks), the full span in steps, and bits = log2(span in steps); missing codes inside the span; plus the value
histogram figure (one bin per possible code).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np


def values(src: Path, part: int, of: int, out: Path) -> None:
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    ncase = defaultdict(int)
    meta = {}
    for f in sorted(src.glob("*.npz")):
        if not f.stem.isdigit() or int(f.stem) % of != part:
            continue
        z = np.load(f)
        for k in z.files:
            if not k.endswith("|code"):
                continue
            tr = k[:-5]
            codes, counts = z[k], z[tr + "|count"]
            d = acc[tr]
            if codes.dtype.kind == "f":
                codes = np.round(codes, 6)
            for c_, n_ in zip(codes.tolist(), counts.tolist()):
                d[c_] += n_
            ncase[tr] += 1
            if tr not in meta:
                meta[tr] = json.loads(str(z[tr + "|meta"]))
    out.mkdir(parents=True, exist_ok=True)
    res = {tr: {"cases": ncase[tr], "meta": meta[tr], "codes": list(d.keys()), "counts": list(d.values())} for tr, d in acc.items()}
    (out / f"values_part{part}.json").write_text(json.dumps(res))
    print(part, {tr: v["cases"] for tr, v in res.items()})


def resolution(out: Path) -> None:
    tot: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    ncase = defaultdict(int)
    meta = {}
    for p in sorted(out.glob("values_part*.json")):
        for tr, v in json.loads(p.read_text()).items():
            for c_, n_ in zip(v["codes"], v["counts"]):
                tot[tr][c_] += n_
            ncase[tr] += v["cases"]
            meta.setdefault(tr, v["meta"])
    res = {}
    for tr, d in tot.items():
        codes = np.array(sorted(d), dtype=float)
        counts = np.array([d[c] for c in sorted(d)], dtype=np.int64)
        integer = meta[tr]["format"] not in ("float32", "float64")
        diffs = np.diff(codes)
        if integer:
            step = reduce(gcd, diffs.astype(np.int64).tolist(), 0) if diffs.size else 1
        else:
            dd = np.round(diffs, 6)
            vals, cnt = np.unique(dd[dd > 0], return_counts=True)
            step = float(vals[np.argmax(cnt)]) if vals.size else 0.0
            # float32 storage blurs a decimal step (0.05 reads 0.049988...); snap to 1, 2 or 5 x 10^k within 0.1%
            if step > 0:
                k = np.floor(np.log10(step))
                for m in (1, 2, 5, 10):
                    if abs(step / (m * 10 ** k) - 1) < 1e-3:
                        step = float(m * 10 ** k)
        span = (codes[-1] - codes[0]) / step if step else 0
        possible = int(round(span)) + 1
        res[tr] = {"cases": ncase[tr], "format": meta[tr]["format"], "storage_bits": meta[tr]["storage_bits"],
                   "gain": meta[tr]["gain"], "offset": meta[tr]["offset"], "fs": meta[tr]["fs"],
                   "min_code": codes[0], "max_code": codes[-1], "step": step, "possible_codes": possible,
                   "codes_used": int(codes.size), "missing_codes": possible - int(codes.size),
                   "bits_to_store_full_range": int(np.ceil(np.log2(possible))) if possible > 1 else None,
                   "samples": int(counts.sum()),
                   "share_at_min": round(float(counts[0] / counts.sum()), 6),
                   "share_at_max": round(float(counts[-1] / counts.sum()), 6)}
        # Robust range: a few garbage samples (BIS values to 1.6e15, CardioQ ABP to 95,124) must not set the bit
        # count. The core is the span between the 1e-7 and 1 - 1e-7 sample quantiles, widened to the nearest code that
        # is not separated from the body by more than 64 steps.
        cum = np.cumsum(counts) / counts.sum()
        lo = int(np.searchsorted(cum, 1e-7)); hi = int(np.searchsorted(cum, 1 - 1e-7))
        gapmax = 64 * step if step else 0
        while lo > 0 and codes[lo] - codes[lo - 1] <= gapmax:
            lo -= 1
        while hi < codes.size - 1 and codes[hi + 1] - codes[hi] <= gapmax:
            hi += 1
        core_possible = int(round((codes[hi] - codes[lo]) / step)) + 1 if step else 0
        out_samples = int(counts[:lo].sum() + counts[hi + 1:].sum())
        # Local step by range: the commonest spacing between used codes in 8 bands of equal sample count; a
        # non-uniform scale (companding) shows as different steps across bands.
        bands = []
        edges = np.searchsorted(cum, np.linspace(0, 1, 9)[1:-1])
        for a_, b_ in zip(np.r_[lo, edges], np.r_[edges, hi]):
            cc = codes[a_:b_ + 1]
            dd = np.round(np.diff(cc), 6)
            dd = dd[dd > 0]
            if dd.size:
                v_, n_ = np.unique(dd, return_counts=True)
                bands.append([float(cc[0]), float(cc[-1]), float(v_[np.argmax(n_)])])
        # Two different quantities, kept apart: the span of codes the data use (and the whole bits needed to store
        # that span), and the converter's resolution, which is an estimate made elsewhere and can only be >= the span.
        lattice = int(np.unique(np.round((codes[lo:hi + 1] - codes[lo]) / step)).size) if step else 0
        res[tr].update({"core_min": float(codes[lo]), "core_max": float(codes[hi]), "core_span_codes": core_possible,
                        "core_codes_used": lattice,
                        "bits_to_store_span": int(np.ceil(np.log2(core_possible))) if core_possible > 1 else None,
                        "outlier_samples": out_samples,
                        "outlier_codes": [float(c) for c in np.r_[codes[:lo], codes[hi + 1:]][:20]],
                        "step_by_band": bands})
        np.savez_compressed(out / f"hist_{tr}.npz", codes=codes, counts=counts)
    (out / "resolution.json").write_text(json.dumps(res, indent=1))
    for tr, r in sorted(res.items(), key=lambda kv: -kv[1]["samples"]):
        print(tr, {k: r[k] for k in ("cases", "format", "step", "core_min", "core_max", "core_span_codes",
                                     "core_codes_used", "bits_to_store_span", "outlier_samples")},
              "bands", [b[2] for b in r["step_by_band"]])


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "values":
        a = dict(zip(sys.argv[3::2], sys.argv[4::2]))
        values(Path(sys.argv[2]), int(a["--part"]), int(a["--of"]), Path(a["--out"]))
    else:
        resolution(Path(sys.argv[2]))


PANELS = [("SNUADC__ECG_II", 0, 1023), ("SNUADC__ECG_V5", 0, 1023), ("SNUADC__PLETH", 0, 1023), ("SNUADC__ART", 0, 1023),
          ("SNUADC__CVP", 0, 1023), ("SNUADC__FEM", 0, 1023), ("Primus__CO2", 0, 4095), ("Primus__AWP", 0, 4095),
          ("BIS__EEG1_WAV", None, None), ("CardioQ__FLOW", 0, 1023)]


def value_figure(out: Path) -> None:
    """Value histogram per waveform: one bin per possible code over the converter's range, log count axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = json.loads((out / "resolution.json").read_text())
    fig, axs = plt.subplots(5, 2, figsize=(12, 13))
    notes = {}
    for a_, (tr, lo, hi) in zip(axs.flat, PANELS):
        z = np.load(out / f"hist_{tr}.npz")
        c, n = z["codes"], z["counts"]
        r = R[tr]
        if tr.startswith("BIS"):
            step = 0.05
            k = np.round(c / step).astype(np.int64)
            keep = (c > -1500) & (c < 1850)
            lo_k, hi_k = -29534, 36001          # -1476.7 .. +1800.05 uV on the 0.05 grid
            x = np.arange(lo_k, hi_k + 1)
            y = np.zeros(x.size)
            np.add.at(y, k[keep] - lo_k, n[keep])
            xs = x * step
            out_n = int(n[~keep].sum())
            a_.set_xlabel("value (µV), one bin per 0.05 µV step", fontsize=8)
            title = f"{tr.replace('__', '/')}: 0.05 µV grid over 3,276.7 µV = 65,536 steps → 16 bits"
        else:
            x = np.arange(lo, hi + 1)
            y = np.zeros(x.size)
            keep = (c >= lo) & (c <= hi)
            np.add.at(y, (c[keep] - lo).astype(np.int64), n[keep])
            xs = x
            out_n = int(n[~keep].sum())
            a_.set_xlabel(f"stored code, one bin per code ({hi - lo + 1:,} codes shown)", fontsize=8)
            used = int((y > 0).sum())
            if tr.startswith("SNUADC"):
                head = f"10-bit (codes 0–1023 observed in the release), {used:,} codes used here"
            else:
                head = f"codes {int(r['min_code'])}–{int(r['max_code'])} observed, {used:,} used; bit depth not established"
            title = f"{tr.replace('__', '/')}: {head}; 1 code = {r['gain']:.4g} units"
        a_.step(xs, np.where(y > 0, y, np.nan), where="mid", lw=0.6, color="#08519c")
        a_.set_yscale("log")
        a_.set_title(title + (f"; {out_n:,} outlier samples not shown" if out_n else ""), fontsize=8, loc="left")
        a_.set_ylabel("samples", fontsize=8)
        a_.tick_params(labelsize=7.5)
        a_.spines[["top", "right"]].set_visible(False)
        notes[tr] = {"outliers_not_shown": out_n}
    fig.tight_layout()
    fig.savefig(out / "survey_value_histograms.png", dpi=140)
    plt.close(fig)
    print(notes)
