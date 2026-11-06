# SPDX-License-Identifier: MIT
"""Summaries and figures from the release-wide spectra (metrology-atlas: the spectrum figure on each signal page and
the study of the waveforms' frequency content). Reads the output of tools/spectra_scan.py.

    python tools/spectra_agg.py $W/spectra --out $W/spectra/agg [--units $W/results/sync_full/units.csv]

Per part, a cache (OUT/cache_part_K.json) holds each recording's results, so a part already done is not read again.
Then, per track (each recording weighted once):

* **Spectrum:** across recordings, the median and the 25th-75th percentiles of each recording's median spectrum, in
  physical units per Hz (dB above rounding times q = 1/(6 fs) times the recording's step squared); the rounding floor
  q times the median step squared.
* **Low-pass:** spectra_scan.fit_rolloff on each recording's median spectrum. A fit is kept when its corner is clear
  of the bounds (above 5.5 Hz, below 0.95 of Nyquist), its RMS residual is under 1.5 dB and the spectrum at the corner
  stands at least 10 dB above the fitted floor (a corner fitted where the signal has already reached the floor
  describes the floor, not a filter); the corners' quantiles
  and a histogram (1 Hz bins below 60 Hz, 5 Hz above), by the SNUADC unit (from units.csv) where there is one.
* **Lines:** in each recording's median spectrum, a bin at 8 Hz or above (below it, the heart and breathing
  harmonics) more than 3 dB above the running median of the 9 bins around it is a peak; per bin, the share of
  recordings with a peak there. Bins where 10% or more do are listed.
* **Mains:** each recording's median over minutes of the 60, 120 and 180 Hz levels (line over its ring, dB): line
  (above +3 dB), notch (below -3 dB) or can't tell; and the share of minutes with a line above +6 dB.
* **Noisy and flat minutes:** a minute whose floor (floor_db) is 6 dB or more above the recording's median floor
  is noisy (such as an unattached lead picking up interference); one whose 5-15 Hz level is within 6 dB of its floor
  is flat (no signal). Their shares, and the run of noisy minutes that ends the recording (minutes).
* **Changes within a recording:** the minute series of hi_db - mid_db (30-45 Hz against 5-15 Hz), leaving out
  noisy and flat minutes. The best single split with at least 20 minutes either side; a change when the two medians
  differ by 10 dB or more and each side's median absolute deviation is under a third of that. Not for the EEG, whose
  spectrum follows the depth of anaesthesia, nor for tracks without a 30-45 Hz band.
* **Pairs:** quantiles of coherence at 60, 120 and 180 Hz and in the reference band; the time difference where
  coherence is 0.5 or more, as stored and folded into a quarter cycle either side of zero (a channel of opposite
  polarity is half a cycle out).

Writes OUT/summary.json and, with matplotlib, a figure per signal page (OUT/spectrum-VARIABLE.png) and one of all
tracks (OUT/spectra-all.png).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from spectra_scan import FEAT, MAINS, bins, fit_rolloff  # noqa: E402

PAGES = {"snuadc-ecg-ii": ["SNUADC/ECG_II"], "snuadc-art": ["SNUADC/ART"], "snuadc-pleth": ["SNUADC/PLETH"],
         "bis-eeg": ["BIS/EEG1_WAV", "BIS/EEG2_WAV"], "primus-co2": ["Primus/CO2"], "primus-awp": ["Primus/AWP"],
         "cardioq-abp": ["CardioQ/ABP"]}
FI = {k: i for i, k in enumerate(FEAT)}


def arr(x) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in x], float)


def centres(fs: float) -> np.ndarray:
    e = bins(fs)
    return np.sqrt(e[:-1] * e[1:])


def peaks(d: np.ndarray) -> np.ndarray:
    out = np.zeros(d.size, bool)
    for i in range(4, d.size - 4):
        w = d[i - 4:i + 5]
        if np.isfinite(d[i]) and np.isfinite(w).sum() >= 7 and d[i] - np.nanmedian(w) > 3 and d[i] == np.nanmax(w):
            out[i] = True
    return out


def change(y: np.ndarray, minlen: int = 20) -> dict | None:
    y = y[np.isfinite(y)]
    if y.size < 2 * minlen:
        return None
    best = None
    for k in range(minlen, y.size - minlen + 1, max(1, y.size // 200)):
        a, b = y[:k], y[k:]
        d = float(np.median(b) - np.median(a))
        mad = max(float(np.median(np.abs(a - np.median(a)))), float(np.median(np.abs(b - np.median(b)))))
        if best is None or abs(d) > abs(best["delta_db"]):
            best = {"at": k, "delta_db": round(d, 2), "mad_db": round(mad, 2)}
    best["change"] = abs(best["delta_db"]) >= 10 and best["mad_db"] < abs(best["delta_db"]) / 3
    return best


def minute_results(npzs: list[Path]) -> dict:
    """(case, track) -> mains medians, line-minute shares, flat share, change."""
    out = {}
    for p in npzs:
        z = np.load(p)
        names, case, track, feat = z["track_names"], z["case"], z["track"], z["feat"].astype(float)
        key = case.astype(np.int64) * 1000 + track
        order = np.argsort(key, kind="stable")
        ks, starts = np.unique(key[order], return_index=True)
        for k, s, e in zip(ks, starts, list(starts[1:]) + [key.size]):
            rows = order[s:e]
            rows = rows[np.argsort(z["minute"][rows])]
            F = feat[rows]
            c, name = int(k // 1000), str(names[k % 1000])
            fl = F[:, FI["floor_db"]]
            noisy = fl >= np.nanmedian(fl) + 6
            flat = F[:, FI["mid_db"]] < fl + 6
            tail = int(np.argmin(noisy[::-1])) if not noisy.all() else int(noisy.size)
            r = {"minutes": int(rows.size), "flat_share": round(float(np.mean(flat)), 4),
                 "noisy_share": round(float(np.mean(noisy)), 4), "noisy_tail_min": tail}
            for m in MAINS:
                v = F[:, FI[f"m{int(m)}_db"]]
                if np.isfinite(v).any():
                    r[f"m{int(m)}_db"] = round(float(np.nanmedian(v)), 2)
                    r[f"m{int(m)}_line_share"] = round(float(np.mean(v[np.isfinite(v)] > 6)), 4)
            keep = ~flat & ~noisy
            y = F[keep, FI["hi_db"]] - F[keep, FI["mid_db"]]
            if "EEG" not in name and np.isfinite(y).sum() >= 40:
                r["change"] = change(y)
            out[f"{c}|{name}"] = r
    return out


def part_cache(jl: Path, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    M = minute_results(sorted(jl.parent.glob(jl.stem + "_m*.npz")))
    recs = {}
    for line in jl.read_text().splitlines():
        r = json.loads(line)
        if r.get("error"):
            recs[str(r["case"])] = {"error": r["error"]}
            continue
        if r["tracks"] and not any(k.startswith(f"{r['case']}|") for k in M):
            continue                       # a line whose minutes were lost in a stopped run; scanned again elsewhere
        T = {}
        for name, t in r["tracks"].items():
            d = arr(t["psd_db"])
            fit = fit_rolloff(centres(t["fs"]), d, t["fs"])
            T[name] = {k: t[k] for k in ("fs", "unit", "step_phys", "minutes", "psd_db", "psd_db_p25", "psd_db_p75")}
            T[name]["fit"] = fit
            T[name]["peaks"] = np.flatnonzero(peaks(d)).tolist()
            T[name].update(M.get(f"{r['case']}|{name}", {}))
        recs[str(r["case"])] = {"tracks": T, "pairs": r["pairs"]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(recs))
    return recs


def q(v, p=(5, 25, 50, 75, 95), nd=2):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {"q": [round(float(x), nd) for x in np.percentile(v, p)], "n": int(v.size)} if v.size else None


def fit_ok(f: dict | None, fs: float, d: np.ndarray | None = None) -> bool:
    if not (bool(f) and 5.5 < f["fc_hz"] < 0.95 * fs / 2 and f["rms_db"] < 1.5):
        return False
    if d is None:
        return True
    i = int(np.argmin(np.abs(centres(fs) - f["fc_hz"])))
    return bool(np.isfinite(d[i]) and d[i] - f["floor_db"] >= 10)


def summarise(recs: dict, units: dict) -> tuple[dict, dict]:
    by = collections.defaultdict(list)
    for c, r in recs.items():
        for name, t in r.get("tracks", {}).items():
            by[name].append((int(c), t))
    S, curves = {}, {}
    for name, L in sorted(by.items()):
        fss = collections.Counter(t["fs"] for _, t in L)
        fs = fss.most_common(1)[0][0]
        L = [(c, t) for c, t in L if t["fs"] == fs]
        f = centres(fs)
        qq = 1 / (6 * fs)
        D = np.array([arr(t["psd_db"]) for _, t in L])
        steps = np.array([t["step_phys"] or np.nan for _, t in L])
        phys = qq * 10 ** (D / 10) * steps[:, None] ** 2
        with np.errstate(all="ignore"):
            curves[name] = {"f": f, "med": np.nanmedian(phys, 0), "p25": np.nanpercentile(phys, 25, 0),
                            "p75": np.nanpercentile(phys, 75, 0), "floor": qq * float(np.nanmedian(steps)) ** 2,
                            "unit": L[0][1]["unit"], "n": len(L), "fs": fs, "db_med": np.nanmedian(D, 0)}
        ok = [(c, t["fit"]) for c, t in L if fit_ok(t["fit"], fs, arr(t["psd_db"]))]
        fc = np.array([x["fc_hz"] for _, x in ok])
        hist = collections.Counter(int(x) if x < 60 else int(x // 5 * 5) for x in fc)
        by_unit = collections.defaultdict(list)
        for c, x in ok:
            if units.get(c):
                by_unit[units[c]].append(x["fc_hz"])
        P = np.array([np.isin(np.arange(f.size), t["peaks"]) for _, t in L]) & (f >= 8)[None, :]
        share = P.mean(0)
        lines = [{"f_hz": round(float(f[i]), 2), "share": round(float(share[i]), 3),
                  "db_above_rounding": None if not np.isfinite(curves[name]["db_med"][i])
                  else round(float(curves[name]["db_med"][i]), 1)} for i in np.flatnonzero(share >= 0.10)]
        mains = {}
        for m in MAINS:
            v = [t.get(f"m{int(m)}_db") for _, t in L]
            v = np.array([x for x in v if x is not None], float)
            if v.size:
                mains[str(int(m))] = {"recordings": int(v.size), "line": int(np.sum(v > 3)), "notch": int(np.sum(v < -3)),
                                      "cant_tell": int(np.sum(np.abs(v) <= 3)), "level_db": q(v),
                                      "line_minutes_share": q([t.get(f"m{int(m)}_line_share") for _, t in L], nd=4)}
        ch = [(c, t["change"]) for c, t in L if t.get("change")]
        S[name] = {
            "recordings": len(L), "other_rates": {str(k): v for k, v in fss.items() if k != fs},
            "unit": L[0][1]["unit"], "fs": fs, "minutes": int(sum(t["minutes"] for _, t in L)),
            "steps": {str(round(k, 6)): v for k, v in collections.Counter(round(s, 6) for s in steps).most_common(6)},
            "floor_db": q([t.get("psd_db")[-3] if t.get("psd_db")[-3] is not None else np.nan for _, t in L]),
            "flat_share": q([t.get("flat_share") for _, t in L], nd=4),
            "noisy_share": q([t.get("noisy_share") for _, t in L], nd=4),
            "noisy_tail_min": q([t.get("noisy_tail_min") for _, t in L], (50, 75, 90, 95, 99), nd=0),
            "noisy_tail_hist": dict(sorted(collections.Counter(min(int(t.get("noisy_tail_min") or 0), 30)
                                                               for _, t in L).items())),
            "lowpass": {"fitted": len(ok), "fc_hz": q(fc), "order_ok_order": q([x["order"] for _, x in ok if x["order_ok"]]),
                        "beta": q([x["beta"] for _, x in ok]), "hist": dict(sorted(hist.items())),
                        "by_snuadc_unit": {u: q(v) for u, v in sorted(by_unit.items()) if len(v) >= 5}},
            "lines": lines, "mains": mains,
            "changes": {"assessed": len(ch), "changed": sum(x["change"] for _, x in ch),
                        "delta_db": q([x["delta_db"] for _, x in ch if x["change"]]),
                        "examples": sorted([c for c, x in ch if x["change"]])[:20]},
        }
    pairs = collections.defaultdict(list)
    for r in recs.values():
        for k, p in r.get("pairs", {}).items():
            pairs[k].append(p)
    PS = {}
    for k, L in sorted(pairs.items()):
        e = {"recordings": len(L), "coh_ref": q([p.get("coh_ref") for p in L], nd=4)}
        for m in MAINS:
            m = int(m)
            if any(f"coh_{m}" in p for p in L):
                e[f"coh_{m}"] = q([p.get(f"coh_{m}") for p in L], nd=4)
                v = [p[f"later_us_{m}"] for p in L if (p.get(f"coh_{m}") or 0) >= 0.5]
                T = 1e6 / m
                e[f"later_us_{m}_coh05"] = q(v, nd=1)
                e[f"later_us_{m}_coh05_folded"] = q([(x + T / 4) % (T / 2) - T / 4 for x in v], nd=1)
        PS[k] = e
    return {"tracks": S, "pairs": PS}, curves


def figures(curves: dict, S: dict, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []

    def draw(ax, name, colour="#1f4e79", label=None):
        c = curves[name]
        ok = np.isfinite(c["med"])
        ax.fill_between(c["f"][ok], c["p25"][ok], c["p75"][ok], color=colour, alpha=0.18, lw=0)
        ax.loglog(c["f"][ok], c["med"][ok], color=colour, lw=1.4, label=label or f"median, {c['n']:,} recordings")
        ax.axhline(c["floor"], color="#b03a2e", ls="--", lw=0.9)
        ax.text(c["f"][ok][0], c["floor"] * 1.25, "rounding floor", color="#b03a2e", fontsize=7, va="bottom")
        ax.set_xlim(c["f"][ok][0], c["fs"] / 2)
        fc = S[name]["lowpass"]["fc_hz"]
        if fc and S[name]["lowpass"]["fitted"] >= 0.5 * S[name]["recordings"]:
            x = fc["q"][2]
            ax.axvline(x, color="0.4", ls=":", lw=0.9)
            ax.text(x, ax.get_ylim()[1], f" low-pass {x:.0f} Hz", fontsize=7, va="top", color="0.3", rotation=90)
        for ln in S[name]["lines"]:
            if ln["share"] >= 0.2:
                i = int(np.argmin(np.abs(c["f"] - ln["f_hz"])))
                ax.annotate(f"{ln['f_hz']:.3g} Hz\n{ln['share']:.0%}", (c["f"][i], c["med"][i]), fontsize=6,
                            xytext=(0, 12), textcoords="offset points", ha="center", color="0.25",
                            arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.5})
        unit = c["unit"] or "a.u."
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel(f"power ({unit}²/Hz)")
        ax.grid(True, which="major", color="0.9", lw=0.5)

    for page, names in PAGES.items():
        names = [n for n in names if n in curves]
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for n, col in zip(names, ("#1f4e79", "#2e8b57")):
            draw(ax, n, col, label=f"{n.split('/')[1]}: median, {curves[n]['n']:,} recordings" if len(names) > 1 else None)
        ax.legend(fontsize=7, loc="upper right", frameon=False)
        fig.tight_layout()
        p = out / f"spectrum-{page}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p.name)
    names = sorted(n for n in curves if "#" not in n and curves[n]["n"] >= 10)   # not the rare second copies
    k = int(np.ceil(len(names) / 3))
    fig, axs = plt.subplots(k, 3, figsize=(14, 3.6 * k))
    for ax, n in zip(axs.ravel(), names):
        draw(ax, n)
        ax.set_title(f"{n} ({curves[n]['fs']:g} Hz)", fontsize=9)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "spectra-all.png", dpi=110)
    plt.close(fig)
    return made + ["spectra-all.png"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="the scan's folder: part_K.jsonl and part_K_mNNN.npz")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--units", type=Path, help="units.csv (case, snuadc_unit, ...) for the corners by unit")
    ap.add_argument("--parts-only", action="store_true", help="make the part caches and stop")
    a = ap.parse_args()
    recs = {}
    for jl in sorted(a.folder.glob("part_*.jsonl")):
        recs.update(part_cache(jl, a.out / f"cache_{jl.stem}.json"))
        print(f"{jl.name}: {len(recs)} recordings so far", flush=True)
    if a.parts_only:
        return
    units = {}
    if a.units and a.units.exists():
        units = {int(r["case"]): r["snuadc_unit"] for r in csv.DictReader(open(a.units)) if r.get("snuadc_unit")}
    S, curves = summarise(recs, units)
    S["recordings"] = len(recs)
    S["errors"] = sum(1 for r in recs.values() if r.get("error"))
    S["curves"] = {n: {"f_hz": np.round(c["f"], 4).tolist(), "floor": c["floor"], "unit": c["unit"], "n": c["n"],
                       **{k: [None if not np.isfinite(v) else float(f"{v:.4g}") for v in c[k]]
                          for k in ("med", "p25", "p75")}} for n, c in curves.items()}
    try:
        S["figures"] = figures(curves, S["tracks"], a.out)
    except ImportError:
        S["figures"] = []
    (a.out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: {kk: v[kk] for kk in ("recordings", "lowpass", "mains")} for k, v in S["tracks"].items()},
                     indent=1)[:6000])


if __name__ == "__main__":
    main()
