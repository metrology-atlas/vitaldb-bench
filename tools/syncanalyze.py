# SPDX-License-Identifier: MIT
"""Pass 2 of the clock re-synchronisation study: tests on each run boundary, and drift per device.

    python tools/syncanalyze.py --scan DIR --out analysis.json [--plots DIR]

Reads what ``syncscan.py`` wrote. For every waveform run boundary:

- **device-wide**: do the device's other waveform tracks have a boundary at the same moment (within
  0.25 s) with the same jump (within 2 ms)? A clock correction moves every track of a device at once;
  a dropout in one channel does not.
- **beats** (ECG_II, ART, PLETH, where snippets were kept): beats are detected on each side, and the
  interval across the join is measured twice: with the two sides butted together (the jump was a clock
  correction, nothing is missing) and with the jump inserted (the gap is real). Each is scored against
  the neighbouring beat intervals as a robust z (median, 1.4826 x MAD).
- **step**: the sample-to-sample step across the join against the steps inside the snippets.

The same tests run at record boundaries inside long runs (controls: continuous by construction), which
gives each statistic's distribution when nothing is missing.

Clock corrections: a boundary is a correction candidate when its jump is at most ``--max-jump-s``,
it is not a step back in time, and every other waveform track of the device has the same jump at the
same moment. Drift comes only from a candidate whose previous boundary on that track was a candidate
too (so the time since the last correction is known, and a case's first interval is never used):
ppm = jump / time since that correction; negative = the device clock runs fast against the recording
PC. Per case and device the duration-weighted median of those gives one rate. Output holds only
aggregates: counts, quantiles, histograms, and one drift value per case and device
(``<out>.drift.csv``).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy import signal

BEAT_BAND = {"SNUADC/ECG_II": (8.0, 25.0), "SNUADC/ART": (0.5, 8.0), "SNUADC/PLETH": (0.5, 8.0)}
EDGE_S = 0.15  # beats this close to a snippet edge are unreliable after filtering


def beat_times(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Beat times (s) in one snippet: band-pass, then peaks at least 0.3 s apart."""
    if x.size < 2 * fs:
        return np.empty(0)
    sos = signal.butter(2, band, btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x.astype(np.float64))
    if band[0] >= 5:  # ECG: energy of the QRS
        y = np.abs(y)
    prom = 0.35 * (np.quantile(y, 0.99) - np.quantile(y, 0.01))
    pk, _ = signal.find_peaks(y, distance=int(0.3 * fs), prominence=prom)
    t = pk / fs
    return t[(t > EDGE_S) & (t < x.size / fs - EDGE_S)]


def robust_z(v: float, ref: np.ndarray) -> float | None:
    if ref.size < 3:
        return None
    med = float(np.median(ref))
    mad = 1.4826 * float(np.median(np.abs(ref - med)))
    return (v - med) / max(mad, 0.02 * med, 0.01)


def join_tests(x: np.ndarray, pre: int, between: int, fs: float, band: tuple[float, float], jump: float) -> dict:
    """Is the signal continuous across a boundary?

    ``x`` = ``pre`` samples of the run before, ``between`` samples of any short runs, then the run after,
    butted together as if nothing were missing. The beat interval that spans the boundary (the cut at
    ``pre``) is scored against the beat intervals wholly before or after it, as it stands (butted) and
    with the jump inserted at the cut (a real gap of ``jump`` seconds).
    """
    out: dict = {}
    xf = x.astype(np.float64)
    d = np.abs(np.diff(xf))
    step = abs(xf[pre] - xf[pre - 1])
    out["step_pct"] = float((d < step).mean())  # where the step across the cut falls among all steps
    t = beat_times(x, fs, band)
    cut = pre / fs
    hi = (pre + between) / fs
    before, after = t[t < cut], t[t >= hi]
    if before.size >= 3 and after.size >= 3:
        rr = np.concatenate([np.diff(before), np.diff(after)])
        span = after[0] - before[-1]
        inside = int(((t >= cut) & (t < hi)).sum())  # beats detected in the short runs' samples
        med = float(np.median(rr))
        out["rr_med_s"] = med
        out["z_butt"] = robust_z(span / (inside + 1), rr)
        # a real gap may hide whole beats: the nearest whole number of intervals is the fairest reading
        out["z_gap"] = robust_z((span + jump) / max(1, round((span + jump) / med)), rr) if span + jump > 0 else None
        out["beats_between"] = inside
    return out


def jump_class(j: float, period: float) -> str:
    p = j / period
    if p < -0.5:
        return "overlap>1s" if j < -1 else "overlap<=1s"
    if p <= 0.5:
        return "join"
    if j <= 1:
        return "gap<=1s"
    if j <= 10:
        return "gap1-10s"
    return "gap>10s"


STEP_ONLY = ("BIS/EEG1_WAV", "Primus/CO2")  # no beats: sample-level continuity only
PREFERRED = {"SNUADC": "SNUADC/ECG_II", "BIS": "BIS/EEG1_WAV"}


def collect_case(c: dict, snipdir: Path, a) -> tuple[list, list]:
    """Rows (one per macro boundary of each track) and control tests for one case."""
    rows, controls = [], []
    snips_path = snipdir / f"{c['case']:04d}.npz"
    snips = np.load(snips_path) if snips_path.exists() else {}
    by_dev = defaultdict(list)
    for name, t in c["tracks"].items():
        by_dev[t["device"]].append(name)
    for name, t in c["tracks"].items():
        fs, period = t["srate"], 1.0 / t["srate"]
        dev = t["device"]
        typ = c["devices"].get(dev, {}).get("type", dev)
        others = [o for o in by_dev[dev] if o != name]
        prev_event = False
        for k, b in enumerate(t["macro"]):
            r = {"case": c["case"], "track": name, "device": dev, "type": typ, "port": c["devices"].get(dev, {}).get("port", ""),
                 "end_s": b["end_s"], "jump_s": b["jump_s"], "span_s": b["span_s"], "dur_before_s": b["dur_before_s"],
                 "dur_after_s": b["dur_after_s"], "short_runs_between": b["short_runs_between"], "ppm": b["ppm"],
                 "step_back": b["step_back"], "cls": jump_class(b["jump_s"], period), "first": k == 0}
            matched = []
            for o in others:
                for ob in c["tracks"][o]["macro"]:
                    if abs(ob["end_s"] - b["end_s"]) <= 0.5:
                        matched.append(ob["jump_s"] - b["jump_s"])
                        break
            r["others_on_device"] = len(others)
            r["others_same_moment"] = len(matched)
            r["others_same_jump"] = sum(abs(m) <= 0.002 for m in matched)
            r["device_wide"] = bool(others) and r["others_same_jump"] == len(others)
            # a clock-correction candidate: small, not a step back, and (where the device has other
            # waveform tracks) the same jump on all of them at once
            r["event"] = abs(b["jump_s"]) <= a.max_jump_s and not b["step_back"] and (r["device_wide"] or not others)
            # its drift is usable only when the time since the previous correction is known: the
            # previous boundary on this track was a correction too (so never the first interval)
            r["drift_ok"] = r["event"] and prev_event and b["span_s"] >= a.min_span_s
            prev_event = r["event"]
            if b.get("snip"):
                x = snips[b["snip"]]
                between = x.size - 2 * b["snip_pre"]
                if name in BEAT_BAND:
                    r |= join_tests(x, b["snip_pre"], between, fs, BEAT_BAND[name], b["jump_s"])
                elif name in STEP_ONLY:
                    r |= step_test(x, b["snip_pre"])
            rows.append(r)
        for ctl in t.get("controls", []):
            if name in BEAT_BAND:
                controls.append({"track": name, **join_tests(snips[ctl["snip"]], ctl["snip_pre"], 0, fs, BEAT_BAND[name], 0.0)})
            elif name in STEP_ONLY:
                controls.append({"track": name, **step_test(snips[ctl["snip"]], ctl["snip_pre"])})
    return rows, controls


def main() -> None:  # noqa: C901, PLR0912, PLR0915 - one linear analysis
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True, nargs="+", help="one or more syncscan output directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plots")
    ap.add_argument("--max-jump-s", type=float, default=2.0, help="largest jump that can be a clock correction")
    ap.add_argument("--min-span-s", type=float, default=600.0, help="shortest time between corrections used for drift")
    ap.add_argument("--stage", choices=("collect", "all"), default="all", help="collect: only cache per-case rows")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=1e9)
    a = ap.parse_args()
    import time  # noqa: PLC0415

    # stage 1 (resumable, can run as several parts): per-case rows cached beside the output. The scan
    # files are streamed a line at a time; only this part's cases are parsed in full.
    cache_dir = Path(a.out).parent / (Path(a.out).stem + "_rows")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"rows_{a.part}of{a.of}.jsonl"
    done = {json.loads(x)["case"] for x in cache.read_text().splitlines() if x.strip()} if cache.exists() else set()
    t0 = time.time()
    n_cases, errors, secs, mb, left, all_ids = 0, [], 0.0, 0.0, 0, set()
    with cache.open("a") as fh:
        for d in map(Path, a.scan):
            with (d / "boundaries.jsonl").open() as src:
                for x in src:
                    if not x.strip():
                        continue
                    caseid = int(x[x.index('"case": ') + 8 : x.index(",", x.index('"case": '))])
                    mine = caseid % a.of == a.part and caseid not in done
                    if not mine and a.stage == "collect":
                        continue
                    c = json.loads(x)
                    if "error" in c:
                        errors.append({"case": c["case"], "error": c["error"]})
                        continue
                    n_cases += 1
                    all_ids.add(caseid)
                    secs += c["seconds"]
                    mb += c["file_mb"]
                    if not mine:
                        continue
                    if time.time() - t0 > a.time_budget:
                        left += 1
                        continue
                    r, k = collect_case(c, d / "snips", a)
                    fh.write(json.dumps({"case": caseid, "rows": r, "controls": k}, default=float) + "\n")
                    done.add(caseid)
    if a.stage == "collect" or left:
        print(json.dumps({"part": a.part, "of": a.of, "cases_left": left, "seconds": round(time.time() - t0, 1)}))
        return
    rows, controls, have = [], [], set()
    for f in sorted(cache_dir.glob("rows_*.jsonl")):
        for x in f.read_text().splitlines():
            if x.strip():
                e = json.loads(x)
                if e["case"] in have:
                    continue
                have.add(e["case"])
                rows += e["rows"]
                controls += e["controls"]
    missing = sorted(all_ids - have)
    if missing:
        print(json.dumps({"cases_not_collected": len(missing)}))
        return

    def q(vals, ps=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict | None:
        v = np.array([x for x in vals if x is not None and np.isfinite(x)])
        return {str(p): round(float(np.quantile(v, p)), 4) for p in ps} | {"n": int(v.size)} if v.size else None

    res: dict = {"cases": n_cases, "cases_with_errors": errors, "seconds_per_mb": secs / max(mb, 1e-9), "boundaries": len(rows),
                 "short_runs_in_macro": sum(r["short_runs_between"] for r in rows)}
    types = sorted({r["type"] for r in rows})
    res["by_type_and_class"] = {t: dict(Counter(r["cls"] for r in rows if r["type"] == t)) for t in types}
    res["device_wide"] = {t: {cls: {"n": len(x), "same_moment": sum(r["others_same_moment"] == r["others_on_device"] for r in x),
                                    "same_jump": sum(r["device_wide"] for r in x)}
                              for cls in sorted({r["cls"] for r in rows if r["type"] == t and r["others_on_device"]})
                              for x in [[r for r in rows if r["type"] == t and r["cls"] == cls and r["others_on_device"]]]}
                          for t in types}
    # one row per device-level event: keep the preferred track (ECG_II for SNUADC, EEG1 for BIS), else the first
    pref = {}
    for r in rows:
        key = (r["case"], r["device"])
        best = PREFERRED.get(r["device"])
        cur = pref.get(key)
        if cur is None or (r["track"] == best and cur != best) or (cur != best and r["track"] < cur):
            pref[key] = r["track"]
    dev_rows = [r for r in rows if pref[(r["case"], r["device"])] == r["track"]]
    res["events"] = {t: {"candidates": sum(r["event"] for r in dev_rows if r["type"] == t),
                         "drift_usable": sum(r["drift_ok"] for r in dev_rows if r["type"] == t),
                         "jump_s": q([r["jump_s"] for r in dev_rows if r["type"] == t and r["event"]], (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)),
                         "jump_mode_10ms": Counter(round(r["jump_s"], 2) for r in dev_rows if r["type"] == t and r["event"]).most_common(8),
                         "span_s": q([r["span_s"] for r in dev_rows if r["type"] == t and r["drift_ok"]])}
                     for t in types}
    res["controls"] = {tr: {"z_butt": q(c.get("z_butt") for c in controls if c["track"] == tr),
                            "step_pct": q(c.get("step_pct") for c in controls if c["track"] == tr)} for tr in (*BEAT_BAND, *STEP_ONLY)}
    res["join_tests"] = {}
    for tr in (*BEAT_BAND, *STEP_ONLY):
        for ev in (True, False):
            for cls in ("overlap<=1s", "gap<=1s", "gap1-10s"):
                x = [r for r in rows if r["track"] == tr and r["cls"] == cls and r["event"] == ev and "step_pct" in r]
                if not x:
                    continue
                e = {"n": len(x), "step_pct": q(r["step_pct"] for r in x)}
                if tr in BEAT_BAND:
                    e |= {"z_butt": q(r.get("z_butt") for r in x), "z_gap": q(r.get("z_gap") for r in x),
                          "butt_fits_better": sum(abs(r["z_butt"]) < abs(r["z_gap"]) for r in x
                                                  if r.get("z_butt") is not None and r.get("z_gap") is not None)}
                res["join_tests"][f"{tr} | {'event' if ev else 'not event'} | {cls}"] = e
    # drift: per device and case, from corrections whose preceding boundary was also a correction
    per_case = defaultdict(dict)
    within = defaultdict(list)
    for (case, dev), track in pref.items():
        v = [r for r in dev_rows if r["case"] == case and r["device"] == dev and r["drift_ok"]]
        if not v:
            continue
        ppm = np.array([r["ppm"] for r in v])
        w = np.array([r["span_s"] for r in v])
        order = np.argsort(ppm)
        med = float(ppm[order][np.searchsorted(np.cumsum(w[order]) / w.sum(), 0.5)])
        typ = v[0]["type"]
        per_case[case][typ] = {"ppm": med, "n": int(ppm.size), "span_s": float(w.sum()), "track": track, "port": v[0]["port"],
                               "iqr": float(np.subtract(*np.quantile(ppm, [0.75, 0.25]))) if ppm.size > 1 else None}
        if ppm.size >= 3:
            within[typ].extend((ppm - med).tolist())
    res["drift_per_case"] = {str(k): v for k, v in sorted(per_case.items())}
    pc = (0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995)
    res["drift_between_cases"] = {t: q([v[t]["ppm"] for v in per_case.values() if t in v], pc) for t in types}
    res["drift_within_case"] = {t: q(v, pc) for t, v in within.items()}
    res["ports"] = {t: dict(Counter(r["port"] for r in dev_rows if r["type"] == t and r["first"]).most_common(12)) for t in types}
    Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    with Path(a.out).with_suffix(".drift.csv").open("w") as fh:
        fh.write("case,device_type,port,drift_ppm,n_corrections,span_s,iqr_ppm\n")
        for case, d in sorted(per_case.items()):
            for t, v in sorted(d.items()):
                fh.write(f"{case},{t},{v['port']},{v['ppm']:.4f},{v['n']},{v['span_s']:.0f},{'' if v['iqr'] is None else round(v['iqr'], 4)}\n")
    if a.plots:
        plots(res, dev_rows, rows, controls, within, Path(a.plots))
    print(json.dumps({k: res[k] for k in ("cases", "boundaries", "seconds_per_mb")}))


def step_test(x: np.ndarray, pre: int) -> dict:
    """Where the sample-to-sample step across the cut falls among all the steps in the snippet."""
    xf = x.astype(np.float64)
    d = np.abs(np.diff(xf))
    return {"step_pct": float((d < abs(xf[pre] - xf[pre - 1])).mean())}


def _range(v: list[float], lo: float = 0.005, hi: float = 0.995, pad: float = 2.0) -> tuple[float, float]:
    a = np.asarray(v)
    return (float(np.floor(np.quantile(a, lo) - pad)), float(np.ceil(np.quantile(a, hi) + pad))) if a.size else (-1.0, 1.0)


def plots(res: dict, dev_rows: list, rows: list, controls: list, within: dict, outdir: Path) -> None:  # noqa: C901
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    outdir.mkdir(parents=True, exist_ok=True)
    types = [t for t in sorted(res["drift_between_cases"]) if res["drift_between_cases"][t]]
    note = "ppm = jump / time since the previous correction; negative = device clock fast against the recording PC"
    # 1. what a correction looks like: the jumps of device-level correction candidates
    fig, axes = plt.subplots(len(res["events"]), 1, figsize=(11, 2.2 * len(res["events"])), squeeze=False)
    for ax, t in zip(axes[:, 0], res["events"], strict=True):
        v = [r["jump_s"] for r in dev_rows if r["type"] == t and r["event"]]
        ax.hist(v, bins=np.arange(-2, 2.005, 0.005))
        ax.set_title(f"{t}: jump at each correction candidate (n={len(v)}), s", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "1_correction_jumps.png", dpi=110)
    plt.close(fig)
    if types:
        # 2. one drift per case, fine bins
        fig, axes = plt.subplots(len(types), 1, figsize=(11, 2.4 * len(types)), squeeze=False)
        for ax, t in zip(axes[:, 0], types, strict=True):
            v = [c[t]["ppm"] for c in res["drift_per_case"].values() if t in c]
            lo, hi = _range(v)
            ax.hist(v, bins=np.arange(lo, hi + 0.25, 0.25))
            ax.set_title(f"{t}: drift per case (n={len(v)} cases), 0.25 ppm bins", fontsize=9)
        axes[-1, 0].set_xlabel(note, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "2_drift_per_case.png", dpi=110)
        plt.close(fig)
        # 3. scatter within a case
        fig, axes = plt.subplots(len(types), 1, figsize=(11, 2.2 * len(types)), squeeze=False)
        for ax, t in zip(axes[:, 0], types, strict=True):
            w = np.clip(within.get(t, []), -5, 5)
            ax.hist(w, bins=np.arange(-5, 5.05, 0.1))
            ax.set_title(f"{t}: each correction minus its case's drift, cases with >= 3 corrections (n={len(w)}), ppm, clipped at +/-5", fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / "3_drift_within_case.png", dpi=110)
        plt.close(fig)
        # 4. devices that travel together: SNUADC against BIS and Primus, per case
        pairs = [(x, y) for x, y in (("SNUADC", "BIS"), ("SNUADC", "Primus"), ("BIS", "Primus")) if x in types and y in types]
        if pairs:
            fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.6), squeeze=False)
            for ax, (x, y) in zip(axes[0], pairs, strict=True):
                pts = [(c[x]["ppm"], c[y]["ppm"]) for c in res["drift_per_case"].values() if x in c and y in c]
                if pts:
                    px, py = zip(*pts, strict=True)
                    ax.scatter(px, py, s=4, alpha=0.5)
                    ax.set_xlim(*_range(px))
                    ax.set_ylim(*_range(py))
                ax.set_xlabel(f"{x} drift, ppm", fontsize=8)
                ax.set_ylabel(f"{y} drift, ppm", fontsize=8)
                ax.set_title(f"per case (n={len(pts)})", fontsize=9)
            fig.tight_layout()
            fig.savefig(outdir / "4_drift_pairs.png", dpi=110)
            plt.close(fig)
    # 5. continuity across the join: beat test (ECG/ART/PLETH) and step test (EEG/CO2)
    tracks = [*BEAT_BAND, *STEP_ONLY]
    fig, axes = plt.subplots(1, len(tracks), figsize=(4 * len(tracks), 3.2))
    for ax, tr in zip(axes, tracks, strict=True):
        key, bins, lab = ("z_butt", np.linspace(-15, 40, 111), "beat interval across the join, butted (z)") if tr in BEAT_BAND \
            else ("step_pct", np.linspace(0, 1, 51), "step across the join, percentile")
        series = (([c.get(key) for c in controls if c["track"] == tr], "controls"),
                  ([r.get(key) for r in rows if r["track"] == tr and r["event"]], "correction candidates"),
                  ([r.get(key) for r in rows if r["track"] == tr and not r["event"] and abs(r["jump_s"]) <= 10], "other boundaries <= 10 s"))
        for v, name in series:
            v = [x for x in v if x is not None]
            if v:
                ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", density=True, label=f"{name} (n={len(v)})")
        ax.set_title(f"{tr}: {lab}", fontsize=8)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(outdir / "5_join_tests.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
