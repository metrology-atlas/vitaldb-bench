# SPDX-License-Identifier: MIT
"""Look for time-base breaks in paced ECG: places where the pacing train jumps in phase. This machine only.

    python tools/paced_breaks.py --root R --screen 'screen_*.jsonl' --out breaks.jsonl --time-budget 165

A fixed-rate pacemaker beats on its own crystal clock: beat n falls at t0 + n * P. Lost or joined samples
show as a phase step in that train. Candidates come from tools/paced_screen.py: a lead whose regular runs
share one interval (at least ``--min-dominant`` beats within 2 ms of it, and at least ``--min-share`` of all
run beats). For each, beats are timed to a fraction of a sample by matching a template of the case's own
paced beat, and every beat is placed on three timelines:

- count: sample index / nominal rate (the device's own sample clock);
- stamp: the record's VitalDB time stamp + offset within the record (what a reader of the file gets);
- line: the stamp sawtooth model (physiosync.stamp_clock), the corrected time.

An event is an interval that is not P (beyond ``--tol-ms``) between two paced stretches of at least
``--side`` beats each at the same P. Its phase step on each timeline is the first beat after it, against
the line fitted to the beats before, wrapped into [-P/2, P/2). Also recorded: beats detected inside the
interval (an intrinsic beat resets a demand pacemaker: physiology, not the time base), the stamp jump between
the two beats (the record stamps' own step), and how many records the interval spans. Events with phase steps
under ``--min-step-ms`` and no stamp jump are counted, not listed: fiducial noise is about 1 ms. One JSON line
per case: the paced stretches (P on count and stamp time) and the events. Resumable.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
from physiosync import corrected_times, stamp_clock
from scipy import signal
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

HALF = 50          # template half-width, samples (100 ms at 500 Hz)


def candidates(paths: list[str], min_dominant: int, min_share: float) -> dict[int, list[tuple[str, float]]]:
    """Case -> [(lead, dominant period ms)] from the screen output."""
    out: dict[int, list] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            for lead, d in r.items():
                if not isinstance(d, dict) or not d.get("runs"):
                    continue
                P = np.array([x[2] for x in d["runs"]])
                N = np.array([x[1] for x in d["runs"]])
                n, p = max((int(np.sum(N[np.abs(P - q) <= 2])), float(q)) for q in P)
                if n >= min_dominant and n / max(d["run_beats"], 1) >= min_share:
                    out.setdefault(int(r["case"]), []).append((lead, p))
    return out


def refine(x: np.ndarray, pk: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Sub-sample beat positions by matching the median beat; returns (positions, correlation)."""
    ok = pk[(pk > HALF + 12) & (pk < x.size - HALF - 12)]
    seg = np.stack([x[i - HALF:i + HALF] for i in ok])
    seg = seg - seg.mean(axis=1, keepdims=True)
    tpl = np.median(seg, axis=0)
    tpl = (tpl - tpl.mean()) / (np.linalg.norm(tpl - tpl.mean()) + 1e-12)
    pos, cor = np.empty(ok.size), np.empty(ok.size)
    for j, i in enumerate(ok):
        w = x[i - HALF - 10:i + HALF + 10]
        c = np.array([np.dot(tpl, w[s:s + 2 * HALF] - w[s:s + 2 * HALF].mean()) /
                      (np.linalg.norm(w[s:s + 2 * HALF] - w[s:s + 2 * HALF].mean()) + 1e-12) for s in range(21)])
        k = int(np.argmax(c))
        d = 0.0
        if 0 < k < 20:
            den = c[k - 1] - 2 * c[k] + c[k + 1]
            d = 0.5 * (c[k - 1] - c[k + 1]) / den if den < 0 else 0.0
        pos[j], cor[j] = i - 10 + k + d, c[k]
    return pos, cor


def on_timelines(pos: np.ndarray, tr, fs: float, clk: dict) -> dict[str, np.ndarray]:
    """Beat times (s) on the count, stamp and line timelines, from fractional sample positions."""
    st = tr.record_offsets[: tr.record_times.size].astype(np.float64)
    r = np.clip(np.searchsorted(st, pos, side="right") - 1, 0, st.size - 1)
    within = (pos - st[r]) / fs
    line_rec = corrected_times(clk, st)
    b = clk["rate_ppm"] * 1e-6
    return {"count": pos / fs, "stamp": tr.record_times[r] + within, "line": line_rec[r] + within * (1 + b), "record": r}


def phase_step(t: np.ndarray, i: int, side: int, P: float) -> float:
    """Phase step (ms) at interval i (between beats i and i+1), from the fit to the ``side`` beats before."""
    a = np.arange(i - side + 1, i + 1)
    n = np.round((t[a] - t[a[0]]) / P)
    slope, icpt = np.polyfit(n, t[a], 1)
    nb = np.round((t[i + 1] - icpt) / slope)
    d = t[i + 1] - (icpt + nb * slope)
    return float((d + slope / 2) % slope - slope / 2) * 1000.0


def analyse(path: Path, lead: str, p_ms: float, a) -> dict:
    """Paced stretches and phase-step events of one lead."""
    v = read_vital(path, track_names={lead})
    tr = v.tracks[lead]
    fs = float(tr.srate)
    x = np.nan_to_num(tr.physical().astype(np.float64))
    b, aa = signal.butter(2, [5 / (fs / 2), 25 / (fs / 2)], "band")
    y = np.abs(signal.filtfilt(b, aa, x))
    pk, _ = signal.find_peaks(y, distance=int(0.25 * fs), height=np.percentile(y, 98) * 0.4)
    pos, cor = refine(x, pk, int(round(p_ms / 1000 * fs)))
    st = tr.record_offsets[: tr.record_times.size]
    clk = stamp_clock(tr.record_times, st, fs, correction_sign=-1)
    T = on_timelines(pos, tr, fs, clk)
    P = p_ms / 1000.0
    rr = np.diff(T["count"])
    paced = np.abs(rr - P) <= a.tol_ms / 1000.0
    # stretches of paced intervals
    edges = np.flatnonzero(np.diff(np.r_[0, paced.astype(int), 0]))
    runs = list(zip(edges[::2], edges[1::2]))           # [start interval, end interval)
    stretches = []
    for s, e in runs:
        if e - s >= a.side:
            sl = slice(s, e + 1)
            n = np.round((T["count"][sl] - T["count"][s]) / P)
            pc = np.polyfit(n, T["count"][sl], 1)[0]
            ps = np.polyfit(n, T["stamp"][sl], 1)[0]
            res = T["count"][sl] - np.polyval(np.polyfit(n, T["count"][sl], 1), n)
            stretches.append({"t_stamp": round(float(T["stamp"][s] - v.origin), 2), "beats": int(e - s + 1),
                              "P_count_ms": round(pc * 1000, 4), "P_stamp_ms": round(ps * 1000, 4),
                              "jitter_ms": round(float(np.std(res)) * 1000, 3)})
    events, n_minor = [], 0
    good = [(s, e) for s, e in runs if e - s >= a.side]
    for (s1, e1), (s2, e2) in zip(good, good[1:]):
        if s2 - e1 > a.max_gap_beats:
            continue
        i_before, i_after = e1, s2                        # last beat of stretch 1, first beat of stretch 2
        steps = {}
        for k in ("count", "stamp", "line"):
            t = T[k]
            # treat beats e1..s2 as one interval: predict the first beat of stretch 2 from stretch 1
            tt = np.r_[t[:i_before + 1], t[i_after]]
            steps[k] = round(phase_step(tt, i_before, a.side, P), 3)
        rb, ra = int(T["record"][i_before]), int(T["record"][i_after])
        jump = float((T["stamp"][i_after] - T["stamp"][i_before]) - (T["count"][i_after] - T["count"][i_before])) * 1000
        ev = {"t_stamp": round(float(T["stamp"][i_before] - v.origin), 2), "beats_inside": int(i_after - i_before - 1),
                       "interval_ms": round(float(T["count"][i_after] - T["count"][i_before]) * 1000, 2),
                       "step_count_ms": steps["count"], "step_stamp_ms": steps["stamp"], "step_line_ms": steps["line"],
                       "stamp_minus_count_ms": round(jump, 3), "records_spanned": ra - rb,
                       "corr_before": round(float(cor[i_before]), 3), "corr_after": round(float(cor[i_after]), 3)}
        minor = max(abs(steps["count"]), abs(steps["stamp"])) < a.min_step_ms and abs(jump) < 2.0
        if minor:
            n_minor += 1
        else:
            events.append(ev)
    return {"lead": lead, "P_screen_ms": p_ms, "fs": fs, "beats": int(pos.size), "clock": {"rate_ppm": round(clk["rate_ppm"], 2),
            "source": clk["rate_source"], "events": clk["corrections"]["events"], "gaps": clk["gaps"]},
            "stretches": stretches, "minor_events": n_minor, "events": events}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--screen", required=True, help="glob of paced_screen.py outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="", help="comma-separated cases to add (lead II, period from the screen)")
    ap.add_argument("--min-dominant", type=int, default=300)
    ap.add_argument("--min-share", type=float, default=0.8)
    ap.add_argument("--tol-ms", type=float, default=4.0)
    ap.add_argument("--side", type=int, default=10)
    ap.add_argument("--max-gap-beats", type=int, default=3)
    ap.add_argument("--min-step-ms", type=float, default=6.0, help="smaller steps (with no stamp jump) are only counted")
    ap.add_argument("--time-budget", type=float, default=165)
    a = ap.parse_args()
    cands = candidates(sorted(glob.glob(a.screen)), a.min_dominant, a.min_share)
    out = Path(a.out)
    done = {(json.loads(x)["case"], json.loads(x).get("lead")) for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t0, slowest, n = time.time(), 0.0, 0
    jobs = [(c, lead, p) for c, ls in sorted(cands.items()) for lead, p in ls if (c, lead) not in done]
    with out.open("a") as fh:
        for c, lead, p in jobs:
            if time.time() - t0 + max(1.5 * slowest, 10) > a.time_budget:
                break
            t1 = time.time()
            try:
                r = {"case": c, **analyse(Path(a.root) / "vital_files" / f"{c:04d}.vital", lead, p, a)}
            except Exception as ex:  # noqa: BLE001
                r = {"case": c, "lead": lead, "error": f"{type(ex).__name__}: {ex}"[:300]}
            r["seconds"] = round(time.time() - t1, 1)
            slowest = max(slowest, r["seconds"])
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
    print(json.dumps({"candidates": sum(len(v) for v in cands.values()), "done_now": n, "left": len(jobs) - n}))


if __name__ == "__main__":
    main()
