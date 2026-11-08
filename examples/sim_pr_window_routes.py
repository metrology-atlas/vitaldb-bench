# SPDX-License-Identifier: MIT
"""Synthetic test of the two routes to PLETH_HR's window (examples/pr_window_routes.py): does the comparison with
heart rate recover the truth when everything is known?

    python examples/sim_pr_window_routes.py --out $W/pr_window_routes/sim [--recordings 300] [--hours 2] [--seed 1]

One heartbeat series per recording: a rate that wanders (an Ornstein-Uhlenbeck process, 75 bpm mean, SD 6 bpm, time
constant 60 s, plus breathing at 0.25 Hz) and small beat-to-beat jitter; pulses at the finger a fixed pulse-arrival
time (PAT) after each R peak. The monitor, after each beat, computes HR = floor(60 / mean of the last 8 R-R
intervals within 15% of their median) and PLETH_HR = round(60 / mean of the last 10 pulse intervals within 25% of
their median); each value is ready a delay after its beat (HR: d_HR, by group; PLETH_HR: d_PR) and the recorder,
polling every 2 s at a random phase, stamps the latest ready value of each. So stamp = close + d + phi exactly, with
phi the wait for the poll.

The stamped series then go through the comparison's own scan (values held, read at HR stamps + L, changes over 10 s
blocks) and its lag reading (parabola through the whole-record lags), and route 1 = L + d_HR + poll/2 + 4 x interval
is set against the truth, the PLETH_HR window centre in the heartbeat time base: d_PR + poll/2 + 5 x interval + PAT.
Writes OUT/summary.json (true values, route 1, its error) and prints it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hr_vs_pulse_rate import BLOCK, LAGS, RECORD_S, _corr, _held  # noqa: E402

D_HR = {"early": 1.88, "late": 4.17}
D_PR = 3.73
PAT = 0.27
POLL = 2.0


def beats(rng, hours):
    """R-peak times (s): an OU heart rate plus breathing, integrated beat by beat."""
    T = hours * 3600
    dt, tau, sd = 0.25, 60.0, 6.0
    n = int(T / dt) + 1
    x = np.zeros(n)
    a = np.exp(-dt / tau)
    e = rng.normal(0, sd * np.sqrt(1 - a * a), n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    tg = np.arange(n) * dt
    rate = 75 + x + 2.0 * np.sin(2 * np.pi * 0.25 * tg)
    out, t = [], 1.0
    while t < T - 5:
        r = np.interp(t, tg, rate)
        t += 60.0 / r * (1 + rng.normal(0, 0.01))
        out.append(t)
    return np.array(out)


def monitor(ev, n_iv, band, trunc, d):
    """Value after each event: 60 / mean of the last n_iv intervals within band of their median; ready at event + d."""
    iv = np.diff(ev)
    w = np.lib.stride_tricks.sliding_window_view(iv, n_iv)
    med = np.median(w, axis=1)[:, None]
    keep = np.abs(w - med) <= band * med
    m = np.where(keep, w, 0).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)
    v = 60.0 / m
    v = np.floor(v) if trunc else np.round(v)
    close = ev[n_iv:]
    return close + d, v


def stamped(ready, v, polls):
    i = np.searchsorted(ready, polls, side="right") - 1
    ok = i >= 0
    return polls[ok], v[i[ok]]


def scan(th, xh, tp, xp):
    """The comparison's scan for one recording: per lag, sufficient statistics of 10 s block changes."""
    t0 = th[0]
    blk = np.floor((th - t0) / BLOCK).astype(int)
    out = {}
    for L in LAGS:
        p = _held(tp, xp, th + L)
        ok = np.isfinite(p)
        b = blk[ok]
        nb = np.bincount(b)
        mh = np.bincount(b, xh[ok]) / np.where(nb, nb, 1)
        mp = np.bincount(b, p[ok]) / np.where(nb, nb, 1)
        full = nb >= 3
        k = full[1:] & full[:-1]
        dh, dp = np.diff(mh)[k], np.diff(mp)[k]
        out[str(L)] = {"ch": [dh.size, dh.sum(), dp.sum(), (dh * dh).sum(), (dp * dp).sum(), (dh * dp).sum()]}
    return out


def step_peak(sub):
    rec = list(range(-8, 9))
    c = []
    for k in rec:
        t = np.zeros(6)
        for r in sub:
            t += np.array(r[str(float(RECORD_S * k))]["ch"], float)
        c.append(_corr(t))
    c = np.array(c)
    i = int(np.nanargmax(c))
    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
    return RECORD_S * (rec[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recordings", type=int, default=300)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    res = {"pat_s": PAT, "d_pr_s": D_PR, "poll_s": POLL, "recordings_per_group": a.recordings, "hours": a.hours,
           "groups": {}}
    for g, dhr in D_HR.items():
        rows, ivs = [], []
        for _ in range(a.recordings):
            r = beats(rng, a.hours)
            p = r + PAT
            polls = np.arange(rng.uniform(0, POLL), r[-1], POLL)
            th, xh = stamped(*monitor(r, 8, 0.15, True, dhr), polls)
            tp, xp = stamped(*monitor(p, 10, 0.25, False, D_PR), polls)
            rows.append(scan(th, xh, tp, xp))
            ivs.append(np.median(np.diff(r)))
        iv = float(np.mean(ivs))
        L = step_peak(rows)
        route1 = L + dhr + POLL / 2 + 4 * iv
        truth = D_PR + POLL / 2 + 5 * iv + PAT
        res["groups"][g] = {"lag_s": round(L, 3), "true_lag_s": round(truth - (dhr + POLL / 2 + 4 * iv), 3),
                            "route1_centre_s": round(route1, 3), "true_centre_s": round(truth, 3),
                            "route1_error_s": round(route1 - truth, 3)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
