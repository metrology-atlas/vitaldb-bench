# SPDX-License-Identifier: MIT
"""Test for a fault at the SNUADC firmware's microsecond-counter wrap (metrology-atlas study snuadc-firmware).

The firmware times its samples with micros(), which wraps every 2^32 us = 4,294.97 s of uptime. A fault there
(extra or lost samples at the wrap) would show as a step in the sample-count timeline of a signal carrying an
outside clock, at the same moment in every SNUADC channel, repeating every 4,295 s. The 240 Hz monitor output line
in ART and PLETH ([[study:monitor-lines]]) is such a clock.

    python tools/wrap_test.py select --lines $W/monitor_lines/lines.jsonl --root $R --out $W/wrap_test/files.json
    python tools/wrap_test.py run --files $W/wrap_test/files.json --root $R --out $W/wrap_test
    python tools/wrap_test.py run --files $W/wrap_test/files.json --root $R --out $W/wrap_test_wide --min-wraps 1

The widened form also repeats the spacing control on the recordings other than the first six (out of sample; --first
names the files.json of the first run when --files comes from a new selection), and
counts the recordings spanning two wraps that show a pair. It tests every selected recording whose continuous stretch spans at least --min-wraps wraps.
``run`` is resumable (OUT/cases.jsonl) and stops after --time-budget seconds; rerun until it writes wrap_test.json.

``select`` takes the recordings where monitor_lines finds the 240 Hz line in both ART and PLETH, measures each
one's longest continuous stretch of both, and keeps the six longest. ``run``, per recording, over that stretch:

* the 240 Hz line's complex amplitude in 10 s windows on the global sample index (monitor_lines.demod), its
  unwrapped phase as a time (tau = -phase / 2 pi 240), less a straight line (the line's rate);
* a step at each window boundary: the mean tau over the next 6 windows minus the previous 6, with its standard
  error from the window-to-window scatter; candidate steps above 5 standard errors, at a local maximum;
* steps shared by ART and PLETH (within 20 s, the same sign), and pairs of steps 4,295 +/- 30 s apart; as a
  control, the number of pairs of shared steps at every other spacing from 2,000 to 6,500 s (10 s apart, more than
  120 s from 4,295 s), counted the same way.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from monitor_lines import FS, demod, line_signal  # noqa: E402
from pat import sample_times  # noqa: E402
from vitaldb_reader.vital_file import read_vital  # noqa: E402

ART, PLETH = "SNUADC/ART", "SNUADC/PLETH"
WRAP_S = 2 ** 32 / 1e6
WIN = 10.0
M = 6
K_SE = 5.0


def stretch(case) -> tuple[int, int, np.ndarray, np.ndarray, float] | None:
    """Longest stretch where ART and PLETH are both continuous: (index into ART, into PLETH, ART raw, PLETH raw,
    start time)."""
    a, p = case.tracks.get(ART), case.tracks.get(PLETH)
    if a is None or p is None or float(a.srate) != FS or float(p.srate) != FS:
        return None
    ta, tp = sample_times(a), sample_times(p)
    best = None
    for t in (ta,):
        gap = np.flatnonzero(np.abs(np.diff(t) - 1 / FS) > 0.01 / FS)
        be = np.r_[0, gap + 1, t.size]
        for s, e in zip(be[:-1], be[1:]):
            t0, t1 = ta[s], ta[e - 1]
            ps, pe = np.searchsorted(tp, t0 - 0.5 / FS), np.searchsorted(tp, t1 + 0.5 / FS)
            if pe - ps < 2 or np.max(np.abs(np.diff(tp[ps:pe]) - 1 / FS)) > 0.01 / FS:
                continue
            n = min(e - s, pe - ps)
            if best is None or n > best[2]:
                best = (s, ps, n, t0)
    if best is None:
        return None
    s, ps, n, t0 = best
    return s, ps, np.asarray(a.values[s:s + n], float), np.asarray(p.values[ps:ps + n], float), float(t0)


def select(a) -> None:
    rows = [json.loads(x) for x in Path(a.lines).read_text().splitlines() if x.strip()]
    cand = [r["case"] for r in rows
            if r.get("ART", {}).get("240", {}).get("strength", 0) > 3 and r.get("PLETH", {}).get("240", {}).get("strength", 0) > 3]
    lengths = []
    for c in cand:
        case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ART, PLETH})
        st = stretch(case)
        if st:
            lengths.append((c, round(st[2].size / FS, 1)))
    lengths.sort(key=lambda x: -x[1])
    out = {"candidates": len(cand), "lengths_s": lengths, "chosen": [c for c, _ in lengths[: a.n]]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"candidates": len(cand), "chosen": lengths[: a.n]}))


def tau_series(x: np.ndarray, n0: int) -> tuple[np.ndarray, np.ndarray]:
    z = demod(line_signal(x), n0, 240.0, WIN)
    ph = np.unwrap(np.angle(z))
    tt = (np.arange(z.size) + 0.5) * WIN
    tau = -ph / (2 * np.pi * 240.0)
    tau = tau - np.polyval(np.polyfit(tt, tau, 1), tt)
    return tt, tau


def steps(tt: np.ndarray, tau: np.ndarray) -> tuple[list[dict], float]:
    n = tau.size
    s = np.full(n, np.nan)
    for k in range(M, n - M + 1):
        s[k] = tau[k:k + M].mean() - tau[k - M:k].mean()
    d = np.diff(tau)
    sd = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)      # window-to-window scatter of tau
    se = sd * np.sqrt(2 / M)
    out = []
    for k in range(M, n - M + 1):
        if np.isfinite(s[k]) and abs(s[k]) > K_SE * se and abs(s[k]) == np.nanmax(np.abs(s[max(M, k - M):k + M + 1])):
            out.append({"t_s": round(float(k * WIN), 1), "step_ms": round(float(s[k] * 1e3), 3), "z": round(float(s[k] / se), 1)})
    return out, float(se)


def run(a) -> None:
    sel = json.loads(Path(a.files).read_text())
    if a.min_wraps is None:
        files = sel["chosen"]
    else:
        files = [c for c, length in sel["lengths_s"] if length >= a.min_wraps * WRAP_S]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    part = out / "cases.jsonl"
    done = {}
    if part.exists():
        for x in part.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                done[r["case"]] = r
    t_start = time.time()
    fh = part.open("a")
    for c in files:
        if c in done:
            continue
        if time.time() - t_start > a.time_budget:
            break
        case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ART, PLETH})
        s, ps, xa, xp, t0 = stretch(case)
        ta, ua = tau_series(xa, s)
        tp, up = tau_series(xp, ps)
        sa, sea = steps(ta, ua)
        sp, sep = steps(tp, up)
        shared = [dict(t_s=x["t_s"], art_ms=x["step_ms"], pleth_ms=y["step_ms"]) for x in sa for y in sp
                  if abs(x["t_s"] - y["t_s"]) <= 20 and np.sign(x["step_ms"]) == np.sign(y["step_ms"])]
        allt = sorted({x["t_s"] for x in sa + sp})
        pairs = [(t1, t2) for i, t1 in enumerate(allt) for t2 in allt[i + 1:] if abs(t2 - t1 - WRAP_S) <= 30]
        art_first = float(case.tracks[ART].record_times[0])
        r = {"case": c, "stretch_s": round(xa.size / FS, 1), "wraps_due": int(xa.size / FS // WRAP_S),
             # where the stretch starts: ART samples since the track's first sample, and seconds since its first stamp
             "stretch_start_index": int(s), "stretch_start_after_art_s": round(t0 - art_first, 3),
             "se_ms": {"ART": round(sea * 1e3, 3), "PLETH": round(sep * 1e3, 3)},
             "steps": {"ART": sa, "PLETH": sp}, "shared": shared, "pairs_4295s": pairs}
        fh.write(json.dumps(r) + "\n")
        fh.flush()
        done[c] = r
        print(json.dumps({k: v for k, v in r.items() if k != "steps"}))
    fh.close()
    if any(c not in done for c in files):
        print(f"not finished: {sum(c in done for c in files)} of {len(files)} recordings; run again")
        return
    res = [done[c] for c in files]
    # control: pairs of shared steps at other spacings (2,000-6,500 s in 10 s steps, away from the wrap), same +/-30 s
    def shared_pairs(D, rs=None):
        n = 0
        for r in (res if rs is None else rs):
            t = sorted(x["t_s"] for x in r["shared"])
            n += sum(1 for i, t1 in enumerate(t) for t2 in t[i + 1:] if abs(t2 - t1 - D) <= 30)
        return n
    at_wrap = shared_pairs(WRAP_S)
    ctrl = [shared_pairs(D) for D in np.arange(2000, 6501, 10) if abs(D - WRAP_S) > 120]
    S_ctrl = {"shared_pairs_at_4295s": at_wrap, "control_spacings": len(ctrl), "control_mean": round(float(np.mean(ctrl)), 3),
              "control_max": int(np.max(ctrl)), "control_share_at_least_as_many": round(float(np.mean(np.array(ctrl) >= at_wrap)), 4)}
    due = sum(r["wraps_due"] for r in res)
    # the same control on the recordings not among the six first tested (an out-of-sample check)
    first = set(json.loads(Path(a.first).read_text())["chosen"]) if a.first else set(sel["chosen"])
    rest = [r for r in res if r["case"] not in first]
    if a.min_wraps is not None and rest:
        at_r = shared_pairs(WRAP_S, rest)
        c_r = np.array([shared_pairs(D, rest) for D in np.arange(2000, 6501, 10) if abs(D - WRAP_S) > 120])
        S_ctrl["not_first_six"] = {"files": len(rest), "shared_pairs_at_4295s": at_r, "control_mean": round(float(c_r.mean()), 3),
                                   "control_max": int(c_r.max()), "control_share_at_least_as_many": round(float(np.mean(c_r >= at_r)), 4)}
    two = [r for r in res if r["stretch_s"] >= 2 * WRAP_S]
    S_ctrl["recordings_spanning_2_wraps"] = {"n": len(two), "with_shared_pair_4295s": sum(
        1 for r in two if any(abs(y["t_s"] - x["t_s"] - WRAP_S) <= 30 for x in r["shared"] for y in r["shared"]))}
    S = {"control": S_ctrl, "files": len(res), "wraps_due": due,
         "stretch_h": round(sum(r["stretch_s"] for r in res) / 3600, 1),
         "shared_steps_per_wrap_due": round(sum(len(r["shared"]) for r in res) / due, 2) if due else None, "with_pair_4295s": sum(1 for r in res if r["pairs_4295s"]),
         "with_shared_step": sum(1 for r in res if r["shared"]),
         "with_shared_pair_4295s": sum(1 for r in res if any(abs(y["t_s"] - x["t_s"] - WRAP_S) <= 30 for x in r["shared"] for y in r["shared"])),
         "shared_steps": sum(len(r["shared"]) for r in res),
         "shared_steps_ms": sorted(abs(x["art_ms"]) for r in res for x in r["shared"]),
         "se_ms_range": [min(min(r["se_ms"].values()) for r in res), max(max(r["se_ms"].values()) for r in res)],
         "cases": res}
    (out / "wrap_test.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: v for k, v in S.items() if k != "cases"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--lines", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=6)
    p = sub.add_parser("run")
    p.add_argument("--files", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-wraps", type=int, default=None,
                   help="test every selected recording whose stretch spans at least this many wraps (default: the chosen six)")
    p.add_argument("--time-budget", type=float, default=150)
    p.add_argument("--first", help="files.json whose 'chosen' are the first six tested (default: --files' own)")
    a = ap.parse_args()
    {"select": select, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
