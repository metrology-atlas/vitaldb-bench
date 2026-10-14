# SPDX-License-Identifier: MIT
"""PAT through whole cases: does the ECG-to-PPG delay drift and reset (a sawtooth)?

    python tools/pat_ts.py --root R --cases cases.json --scan SCANDIR [SCANDIR ...] --out pat_ts.jsonl [--time-budget 150]

For each case the whole overlap of SNUADC/ECG_II and SNUADC/PLETH is cut into consecutive 30 s windows;
windows with a break in either track are skipped. Per window: the median pulse-foot PAT (method of
``pat.py``) and the beat count. The case's SNUADC clock corrections (macro jumps of -0.50 to -0.478 s
from the drift scan) and other run boundaries are listed with their times, so steps in PAT can be
compared with them. Output: one line per case, window medians only (no signal values).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pat import ECG, PPG, analyse_window, sample_times  # noqa: E402

from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: E402


def boundaries(scan_dirs: list[str], cases: set[int]) -> dict:
    out = {}
    for d in scan_dirs:
        with (Path(d) / "boundaries.jsonl").open() as fh:
            for line in fh:
                c = int(line[line.index('"case": ') + 8 : line.index(",", line.index('"case": '))])
                if c not in cases:
                    continue
                t = json.loads(line)["tracks"].get(ECG)
                if t:
                    out[c] = [{"t": m["end_s"], "jump": m["jump_s"]} for m in t["macro"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--scan", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--time-budget", type=float, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    todo = [c for c in json.loads(Path(a.cases).read_text())["cases"] if c not in done]
    bnd = boundaries(a.scan, set(todo))
    t_start, slowest = time.time(), 0.0
    with out.open("a") as fh:
        for c in todo:
            if time.time() - t_start + max(2 * slowest, 5) > a.time_budget:
                break
            t1 = time.time()
            res: dict = {"case": c, "boundaries": bnd.get(c, [])}
            case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG})
            e, p = case.tracks.get(ECG), case.tracks.get(PPG)
            if e is None or p is None or not e.record_times.size or not p.record_times.size:
                res["skip"] = "no ECG_II or PLETH"
            else:
                fs = float(e.srate)
                origin = case.origin
                te, tp = sample_times(e) - origin, sample_times(p) - origin
                xe, xp = e.physical(), p.physical()
                lo, hi = max(te[0], tp[0]), min(te[-1], tp[-1])
                wins = []
                s = lo
                while s + a.window_s <= hi:
                    ea, eb = np.searchsorted(te, s), np.searchsorted(te, s + a.window_s)
                    pa, pb = np.searchsorted(tp, s - 1), np.searchsorted(tp, s + a.window_s + 1)
                    s += a.window_s
                    if eb - ea < 0.98 * a.window_s * fs or pb - pa < 0.98 * (a.window_s + 2) * fs:
                        continue
                    if np.max(np.abs(np.diff(te[ea:eb]) - 1 / fs)) > 0.01 / fs or np.max(np.abs(np.diff(tp[pa:pb]) - 1 / fs)) > 0.01 / fs:
                        continue
                    xw, pw = xe[ea:eb], xp[pa:pb]
                    if not np.isfinite(xw).all() or not np.isfinite(pw).all() or np.ptp(pw) == 0 or np.ptp(xw) == 0:
                        continue
                    v = np.array(analyse_window(te[ea:eb], xw, tp[pa:pb], pw, fs)["foot"])
                    v = v[(v > 0.05) & (v < 1.0)]  # drop the "no upstroke found" spike at 0
                    if v.size >= 10:
                        wins.append([round(s - a.window_s / 2, 2), round(float(np.median(v)), 5),
                                     round(float(np.subtract(*np.quantile(v, [0.75, 0.25]))), 5), int(v.size)])
                res["windows"] = wins
            res["seconds"] = time.time() - t1
            fh.write(json.dumps(res) + "\n")
            slowest = max(slowest, res["seconds"])
    print(json.dumps({"done": sum(1 for _ in out.open()), "seconds": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
