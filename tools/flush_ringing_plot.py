# SPDX-License-Identifier: MIT
"""Figure for flush_ringing: example releases, damping coefficient vs natural frequency, and within-case trend
of the natural frequency. Args: full_fr.jsonl root out.png"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pat import sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

rs = [json.loads(l) for l in open(sys.argv[1])]
R = [{**x, "case": r["case"]} for r in rs for f in r.get("flushes", []) for x in f["releases"]]
ring = [x for x in R if x.get("ringing") and x.get("fn_hz")]
z = np.array([x["zeta"] for x in ring])
fn = np.array([x["fn_hz"] for x in ring])
rng = np.random.default_rng(3)
pick = {
    "underdamped (low ζ)": min((x for x in ring if 14 < x["fn_hz"] < 20 and x["zeta"] < 0.15), key=lambda x: rng.random()),
    "moderately damped": min((x for x in ring if 15 < x["fn_hz"] < 25 and 0.45 < x["zeta"] < 0.6), key=lambda x: rng.random()),
    "no undershoot (slow fall)": min((x for x in R if x.get("ringing") is False and (x["fall_ms"] or 0) > 18), key=lambda x: rng.random()),
}
fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(2, 3)
for i, (lab, x) in enumerate(pick.items()):
    ax = fig.add_subplot(gs[0, i])
    tr = read_vital(Path(sys.argv[2]) / "vital_files" / f"{x['case']:04d}.vital", track_names={"SNUADC/ART"}).tracks["SNUADC/ART"]
    t = sample_times(tr)
    xx = tr.physical()
    i0, i1 = np.searchsorted(t, x["t"] - 0.15), np.searchsorted(t, x["t"] + 0.5)
    ax.plot((t[i0:i1] - x["t"]) * 1000, xx[i0:i1], lw=1)
    ax.axhline(x["settled"], color="0.6", lw=.6, ls="--")
    ttl = f"{lab}: case {x['case']}, fall {x['fall_ms']:.0f} ms"
    if x.get("fn_hz"):
        ttl += f", ζ {x['zeta']:.2f}, fn {x['fn_hz']:.0f} Hz"
    ax.set_title(ttl, fontsize=9); ax.set_xlabel("ms from release"); ax.set_ylabel("mmHg"); ax.grid(alpha=.3)
ax = fig.add_subplot(gs[1, 0:2])
ax.scatter(fn, z, s=3, alpha=.25)
ax.set_xlim(0, 45); ax.set_ylim(0, 1.0)
ax.axhspan(0.45, 0.7, color="C2", alpha=.08, label="ζ 0.45–0.7 (commonly cited optimal range)")
ax.axvline(10, color="C3", lw=.8, ls="--", label="fn 10 Hz")
ax.set_xlabel("natural frequency fn (Hz)"); ax.set_ylabel("damping coefficient ζ")
ax.set_title(f"{len(ring)} flush releases with ringing in {len({x['case'] for x in ring})} cases: ζ median {np.median(z):.2f}, fn median {np.median(fn):.1f} Hz", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
per = {}
for x in ring:
    per.setdefault(x["case"], []).append((x["t"], x["fn_hz"]))
sl = []
for v in per.values():
    v = np.array(sorted(v))
    if len(v) >= 3 and np.ptp(v[:, 0]) > 3600:
        sl.append(np.polyfit((v[:, 0] - v[0, 0]) / 3600, v[:, 1], 1)[0])
ax = fig.add_subplot(gs[1, 2])
ax.hist(sl, bins=np.arange(-8, 8.5, 0.5), color="C1")
ax.axvline(0, color="k", lw=.6)
ax.set_xlabel("within-case trend of fn (Hz per hour)"); ax.set_ylabel("cases")
ax.set_title(f"{len(sl)} cases with ≥ 3 releases over > 1 h: median {np.median(sl):+.2f} Hz/h, falling {np.mean(np.array(sl) < 0) * 100:.0f}%", fontsize=10)
fig.suptitle("Fast-flush (square-wave) response of the arterial line at flush release (1,440 VitalDB cases)", fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=75)
