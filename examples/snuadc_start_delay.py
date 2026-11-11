"""SNUADC start after the file's first record, over the release (11 Nov 2026).

Atlas parameter snuadc-start-delay (study vitaldb-loader). Reads the segment tables of the waveform survey
(tools/waveform_survey.py: segs_*.parquet, one row per contiguous segment, t in seconds from the file's first record)
and takes, per recording, the start of each device's earliest segment (SNUADC, Primus, BIS).

    python examples/snuadc_start_delay.py $W/signal_survey/tables --out OUT.json
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = pd.concat([pd.read_parquet(f, columns=["case", "track", "t"])
                   for f in sorted(glob.glob(str(Path(a.tables) / "segs_*.parquet")))])
    out = {}
    for dev in ("SNUADC", "Primus", "BIS"):
        t = s[s.track.str.startswith(dev + "__")].groupby("case").t.min().to_numpy()
        q = np.percentile(t, [5, 25, 50, 75, 95, 99])
        out[dev] = {"recordings": int(t.size), "start_s_p5_25_50_75_95_99": [round(float(x), 2) for x in q],
                    "max_s": round(float(t.max()), 1)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
