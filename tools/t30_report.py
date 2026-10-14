# SPDX-License-Identifier: MIT
"""Collect the 30-case test matrix results into one table (aggregates only).

    python tools/t30_report.py RESULTS_DIR --out report.json

Reads ``<name>/size.json``, ``fid.json`` and ``reads.jsonl`` for every build under RESULTS_DIR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def fidelity(path: Path) -> dict:
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out = {"cases": d["cases"], "of": d["of"], "excluded": d.get("excluded", ""), "problems": [],
           "time_err_us_max": {}, "model_err_us_max": {}, "pieces": {}}
    for name, dbs in d["tracks"].items():
        for db, g in dbs.items():
            c = g["cases"]
            if g.get("present", 0) < c:
                out["problems"].append(f"{name}: present {g.get('present', 0):.0f}/{c:.0f}")
            if g.get("count_match", 0) < c:
                out["problems"].append(f"{name}: count {g.get('count_match', 0):.0f}/{c:.0f}, missing {g.get('missing_values_total', 0):.0f}")
            if name != "EVENT" and g.get("values_exact", 0) < g.get("count_match", 0):
                out["problems"].append(f"{name}: exact {g.get('values_exact', 0):.0f}/{g.get('count_match', 0):.0f}")
            for k, tgt in (("time_err_ns_max", "time_err_us_max"), ("model_err_ns_max", "model_err_us_max")):
                if k in g:
                    out[tgt][name] = round(g[k] / 1000, 3)
            out["pieces"][name] = int(g.get("pieces_total", 0))
    return out


def reads(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = [r for line in path.read_text().splitlines() if line.strip() for r in json.loads(line)["rows"]]
    med = statistics.median
    return {"n_cases": len(rows),
            "first_window_of_case_ms": med(r["window_first_s"][0] for r in rows) * 1e3,
            "window_ms": med(x for r in rows for x in r["window_first_s"][1:]) * 1e3,
            "window_repeat_ms": med(x for r in rows for x in r["window_repeat_s"]) * 1e3,
            "aligned_60s_5tracks_ms": med(r["aligned_s"] for r in rows) * 1e3,
            "full_ecg_ms": med(r["full_s"] for r in rows) * 1e3,
            "full_ecg_msamples_per_s": med(r["full_n"] / r["full_s"] for r in rows) / 1e6}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    table = {}
    for d in sorted(Path(a.results).iterdir()):
        if not (d / "size.json").exists():
            continue
        s = json.loads((d / "size.json").read_text())
        row = {"db": s["db"], "total_mb": s["total_bytes"] / 1e6, "ingest_s": s.get("ingest_seconds"), "maxrss_mb": s.get("maxrss_mb")}
        k = s.get("kinds", {})
        if s["db"] == "atriumdb":
            row |= {"index_mb": s["index_db_bytes"] / 1e6, "labels": s["labels"]}
            for kind in ("wave", "numeric"):
                g = k.get(kind, {})
                row[f"{kind}_mb"] = g.get("tsc_bytes", 0) / 1e6
                row[f"{kind}_bits"] = g.get("bits_per_value")
                row[f"{kind}_intervals"] = g.get("intervals")
                row[f"{kind}_blocks"] = g.get("blocks")
            row["ecg_bits"] = s["measures"].get("SNUADC/ECG_II", {}).get("bits_per_value")
            row["eeg_bits"] = s["measures"].get("BIS/EEG1_WAV", {}).get("bits_per_value")
        else:
            row |= {"control_mb": s["control_bytes"] / 1e6, "partitions": s["partitions"], "signals": s.get("signals"),
                    "irregular_time_offsets": s.get("irregular_time_offsets")}
            for kind in ("wave", "numeric", "string"):
                g = k.get(kind, {})
                row[f"{kind}_mb"] = g.get("value_bytes", 0) / 1e6
                row[f"{kind}_bits"] = g.get("bits_per_value")
            st = s.get("spec_types", {})
            ecg = st.get("vitaldb_snuadc_ecg_ii_counts")
            row["ecg_bits"] = ecg["value_bytes"] * 8 / ecg["values"] if ecg else None
            eeg = next((v for kk, v in st.items() if kk.startswith("vitaldb_bis_eeg1_wav")), None)
            row["eeg_bits"] = eeg["value_bytes"] * 8 / eeg["values"] if eeg else None
        row["fidelity"] = fidelity(d / "fid.json")
        row["reads"] = reads(d / "reads.jsonl")
        table[d.name] = row
    Path(a.out).write_text(json.dumps(table, indent=1))
    for name, r in table.items():
        f = r["fidelity"]
        print(f"{name:20s} {r['total_mb']:7.1f} MB  wave {r.get('wave_bits') or 0:5.2f} b/v  ecg {r.get('ecg_bits') or 0:5.2f}  "
              f"num {r.get('numeric_bits') or 0:6.2f}  ingest {r['ingest_s'] or 0:6.0f}s  fid {f.get('cases')}/{f.get('of')} "
              f"problems {len(f.get('problems', []))}  win {r['reads'].get('window_ms', 0):.2f} ms")


if __name__ == "__main__":
    main()
