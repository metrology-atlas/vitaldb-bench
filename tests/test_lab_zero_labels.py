# SPDX-License-Identifier: MIT
import csv
import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _rows(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def test_lab_draw_labels(tmp_path):
    art = tmp_path / "art.csv"
    with open(art, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "label", "start_s", "end_s", "start_rel_s", "end_rel_s", "n_events", "kinds", "max_mmhg", "recovered_ok", "lab_matched"])
        w.writerow([1, "art_line:unavailable", 1000.0, 1100.0, 0.0, 100.0, 1, "flat", "", True, False])
        w.writerow([1, "art_line:access", 1500.0, 1560.0, 500.0, 560.0, 2, "flat+high", 300, True, True])
        w.writerow([1, "art_line:access", 1700.0, 1720.0, 700.0, 720.0, 1, "high", 300, True, True])
        w.writerow([1, "art_line:access", 3000.0, 3050.0, 2000.0, 2050.0, 1, "high", 300, True, True])
    lab = tmp_path / "lab.csv"
    # electrolytes only (other), a CBC marker (central), a blood gas, and one sample beyond the ART span
    lab.write_text("caseid,dt,name,result\n1,900,na,140\n1,930,k,4\n1,2300,hb,10\n1,2600,ph,7.4\n1,2610,po2,200\n"
                   "1,5000,na,139\n")
    d, r = tmp_path / "d.csv", tmp_path / "r.csv"
    subprocess.run([sys.executable, str(TOOLS / "lab_draw_labels.py"), "--lab", str(lab), "--art-labels", str(art),
                    "--out-draw", str(d), "--out-time", str(r)], check=True, capture_output=True)
    draws, times = _rows(d), _rows(r)
    assert [x["panel"] for x in draws] == ["other", "central", "blood_gas"]
    # other and central panels use the 30-min window: the CBC at 2300 s now sees all three accesses
    assert [x["label"] for x in draws] == ["lab_draw:multiple", "lab_draw:multiple", "lab_draw:single"]
    assert draws[0]["access_start_rel_s"] == "700.0" and draws[0]["lag_s"] == "200.0" and draws[0]["tests"] == "k;na"
    assert draws[1]["lag_s"] == "300.0" and draws[1]["window_s"] == "1800.0"
    # the blood gas keeps the 15-min window: only the access at 2000 s
    assert draws[2]["lag_s"] == "600.0" and draws[2]["n_candidates"] == "1" and draws[2]["window_s"] == "900.0"
    assert len(times) == 3              # the sample at 5000 s is outside the ART span + window
    assert [x["label"] for x in times] == ["lab_time:other", "lab_time:central", "lab_time:blood_gas"]
    assert all(x["start_s"] == x["end_s"] for x in times)
    # the availability rule: a lab value is never known at its timestamp
    assert {x["available_at"] for x in times} == {"lab_time_plus_turnaround"}


def test_art_zero_labels(tmp_path):
    rec = {"case": 7, "art_start_abs": 100.0, "win": 20, "guard": 5, "accesses": [
        {"start_abs": 200.0, "end_abs": 210.0, "recovered_abs": 230.0, "kinds": ["low"], "delta": {"map": 1, "sbp": 1, "dbp": 2},
         "pre": {"map": 80}, "air": [{"first": 3.0, "last": 3.0, "n": 5}]},
        {"start_abs": 500.0, "end_abs": 510.0, "recovered_abs": 530.0, "kinds": ["high"], "delta": {"dbp": 20}, "air": []},
        {"start_abs": 900.0, "end_abs": 910.0, "recovered_abs": 940.0, "kinds": ["low"], "delta": {"dbp": -12},
         "pre": {}, "air": [{"first": 8.0, "last": 0.5, "n": 6}]}]}
    j = tmp_path / "j.jsonl"
    j.write_text(json.dumps(rec) + "\n")
    z, s = tmp_path / "z.csv", tmp_path / "s.csv"
    subprocess.run([sys.executable, str(TOOLS / "art_zero_labels.py"), "--jumps", str(j), "--out-zero", str(z),
                    "--out-suspect", str(s)], check=True, capture_output=True)
    zs, ss = _rows(z), _rows(s)
    assert [x["label"] for x in zs] == ["art_zero:ok", "art_zero:suspect"]   # the flush without air is not a zero
    assert ss == [{"case": "7", "label": "art_pressure_suspect", "start_s": "230.0", "end_s": "900.0",
                   "zero_start_s": "900.0", "zero_d_dbp": "-12", "since": "previous zero"}]
    assert zs[1]["air_step"] == "-7.5"


def test_flush_response_labels(tmp_path):
    rec = {"case": 3, "flushes": [
        {"start": 100.0, "recovered": 140.0, "kinds": "high", "pre": [80, 120, 60], "post": None, "releases": [
            {"t": 110.0, "plateau": 250, "settled": 80, "fall_ms": 8, "ringing": True, "undershoot": 100, "overshoot": 40,
             "zeta": 0.28, "fd_hz": 20.0, "fn_hz": 20.8}]},
        {"start": 700.0, "recovered": 730.0, "kinds": "high", "pre": None, "post": None, "releases": [
            {"t": 705.0, "plateau": 250, "settled": 80, "fall_ms": 9, "ringing": True, "undershoot": 100, "overshoot": 5,
             "zeta": 0.55, "fd_hz": 10.0, "fn_hz": 12.0},
            {"t": 720.0, "plateau": 250, "settled": 80, "fall_ms": 25, "ringing": False}]}]}
    j = tmp_path / "fr.jsonl"
    j.write_text(json.dumps(rec) + "\n")
    out = tmp_path / "f.csv"
    subprocess.run([sys.executable, str(TOOLS / "flush_response_labels.py"), "--ringing", str(j), "--out", str(out)],
                   check=True, capture_output=True)
    rows = _rows(out)
    assert [r["label"] for r in rows] == ["art_flush_response:low_damping", "art_flush_response:moderate_damping",
                                          "art_flush_response:no_ringing"]
    assert rows[1]["d_fn_hz"] == "-8.8" and rows[1]["minutes_since_prev"] == "9.9"
    assert rows[0]["pre_sbp"] == "120" and rows[0]["post_map"] == ""
