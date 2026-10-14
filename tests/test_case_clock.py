# SPDX-License-Identifier: MIT
import json

from vitaldb_bench.case_clock import load_shifts_ns, shift_seconds, time_of_day, write_table


def row(adm, *ev):
    return {"adm": str(adm), **dict(zip(("anestart", "aneend", "opstart", "opend"), map(str, ev)))}


def test_time_of_day_from_adm_minute_and_emr_phase():
    # casestart at 07:43:12: adm (midnight 2 days before, casestart truncated to 07:43) = -(2*86400 + 7*3600 + 43*60)
    adm = -(2 * 86400 + 7 * 3600 + 43 * 60)
    # EMR minute marks fall 48 s after a whole-minute offset from casestart (casestart is 12 s past the minute)
    r = time_of_day(row(adm, -552, 10848, 1668, 10368))
    assert r["status"] == "ok"
    assert r["minute_s"] == 7 * 3600 + 43 * 60
    assert r["second"] == 12
    assert r["tod_s"] == 7 * 3600 + 43 * 60 + 12


def test_majority_phase_and_refusals():
    adm = -86400
    r = time_of_day(row(adm, -552, 10848, 1668, 10371))        # opend off the phase: 3 of 4 agree
    assert r["status"] == "ok" and r["events_agree"] == 3
    assert time_of_day(row(adm, -552, 10849, 1669, 10371))["status"].startswith("EMR events disagree")
    assert time_of_day(row(adm + 30, -552, 10848, 1668, 10368))["status"] == "adm not on a minute"


def test_shift_and_table(tmp_path):
    assert shift_seconds(7 * 3600, 9) == -2 * 3600
    tab = {1: {"tod_s": 3600, "status": "ok"}, 2: {"status": "no adm"}}
    write_table(tmp_path / "case_clock.json", tab, 9)
    doc = json.loads((tmp_path / "case_clock.json").read_text())
    assert doc["unshifted"] == [2]
    assert load_shifts_ns(tmp_path) == {1: (3600 - 9 * 3600) * 10**9}
    assert load_shifts_ns(tmp_path / "missing") == {}
