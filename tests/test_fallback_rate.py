# SPDX-License-Identifier: MIT
"""tools/fallback_rate.py: the SNUADC fallback rate (setup, else port, else overall)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fallback_rate import StepRates, load_step_rates  # noqa: E402

PORTS = {"COM4": -65.4, "COM13": -66.9, "*": -64.5}


def test_without_cases_is_the_plain_port_dict(tmp_path):
    f = tmp_path / "rates.json"; f.write_text(json.dumps(PORTS))
    r = load_step_rates(f)
    for case in (1, 2, None):
        rc = r.for_case(case)
        for port in ("COM4", "COM13", "COM99", None):
            assert rc.get(port, rc.get("*", 0.0)) == PORTS.get(port, PORTS["*"])
        assert rc.tier == "port_step_rate"


def test_setup_rate_wins_for_its_cases():
    r = StepRates({**PORTS, "cases": {"7": -79.6}})
    rc = r.for_case(7)
    assert rc.get("COM4", rc.get("*", 0.0)) == -79.6 and rc["COM13"] == -79.6
    assert rc.tier == "setup_step_rate" and rc.setup_rate == -79.6
    other = r.for_case(8)
    assert other.get("COM4", other.get("*", 0.0)) == -65.4 and other.tier == "port_step_rate"


def test_empty_file_gives_empty_table():
    r = load_step_rates(None)
    assert r.for_case(1).get("COM4", r.get("*", 0.0)) == 0.0
