# SPDX-License-Identifier: MIT
"""The VitalDB clock model on the connector's fixture: tables beside a TimeNet build, read back corrected."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

vdb = pytest.importorskip("timenet_connectors.datasets.vitaldb.vitaldb")
physiosync = pytest.importorskip("physiosync")

from vitaldb_bench import clock_model as cm  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _fit_tool():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("clock_model_fit", TOOLS / "clock_model_fit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_population_constants_are_leave_one_out():
    def fit(rate, lags_ms):
        rows = [[i * 300.0, x, 0.6, 0.1, 0.0] for i, x in enumerate(lags_ms)]
        return {"model": {"clocks": {"ecg": {"rate_ppm": rate}}, "burst": {"ecg~eeg1": rows}}}
    fits = {1: fit(-60, [-800] * 5), 2: fit(-60, [-810] * 5), 3: fit(-60, [-700] * 5), 4: fit(0, [-690] * 5)}
    pop = cm.population(fits)
    assert {c: p["mode"] for c, p in pop.items()} == {1: "step", 2: "step", 3: "step", 4: "small"}
    assert cm.loo_constant(pop, 3, "step") == pytest.approx(-0.805)     # 3 itself left out
    assert cm.loo_constant(pop, 4, "small") is None                      # nobody else in its mode
    parts = cm.error_parts(pop)
    assert set(parts) == {"step"} and parts["step"]["between"].n == 3


def test_tables_and_corrected_read_back(tmp_path):
    from test_timenet_labels import FIXTURES, _build  # noqa: PLC0415
    from timenet_connectors.datasets.vitaldb.vitaldb.resets import RULE_VERSION, reorder_case  # noqa: PLC0415
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: PLC0415

    build = _build(tmp_path)
    tool = _fit_tool()
    v, _ = reorder_case(read_vital(FIXTURES / "vital_files" / "0001.vital"))   # as the build and the fit tool order it
    tracks, clocks = {}, {}
    for name in ("SNUADC/ECG_II", "BIS/EEG1_WAV"):
        tracks[name], clocks[name] = tool.fit_track(v.tracks[name], name.split("/")[0], {}, {"*": -64.5})
    fits = {1: {"case": 1, "origin_s": v.origin, "tracks": tracks, "snuadc_mode": "small", "reset_order": RULE_VERSION}}
    n = cm.write_tables(build, fits)
    assert n["clock_model"] == len(tracks["SNUADC/ECG_II"]["k_breaks"])   # BIS: no bursts, no reference shift

    cs = cm.corrected_timenet(build)
    w = cs.window(1, "SNUADC/ECG_II", -1e6, 1e6)
    tr = v.tracks["SNUADC/ECG_II"]
    st = tr.record_offsets[: tr.record_times.size]
    expect = physiosync.corrected_times(clocks["SNUADC/ECG_II"], st, tr.record_sizes[: st.size]) - v.origin
    assert w.corrected and w.t.size == expect.size
    assert np.max(np.abs(w.t - np.sort(expect))) < 1e-6
    assert np.allclose(w.t_hi - w.t, 0.5e-6, atol=1e-8)                   # the reference: stamp rounding only
    eeg = cs.window(1, "BIS/EEG1_WAV", -1e6, 1e6)
    assert not eeg.corrected and eeg.notes                                 # left on its stored times

    none = cm.load_clock_model(build, corrections="none")
    assert none.clocks == {}

    # a fit made on file order is not applied to a build in sampling order
    fits[1]["reset_order"] = "file"
    cm.write_tables(build, fits)
    w = cm.corrected_timenet(build).window(1, "SNUADC/ECG_II", -1e6, 1e6)
    assert not w.corrected and "sample order" in w.notes[0]
