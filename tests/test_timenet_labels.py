# SPDX-License-Identifier: MIT
"""The TimeNet label pass: derived labels become record annotations, on the record's own timeline."""

import csv
import os
from fractions import Fraction
from pathlib import Path

import pytest

vdb = pytest.importorskip("timenet_connectors.datasets.vitaldb.vitaldb")

from vitaldb_bench.label_families import FAMILIES  # noqa: E402
from vitaldb_bench.stores import TimeNetStore, newest_version  # noqa: E402
from vitaldb_bench.timenet_labels import family_annotations, label_version, origin_of  # noqa: E402

FIXTURES = Path(vdb.__file__).parent / "tests" / "fixtures"


def test_points_intervals_and_metadata():
    fam = FAMILIES["lab_time"]
    origin = Fraction(4102444800)
    rows = [{"case": "1", "label": "lab_time:blood_gas", "start_s": "4102444810.5", "end_s": "4102444810.5",
             "panel": "blood_gas", "available_at": "lab_time_plus_turnaround"},
            {"case": "1", "label": "lab_time:central", "start_s": "4102444820", "end_s": "4102444819"}]
    anns, skipped = family_annotations(fam, rows, origin, "vitaldb-0001")
    assert skipped == 1                                   # end before start
    (a,) = anns
    assert a.key == "lab_time" and a.value == "lab_time:blood_gas"
    assert a.span.is_point and a.span.start_us == 10_500_000
    assert a.metadata["available_at"] == "lab_time_plus_turnaround" and a.metadata["panel"] == "blood_gas"
    assert a.metadata["measure_tag"] == ""

    art = FAMILIES["art_line"]
    (b,), _ = family_annotations(art, [{"case": "1", "label": "art_line:flush", "start_s": "4102444801",
                                        "end_s": "4102444803"}], origin, "vitaldb-0001")
    assert not b.span.is_point and (b.span.start_us, b.span.end_us) == (1_000_000, 3_000_000)
    assert b.metadata["measure_tag"] == "SNUADC/ART" and b.metadata["available_at"] == "case_end"


def _build(tmp_path: Path) -> Path:
    """One-case TimeNet build of the connector's fixture, in the layout ``vitaldb_bench.build`` writes."""
    from timenet.writer import TimeFWriter
    from timenet_connectors.datasets.vitaldb.vitaldb.connector import CONNECTOR

    os.environ.update(VITALDB_ROOT=str(FIXTURES), VITALDB_CASES="1", VITALDB_TIMING="nominal")
    conn = CONNECTOR()
    dataset = conn.convert(conn.download(tmp_path / "dl"))
    if dataset.schema is None:
        dataset.derive_schema()
    out = tmp_path / "build"
    with TimeFWriter(out / "part-0000", dataset) as writer:
        writer.write()
    (out / "build_log.jsonl").write_text('{"kind": "timenet", "unit": "part-0000", "status": "ok", "cases": [1]}\n')
    return out


def test_copy_on_write_label_pass(tmp_path):
    out = _build(tmp_path)
    base = newest_version(out / "part-0000" / "vitaldb" / "vitaldb")
    assert base.name == "1.0.0"
    store = TimeNetStore(out)
    origin = origin_of(store.record(1))
    t0 = float(origin)
    results = tmp_path / "results"
    results.mkdir()
    with (results / "lab_time_labels.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["case", "label", "start_s", "end_s", "panel"])
        w.writeheader()
        w.writerow({"case": 1, "label": "lab_time:blood_gas", "start_s": t0 + 2, "end_s": t0 + 2, "panel": "blood_gas"})
    stats = label_version(base, out / "part-0000", results, ["lab_time", "art_line"])
    assert stats["labels"] == {"lab_time": 1, "art_line": 0}
    assert newest_version(out / "part-0000" / "vitaldb" / "vitaldb").name == "1.1.0"
    rec = TimeNetStore(out).record(1)                     # reads the newest version
    labs = [a for a in rec.walk_annotations() if a.key == "lab_time"]
    assert len(labs) == 1 and labs[0].span.start_us == 2_000_000
    assert labs[0].metadata["available_at"] == "lab_time_plus_turnaround"
    # the connector's own annotations survive
    assert any(a.key == "clinical_time" for a in rec.walk_annotations())
    # a rerun replaces rather than duplicates
    label_version(newest_version(out / "part-0000" / "vitaldb" / "vitaldb"), out / "part-0000", results,
                  ["lab_time"], "1.2.0")
    rec = TimeNetStore(out).record(1)
    assert sum(a.key == "lab_time" for a in rec.walk_annotations()) == 1
