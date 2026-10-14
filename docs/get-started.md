# Get started

## Install

You need Python 3.11 or later and a copy of the VitalDB release (from PhysioNet). The package depends on
TimeNet and its VitalDB connector.

```bash
git clone https://github.com/meshlab/vitaldb-bench.git
cd vitaldb-bench
pip install -e .            # the package: builds, stores, checks
pip install -e ".[tools]"   # also the analysis scripts in tools/
```

For AtriumDB builds, also install the GPL companion package `vitaldb-bench-atriumdb` (it lives beside this
one in the code folder).

## Tell the scripts where your data are

The shell drivers in `scripts/` read machine-specific paths from `scripts/local.env`, which git ignores.
Copy the example and edit it:

```bash
cp scripts/local.env.example scripts/local.env
```

| Variable | Meaning |
|---|---|
| `VITALDB_ROOT` | The VitalDB release folder (holds `vital_files/` and `clinical_data.csv`). |
| `VITALDB_PYTHON` | The Python that has this package installed. |
| `VITALDB_WORK` | Work folder; the test-matrix scripts read `bench/cases_*.json` from it. |
| `VITALDB_TEST_DBS` | Where `t30_out.sh` copies finished test builds. |

The Python commands below take the same paths as arguments (`--root` and so on).

## A first build

Pick a reproducible sample of cases, build TimeNet from it, and check the build:

```bash
python -m vitaldb_bench.build select  --root "$VITALDB_ROOT" --out cases.json --n 20 --seed 0
python -m vitaldb_bench.build timenet --root "$VITALDB_ROOT" --cases cases.json --out build/tn
python -m vitaldb_bench.fidelity     --root "$VITALDB_ROOT" --timenet build/tn --out fidelity.json
```

Each command stops cleanly after `--time-budget` seconds (default 150) and prints what is left. Run it
again until nothing is: see [Build TimeNet](how-to/build-timenet.md).

`fidelity.json` is an aggregate by track and holds no values. `fidelity.jsonl`, beside it, is per case:
keep it on the analysis machine.

## Read a track back

```python
# requires a TimeNet build
from vitaldb_bench.stores import TimeNetStore

store = TimeNetStore("build/tn")
case = store.caseids[0]
art = store.track(case, "SNUADC/ART")             # TrackData: raw, decoded, times_ns, pieces
values, times_us = store.window(case, "SNUADC/ART", 0, 10_000_000)   # the first 10 s only
```
