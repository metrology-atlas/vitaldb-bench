# Build TimeNet in time-boxed chunks

A full build takes longer than one command should run. Every build command therefore does as much as fits
in `--time-budget` seconds, stops between units of work, and logs each finished unit in
`build_log.jsonl` beside the output. Run the same command again and it continues.

## 1. Select the cases

```bash
python -m vitaldb_bench.build select --root "$VITALDB_ROOT" --out cases.json --n 200 --strata 10 --seed 0
```

The .vital files are sorted by size and cut into `--strata` equal strata; the same number of cases is drawn
from each, so short and long cases both appear. The same seed gives the same sample. `cases.json` lists
case ids: it stays on the analysis machine with the rest of the per-case material.

## 2. Build, chunk by chunk

```bash
python -m vitaldb_bench.build timenet --root "$VITALDB_ROOT" --cases cases.json --out build/tn \
    --timing nominal --values integer --time-budget 150
```

Repeat until it prints `"partitions_left": 0`. Cases are written in partitions of `--partition-size`
cases (default 5), each its own TimeNet registry root (`build/tn/part-0000`, ...), because the TimeNet
writer holds a whole dataset in memory until it commits. Before starting a partition the command predicts
how long it will take, from the seconds per source byte of earlier partitions, and leaves it for the next
call if it would overrun the budget.

| Option | Choices | Meaning |
|---|---|---|
| `--timing` | `nominal` (default), `file`, `resync` | How waveform samples are placed in time: see [Timing modes](../concepts/timing-modes.md). |
| `--values` | `source` (default), `integer` | `integer` stores a float track that sits exactly on a lattice as its integers plus a scale. Lossless. |
| `--tolerance-us` | float, default 1 | For `file` timing: largest allowed distance from the file's stamps before a run is split. |
| `--exclude-cases` | comma list | Leave cases out (their whole partition). |
| `--backend`, `--encoding`, `--level`, `--compression`, `--chunk-bytes` | | TimeNet writer settings under test; each defaults to the TimeNet default. |

## 3. Drive several builds with the scripts

`scripts/` holds shell drivers that advance builds by one call of about 170 s each, so a long build can be
run as a series of short steps:

| Script | What one call does |
|---|---|
| `chunk.sh CASES SET variant...` | Advances the TimeNet and the AtriumDB build of each variant (`timing-values`, e.g. `nominal-integer`). |
| `chunk_par.sh CASES SET DB variant...` | Advances one database's builds of several variants side by side (timings measured under contention). |
| `t30.sh [filter]` | Advances the 30-case test matrix in `scripts/t30_jobs.txt`. |
| `t30_measure.sh`, `t30_out.sh` | Measure finished test builds; copy them out. |
| `tick.sh` | One step of the whole test matrix. |

They read `scripts/local.env` (see [Get started](../get-started.md)), write builds under `$HOME/build`
on the machine that runs them, and put the GPL package on `PYTHONPATH` for the AtriumDB builds.

!!! note "Build on a local disk"
    The builds go to a local disk rather than a mounted network or Windows folder, where SQLite and DuckDB
    misbehave. Copy finished builds out afterwards.

## AtriumDB

AtriumDB builds use the same case list and the same shared arguments, from the GPL package:

```bash
python -m vitaldb_bench_atriumdb.build atriumdb --root "$VITALDB_ROOT" --cases cases.json --out build/adb
```
