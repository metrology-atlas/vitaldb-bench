# Data sharing and licensing

## Code only, never data

This repository holds code, aggregate results and figures. It holds no VitalDB data and no per-case
results:

- **Per-case output stays on the analysis machine.** Every per-case pass (`*.jsonl`, label CSVs, case
  lists) is written next to the analysis, and `.gitignore` excludes the usual formats (`*.vital`, `*.jsonl`,
  `*.parquet`, `*.db`, `*.sqlite`).
- **Only aggregates and plots leave.** Summary JSON files are counts, quantiles and fitted constants, by
  track or by group; they carry no values.
- **Machine paths are not published.** Scripts read them from `scripts/local.env` or the environment.
- **Case ids where they help.** VitalDB is open data (CC BY 4.0) and its case ids are public. A doc, figure
  or example that discusses a specific case (an edge case, a clear example of a mechanism, a comparison
  between cases) names it; population and group results do not need ids. Bulk per-case output (one row per
  case or per beat) still stays on the analysis machine: `tests/test_no_per_case_files.py` fails if a
  `*.jsonl`, `*.parquet`, `*.vital` or database file, or a CSV outside `results/`, is tracked.

To reproduce the results, download the release from PhysioNet under its own licence and point `--root` at it.

## Licences

| Package | Licence | Why |
|---|---|---|
| `vitaldb-bench` | MIT | Uses only MIT and permissive libraries (TimeNet, `vitaldb`, NumPy, ...). |
| `vitaldb-bench-atriumdb` | GPL-3.0-or-later | Uses the AtriumDB SDK, which is GPL-3.0. |

The split is one-way: the GPL package imports this one, and this package never imports it. Hooks such as
`vitaldb_bench.perf.STORES` and the `adb` argument of `vitaldb_bench.fidelity.run` let the GPL package
plug its store in without this package depending on it.
