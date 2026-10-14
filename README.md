# vitaldb-bench

Timing and data-quality audit of [VitalDB](https://vitaldb.net): device clock drift and synchronisation,
the delay and averaging window of every numeric value (stamped at the end of its window, the
availability rule), derived per-case labels, and resumable TimeNet builds. **MIT licence.**

AtriumDB builds, labels and benchmarks are in the separate GPL-3.0-or-later package
[`vitaldb-bench-atriumdb`](../vitaldb-bench-atriumdb), because the AtriumDB SDK is GPL. Nothing here
imports it; it imports this package.

| Folder | Contents |
|---|---|
| `vitaldb_bench/` | Installed package: TimeNet builds (`build`), stores, fidelity and performance checks, case clock, timing report. |
| `tools/` | Analysis scripts behind the report's tables and figures (extra dependencies: `pip install .[tools]`). |
| `results/` | Aggregate results and figures only. No per-case data. |
| `docs/` | Bibliography and the LaTeX timing guide. |
| `scripts/` | Shell drivers for time-boxed build chunks. They expect the VitalDB release beside the code folder. |
| `tests/` | `pytest`. |

Related MIT packages: `physiosync` (clock and ECG-PPG timing corrections) and `lineaccess` (pressure-line
access detection).

No VitalDB data are included. To reproduce, download the release from PhysioNet and point `--root` at it.
