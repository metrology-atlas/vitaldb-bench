# vitaldb-bench

Timing and data-quality audit of [VitalDB](https://vitaldb.net): device clock drift and synchronisation,
the delay and averaging window of every numeric value (stamped at the end of its window, the
availability rule), derived per-case labels, and resumable TimeNet builds. **MIT licence.**

Most studies in the [VitalDB metrology atlas](https://vitaldb.metrology-atlas.org) were run with the scripts here;
each study page links to the code it used. AtriumDB builds live in a separate GPL-licensed package (not yet public),
because the AtriumDB SDK is GPL; nothing here imports it.

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

## Running the whole-release scans

The scan drivers (`tools/run_tool_scan.sh`, `tools/run_burst_validation.sh`, `examples/run_scan.sh`,
`examples/run_ppv_by_breath.sh`) run a script over every recording in 8 parallel parts. They need three settings,
taken from the environment or from `scripts/run.env` (copy `scripts/run.env.example`; git ignores the copy):

| Variable | Meaning |
|---|---|
| `W` | Work folder for per-case output. It stays on your machine; only aggregates are published. |
| `R` | The VitalDB 1.0.0 release folder. |
| `PYTHONPATH` | Where `timenet-connectors` and `physiosync` live, if they are not pip-installed. |

```bash
bash tools/run_tool_scan.sh continuity_sync          # writes $W/continuity_sync/part_*.jsonl
python tools/continuity_sync.py agg $W/continuity_sync/part_*.jsonl --out $W/continuity_sync/agg
```

