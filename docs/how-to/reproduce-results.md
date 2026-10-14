# Reproduce the results

`results/` holds the aggregate results and figures behind the drift document and the timing guide. Each was
made by a script in `tools/`, run on a machine that holds the VitalDB release. The per-case files those
scripts write along the way stay on that machine; only these aggregates are published.

## The pattern

Most analyses run in two steps:

1. **A per-case pass** reads the .vital files and writes one JSON line per case (resumable, time-boxed).
   This output is per case and is not published.
2. **An aggregate or plot step** reads the per-case file and writes a summary JSON or a figure. These go
   in `results/`.

Each script's docstring gives its exact commands; `python tools/<script>.py --help` lists the options.
Install the extra dependencies first with `pip install -e ".[tools]"`.

## Which script made which file

Files in `results/pat/`:

| File | Script(s) in `tools/` |
|---|---|
| `drug_effect_latency_summary.json` | `drug_effect_latency.py` |
| `emr_markers_summary.json` | `emr_markers.py` |
| `loop_closure_art_summary.json` | `loop_closure_art.py` |
| `numeric_delay_labels_summary.json` | `numeric_delay_labels.py` |
| `numeric_delay_variability*.json`, `numeric_delay_variability.png` | `numeric_delay_variability.py`, `numeric_delay_variability_plot.py` |
| `numeric_derivation_pilot_summary.json`, `numeric_derivation_pilot.png` | `numeric_derivation_pilot.py`, `numeric_derivation_agg.py`, `numeric_derivation_plot.py` |
| `numeric_intervals_1000.json`, `numeric_intervals_dist.png`, `numeric_intervals_jitter.png` | `numeric_intervals.py`, `numeric_intervals_plot.py` |
| `numeric_window_timing.png` | `numeric_window_diagram.py` (schematic; no data) |
| `rare_device_sync_summary.json` | `rare_device_sync.py` |
| `spo2_event_pilot_summary.json`, `spo2_event_pilot.png` | `spo2_event_pilot.py`, `spo2_event_agg.py`, `spo2_event_plot.py` |
| `time_source_accuracy.png`, `.svg` | `time_source_accuracy_figure.py` |
| `timestamp_resolution_*.json`, `timestamp_resolution_inventory.csv` | `timestamp_resolution.py` (`tables`, `tracks`, `merge`, `rounding`) |
| `value_range_audit_summary.json` | `value_range_audit.py` |
| `value_resolution_tracks.json`, `value_resolution_labs.json` | `value_resolution.py` (`tracks`, `merge`, `labs`) |
| `vigilance_window_summary.json` | `vigilance_window.py` |
| `sync/*.png` | `syncscan.py`, `syncanalyze.py`, then `sync_figures.py` |
| `lab_time_lag_summary.json` | `lab_draw_labels.py` (its `lag_s` column), aggregated by a one-off step that is not in `tools/`. |
| `hr_window_groups_summary.json` | The producing script is not in `tools/`. |

The scripts in `tools/adhoc/` are short aggregation and plotting helpers written during the audit. They are
kept because numbers or figures came from them, not as maintained tools; see `tools/adhoc/README.md`.

A catalogue of every script, with its one-line description, is in [Analysis scripts](../reference/tools.md).

## Label CSVs for the AtriumDB label pass

Some scripts write per-case **label CSVs** (`numeric_delay_labels.py`, `bis_smoothing_labels.py`,
`lab_draw_labels.py`, `art_zero_labels.py`, `flush_response_labels.py`, and the physiosync and lineaccess
exports). Those are per case and stay on the machine; `vitaldb-bench-atriumdb` writes them into a build.
