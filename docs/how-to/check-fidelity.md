# Check a build against the source files

`vitaldb_bench.fidelity` reads every case of a build back and compares every stored track with what the
.vital file holds, read by the same shared reader the importers use.

```bash
python -m vitaldb_bench.fidelity --root "$VITALDB_ROOT" --timenet build/tn --out fidelity.json --timing nominal
```

With an AtriumDB build as well (GPL package):

```bash
python -m vitaldb_bench_atriumdb.fidelity --root "$VITALDB_ROOT" --timenet build/tn --atriumdb build/adb \
    --out fidelity.json --timing nominal
```

Pass the `--timing` the builds used: it decides where the importer *meant* to put each sample, so the check
can tell the cost of the timing model from the cost of the store's time grid.

## What is checked

| Check | Meaning |
|---|---|
| `values_exact` | Stored values decode to the file's values bit for bit (counts for integer formats, the stored floats otherwise). |
| `physical_exact` | AtriumDB: physical values (`count * gain + offset`) match the reader's. |
| `scale_kept` | TimeNet: each signal keeps the track's gain and offset. |
| `time_err_ns_*` | Distance between where the store puts each value and where the file puts it. |
| `model_err_ns_*` | The same, against the importer's own timing model: what the store's time grid costs. |
| `count_match`, `missing_values_total` | Whether every value survived. |
| `pieces_total` | Signals (TimeNet) or intervals (AtriumDB) a track was stored as. |
| `labels_equal` | Event annotations and labels match the file's `EVENT` track. |

AtriumDB returns values sorted by time, so where records overlap or step back in time the file's order
cannot be kept. Those tracks are compared as sets of (time, value), paired by time in bins of a quarter
sample and then by value.

## Outputs

- `fidelity.jsonl`: one line per case. **Per case: keep it on the analysis machine.** It also makes the
  check resumable; run the command again to continue.
- `fidelity.json`: the summary, by track and database. Counts and worst-case errors only; no values and no
  case ids. This is the file that can leave the machine.

Use `--exclude ID,ID` (case ids, comma-separated) to skip cases (for example one whose build was interrupted).
