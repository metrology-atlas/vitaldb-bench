# Timing modes

A .vital waveform is a series of records. Each record has a float64 start time, and its samples follow at
the nominal rate. Records that join without a break (within half a sample) form a **run**; a gap, an
overlap or a step back in time starts a new run.

Within a run the recorder stamps each record as the previous stamp plus the record's duration at the
nominal rate, so a run looks perfectly sampled. The device's own clock still drifts against the recording
PC by tens of ppm, and the recorder corrects that with a jump at the start of the next run: an overlap when
the device ran fast, a short gap when it ran slow.

Builds choose how to place samples in time with `--timing`:

| Mode | Places samples | Trade-off |
|---|---|---|
| `nominal` (default) | At the nominal rate, exactly, from the first record of each run. | Simple and compact; late in a long run a sample can sit milliseconds from the file's stamp. The distance is reported. |
| `file` | On a straight line through each run's first and last record, split wherever a record would land more than `--tolerance-us` from its stamp. | Reproduces the file's stamps to within the tolerance. |
| `resync` | At a rate estimated from the re-synchronisation jump that ends the run, so it ends where the next run starts. | Spreads the clock correction back over the run. Assumes the jump undoes only drift since the last jump. |

`vitaldb_bench.fidelity` reports both errors: against the file's stamps (`time_err_ns_*`) and against the
timing model the build used (`model_err_ns_*`, what the store's time grid costs).

`vitaldb_bench.timing_report` summarises the rates a `resync` build estimated.

## Values

`--values source` stores values as the file has them. `--values integer` stores a float track that sits
exactly on a lattice (a constant step) as integers with a scale and offset; integer-format waveforms
(converter counts) are always stored as counts with their gain and offset. Both are lossless, which
`values_exact` in the fidelity check confirms.
