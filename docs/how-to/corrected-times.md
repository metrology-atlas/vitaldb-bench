# Read data with corrected times

The builds keep every value on its **nominal** time (what the recorder stamped). The clock model is applied
when you read, from small tables beside the build, so a better synchronisation model only means new tables:
the database is never re-encoded.

!!! note "Proof of principle"
    This covers the SNUADC and BIS waveform tracks of TimeNet builds, with empirical error distributions and
    whole-recording fits. Fits that respect availability (`as_of` queries), parametric error tails, the
    AtriumDB reader and the joint time-value density are still to come.

## 1. Fit the clocks

```bash
python tools/clock_model_fit.py --root "$VITALDB_ROOT" --cases-json cases.json \
    --units sync_full/units.csv --step-rates snuadc_step_rates.json --out clock_fits.jsonl
```

For every SNUADC and BIS waveform track this fits the stamp sawtooth (`physiosync.stamp_clock`, with the
adopted VitalDB rules for tracks without whole teeth) and runs the ECG~EEG burst probe on the corrected
times. About 2 s per case; resumable.

## 2. Write the tables beside the build

```bash
python -m vitaldb_bench.build label-tables --out build/tn --results results/pat \
    --clock-fits clock_fits.jsonl --population more_fits.jsonl
```

This writes `build/tn/label_params/`:

| Table | Rows | What it holds |
|---|---|---|
| `<family>.parquet` | one per label | The derived label families, as the AtriumDB pass writes them; times in ns from the case origin. |
| `clock_model.parquet` | one per track stretch | Where the stretch starts (sample count), its offset, the track's rate and rate source, the reference shifts. |
| `clock_errors.parquet` | one per error part | Empirical distributions (1001 quantiles) of true minus corrected time. |

`--population` adds cases whose bursts help estimate the constants and errors (the build's own fits are
always used). Rerun the command after any model change.

## 3. Read

```python
from vitaldb_bench.clock_model import corrected_timenet

cs = corrected_timenet("build/tn", corrections="population", coverage=0.99)
w = cs.window(case, "BIS/EEG1_WAV", 600.0, 610.0)       # seconds from the case origin
w.t, w.t_lo, w.t_hi            # corrected times and 0.99 bounds of the true time
w.t_nominal, w.stretch         # as stored, and which stretch each value is in
w.error                        # the error parts, each as median and central interval

w = cs.window(case, "BIS/EEG1_WAV", 600.0, 610.0, pdf=True, realisations=200)
w.dists[0].pdf(x), w.dists[0].cdf(x)    # a distribution object per value
w.realisations                 # 200 possible true timelines, shared errors drawn once
```

Windows are selected on corrected time. Every time is on the **SNUADC clock** (the reference): SNUADC
tracks need no shift, BIS tracks are moved by the EEG-ECG offset the stamp model leaves.

| `corrections=` | BIS reference shift | Error |
|---|---|---|
| `"population"` | median of the other cases in the same SNUADC mode (leave-one-out) | the population's spread between cases, plus within a case |
| `"per_case"` | the case's own burst median (at least 5 good windows; else population) | within a case, plus the uncertainty of the median |
| `"none"` | no clocks | the store's own times, unchanged |

A track that cannot be corrected (no clock, or the store holds a different number of samples than the fit
saw) comes back on its stored times with `corrected=False`, NaN bounds and a note saying why.

## How good is it?

Measured on the dev and main builds with `tools/clock_model_check.py` (46 cases with both ECG and EEG, 420
ten-minute windows; the EEG's remaining lag behind the ECG, read back through the corrected store):

| Times | Median lag | \|lag\| p50 | \|lag\| p90 | Inside the 0.99 bounds |
|---|---|---|---|---|
| Stored (nominal) | -805 ms | 805 ms | 1,035 ms | - |
| Corrected, population constant | -6 ms | 53 ms | 225 ms | 97.4% |
| Corrected, own bursts | 0 ms | 23 ms | 106 ms | 97.9% |

Five-fold held-out coverage of the 0.99 bounds over all 208 cases with bursts is 98.4% (step mode) and
98.6% (frequent-small-correction mode): the empirical tails are slightly short, which is what a parametric
tail would address. The errors come from ECG~EEG bursts, so they include the burst probe's own noise.
