# Plan a multi-signal analysis: the timing budget

Before combining two or more VitalDB signals, check how well their times can line up.
`tools/pair_timing_budget.py` reads the accuracy and resolution of every kind of VitalDB time stamp (the
entries behind the time-source accuracy figure) and reports what a joint analysis can resolve. It reads no
data.

!!! note "Superseded by the metrology atlas's planner (October 2026)"
    This tool uses the "accuracy and resolution" entries of an earlier version of the time-source figure. The
    [metrology atlas](https://vitaldb.metrology-atlas.org/planner.html) now describes every kind of time stamp by its
    bias and standard uncertainty (VIM/GUM), with shared errors cancelling between signals, and its synchronisation
    planner replaces this budget. The tool is kept because released results cite it.

```bash
python tools/pair_timing_budget.py --list                 # the sources and their clock domains
python tools/pair_timing_budget.py snu_model pleth        # ECG (SNUADC reference) + PPG
python tools/pair_timing_budget.py snu_model hr_lab --resolve-ms 5
python tools/pair_timing_budget.py --selftest
```

For ECG and PPG from the SNUADC converter:

```text
Relative timing sigma        13.2 ms
Coherent cross-channel band  10 Hz
Finest useful shared grid    45.8 ms
Latency floor (avail. rule)  400.0 ms
Oldest data in the vector    433.0 ms
Longest window before drift  unbounded  (one clock: drift cancels)
```

## The model

A screening tool, not a fit. For each source, the stamp's delay range is `[x_lo, x_hi]` and the width of
the interval a value describes is up to `y_hi`.

| Quantity | Formula | Meaning |
|---|---|---|
| delay spread | `σ_delay = (x_hi − x_lo) / √12` | What is left after correcting the delay to the middle of its range. |
| interval | `σ_res = y_hi / √12` | Where in its interval a value's time lies. |
| relative sigma | `σ = √(Σ σ_delay² + σ_res²)` | Over all sources, taken as independent. |
| coherent band | `0.1325 / σ` | The −3 dB frequency of a Gaussian timing blur. |
| finest grid | `√12 · σ` | A shared grid finer than this adds no cross-channel information. |
| latency floor | `max x_hi` | Availability rule: a joint vector at `t` can only hold windows that ended this long before `t`. |
| beats to average | `(σ / δ)²` | To resolve `δ` (`--resolve-ms`); the random part only, bias does not average away. |
| longest window | `σ / abs(Δdrift)` | Between two clocks with known drift and no resync. |

Sources on one clock domain share their drift, which then cancels. Entries marked as estimated rather than
measured are flagged in the output.
