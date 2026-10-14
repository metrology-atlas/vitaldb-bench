# vitaldb-bench

Timing and data-quality audit of [VitalDB](https://vitaldb.net). It measures how far every time stamp in the
release can be trusted, and builds databases (TimeNet here; AtriumDB in the companion package) that carry
what was found as labels.

- **Clock drift and synchronisation.** Each device in an operating room keeps its own clock. The audit
  measures drift, re-synchronisation jumps and the offsets between devices.
- **Numeric delays and averaging windows.** A monitor numeric (HR, SpO2, ETCO2, ...) describes a window of
  signal that ended *before* its time stamp. The audit measures each window and delay.
- **Derived labels.** Per-case results (delay groups, line accesses, lab draws, ...) become label CSVs
  that the AtriumDB label pass writes into a build, each with `available_at`, the earliest time its
  information could have been known.
- **Resumable builds and checks.** TimeNet builds run in short, time-boxed chunks; every stored value and
  time is then checked against the .vital files, and the builds are benchmarked.

!!! warning "The availability rule"
    No value may be used before it could have been known. A numeric is known at its stamp, but it
    describes a window that ended earlier; a lab value is known only after a turnaround that VitalDB does
    not record. See [Availability](concepts/availability.md).

## Where to go next

| You want to | Read |
|---|---|
| Install it and run a first build | [Get started](get-started.md) |
| Build TimeNet in time-boxed chunks | [Build TimeNet](how-to/build-timenet.md) |
| Check a build against the source files | [Check fidelity](how-to/check-fidelity.md) |
| Benchmark sizes and reads | [Benchmark builds](how-to/benchmark.md) |
| Recover a case's clock time of day | [Recover the case clock](how-to/case-clock.md) |
| Plan a multi-signal analysis | [Timing budget](how-to/timing-budget.md) |
| Find which script made a result | [Reproduce the results](how-to/reproduce-results.md) |
| Understand the timing models | [Timing modes](concepts/timing-modes.md) |

## Related packages

| Package | Licence | What it does |
|---|---|---|
| `vitaldb-bench-atriumdb` | GPL-3.0-or-later | AtriumDB builds, derived labels and benchmarks. Imports this package. |
| `physiosync` | MIT | Clock and ECG-PPG timing corrections. |
| `lineaccess` | MIT | Arterial-line access detection (flushes, draws, zeroing). |
| `vitaldb-resolution` | MIT | Value and time resolution of every VitalDB variable. |

No VitalDB data are included in this repository: only code, aggregate results and figures.
The longer write-up is the [VitalDB timing guide](timing_guide/vitaldb_timing_guide.pdf) (PDF).
