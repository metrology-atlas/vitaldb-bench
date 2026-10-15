# API reference

The installed package, `vitaldb_bench`. The analysis scripts are listed in
[Analysis scripts](tools.md); the AtriumDB side is documented with `vitaldb-bench-atriumdb`.

::: vitaldb_bench
    options:
      show_root_heading: false
      members: false

## Builds

::: vitaldb_bench.build

## Derived labels

::: vitaldb_bench.label_families

::: vitaldb_bench.timenet_labels

## Corrected times

The dataset-neutral layer (it imports nothing VitalDB-specific, so it can become its own package):

::: vitaldb_bench.correct

::: vitaldb_bench.correct.layer

::: vitaldb_bench.correct.clock

::: vitaldb_bench.correct.dist

The VitalDB side:

::: vitaldb_bench.clock_model

## Stores

::: vitaldb_bench.stores

## Fidelity check

::: vitaldb_bench.fidelity

## Benchmarks

::: vitaldb_bench.perf

## Case clock

::: vitaldb_bench.case_clock

## Timing report

::: vitaldb_bench.timing_report
