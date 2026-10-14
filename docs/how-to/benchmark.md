# Benchmark builds

`vitaldb_bench.perf` compares a TimeNet and an AtriumDB build of the same cases. Commands that open the
AtriumDB build need the GPL package, so run them as `python -m vitaldb_bench_atriumdb.perf`; without it,
`--db atriumdb` and `map` stop with a pointer.

## Size and build cost

```bash
python -m vitaldb_bench_atriumdb.perf size --timenet build/tn --atriumdb build/adb --root "$VITALDB_ROOT" --out size.json
```

Bytes on disk for each build and for the source .vital files, total build seconds and peak memory (from
the build logs).

## Read speed

```bash
python -m vitaldb_bench.perf reads --db timenet --build build/tn --root "$VITALDB_ROOT" \
    --cases cases.json --out reads.jsonl --n-cases 10 --windows 10 --seed 0
```

On a seeded random set of cases it times opening the store, reading one whole `SNUADC/ECG_II` track,
10 s windows (first pass and repeat), and one 60 s window of five tracks at once (ECG, ART, PLETH, CO2, HR).
Each call is a fresh process, so its first pass is as cold as the machine allows. The OS page cache of a
virtual machine's host cannot be dropped from inside it, so "cold" means a first access by a new process.

For colder reads (page cache evicted before every read) use the GPL package's `coldread` (AtriumDB builds
that differ in block size) and `ecgcold` (AtriumDB and TimeNet side by side).

## Same answer downstream

```bash
python -m vitaldb_bench_atriumdb.perf map --timenet build/tn --atriumdb build/adb --root "$VITALDB_ROOT" --out map.json
```

Mean arterial pressure over consecutive 60 s windows from the reader, from TimeNet and from AtriumDB, and
how far apart they are (mmHg). Resumable; per-case rows go to `map.jsonl` and stay on the machine.

## Clock rates in a resync build

```bash
python -m vitaldb_bench.timing_report --timenet build/tn-resync --out timing_report.json
```

For a build made with `--timing resync`: per waveform track, the share of samples whose rate came from a
re-synchronisation jump, and the distribution of the estimated rate error in ppm, overall and within
cases. A device's clock should be steady within a case, so a wide within-case spread suggests noise rather
than drift. Aggregates only.
