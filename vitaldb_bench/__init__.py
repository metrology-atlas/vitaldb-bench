# SPDX-License-Identifier: MIT
"""VitalDB timing and data-quality audit: resumable TimeNet builds, stores, fidelity and performance checks.

Modules:
    build: case selection and resumable TimeNet builds.
    stores: read a track back out of a TimeNet build (:class:`~vitaldb_bench.stores.TrackData`).
    fidelity: check every stored value and time against the .vital files.
    perf: size, read-speed and downstream-equivalence benchmarks.
    case_clock: recover each case's clock time of day from the clinical table.
    timing_report: summarise the clock rates a resync-timed build estimated.

AtriumDB builds and labels are in the GPL-3.0-or-later package ``vitaldb_bench_atriumdb``, which imports
this one; this package never imports it.
"""
