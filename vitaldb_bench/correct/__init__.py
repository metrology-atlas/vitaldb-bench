# SPDX-License-Identifier: MIT
"""Timing correction applied when data are read, with its uncertainty.

The stored data keep their nominal times. A clock model (a few numbers per stretch of each track, kept in
label tables beside the build) is applied at read time, so a better synchronisation model only means new
label tables, never a re-encoded database. :class:`CorrectedStore` returns, for every value, its corrected
time, bounds at a chosen coverage, and on request a distribution object per value or Monte Carlo
realisations that respect which errors are shared.

This subpackage is dataset-neutral: it must not import anything dataset-specific (a test enforces it), so
it can move to its own package once the interface settles. The VitalDB side (fitting the clocks, writing
the tables, reading a TimeNet build) is :mod:`vitaldb_bench.clock_model`.
"""

from .clock import ClockModel, TrackClock, TrackError
from .dist import Dist, Empirical, PointMass, Shifted, Uniform, sum_of
from .layer import CorrectedStore, CorrectedWindow, Samples, SampleSource

__all__ = ["ClockModel", "CorrectedStore", "CorrectedWindow", "Dist", "Empirical", "PointMass", "SampleSource",
           "Samples", "Shifted", "TrackClock", "TrackError", "Uniform", "sum_of"]
