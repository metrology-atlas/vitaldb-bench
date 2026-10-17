# SPDX-License-Identifier: MIT
"""Clock models: where each sample of a track really happened, and how sure we are.

A :class:`TrackClock` maps a sample's position in its track (the sample count ``k``, which a recorder never
loses even when its time stamps jump) to a corrected time:

    t(k) = t0 + (k / fs) * (1 + rate) + offset[stretch(k)] + ref_shift

The rate is the device's clock error, each stretch (the samples between two real gaps) has its own offset,
and ``ref_shift`` moves the device onto the reference clock that every track is expressed in. This is the
straight line through the middle of a stamp sawtooth (see ``physiosync.stamp_clock``), stored compactly: a
few numbers per stretch instead of a time per sample.

A :class:`TrackError` says how far the true time may be from ``t(k)``, as independent parts:

- ``between``: error shared by every sample of the track in the case (how far this case's device offset is
  from the constant used);
- ``within``: error shared within a block of ``block_s`` seconds (drift of the offset during the case);
- ``resolution``: error of each value on its own (the stored time stamp's rounding).

All three are distributions of ``true time - corrected time`` in seconds. Nothing here knows about any
dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dist import Dist, PointMass, sum_of

__all__ = ["ClockModel", "TrackClock", "TrackError"]


@dataclass(frozen=True)
class TrackClock:
    """The count-to-time line of one track.

    Args:
        t0_s: Time of sample 0 before correction, seconds on the store's timeline.
        fs: Nominal sampling rate (Hz).
        rate_ppm: Device clock error; positive means its samples last longer than nominal.
        k_breaks: Sample count at which each stretch starts (the first is 0).
        offsets_s: Offset of each stretch (s).
        ref_shift_s: Added to every time to put the device on the reference clock (s).
        n_samples: Samples in the track when the clock was fitted; a store holding a different number is
            not corrected (its counts would not line up).
        sample_order: The order the fit put the samples in, where the source writes some out of order (for
            VitalDB, the SNUADC reset rule); a store stating a different order is not corrected. Empty: not
            stated.

    Examples:
        >>> c = TrackClock(t0_s=10.0, fs=100.0, rate_ppm=100.0, k_breaks=np.array([0, 1000]),
        ...                offsets_s=np.array([0.0, 0.5]), ref_shift_s=0.0, n_samples=2000)
        >>> c.times(np.array([0, 100, 1000])).round(6).tolist()
        [10.0, 11.0001, 20.501]
    """

    t0_s: float
    fs: float
    rate_ppm: float
    k_breaks: np.ndarray
    offsets_s: np.ndarray
    ref_shift_s: float = 0.0
    n_samples: int = -1
    sample_order: str = ""

    def stretch(self, k: np.ndarray) -> np.ndarray:
        """Stretch index of each sample count."""
        return np.searchsorted(self.k_breaks, np.asarray(k), side="right") - 1

    def times(self, k: np.ndarray) -> np.ndarray:
        """Corrected time (s) of each sample count."""
        k = np.asarray(k, dtype=np.float64)
        s = np.clip(self.stretch(k), 0, len(self.offsets_s) - 1)
        return self.t0_s + k / self.fs * (1.0 + self.rate_ppm * 1e-6) + self.offsets_s[s] + self.ref_shift_s


@dataclass(frozen=True)
class TrackError:
    """How far the true time may be from the corrected one, in independent parts (see the module docstring).

    Args:
        between: Shared by the whole track in the case.
        within: Shared within each block of ``block_s`` seconds.
        resolution: Independent for every value.
        block_s: Block length for ``within`` (s).
        group: Name of the population the distributions came from (for reporting).
    """

    between: Dist = field(default_factory=PointMass)
    within: Dist = field(default_factory=PointMass)
    resolution: Dist = field(default_factory=PointMass)
    block_s: float = 300.0
    group: str = ""

    def total(self) -> Dist:
        """Distribution of the whole error of one value (the parts added, as independent)."""
        return sum_of([self.between, self.within, self.resolution])


@dataclass
class ClockModel:
    """Clocks and errors for every corrected track, keyed by ``(case, track)``.

    Args:
        clocks: The count-to-time line per track.
        errors: The error parts per track (a track with a clock and no entry here has unknown error).
        info: Free-form provenance: model version, reference clock, correction kind.
    """

    clocks: dict[tuple, TrackClock] = field(default_factory=dict)
    errors: dict[tuple, TrackError] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
