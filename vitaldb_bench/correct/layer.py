# SPDX-License-Identifier: MIT
"""Read data with corrected times: :class:`CorrectedStore`.

A store only has to say, for a window of one track, which values it holds, their nominal times and their
sample counts (:class:`SampleSource`). :class:`CorrectedStore` puts each value on its track's clock line and
attaches the uncertainty, in up to four forms:

1. bounds (always): ``t_lo`` and ``t_hi`` at the chosen coverage around the corrected time ``t``;
2. the error parts (always, in :attr:`CorrectedWindow.error`);
3. distribution objects (``pdf=True``): one per value, with ``pdf``, ``cdf``, ``quantile`` and ``sample``;
4. realisations (``realisations=n``): ``n`` complete timelines, drawing the shared errors once per track
   (``between``) or once per block (``within``) and the rest per value, so that differences between
   nearby values keep their correct, smaller spread.

A track without a clock comes back on its nominal times with ``corrected=False`` and NaN bounds, so a
model with no clocks returns exactly what the store holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .clock import ClockModel, TrackClock, TrackError
from .dist import Dist, Shifted

__all__ = ["CorrectedStore", "CorrectedWindow", "SampleSource", "Samples"]


@dataclass
class Samples:
    """What a store returns for one window of one track.

    Attributes:
        values: The values (any dtype).
        t_nominal: Nominal time of each value, seconds on the store's timeline.
        k: Sample count of each value in its track, or None when the track has no sample count (irregular
            data): such values cannot be put on a clock line.
        n_track: Values in the whole track as the store holds it (checked against the clock's fit).
        resolution_s: Step of the stored time stamps (their rounding error is uniform over one step).
        sample_order: How the store ordered the samples, when its source writes them out of order in places
            (checked against the clock's fit; empty means not stated, and is not checked).
    """

    values: np.ndarray
    t_nominal: np.ndarray
    k: np.ndarray | None
    n_track: int
    resolution_s: float = 0.0
    sample_order: str = ""


class SampleSource(Protocol):
    """Anything that can return :class:`Samples` for a time window of one track."""

    def samples(self, case, track: str, start_s: float, end_s: float) -> Samples | None:
        """Values with nominal times in ``[start_s, end_s)``, or None when the case has no such track."""
        ...


@dataclass
class CorrectedWindow:
    """One window of one track, with corrected times and their uncertainty.

    Attributes:
        values: The values, in corrected-time order.
        t: Corrected time (s), the point estimate.
        t_nominal: Time as stored (s).
        t_lo: Lower bound of the true time at ``coverage`` (NaN when not corrected).
        t_hi: Upper bound.
        available_hi: Latest time the value could have been available (here the upper bound of ``t``; an
            onward delay is added by the caller's model where one exists).
        stretch: Stretch index of each value (-1 when not corrected).
        corrected: Whether a clock was applied.
        coverage: Probability held between the bounds.
        error: The error parts and their summary (``sd`` and central interval of each part, seconds).
        dists: With ``pdf=True``, one distribution object per value (true time); else None.
        realisations: With ``realisations=n``, an ``(n, len(t))`` array of possible true times; else None.
        notes: Why a track was left uncorrected, or other caveats.
    """

    values: np.ndarray
    t: np.ndarray
    t_nominal: np.ndarray
    t_lo: np.ndarray
    t_hi: np.ndarray
    available_hi: np.ndarray
    stretch: np.ndarray
    corrected: bool
    coverage: float
    error: dict = field(default_factory=dict)
    dists: list | None = None
    realisations: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def _summary(d: Dist, coverage: float) -> dict:
    lo, hi = d.interval(coverage)
    return {"median_s": float(d.quantile(0.5)), "lo_s": lo, "hi_s": hi}


class CorrectedStore:
    """A store read through a clock model.

    Args:
        source: The underlying store (see :class:`SampleSource`).
        model: Clocks and errors per ``(case, track)``.
        coverage: Probability held between ``t_lo`` and ``t_hi`` (default 0.99).
        margin_s: Extra nominal time read on each side of a window, so that values whose corrected time
            falls inside it are found; it must exceed the largest correction of the model.
        seed: Seed for realisations (reproducible by default).

    Examples:
        >>> import numpy as np
        >>> from vitaldb_bench.correct import ClockModel, TrackClock, TrackError, Uniform
        >>> class Toy:
        ...     def samples(self, case, track, a, b):
        ...         k = np.arange(1000); t = k / 100.0
        ...         m = (t >= a) & (t < b)
        ...         return Samples(values=k[m] * 1.0, t_nominal=t[m], k=k[m], n_track=1000, resolution_s=1e-6)
        >>> clk = TrackClock(t0_s=0.0, fs=100.0, rate_ppm=0.0, k_breaks=np.array([0]), offsets_s=np.array([0.0]),
        ...                  ref_shift_s=0.25, n_samples=1000)
        >>> err = TrackError(between=Uniform(-0.01, 0.01))
        >>> cs = CorrectedStore(Toy(), ClockModel(clocks={(1, "x"): clk}, errors={(1, "x"): err}), margin_s=1.0)
        >>> w = cs.window(1, "x", 1.0, 1.05)
        >>> w.t.round(2).tolist(), w.t_nominal.round(2).tolist()
        ([1.0, 1.01, 1.02, 1.03, 1.04], [0.75, 0.76, 0.77, 0.78, 0.79])
        >>> bool(np.all(w.t_hi - w.t < 0.0101))
        True
    """

    def __init__(self, source: SampleSource, model: ClockModel, *, coverage: float = 0.99, margin_s: float = 5.0,
                 seed: int = 0) -> None:
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be between 0 and 1")
        self.source, self.model, self.coverage, self.margin_s, self.seed = source, model, coverage, margin_s, seed
        self._totals: dict[tuple, Dist] = {}

    def clock(self, case, track: str) -> TrackClock | None:
        """The track's clock, or None."""
        return self.model.clocks.get((case, track))

    def _total(self, key, err: TrackError, resolution: Dist) -> Dist:
        if key not in self._totals:
            self._totals[key] = TrackError(err.between, err.within, resolution, err.block_s, err.group).total()
        return self._totals[key]

    def window(self, case, track: str, start_s: float, end_s: float, *, pdf: bool = False,
               realisations: int = 0) -> CorrectedWindow | None:
        """Values of one track whose corrected time is in ``[start_s, end_s)``.

        Args:
            case: Case key, as the store and the model use it.
            track: Track name.
            start_s: Window start, seconds on the store's (and the reference clock's) timeline.
            end_s: Window end (exclusive).
            pdf: Also return one distribution object per value.
            realisations: Also return this many Monte Carlo timelines.

        Returns:
            A :class:`CorrectedWindow`, or None when the store has no such track. Uncorrected tracks are
            selected on their nominal times.
        """
        from .dist import Uniform  # noqa: PLC0415

        clk = self.clock(case, track)
        notes: list[str] = []
        if clk is not None:
            s = self.source.samples(case, track, start_s - self.margin_s, end_s + self.margin_s)
            if s is None:
                return None
            if s.k is None:
                notes.append("no sample counts: not corrected")
                clk = None
            elif clk.n_samples >= 0 and s.n_track != clk.n_samples:
                notes.append(f"store holds {s.n_track} values, the clock was fitted on {clk.n_samples}: not corrected")
                clk = None
            elif clk.sample_order and s.sample_order and clk.sample_order != s.sample_order:
                notes.append(f"store's sample order is {s.sample_order}, the clock was fitted on {clk.sample_order}: "
                             "not corrected")
                clk = None
        if clk is None:
            s = self.source.samples(case, track, start_s, end_s)
            if s is None:
                return None
            if not notes:
                notes.append("no clock model for this track: not corrected")
            n = s.values.size
            nan = np.full(n, np.nan)
            return CorrectedWindow(values=s.values, t=s.t_nominal.astype(float), t_nominal=s.t_nominal, t_lo=nan,
                                   t_hi=nan.copy(), available_hi=nan.copy(), stretch=np.full(n, -1),
                                   corrected=False, coverage=self.coverage, notes=notes)

        t = clk.times(s.k)
        keep = (t >= start_s) & (t < end_s)
        order = np.argsort(t[keep], kind="stable")
        t, values, tn, k = t[keep][order], s.values[keep][order], s.t_nominal[keep][order], s.k[keep][order]
        stretch = clk.stretch(k)
        err = self.model.errors.get((case, track))
        resolution = Uniform(-s.resolution_s / 2, s.resolution_s / 2)
        if err is None:
            notes.append("no error model for this track: bounds are NaN")
            nan = np.full(t.size, np.nan)
            return CorrectedWindow(values=values, t=t, t_nominal=tn, t_lo=nan, t_hi=nan.copy(), available_hi=nan.copy(),
                                   stretch=stretch, corrected=True, coverage=self.coverage, notes=notes)
        total = self._total((case, track, s.resolution_s), err, resolution)
        lo, hi = total.interval(self.coverage)
        out = CorrectedWindow(
            values=values, t=t, t_nominal=tn, t_lo=t + lo, t_hi=t + hi, available_hi=t + hi, stretch=stretch,
            corrected=True, coverage=self.coverage, notes=notes,
            error={"group": err.group, "block_s": err.block_s,
                   "between": _summary(err.between, self.coverage), "within": _summary(err.within, self.coverage),
                   "resolution": _summary(resolution, self.coverage), "total": _summary(total, self.coverage)})
        if pdf:
            out.dists = [Shifted(total, float(x)) for x in t]
        if realisations:
            out.realisations = self._realise(t, stretch, err, resolution, int(realisations))
        return out

    def _realise(self, t: np.ndarray, stretch: np.ndarray, err: TrackError, resolution: Dist, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        r = np.empty((n, t.size))
        between = err.between.sample(n, rng)                       # one per timeline: shared by the track
        block = np.floor(t / err.block_s).astype(np.int64)         # within: one draw per block (and stretch)
        _, inv = np.unique(np.stack([block, stretch]), axis=1, return_inverse=True)
        inv = inv.ravel()
        within = err.within.sample((n, int(inv.max()) + 1 if inv.size else 0), rng)
        for i in range(n):
            r[i] = t + between[i] + (within[i, inv] if inv.size else 0.0) + resolution.sample(t.size, rng)
        return r
