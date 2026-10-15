# SPDX-License-Identifier: MIT
"""Error distributions for corrected times: empirical, uniform, and sums of independent parts.

Every distribution here answers the same four questions: ``pdf``, ``cdf``, ``quantile`` and ``sample``. An
:class:`Empirical` distribution is held as a grid of quantiles, so it keeps the measured shape (skew, heavy
tails, more than one peak) without assuming a family; the price is that it has nothing beyond the most
extreme value it was built from. A parametric family (for example Student-t) can be added later behind the
same four methods.

Nothing in this module knows about any dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Dist", "Empirical", "PointMass", "Shifted", "Uniform", "sum_of"]

_P = np.linspace(0.0, 1.0, 1001)


class Dist:
    """Interface shared by every distribution (all values in seconds)."""

    def cdf(self, x):  # pragma: no cover - interface
        """P(X <= x)."""
        raise NotImplementedError

    def quantile(self, p):  # pragma: no cover - interface
        """Inverse of :meth:`cdf`."""
        raise NotImplementedError

    def pdf(self, x):  # pragma: no cover - interface
        """Density at x (1/s)."""
        raise NotImplementedError

    def sample(self, n: int | tuple, rng: np.random.Generator) -> np.ndarray:
        """Independent draws (inverse-cdf sampling unless a subclass does better)."""
        return self.quantile(rng.random(n))

    def interval(self, coverage: float) -> tuple[float, float]:
        """Central interval holding ``coverage`` of the probability.

        Examples:
            >>> Uniform(-1.0, 1.0).interval(0.5)
            (-0.5, 0.5)
        """
        a = (1.0 - coverage) / 2.0
        return float(self.quantile(a)), float(self.quantile(1.0 - a))


@dataclass(frozen=True)
class PointMass(Dist):
    """All probability at one value (an error known to be exactly ``at``)."""

    at: float = 0.0

    def cdf(self, x):
        """Step from 0 to 1 at ``at``."""
        return np.where(np.asarray(x, float) >= self.at, 1.0, 0.0)

    def quantile(self, p):
        """``at`` for every p."""
        return np.full(np.shape(p), self.at, dtype=float) if np.ndim(p) else self.at

    def pdf(self, x):
        """Zero except at ``at`` (a delta; returned as +inf there)."""
        x = np.asarray(x, float)
        return np.where(x == self.at, np.inf, 0.0)


@dataclass(frozen=True)
class Uniform(Dist):
    """Uniform on ``[lo, hi]``, e.g. the rounding of a time stamp to its resolution step.

    Examples:
        >>> u = Uniform(-0.5e-6, 0.5e-6)
        >>> round(float(u.cdf(0.0)), 3), round(u.sd * 1e6, 4)
        (0.5, 0.2887)
    """

    lo: float
    hi: float

    @property
    def sd(self) -> float:
        """Standard deviation, width / sqrt(12)."""
        return float((self.hi - self.lo) / np.sqrt(12.0))

    def cdf(self, x):
        """Linear from 0 at ``lo`` to 1 at ``hi``."""
        if self.hi <= self.lo:
            return PointMass(self.lo).cdf(x)
        return np.clip((np.asarray(x, float) - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def quantile(self, p):
        """``lo + p * (hi - lo)``."""
        return self.lo + np.asarray(p, float) * (self.hi - self.lo)

    def pdf(self, x):
        """``1 / (hi - lo)`` inside, 0 outside."""
        x = np.asarray(x, float)
        w = self.hi - self.lo
        return np.where((x >= self.lo) & (x <= self.hi), 1.0 / w if w > 0 else np.inf, 0.0)


@dataclass(frozen=True)
class Empirical(Dist):
    """A measured distribution, held as its quantiles on a fixed probability grid.

    Between grid points the cdf is linear, so the density is piecewise constant. ``pdf`` smooths it over
    ``bandwidth`` quantile steps so that a few hundred measurements still give a readable curve.

    Args:
        q: Quantiles at the probabilities ``p`` (non-decreasing, seconds).
        p: The probabilities (default 0, 0.001, ..., 1).
        n: How many measurements it was built from (for reporting).

    Examples:
        >>> d = Empirical.from_samples(np.array([-0.02, -0.01, 0.0, 0.01, 0.02]))
        >>> float(d.quantile(0.5)), float(d.cdf(0.0))
        (0.0, 0.5)
    """

    q: np.ndarray
    p: np.ndarray = field(default_factory=lambda: _P)
    n: int = 0

    @classmethod
    def from_samples(cls, x: np.ndarray, p: np.ndarray | None = None) -> Empirical:
        """Build from raw measurements (NaNs dropped)."""
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if not x.size:
            raise ValueError("no finite samples")
        p = _P if p is None else np.asarray(p, float)
        return cls(q=np.quantile(x, p), p=p, n=int(x.size))

    def cdf(self, x):
        """Linear interpolation of the quantile grid (0 below the minimum, 1 above the maximum)."""
        # tied quantiles (a value measured many times) are a jump in the cdf: keep its top (right-continuous)
        rev_q = self.q[::-1]
        uq, i = np.unique(rev_q, return_index=True)
        return np.interp(np.asarray(x, float), uq, self.p[::-1][i], left=0.0, right=1.0)

    def quantile(self, p):
        """Linear interpolation between grid quantiles."""
        return np.interp(np.asarray(p, float), self.p, self.q)

    def pdf(self, x, bandwidth: int = 20):
        """Density: probability per second between quantiles ``bandwidth`` grid steps apart."""
        k = max(1, int(bandwidth))
        qa, qb = self.q[:-k], self.q[k:]
        mid = (qa + qb) / 2.0
        dens = (self.p[k:] - self.p[:-k]) / np.maximum(qb - qa, 1e-12)
        x = np.asarray(x, float)
        out = np.interp(x, mid, dens, left=0.0, right=0.0)
        return np.where((x < self.q[0]) | (x > self.q[-1]), 0.0, out)

    @property
    def median(self) -> float:
        """The 0.5 quantile."""
        return float(self.quantile(0.5))


@dataclass(frozen=True)
class Shifted(Dist):
    """A distribution moved by ``loc``: the time of one value = its point estimate + its error."""

    base: Dist
    loc: float

    def cdf(self, x):
        """The base cdf at ``x - loc``."""
        return self.base.cdf(np.asarray(x, float) - self.loc)

    def quantile(self, p):
        """Base quantile + ``loc``."""
        return self.base.quantile(p) + self.loc

    def pdf(self, x):
        """Base density at ``x - loc``."""
        return self.base.pdf(np.asarray(x, float) - self.loc)

    def sample(self, n, rng):
        """Base draws + ``loc``."""
        return self.base.sample(n, rng) + self.loc


def sum_of(parts: list[Dist], n: int = 200_000, seed: int = 0) -> Dist:
    """Distribution of a sum of independent errors, by Monte Carlo (fixed seed: reproducible).

    Parts much narrower than the widest one barely matter, so a lone non-trivial part is returned as is.

    Args:
        parts: The independent error components.
        n: Draws used to build the result.
        seed: Random seed.

    Returns:
        An :class:`Empirical` of the sum (or the single part, or a :class:`PointMass` at 0 for no parts).

    Examples:
        >>> s = sum_of([Uniform(-1.0, 1.0), Uniform(-1.0, 1.0)])
        >>> abs(float(s.quantile(0.5))) < 0.02, 1.9 < float(s.quantile(0.9999)) <= 2.0
        (True, True)
    """
    parts = [d for d in parts if not isinstance(d, PointMass) or d.at != 0.0]
    if not parts:
        return PointMass(0.0)
    if len(parts) == 1:
        return parts[0]
    rng = np.random.default_rng(seed)
    total = np.zeros(n)
    for d in parts:
        total += d.sample(n, rng)
    return Empirical.from_samples(total)
