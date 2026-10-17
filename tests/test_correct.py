# SPDX-License-Identifier: MIT
"""The dataset-neutral correction layer: clock lines, distributions, bounds and realisations."""

import ast
from pathlib import Path

import numpy as np
import pytest

from vitaldb_bench.correct import (
    ClockModel,
    CorrectedStore,
    Empirical,
    PointMass,
    Samples,
    TrackClock,
    TrackError,
    Uniform,
    sum_of,
)

CORRECT = Path(__file__).resolve().parents[1] / "vitaldb_bench" / "correct"
ALLOWED = {"__future__", "dataclasses", "typing", "numpy", "math", "collections", "functools"}


def test_correct_imports_nothing_dataset_specific():
    """The layer must stay movable to its own package: stdlib, numpy and its own modules only."""
    for f in CORRECT.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:                     # relative: inside the subpackage
                    continue
                names = [node.module]
            else:
                continue
            for n in names:
                assert n.split(".")[0] in ALLOWED, f"{f.name} imports {n}"


class Toy:
    """1000 samples at 100 Hz, nominal time = k / 100, stamps rounded to 1 us."""

    def samples(self, case, track, a, b):
        if track != "x":
            return None
        k = np.arange(1000)
        t = k / 100.0
        m = (t >= a) & (t < b)
        return Samples(values=k[m] * 2.0, t_nominal=t[m], k=k[m], n_track=1000, resolution_s=1e-6)


def _clock(**kw):
    base = dict(t0_s=0.0, fs=100.0, rate_ppm=0.0, k_breaks=np.array([0, 500]), offsets_s=np.array([0.0, 0.1]),
                ref_shift_s=0.0, n_samples=1000)
    return TrackClock(**{**base, **kw})


def test_no_clock_is_the_store_itself():
    w = CorrectedStore(Toy(), ClockModel()).window(1, "x", 2.0, 3.0)
    s = Toy().samples(1, "x", 2.0, 3.0)
    assert not w.corrected and np.array_equal(w.t, s.t_nominal) and np.array_equal(w.values, s.values)
    assert np.all(np.isnan(w.t_lo)) and w.notes


def test_clock_line_stretches_and_selection_by_corrected_time():
    m = ClockModel(clocks={(1, "x"): _clock(rate_ppm=1000.0, ref_shift_s=-0.05)})
    w = CorrectedStore(Toy(), m, margin_s=1.0).window(1, "x", 4.0, 6.0)
    k = (w.values / 2).astype(int)
    expect = k / 100.0 * 1.001 + np.where(k >= 500, 0.1, 0.0) - 0.05
    assert np.allclose(w.t, expect) and w.t.min() >= 4.0 and w.t.max() < 6.0
    assert set(w.stretch) == {0, 1} and np.all(np.diff(w.t) > 0)


def test_count_mismatch_is_not_corrected():
    m = ClockModel(clocks={(1, "x"): _clock(n_samples=999)})
    w = CorrectedStore(Toy(), m).window(1, "x", 1.0, 2.0)
    assert not w.corrected and "999" in w.notes[0]


def test_sample_order_mismatch_is_not_corrected():
    class Ordered(Toy):
        def samples(self, case, track, a, b):
            s = super().samples(case, track, a, b)
            return None if s is None else Samples(**{**s.__dict__, "sample_order": "rule-b"})

    m = ClockModel(clocks={(1, "x"): _clock(sample_order="rule-a")})
    w = CorrectedStore(Ordered(), m).window(1, "x", 1.0, 2.0)
    assert not w.corrected and "rule-a" in w.notes[0]
    m = ClockModel(clocks={(1, "x"): _clock(sample_order="rule-b")})
    assert CorrectedStore(Ordered(), m).window(1, "x", 1.0, 2.0).corrected
    # an order stated on one side only is not checked
    assert CorrectedStore(Toy(), m).window(1, "x", 1.0, 2.0).corrected


def test_bounds_hold_their_coverage():
    rng = np.random.default_rng(3)
    between = Empirical.from_samples(rng.standard_t(3, 20_000) * 0.02)     # heavy-tailed, as measured
    within = Empirical.from_samples(rng.normal(0, 0.01, 20_000))
    m = ClockModel(clocks={(1, "x"): _clock()}, errors={(1, "x"): TrackError(between=between, within=within)})
    w = CorrectedStore(Toy(), m, coverage=0.99).window(1, "x", 0.0, 10.0)
    truth = w.t + between.sample(w.t.size, rng) + within.sample(w.t.size, rng)
    inside = np.mean((truth >= w.t_lo) & (truth <= w.t_hi))
    assert 0.98 < inside <= 1.0 and np.all(w.available_hi == w.t_hi)
    assert w.error["total"]["hi_s"] > w.error["within"]["hi_s"]


def test_distributions_and_realisations_share_errors():
    between, within = Uniform(-0.05, 0.05), Uniform(-0.01, 0.01)
    m = ClockModel(clocks={(1, "x"): _clock()},
                   errors={(1, "x"): TrackError(between=between, within=within, block_s=100.0)})
    w = CorrectedStore(Toy(), m, seed=1).window(1, "x", 1.0, 1.1, pdf=True, realisations=500)
    d = w.dists[0]
    assert abs(float(d.quantile(0.5)) - w.t[0]) < 0.003 and float(d.cdf(w.t[0] + 0.07)) == 1.0
    r = w.realisations
    assert r.shape == (500, w.t.size)
    spread_abs = np.std(r[:, 0] - w.t[0])
    spread_diff = np.std((r[:, 1] - r[:, 0]) - (w.t[1] - w.t[0]))       # same block: shared errors cancel
    assert spread_abs > 0.02 and spread_diff < 1e-5


def test_empirical_and_sum():
    d = Empirical.from_samples(np.r_[np.zeros(50), np.ones(50)])
    assert float(d.cdf(0.0)) == pytest.approx(0.5, abs=0.01) and float(d.quantile(1.0)) == 1.0
    assert float(d.pdf(5.0)) == 0.0
    assert isinstance(sum_of([]), PointMass) and sum_of([d]) is d
    s = sum_of([Uniform(0, 1), Uniform(0, 1)])
    assert float(s.quantile(0.5)) == pytest.approx(1.0, abs=0.01)
