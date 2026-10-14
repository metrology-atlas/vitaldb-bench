# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from numeric_delay_labels import em_two, posterior_late, se_of  # noqa: E402


def test_mixture_recovers_two_groups_and_classifies_with_case_se():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(2.0, 0.4, 300), rng.normal(4.2, 0.5, 500)]
    se = np.full(x.size, 0.1)
    m = em_two(x, se)
    assert abs(m["mu"][0] - 2.0) < 0.15 and abs(m["mu"][1] - 4.2) < 0.15
    assert abs(m["w"][1] - 0.625) < 0.05
    assert posterior_late(4.5, 0.1, m) > 0.99
    assert posterior_late(1.8, 0.1, m) < 0.01
    # a delay between the groups with a large SE stays uncertain
    assert 0.1 < posterior_late(3.1, 1.5, m) < 0.9


def test_se_from_blocks_with_floor_and_fallback():
    t = {"blocks": [{"d": 2.0, "edge": False}, {"d": 2.0, "edge": False}, {"d": 9.0, "edge": True}]}
    assert se_of(t, 0.8) == (0.05, 2)
    assert se_of({"blocks": [{"d": 2.0, "edge": False}]}, 0.8) == (0.8, 1)
