# SPDX-License-Identifier: MIT
"""The SNUADC fallback rate: what the stamp model and the tools built on it use for a step-mode track whose own stamps
hold no whole tooth (physiosync.stamp_clock rate_measured False).

The rates file (tools/port_rates.py) maps port -> ppm, "*" -> the overall rate, and optionally "cases" -> {case: the
recording's setup rate}: the median whole-teeth rate of its recording station (its device ports), from at least 5
tracks. A setup rate predicts a recording's own rate to a robust SD of 0.75 ppm (examples/setup_rate_check.py); a port
rate can be off by 15 ppm or more where one port serves stations whose converters differ (COM4).

    rates = load_step_rates(path)        # a dict port -> ppm, as before
    r = rates.for_case(c)                # the same, but every lookup gives case c's setup rate if it has one
    r.get(port, r.get("*", 0.0))         # unchanged call sites: setup rate, else port rate, else overall
    r.tier                               # "setup_step_rate" or "port_step_rate"

A file without "cases" behaves exactly as the plain port dict did.

The no-correction bound (H.-C. Lee, 9 Nov 2026). Vital Recorder re-anchors the SNUADC count when it runs more than
0.5 s ahead of the recorder's clock and inserts blank samples when it falls behind, so a track with no correction at
all gained less than 0.5 s over each stretch between gaps (stamp_clock re-anchors at gaps): |rate| < 0.5 s / T, T the
longest stretch. A rates file with "no_correction_bound_s" (0.5) caps a borrowed rate at that bound for such tracks:

    fall, capped = r.no_correction_rate(port_rate, clk, record_starts, fs)   # clk: stamp_clock's first (fit) result

A file without the key leaves the rate unchanged (capped False), so earlier runs reproduce. In the burst check
(examples/snuadc_rate_bound.py) borrowed rates broke the bound in 1,040 of 3,410 such recordings and left a median
drift of +111 ms/h in those without gaps.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

BOUND_KEY = "no_correction_bound_s"


def longest_stretch_s(breaks, record_starts, fs) -> float:
    """Longest stretch between gaps (s): from its first record's start to its last record's start."""
    st = np.asarray(record_starts, dtype=float)
    edges = [int(b) for b in breaks] + [st.size]
    spans = [(st[e - 1] - st[s]) / fs for s, e in zip(edges[:-1], edges[1:]) if e - 1 > s]
    return float(max(spans)) if spans else 0.0


class StepRates(dict):
    """Port -> SNUADC step-mode rate (ppm), with optional per-recording setup rates."""

    tier = "port_step_rate"
    setup_rate = None
    bound_s = None

    def __init__(self, data: dict | None = None):
        data = dict(data or {})
        cases = data.pop("cases", {}) or {}
        b = data.pop(BOUND_KEY, None)
        self.bound_s = float(b) if b else None
        super().__init__(data)
        self.cases = {int(k): float(v) for k, v in cases.items()}

    def for_case(self, case) -> "StepRates":
        r = self.cases.get(int(case)) if case is not None else None
        if r is None:
            return self
        out = _CaseRate(self)
        out.setup_rate = r
        return out

    def no_correction_rate(self, rate_ppm, clk, record_starts, fs):
        """The rate for a track with no correction at all: rate_ppm, capped at bound_s / longest stretch when the
        file sets a bound. Returns (rate, capped)."""
        if not self.bound_s or rate_ppm is None:
            return rate_ppm, False
        t = longest_stretch_s(clk["breaks"], record_starts, fs)
        if t <= 0:
            return rate_ppm, False
        b = self.bound_s / t * 1e6
        if abs(rate_ppm) <= b:
            return rate_ppm, False
        return math.copysign(b, rate_ppm), True


class _CaseRate(StepRates):
    """A recording's view: any lookup gives its setup rate."""

    tier = "setup_step_rate"

    def __init__(self, base: StepRates):
        dict.__init__(self, base)
        self.cases = base.cases
        self.bound_s = base.bound_s

    def get(self, key, default=None):  # noqa: D102 - the setup rate whatever the port
        return self.setup_rate

    def __getitem__(self, key):
        return self.setup_rate


def load_step_rates(path) -> StepRates:
    """Read a rates file (None or "" gives an empty table, as tools without --step-rates had)."""
    return StepRates(json.loads(Path(path).read_text()) if path else {})
