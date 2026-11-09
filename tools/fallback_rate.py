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
"""
from __future__ import annotations

import json
from pathlib import Path


class StepRates(dict):
    """Port -> SNUADC step-mode rate (ppm), with optional per-recording setup rates."""

    tier = "port_step_rate"
    setup_rate = None

    def __init__(self, data: dict | None = None):
        data = dict(data or {})
        cases = data.pop("cases", {}) or {}
        super().__init__(data)
        self.cases = {int(k): float(v) for k, v in cases.items()}

    def for_case(self, case) -> "StepRates":
        r = self.cases.get(int(case)) if case is not None else None
        if r is None:
            return self
        out = _CaseRate(self)
        out.setup_rate = r
        return out


class _CaseRate(StepRates):
    """A recording's view: any lookup gives its setup rate."""

    tier = "setup_step_rate"

    def __init__(self, base: StepRates):
        dict.__init__(self, base)
        self.cases = base.cases

    def get(self, key, default=None):  # noqa: D102 - the setup rate whatever the port
        return self.setup_rate

    def __getitem__(self, key):
        return self.setup_rate


def load_step_rates(path) -> StepRates:
    """Read a rates file (None or "" gives an empty table, as tools without --step-rates had)."""
    return StepRates(json.loads(Path(path).read_text()) if path else {})
