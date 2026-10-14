# SPDX-License-Identifier: MIT
"""Read a track of a case back out of a TimeNet build, in a common shape for comparison.

A :class:`TrackData` holds raw stored values, decoded values, and per-value times in nanoseconds
relative to the case origin. The GPL package vitaldb-bench-atriumdb (`vitaldb_bench_atriumdb.store`) provides the same for AtriumDB.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from timenet.dataset import Record, Signal, TimeFDataset
from timenet.dataset.axis import RegularAxis


@dataclass
class TrackData:
    """One track of one case as a store returns it, in a shape every store shares."""
    raw: np.ndarray
    """Values as stored (counts, lattice integers, or floats)."""
    decoded: np.ndarray
    """Values in the source's own representation: counts for integer formats, floats otherwise."""
    times_ns: np.ndarray
    """Per-value time, int64 ns from the case origin (for TimeNet: whole us * 1000)."""
    pieces: int
    """Signals (TimeNet) or intervals (AtriumDB) the track was stored as."""


def floor_mul(i: np.ndarray, num: int, den: int) -> np.ndarray:
    """Compute ``floor(i * num / den)`` for an int64 index array, exactly and without int64 overflow.

    Args:
        i: Non-negative int64 indices.
        num: Numerator of the multiplier.
        den: Denominator of the multiplier (positive).

    Returns:
        int64 array of the same shape as ``i``.

    Examples:
        >>> import numpy as np
        >>> floor_mul(np.array([0, 1, 2, 3]), 1000, 3).tolist()
        [0, 333, 666, 1000]
    """
    q, r = divmod(num, den)
    if r == 0:
        return i * q
    if int(i.max(initial=0)) * r < 2**62 and int(i.max(initial=0)) * q < 2**62:
        return i * q + (i * r) // den
    return np.array([(int(k) * num) // den for k in i], dtype=np.int64)


def regular_times_us(axis: RegularAxis, n: int, a: int = 0, b: int | None = None) -> np.ndarray:
    """Per-sample times of a regular TimeNet axis in microseconds, vectorised and exact.

    Args:
        axis: The signal's ``RegularAxis``.
        n: Number of values in the signal.
        a: First index (default 0).
        b: One past the last index (default ``n``).

    Returns:
        int64 times in us, as ``RegularAxis.time_offset_us`` gives them one at a time.
    """
    b = n if b is None else b
    i = np.arange(axis.start_index + a, axis.start_index + b, dtype=np.int64)
    return axis.offset_us + floor_mul(i, axis.period_us.numerator, axis.period_us.denominator)


def decode_signal(signal: Signal, values: np.ndarray) -> np.ndarray:
    """Return stored values in the source's own representation.

    Lattice integers (a float track stored as integers) become floats again; counts and floats are unchanged.

    Args:
        signal: The TimeNet signal (its metadata holds ``scale``, ``offset`` and ``float_dtype``).
        values: Values as stored.

    Returns:
        The decoded values.
    """
    md = signal.metadata
    if md.get("float_dtype"):
        return (values.astype(np.float64) * md["scale"] + md["offset"]).astype(md["float_dtype"])
    return values


class TimeNetStore:
    """All partitions of one TimeNet build directory, read one partition at a time.

    Only one partition is held open, so memory stays bounded by the largest partition; the build log
    (``build_log.jsonl``) maps each case to its partition.

    Args:
        build_dir: The build folder written by ``python -m vitaldb_bench.build timenet``.
    """

    def __init__(self, build_dir: str | Path) -> None:
        import json  # noqa: PLC0415

        self.build_dir = Path(build_dir)
        self._part_of: dict[int, str] = {}
        log = self.build_dir / "build_log.jsonl"
        for line in log.read_text().splitlines() if log.exists() else []:
            e = json.loads(line)
            if e.get("status") == "ok" and e.get("kind") == "timenet":
                for c in e["cases"]:
                    self._part_of[int(c)] = e["unit"]
        self._open_part: str | None = None
        self._records: dict[str, Record] = {}

    @cached_property
    def caseids(self) -> list[int]:
        """Case ids the build holds, sorted."""
        return sorted(self._part_of)

    def record(self, caseid: int) -> Record:
        """Open the TimeNet record of one case (opening its partition if another one is open).

        Args:
            caseid: VitalDB case id.

        Returns:
            The case's ``timenet.dataset.Record``.
        """
        part = self._part_of[caseid]
        if part != self._open_part:
            self._records = {}  # drop the previous partition (and any values its loaders cached)
            version = self.build_dir / part / "vitaldb" / "vitaldb" / "1.0.0"
            self._records = {r.record_id: r for r in TimeFDataset.open(path=version).records}
            self._open_part = part
        return self._records[f"vitaldb-{caseid:04d}"]

    def signals(self, caseid: int, dtname: str) -> list[Signal]:
        """The signals one VitalDB track was stored as, in time order.

        Args:
            caseid: VitalDB case id.
            dtname: VitalDB track name, e.g. ``"SNUADC/ART"``.

        Returns:
            The signals (one per stored piece); empty when the case has no such track.
        """
        sigs = [s for s in self.record(caseid).signals if s.metadata.get("vital_track") == dtname]
        return sorted(sigs, key=lambda s: (s.metadata.get("segment", s.metadata.get("piece", 0))))

    def track(self, caseid: int, dtname: str) -> TrackData | None:
        """Read one whole track of one case.

        Args:
            caseid: VitalDB case id.
            dtname: VitalDB track name.

        Returns:
            The track as :class:`TrackData`, or None when the case has no such track.
        """
        sigs = self.signals(caseid, dtname)
        if not sigs:
            return None
        raws, decs, times = [], [], []
        for s in sigs:
            arrow = s.to_arrow()
            raw = np.asarray(arrow.to_pylist(), dtype=object) if s.spec.dtype == "str" else arrow.to_numpy(zero_copy_only=False)
            raws.append(raw)
            decs.append(decode_signal(s, raw) if s.spec.dtype != "str" else raw)
            if isinstance(s.time_axis, RegularAxis):
                times.append(regular_times_us(s.time_axis, s.n_values) * 1000)
            else:
                times.append(s.time_offsets_us() * 1000)
        return TrackData(raw=np.concatenate(raws), decoded=np.concatenate(decs), times_ns=np.concatenate(times), pieces=len(sigs))

    def window(self, caseid: int, dtname: str, start_us: int, end_us: int) -> tuple[np.ndarray, np.ndarray]:
        """Values and times (us) of one track inside ``[start_us, end_us)``, reading only what is needed.

        Args:
            caseid: VitalDB case id.
            dtname: VitalDB track name.
            start_us: Window start, microseconds from the case origin.
            end_us: Window end (exclusive).

        Returns:
            ``(values, times_us)``: physical float64 values and int64 times; both empty when nothing falls in the
            window.
        """
        vals, times = [], []
        for s in self.signals(caseid, dtname):
            span = s.span_us
            if span is None or span[1] <= start_us or span[0] >= end_us:
                continue
            if isinstance(s.time_axis, RegularAxis):
                a = max(0, s.time_axis.index_at_or_after(start_us))
                b = min(s.n_values, s.time_axis.index_at_or_after(end_us))
                if b <= a:
                    continue
                raw = s.read_steps(a, b).to_numpy(zero_copy_only=False)
                t = regular_times_us(s.time_axis, s.n_values, a, b)
            else:
                off = s.time_offsets_us()
                a, b = np.searchsorted(off, start_us), np.searchsorted(off, end_us)
                if b <= a:
                    continue
                raw = s.read_steps(int(a), int(b)).to_numpy(zero_copy_only=False)
                t = off[a:b]
            vals.append(physical(s, raw))
            times.append(t)
        if not vals:
            return np.empty(0), np.empty(0, dtype=np.int64)
        return np.concatenate(vals), np.concatenate(times)


def physical(signal: Signal, raw: np.ndarray) -> np.ndarray:
    """Physical float64 values of stored values, by the rule in the signal's metadata.

    Args:
        signal: The TimeNet signal.
        raw: Values as stored.

    Returns:
        ``raw * scale + offset`` (rounded through the source's float type where it had one), else ``raw``.
    """
    md = signal.metadata
    if "scale" in md:
        v = raw.astype(np.float64) * md["scale"] + md["offset"]
        return v.astype(md["float_dtype"]).astype(np.float64) if md.get("float_dtype") else v
    return raw.astype(np.float64)
