# SPDX-License-Identifier: MIT
"""The VitalDB clock model: tables beside a build, and a corrected view of a TimeNet build.

The general machinery is :mod:`vitaldb_bench.correct`; this module holds the VitalDB side.

**Clock lines.** ``tools/clock_model_fit.py`` fits every SNUADC and BIS waveform track's stamp sawtooth
(physiosync.stamp_clock with the adopted VitalDB rules) and probes the ECG~EEG bursts on the corrected
times. Each track becomes a few rows in ``<build>/label_params/clock_model.parquet``: one per stretch, with
the sample count where the stretch starts, its offset, the track's rate and where the rate came from.

**Reference clock.** Every time is expressed on the SNUADC clock (the ECG): SNUADC tracks need no shift,
BIS tracks are shifted by the EEG-ECG offset that the stamp model leaves. Two corrections are offered:

- ``population``: the median offset of the other cases in the same SNUADC mode (step mode or
  frequent-small-correction mode), leave-one-out, so a case's own bursts never set its own constant;
- ``per_case``: the case's own burst median, where it has at least ``MIN_CASE_WINDOWS`` good windows
  (otherwise the population constant).

**Errors.** ``clock_errors.parquet`` holds empirical distributions (1001 quantiles each) of
``true time - corrected time`` for BIS tracks, per SNUADC mode: ``between`` (case median minus its
leave-one-out constant), ``within`` (each window's lag minus its case median) and ``pooled``. They are
measured on ECG~EEG bursts, so they also hold the burst probe's own noise (they over- rather than
under-state the error), and they do not yet separate the rate term. SNUADC tracks, being the reference,
carry only the time stamps' rounding.

All of these are whole-recording fits (``available_at = case_end``): right for retrospective alignment, not
for queries that must respect availability (deferred; see the drift document).
"""

from __future__ import annotations

import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from vitaldb_bench.correct import ClockModel, Empirical, Samples, TrackClock, TrackError

MODEL_VERSION = "clock-model-2026.10.15"
REFERENCE = "SNUADC"
MIN_CASE_WINDOWS = 5
GOOD_PEAK, GOOD_MARGIN, LAG_RANGE_MS = 0.3, 0.15, (-2500.0, 500.0)   # burst window quality, as in the validation


# ---------------------------------------------------------------------------------------------- the fits

def load_fits(*paths: str | Path) -> dict[int, dict]:
    """Per-case lines of ``clock_model_fit.py`` (or sa_corrected) output; later files win for a case.

    Lines with an ``error`` are skipped.
    """
    out: dict[int, dict] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    out[int(r["case"])] = r
    return out


def mode_of(fit: dict) -> str | None:
    """SNUADC mode of a case: ``step`` when its ECG clock rate is below -20 ppm, else ``small``."""
    if fit.get("snuadc_mode"):
        return fit["snuadc_mode"]
    try:
        return "step" if fit["model"]["clocks"]["ecg"]["rate_ppm"] < -20 else "small"
    except (KeyError, TypeError):
        return None


def burst_lags_s(fit: dict) -> np.ndarray:
    """Good ECG~EEG1 burst lags (s) on corrected times: the EEG's delay behind the ECG in each window."""
    rows = (fit.get("model") or {}).get("burst", {}).get("ecg~eeg1", [])
    a = np.asarray(rows, dtype=float).reshape(-1, 5)
    g = a[(a[:, 2] >= GOOD_PEAK) & (a[:, 2] - a[:, 3] >= GOOD_MARGIN) & (a[:, 1] > LAG_RANGE_MS[0])
          & (a[:, 1] < LAG_RANGE_MS[1])]
    return g[:, 1] / 1000.0


def population(fits: dict[int, dict]) -> dict[int, dict]:
    """Per case with good bursts: mode, its burst lags, their median and count."""
    pop = {}
    for c, f in fits.items():
        m, lags = mode_of(f), burst_lags_s(f)
        if m and lags.size:
            pop[c] = {"mode": m, "lags": lags, "median": float(np.median(lags)), "n": int(lags.size)}
    return pop


def loo_constant(pop: dict[int, dict], case: int, mode: str) -> float | None:
    """Median of the other cases' burst medians in the same mode (the case itself left out)."""
    v = [p["median"] for c, p in pop.items() if c != case and p["mode"] == mode]
    return float(np.median(v)) if v else None


def error_parts(pop: dict[int, dict], exclude: set[int] = frozenset()) -> dict[str, dict[str, Empirical]]:
    """Empirical distributions of ``true - corrected`` time per mode: between, within and pooled.

    Args:
        pop: Output of :func:`population`.
        exclude: Cases left out (for held-out checks).

    Returns:
        ``{mode: {"between": ..., "within": ..., "pooled": ...}}``.
    """
    out = {}
    for mode in ("step", "small"):
        cases = [c for c, p in pop.items() if p["mode"] == mode and c not in exclude]
        between, within, pooled = [], [], []
        for c in cases:
            k = loo_constant({x: pop[x] for x in cases}, c, mode)
            if k is None:
                continue
            p = pop[c]
            # the EEG is late by (lag - constant) after the shift, so true - corrected = -(lag - constant)
            between.append(-(p["median"] - k))
            pooled.extend(-(p["lags"] - k))
            if p["n"] >= 3:
                within.extend(-(p["lags"] - p["median"]))
        if between and within:
            out[mode] = {"between": Empirical.from_samples(np.array(between)),
                         "within": Empirical.from_samples(np.array(within)),
                         "pooled": Empirical.from_samples(np.array(pooled))}
    return out


# ---------------------------------------------------------------------------------------------- tables

def clock_rows(fits: dict[int, dict], pop: dict[int, dict]) -> list[dict]:
    """One row per (case, track, stretch) for ``clock_model.parquet``."""
    rows = []
    for c, f in sorted(fits.items()):
        mode = mode_of(f)
        k_pop = loo_constant(pop, c, mode) if mode else None
        own = pop.get(c)
        for name, t in sorted(f.get("tracks", {}).items()):
            dev = t["device"]
            if dev == REFERENCE:
                shift_pop, shift_case, group = 0.0, 0.0, "reference"
            elif mode is None or k_pop is None:
                continue                         # no SNUADC ECG: the BIS track cannot be put on the reference
            else:
                # the EEG lags the ECG by the constant: subtract it to put the EEG on the ECG's clock
                shift_pop = -k_pop
                shift_case = -own["median"] if own and own["n"] >= MIN_CASE_WINDOWS else None
                group = f"bis~snuadc:{mode}"
            for i, (kb, off) in enumerate(zip(t["k_breaks"], t["offsets_s"], strict=True)):
                rows.append({"case": c, "track": name, "device": dev, "stretch": i, "k_start": int(kb), "offset_s": off,
                             "t0_s": t["t0_s"], "origin_s": f["origin_s"], "fs": t["fs"], "rate_ppm": t["rate_ppm"],
                             "rate_source": t["rate_source"], "rate_tier": t["rate_tier"], "n_samples": t["n_samples"],
                             "max_shift_s": t["max_shift_s"], "snuadc_mode": mode, "ref_shift_population_s": shift_pop,
                             "ref_shift_case_s": shift_case, "case_windows": own["n"] if own else 0,
                             "error_group": group, "constant_fit": "leave-one-out", "available_at": "case_end",
                             "model_version": MODEL_VERSION})
    return rows


def error_rows(parts: dict[str, dict[str, Empirical]]) -> list[dict]:
    """Rows of ``clock_errors.parquet``: one per (group, part), quantiles as a list."""
    return [{"error_group": f"bis~snuadc:{mode}", "part": name, "n": d.n, "p": d.p.tolist(), "q_s": d.q.tolist(),
             "quantity": "true time - corrected time (s)", "model_version": MODEL_VERSION}
            for mode, ds in parts.items() for name, d in ds.items()]


def write_tables(build_dir: str | Path, fits: dict[int, dict], population_fits: dict[int, dict] | None = None) -> dict:
    """Write ``clock_model.parquet`` and ``clock_errors.parquet`` into ``<build>/label_params``.

    Args:
        build_dir: The build folder (TimeNet), or an AtriumDB dataset folder.
        fits: Fits of the build's cases.
        population_fits: More fits to estimate the constants and errors from (the build's own are added).

    Returns:
        Row counts written.
    """
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    pop = population({**(population_fits or {}), **fits})
    out = Path(build_dir) / "label_params"
    out.mkdir(parents=True, exist_ok=True)
    cr, er = clock_rows(fits, pop), error_rows(error_parts(pop))
    pq.write_table(pa.Table.from_pylist(cr), out / "clock_model.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(er), out / "clock_errors.parquet", compression="zstd")
    return {"clock_model": len(cr), "clock_errors": len(er), "population_cases": len(pop)}


def load_clock_model(build_dir: str | Path, corrections: str = "population", timeline: str = "origin") -> ClockModel:
    """Read the tables back as a :class:`~vitaldb_bench.correct.ClockModel`, keyed by ``(case, track)``.

    Args:
        build_dir: Folder holding ``label_params/``.
        corrections: ``population`` (leave-one-out constants), ``per_case`` (the case's own bursts where it
            has enough of them) or ``none`` (no clocks: the store's own times).
        timeline: ``origin`` for times in seconds from the case origin (TimeNet), ``absolute`` for the
            files' own time stamps.

    Returns:
        The model.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    info = {"model_version": MODEL_VERSION, "reference": REFERENCE, "corrections": corrections}
    if corrections == "none":
        return ClockModel(info=info)
    if corrections not in ("population", "per_case"):
        raise ValueError("corrections must be 'population', 'per_case' or 'none'")
    base = Path(build_dir) / "label_params"
    rows = pq.read_table(base / "clock_model.parquet").to_pylist()
    parts: dict[str, dict[str, Empirical]] = defaultdict(dict)
    for r in pq.read_table(base / "clock_errors.parquet").to_pylist():
        parts[r["error_group"]][r["part"]] = Empirical(q=np.asarray(r["q_s"]), p=np.asarray(r["p"]), n=r["n"])
    by_track: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_track[(r["case"], r["track"])].append(r)
    model = ClockModel(info={**info, "model_version": rows[0]["model_version"] if rows else MODEL_VERSION})
    for key, rs in by_track.items():
        rs.sort(key=lambda r: r["stretch"])
        r0 = rs[0]
        own = corrections == "per_case" and r0["ref_shift_case_s"] is not None
        shift = r0["ref_shift_case_s"] if own else r0["ref_shift_population_s"]
        t0 = r0["t0_s"] - (r0["origin_s"] if timeline == "origin" else 0.0)
        model.clocks[key] = TrackClock(t0_s=t0, fs=r0["fs"], rate_ppm=r0["rate_ppm"],
                                       k_breaks=np.array([r["k_start"] for r in rs], dtype=np.int64),
                                       offsets_s=np.array([r["offset_s"] for r in rs]), ref_shift_s=shift,
                                       n_samples=r0["n_samples"])
        g = r0["error_group"]
        if g == "reference":
            model.errors[key] = TrackError(group="reference (SNUADC)")
        elif g in parts:
            within = parts[g]["within"]
            between = (Empirical(q=within.q / np.sqrt(r0["case_windows"]), p=within.p, n=within.n) if own
                       else parts[g]["between"])
            model.errors[key] = TrackError(between=between, within=within,
                                           group=g + (" (own bursts)" if own else " (leave-one-out constant)"))
    return model


# ---------------------------------------------------------------------------------------------- TimeNet

class TimeNetSource:
    """A TimeNet build as a :class:`~vitaldb_bench.correct.SampleSource` (times in s from the case origin).

    Args:
        store: A :class:`~vitaldb_bench.stores.TimeNetStore`, or the build folder.
    """

    resolution_s = 1e-6          # TimeNet times are whole microseconds

    def __init__(self, store) -> None:
        from vitaldb_bench.stores import TimeNetStore  # noqa: PLC0415

        self.store = store if isinstance(store, TimeNetStore) else TimeNetStore(store)

    def samples(self, case: int, track: str, start_s: float, end_s: float) -> Samples | None:
        """Values of one track with nominal times in ``[start_s, end_s)``, with their sample counts."""
        from timenet.dataset.axis import RegularAxis  # noqa: PLC0415

        from vitaldb_bench.stores import physical, regular_times_us  # noqa: PLC0415

        sigs = self.store.signals(case, track)
        if not sigs:
            return None
        a_us, b_us = int(np.floor(start_s * 1e6)), int(np.ceil(end_s * 1e6))
        vals, times, ks = [], [], []
        k0, regular = 0, True
        for s in sigs:
            n = s.n_values
            span = s.span_us
            if isinstance(s.time_axis, RegularAxis):
                if span is not None and span[1] > a_us and span[0] < b_us:
                    a = max(0, s.time_axis.index_at_or_after(a_us))
                    b = min(n, s.time_axis.index_at_or_after(b_us))
                    if b > a:
                        raw = s.read_steps(a, b).to_numpy(zero_copy_only=False)
                        vals.append(physical(s, raw))
                        times.append(regular_times_us(s.time_axis, n, a, b) / 1e6)
                        ks.append(np.arange(k0 + a, k0 + b, dtype=np.int64))
            else:
                regular = False
                off = s.time_offsets_us()
                a, b = np.searchsorted(off, a_us), np.searchsorted(off, b_us)
                if b > a:
                    vals.append(physical(s, s.read_steps(int(a), int(b)).to_numpy(zero_copy_only=False)))
                    times.append(off[a:b] / 1e6)
            k0 += n
        if not vals:
            return Samples(values=np.empty(0), t_nominal=np.empty(0), k=np.empty(0, np.int64) if regular else None,
                           n_track=k0, resolution_s=self.resolution_s)
        t = np.concatenate(times)
        m = (t >= start_s) & (t < end_s)
        return Samples(values=np.concatenate(vals)[m], t_nominal=t[m], k=np.concatenate(ks)[m] if regular else None,
                       n_track=k0, resolution_s=self.resolution_s)


def corrected_timenet(build_dir: str | Path, corrections: str = "population", coverage: float = 0.99):
    """A :class:`~vitaldb_bench.correct.CorrectedStore` over a TimeNet build and its clock tables.

    Examples:
        >>> # cs = corrected_timenet("timenet_db/main-nominal-integer")
        >>> # w = cs.window(case, "BIS/EEG1_WAV", 600.0, 610.0, pdf=True)
    """
    from vitaldb_bench.correct import CorrectedStore  # noqa: PLC0415

    model = load_clock_model(build_dir, corrections=corrections, timeline="origin")
    margin = 1.0 + max([abs(c.ref_shift_s) for c in model.clocks.values()] + [0.0])
    return CorrectedStore(TimeNetSource(build_dir), model, coverage=coverage, margin_s=margin)


# ---------------------------------------------------------------------------------------------- label tables

def write_family_tables(build_dir: str | Path, results: str | Path, families: list[str] | None = None) -> dict:
    """The derived label families as ``<build>/label_params/<family>.parquet`` (the AtriumDB layout).

    Times are ns from each case's origin (the TimeNet record timeline); ``start_s``/``end_s`` keep the
    files' own time stamps as in the CSVs. Only the build's own cases are written.

    Returns:
        Rows written per family.
    """
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    from vitaldb_bench.label_families import FAMILIES, read_family, seconds_to_ns, versions_from  # noqa: PLC0415
    from vitaldb_bench.stores import TimeNetStore  # noqa: PLC0415

    store, results = TimeNetStore(build_dir), Path(results)
    versions = versions_from(results)
    origin_ns: dict[int, int] = {}
    for c in store.caseids:
        origin_ns[c] = round(Fraction(float.fromhex(store.record(c).metadata["vital_origin_s"])) * 10**9)
    out_dir = Path(build_dir) / "label_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in families or list(FAMILIES):
        fam = FAMILIES[name]
        rows = []
        for c, rs in sorted(read_family(results, fam).items()):
            if c not in origin_ns:
                continue
            for r in rs:
                rows.append({"case": c, "record_id": f"vitaldb-{c:04d}", "label_name": fam.label(r),
                             "label_source": fam.source, "source_version": versions.get(fam.source, ""),
                             "measure_tag": r.get(fam.measure_col) if fam.measure_col else (fam.measure or ""),
                             "start_ns": seconds_to_ns(r[fam.start]) - origin_ns[c],
                             "end_ns": seconds_to_ns(r[fam.end]) - origin_ns[c],
                             "available_at": r.get("available_at") or fam.available_at,
                             **{k: (None if v in ("", None) else str(v)) for k, v in r.items() if k != "case"}})
        if rows:
            cols = list(dict.fromkeys(k for r in rows for k in r))
            pq.write_table(pa.Table.from_pylist([{k: r.get(k) for k in cols} for r in rows]),
                           out_dir / f"{name}.parquet", compression="zstd")
        written[name] = len(rows)
    return written
