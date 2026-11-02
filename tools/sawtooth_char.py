# SPDX-License-Identifier: MIT
"""Exhaustive characterisation of the ECG-PPG sawtooth: slip timing, amplitude, period, mechanism and pauses.

Stage 1 (this machine only; per-case files are not shared):

    python tools/sawtooth_char.py scan --root $R --cases CASES.json --out $W/sawtooth_char/scan \
        [--part 0 --of 8] [--minutes 90] [--time-budget 150]

For each case, on the longest continuous stretch of SNUADC ECG II and PLETH (500 Hz, at least 20 min, at most
``--minutes``; the rule of tools/beats_export.py, so cases_pat300.json gives the 266-recording benchmark):

* **R peaks** (5-25 Hz band, sub-sample) and, for each matched beat, four PPG fiducials on the 0.5-10 Hz
  band-passed PLETH (zero phase): the foot (intersecting tangent), the steepest upstroke, the systolic peak and
  the half-height point on the downslope (``down50``); all sub-sample. Four points per pulse divide each beat,
  so a slip can be placed between two fiducials rather than between two beats.
* **The arterial upstroke** for every R peak (steepest rise of the 0.5-10 Hz ART 0.05-0.45 s after it; NaN where
  the line is not physiological: 10 s windows with median 30-200 mmHg and pulse pressure 15-150 mmHg). ART
  shares the converter but has no frame and no sawtooth (atlas study ppg-art-reference).
* **The frame clock** (physiosync.frame_clock on the raw PLETH maximum of each pulse, which locks to the 33.3 ms
  frame): its rate against the SNUADC sample clock, strength, and phase (the position of one frame tick).
* **The current physiosync fit** (frame 1/30 s, ART as reference where present): slips, teeth and tiers, as
  the baseline the new methods are compared with.
* **Injected ground truth.** Where that fit finds no slip in the whole stretch (no sawtooth running), a known
  sawtooth is written into the PLETH waveform itself: the content is delayed by D(t) = frame x (slips so far)
  - rate x (t - t0), holding 33.3 ms of content at each slip (x'(t) = x(t - D(t)) by interpolation), at a rate
  drawn from 15-55 ppm and a random phase (seeded by case id). The four fiducials are then found again on the
  warped waveform, so the test covers the filter's smearing of the step, not only the bookkeeping.

Writes OUT/<case>.npz (or OUT/<case>.skip with the reason).

Stage 2 (per-case numbers, no signal values):

    python tools/sawtooth_char.py slips --scan $W/sawtooth_char/scan --out $W/sawtooth_char/slips.jsonl [--part 0 --of 8]

For the real beats and, where present, the injected ones:

* **Slip detection** on ECG->PPG, on the time-ordered series of every foot, upstroke and peak (each type moved
  onto the upstroke's level by its running median offset; isolated bad points replaced by the running median), so
  a slip inside a pulse is a clean step between two points: at every point boundary, with 15 beats of points each
  side and one
  straight line through both, the log likelihood ratio of a sharp step of exactly one frame against none, the
  freely estimated step, and the log ratio of a sharp step against the same change spread over 7 beats (physiology
  changes gradually). Strong slips: ratio > 25, step 0.6-1.4 frames, sharp, persistent (the level 15-75 or 15-40
  beats after differs from the level as far before by 0.5-1.5 frames: transient bumps fail) and in a stretch whose
  local scatter is at most 2.5 times the recording's. Where there is a clean arterial line, a slip must also step
  ART->PPG (0.5-1.5 frames) and not ECG->ART (under 0.3 frame); those that fail are reported apart. Then the grid of the strong slips
  (constant tooth length, each run of the piecewise grid over its own stretch) predicts where the others should
  be, and a step there is accepted on looser tests (ratio > 8, step 0.5-1.6 frames, sharper than a ramp, holding
  0.4-1.6 frames): the constancy is used to find weak slips, and slips off the grid are kept. The current fit's
  slips (physiosync, moved from ECG to PPG time) are candidates too, tested the same way within 10 s. The same search on the sign-flipped series is the control: no mechanism makes the delay
  step *down* a frame.
* **Placement within the beat.** The fiducials of the three beats either side, in time order, are each pre-slip
  or post-slip (Gaussian about their type's level before and after, with the type's beat-to-beat noise); the
  posterior over the split point gives the most likely gap between two fiducials and a 95% bracket. Also: the
  step of each fiducial type, the ECG->ART step (physiology moves it, a slip does not), fiducials only partly
  shifted (the filter's smearing), and the number of frame ticks inside the bracket where the frame clock is found.
* **Period, piecewise.** The tooth length is the one that best explains the gaps between consecutive slips (a
  fraction of it, which fits every gap too, is penalised for the slips it implies but nobody found). Slips form
  runs on one grid (each within 3% of a tooth of a whole number of teeth from the previous); a break is a phase
  jump, reported with its shift, the longest gap between beats there (was the signal interrupted?), and the
  largest freely sized step of the delay between the two runs (its size, sharpness and likelihood ratio): if the
  phase moves by a share of a tooth, the delay should step by the same share of a frame. One tooth
  length for all runs, each with its own phase; slips in a run of their own are "off grid". The interval fits
  below use the longest run. Interval fits by linear programming: does one
  constant period pass through every 95% bracket (and through every best gap), with its feasible range; the
  same with a slowly changing period (quadratic). Least-squares residuals, the lag-1 correlation of tooth lengths
  (accumulator vs renewal), and the creep within each tooth (the slope of the delay, an independent rate).
* **Gaps and pauses** longer than 1.5 teeth: the slips the constant grid predicts there, the largest step near
  each prediction, the slope, and the kind (start, middle, end).
* **Setups** (``agg --ports --assigned --rates``): per recording the PC key (ports of SNUADC, Solar8000, Primus
  and BIS; tools/setup_assign.py where one is missing), the SNUADC correction mode (step when its rate is below
  -20 ppm), the frame clock, the direction of its sawtooth (at least 3 one-frame steps one way and twice as many
  as the other way) and its signed rate; written to recordings.csv and summarised by setup and mode.
Stage 4, mechanism (``mech`` then ``mech-agg``; from a finished ``slips`` run):

* **Sub-frame steps:** sharp, persistent steps of a fraction of a frame (0.12-0.88), 60 s clear of any one-frame
  step, with the arterial check (in the PPG chain, or before it); whether their sizes lock to quarter frames
  (within 0.03 frame; binomial against uniform sizes), by place (inside a phase jump, inside a run, elsewhere);
  and for each phase jump whether a sub-frame step of its shift lies between the runs.
* **Staircase:** at every slip, one step against two (piecewise constant levels, +/- 12 beats of points): how
  often two steps fit better, and how the frame is split; the same at injected single steps as the control.
* **NIBP:** sub-frame steps, slips and the largest step inside each phase jump within 45 s before to 5 s after
  a new cuff reading (where Solar8000/NIBP_SBP or _DBP changes; the tracks repeat the last reading every 2 s), against the share of the recording's time that is; and how many jumps
  contain a reading, against the recording's reading rate.
* **Runs:** in runs of 4 or more slips, interval fits (constant, slowly changing), residuals against the bracket
  widths, the range of rates a constant tooth allows, and the correlation of consecutive tooth-length deviations
  (about -0.5 for slips on a fixed grid placed with independent error; about 0 if each tooth varies on its own).
* **Pauses** (``mech --recordings``): every stretch longer than 1.5 teeth without a slip in a recording with an
  upward sawtooth (start, middle, end), and every recording without any one-frame step: the slope of the delay as
  a share of the creep (1 creeping, 0 flat), physiosync.sawtooth_evidence with its mirrored control, the frame
  clock and whether the recording's frame comb holds there; for a middle stretch, whether the slips either side
  lie on one grid run (the phase carried on) or not (the shift), and the delay's fall across it in frames against
  that over ordinary teeth. Inside a middle stretch, the slips the grid predicts (evenly spaced between the two
  slips) are looked for: the best one-frame step within 10 s, its size, persistence and the tests it fails; the
  same half way through ordinary teeth is the control.
* **Clocks:** per recording the frame clock against the SNUADC, the signed sawtooth rate, their sum and difference
  (candidate rates of the PPG content's clock) and the SNUADC against the PC (clock network); their spread across
  setups; and for setups seen in both SNUADC modes, the difference between the modes of each quantity (the frame
  clock against the PC, frame +/- SNUADC, should not differ if it is the oximeter's own clock).

* **More injections** (``slips --inj-seeds N``): on each clean host, N more fiducial-shift injections with their own
  rate and phase; seeds 0-2 were used to set the thresholds, the rest are held out. Recall is also given for the
  slips at least 30 beats from either end of the stretch (nearer, persistence cannot be measured).
Manual labels (``label-sample`` then ``label-eval``): a stratified, blinded sample of real events from a slips run
(both detectors' slips, each one's own, grid-predicted slips not found, downward steps, mid-tooth and slip-free
controls), shown as delay traces only. The labeller says what each trace does at the event (ECG->PPG and ART->PPG: up, down,
no step, can't tell or artefact; ECG->ART: flat, steps, can't tell or artefact; for each step, whether it is about one
frame), and may mark the step in each trace separately, flag and annotate each trace, and mark any number of brief dips or spikes, artefacts and other
features in any trace. The verdict comes from the answers through ECG->PPG = ECG->ART + ART->PPG: a PPG
slip moves both PPG delays with ECG->ART flat; a step in ECG->PPG with ART->PPG flat is ECG->ART moving (physiology or
ECG/arterial timing), not the PPG. Scores weight each stratum by its size in the run.

* **Injected ground truth**, two kinds: the waveform warped (above), and the same slips as a shift of the fiducial
  times with the waveform untouched; both scored on clean hosts only: no one-frame step either way in the un-injected beats;
  some recordings without upward slips carry a reversed sawtooth): slips found and false, error and coverage of the brackets, the period error, and
  the current physiosync placement for comparison (its slip times are in ECG time; shifted by the typical delay).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from pat import ECG, PPG, r_peaks, sample_times  # noqa: E402
from pat_res import ecg_subsample, parabolic  # noqa: E402
from vitaldb_reader.vital_file import read_vital  # noqa: E402

from physiosync import fit_ecg_ppg_delay, frame_clock, match_beats  # noqa: E402

warnings.filterwarnings("ignore")

ART = "SNUADC/ART"
FS = 500.0
FRAME = 1 / 30
FRAME_SAMPLES = FS / 30
ADOPTED = 0.27
INJ_RATE_PPM = (15.0, 55.0)
FIDS = ("foot", "up", "peak", "down50")


# ------------------------------------------------------------------------------------------------ fiducials

def band(x: np.ndarray, lo: float = 0.5, hi: float = 10.0) -> np.ndarray:
    sos = signal.butter(2, (lo, hi), btype="bandpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def upstrokes(d: np.ndarray) -> np.ndarray:
    """Integer indices of the steepest rise of each pulse (as pat_res.ppg_onsets)."""
    up, _ = signal.find_peaks(d, distance=int(0.3 * FS), prominence=0.3 * (np.quantile(d, 0.99) - np.quantile(d, 0.5)))
    return up


def ppg_fiducials(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Sub-sample positions (samples) of the four fiducials of every pulse, and the raw maximum (integer)."""
    y = band(raw)
    d = np.gradient(y)
    up = upstrokes(d)
    n = up.size
    out = {k: np.full(n, np.nan) for k in FIDS}
    out["rawmax"] = np.full(n, -1, dtype=np.int64)
    w_back, w_pk = int(0.4 * FS), int(0.4 * FS)
    nxt = np.r_[up[1:], y.size]
    for k, u in enumerate(up):
        if u - w_back < 0 or u + 2 >= y.size or d[u] <= 0:
            continue
        uf = float(parabolic(d, np.array([u]))[0])
        out["up"][k] = uf
        yu = y[u] + (uf - u) * d[u]
        ymin = float(np.min(y[u - w_back:u]))
        f = uf + (ymin - yu) / d[u]
        if u - w_back <= f <= u:
            out["foot"][k] = f
        hi = min(y.size - 1, u + w_pk, int(nxt[k]))
        if hi - u < 3:
            continue
        p = u + int(np.argmax(y[u:hi]))
        out["peak"][k] = float(parabolic(y, np.array([p]))[0])
        # half height on the downslope: between the peak and the lowest point before the next upstroke
        e_ = min(y.size - 1, int(nxt[k]))
        if e_ - p < 3:
            continue
        lowv = float(np.min(y[p:e_]))
        lev = 0.5 * (y[p] + lowv)
        below = np.flatnonzero(y[p:e_] <= lev)
        if below.size:
            j = p + int(below[0])
            if j > p and y[j - 1] != y[j]:
                out["down50"][k] = (j - 1) + (y[j - 1] - lev) / (y[j - 1] - y[j])
        r0 = max(0, u - int(0.05 * FS))
        out["rawmax"][k] = r0 + int(np.argmax(raw[r0:hi]))
    out["up_i"] = up
    return out


def art_upstrokes(case, r_t: np.ndarray) -> np.ndarray:
    """Arterial steepest upstroke 0.05-0.45 s after each R peak (s), NaN where absent or not physiological."""
    out = np.full(r_t.size, np.nan)
    tr = case.tracks.get(ART)
    if tr is None or not tr.record_times.size or float(tr.srate) != FS:
        return out
    ta = sample_times(tr)
    xa = np.asarray(tr.physical(), dtype=np.float64)
    gap = np.flatnonzero(np.abs(np.diff(ta) - 1 / FS) > 0.01 / FS)
    be = np.r_[0, gap + 1, ta.size]
    for a_, b_ in zip(be[:-1], be[1:]):
        if b_ - a_ < 20 * FS or ta[b_ - 1] < r_t[0] or ta[a_] > r_t[-1]:
            continue
        seg_t, seg = ta[a_:b_], xa[a_:b_]
        ok = np.isfinite(seg)
        if ok.mean() < 0.99:
            continue
        seg = np.where(ok, seg, np.nanmedian(seg))
        # 10 s windows: physiological level and pulse pressure
        nw = seg.size // int(10 * FS)
        if nw < 1:
            continue
        X = seg[: nw * int(10 * FS)].reshape(nw, -1)
        med = np.median(X, 1)
        pp = np.quantile(X, 0.95, axis=1) - np.quantile(X, 0.05, axis=1)
        good_w = (med > 30) & (med < 200) & (pp > 15) & (pp < 150)
        d = np.gradient(band(seg))
        sel = np.flatnonzero((r_t > seg_t[0]) & (r_t + 0.45 < seg_t[-1]))
        for i in sel:
            ia, ib = int(np.searchsorted(seg_t, r_t[i] + 0.05)), int(np.searchsorted(seg_t, r_t[i] + 0.45))
            w = min(ia // int(10 * FS), nw - 1)
            if ib - ia < 5 or not good_w[w]:
                continue
            m = ia + int(np.argmax(d[ia:ib]))
            if d[m] <= 0:
                continue
            out[i] = seg_t[0] + float(parabolic(d, np.array([m]))[0]) / FS
    return out


def frame_phase(idx: np.ndarray, ppm: float) -> float:
    """Phase of the frame comb (samples, in [0, P)) at absolute sample index 0, for the period found."""
    P = FRAME_SAMPLES * (1 + ppm * 1e-6)
    z = np.exp(2j * np.pi * idx / P).mean()
    return float((np.angle(z) / (2 * np.pi)) % 1.0 * P)


# ------------------------------------------------------------------------------------------------ injection

def inject(raw: np.ndarray, t0: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Delay the PLETH content by a known sawtooth: x'(t) = x(t - D(t)), D jumping +1 frame at each slip.

    D(t) = FRAME * k(t) - r * (t - t_first), where slips (k += 1) fall where the creep has used up one frame;
    the first slip is at a random phase. Returns the warped samples and the truth (slip times on the stretch's
    clock, rate)."""
    r = float(rng.uniform(*INJ_RATE_PPM)) * 1e-6
    L = FRAME / r
    n = raw.size
    tt = np.arange(n) / FS
    first = float(rng.uniform(0, L))
    slips = first + L * np.arange(int(np.ceil((tt[-1] - first) / L)) + 1)
    slips = slips[slips < tt[-1]]
    k = np.searchsorted(slips, tt, side="right")
    D = FRAME * k - r * tt          # creep down, step up one frame at each slip
    D -= D.min()
    src = tt - D                    # content time shown at each output time
    x = np.interp(src, tt, np.asarray(raw, dtype=np.float64))
    return x, {"slips_t": (t0 + slips), "rate_ppm": r * 1e6, "period_s": L, "first_s": first}


# ------------------------------------------------------------------------------------------------ scan

def beat_table(r_t: np.ndarray, fid: dict, tp0: float, pa: int) -> dict:
    """Match R peaks to pulses (by the upstroke) and return per matched beat: R time, the four fiducial times,
    the raw-maximum sample index (absolute in the PPG stream)."""
    up_t = tp0 + fid["up"] / FS
    ok = np.isfinite(up_t)
    idx_ok = np.flatnonzero(ok)
    i, j, delay = match_beats(r_t, up_t[ok])
    j = idx_ok[j]
    out = {"r_i": i, "delay_mode_s": delay}
    for k in FIDS:
        out[k] = tp0 + fid[k][j] / FS
    out["rawmax"] = np.where(fid["rawmax"][j] >= 0, fid["rawmax"][j] + pa, -1)
    return out


def fit_summary(r: np.ndarray, p: np.ndarray, art: np.ndarray | None) -> dict:
    try:
        m = fit_ecg_ppg_delay(r, p, adopted_pat_s=ADOPTED, frame_s=FRAME, matched=True,
                              ref_t=art if art is not None and np.isfinite(art).sum() > 60 else None)
    except Exception as ex:  # noqa: BLE001
        return {"error": f"{type(ex).__name__}: {ex}"[:200]}
    return {"slips_t": list(map(float, m.slips_t)), "inserted_t": list(map(float, m.inserted_t)),
            "edges_t": list(map(float, m.tooth_edges)), "tiers": list(m.tooth_tier),
            "sources": list(m.tooth_source),
            "rates_ppm": [float(x) * 1e6 if x is not None else float("nan") for x in m.tooth_rate]}


def scan(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for i, c in enumerate(json.loads(Path(a.cases).read_text())["cases"]) if i % a.of == a.part]
    t_start, slowest, n = time.time(), 0.0, 0
    for c in cases:
        f = out / f"{c:04d}.npz"
        if f.exists() or (out / f"{c:04d}.skip").exists():
            continue
        if time.time() - t_start + max(1.5 * slowest, 5) > a.time_budget:
            break
        t1 = time.time()
        reason = None
        try:
            case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG, ART})
            e, p = case.tracks.get(ECG), case.tracks.get(PPG)
            if e is None or p is None or not e.record_times.size or not p.record_times.size or e.srate != 500 \
                    or p.srate != 500:
                reason = "no 500 Hz ECG_II and PLETH"
            else:
                te, tp = sample_times(e), sample_times(p)
                gap = np.flatnonzero(np.abs(np.diff(te) - 1 / FS) > 0.01 / FS)
                be = np.r_[0, gap + 1, te.size]
                k = int(np.argmax(np.diff(be)))
                ea, eb = be[k], min(be[k + 1], be[k] + int(a.minutes * 60 * FS))
                pa, pb = np.searchsorted(tp, te[ea]), np.searchsorted(tp, te[eb - 1])
                if eb - ea < 1200 * FS or pb - pa < 1200 * FS or np.max(np.abs(np.diff(tp[pa:pb]) - 1 / FS)) > 0.01 / FS:
                    reason = "no 20 min continuous stretch"
            if reason is None:
                xe = e.physical()[ea:eb].astype(np.float64)
                rp = r_peaks(xe, FS)
                r_t = te[ea] + ecg_subsample(xe, rp, FS) / FS
                raw = np.asarray(p.values[pa:pb], dtype=np.float64)
                fid = ppg_fiducials(raw)
                bt = beat_table(r_t, fid, tp[pa], pa)
                if bt["r_i"].size < 600:
                    reason = "fewer than 600 matched beats"
            if reason is None:
                art_all = art_upstrokes(case, r_t)
                rm = bt["rawmax"][bt["rawmax"] >= 0]
                fc = frame_clock(rm.astype(np.float64), FRAME_SAMPLES) if rm.size >= 200 else {}
                if fc:
                    fc["phase_samples"] = frame_phase(rm.astype(np.float64), fc["ppm"])
                r_m, art_m = r_t[bt["r_i"]], art_all[bt["r_i"]]
                fit = fit_summary(r_m, bt["up"], art_m)
                port = ""
                for name, dev in case.devices.items():
                    if "SNUADC" in str(name):
                        port = str(getattr(dev, "port", "") or "")
                res = {"r_t": r_m, "art_t": art_m, "rawmax": bt["rawmax"], "ppg_t0": tp[pa], "ppg_first_index": pa,
                       "port": np.array(port), "frame_clock": np.array(json.dumps(fc)),
                       "fit": np.array(json.dumps(fit)), "delay_mode_s": bt["delay_mode_s"]}
                for k_ in FIDS:
                    res[k_] = bt[k_]
                # injected ground truth where no sawtooth is running
                if "error" not in fit and not fit["slips_t"]:
                    rng = np.random.default_rng(10_000 + c)
                    xw, truth = inject(raw, tp[pa], rng)
                    fid_w = ppg_fiducials(xw)
                    bt_w = beat_table(r_t, fid_w, tp[pa], pa)
                    res["inj_r_t"] = r_t[bt_w["r_i"]]
                    res["inj_art_t"] = art_all[bt_w["r_i"]]
                    for k_ in FIDS:
                        res["inj_" + k_] = bt_w[k_]
                    res["inj_truth"] = np.array(json.dumps({k_: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                            for k_, v in truth.items()}))
                    res["inj_fit"] = np.array(json.dumps(fit_summary(res["inj_r_t"], bt_w["up"], res["inj_art_t"])))
                np.savez_compressed(f, **res)
        except Exception as ex:  # noqa: BLE001
            reason = f"error {type(ex).__name__}: {ex}"[:300]
        if reason:
            (out / f"{c:04d}.skip").write_text(reason)
        slowest = max(slowest, time.time() - t1)
        n += 1
    print(json.dumps({"part": a.part, "new": n, "seconds": round(time.time() - t_start, 1),
                      "done": len(list(out.glob("*.npz"))), "skipped": len(list(out.glob("*.skip")))}))


# ------------------------------------------------------------------------------------------------ stage 2: slips

STEP_W = 15          # beats each side for a step
LOC_W = 3            # beats each side of the boundary whose fiducials are ordered to place the slip
GUARD = 3            # beats left out next to the boundary when taking the before/after levels
SIGN_CTRL = (+1, -1)  # +1: slips (delay steps up a frame); -1: the sign-flipped control


def roll_median(x: np.ndarray, w: int) -> np.ndarray:
    """Median of x[i : i + w] for every i (NaN-aware), length x.size - w + 1."""
    from numpy.lib.stride_tricks import sliding_window_view
    return np.nanmedian(sliding_window_view(x, w), axis=1)


def boundary_steps(y: np.ndarray, w: int) -> np.ndarray:
    """Step at each boundary b (between beats b-1 and b): median(y[b:b+w]) - median(y[b-w:b]); NaN at the ends."""
    n = y.size
    out = np.full(n, np.nan)
    if n < 2 * w + 1:
        return out
    m = roll_median(y, w)            # m[i] = median(y[i:i+w])
    b = np.arange(w, n - w + 1)
    out[b] = m[b] - m[b - w]
    return out


def noise_sd(y: np.ndarray) -> float:
    d = np.diff(y[np.isfinite(y)])
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if d.size > 10 else float("nan")


DET_FIDS = ("foot", "up", "peak")


def point_series(F: dict, base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every foot, upstroke and peak as one time-ordered series of delays (fiducial - base), each type moved onto
    the upstroke's level by the 61-beat running median of its difference from the upstroke (pulse shape changes
    slowly). Returns the times, the delays and each point's beat. A slip inside a pulse is then a clean step in
    this series, between two of its points."""
    from scipy.ndimage import median_filter
    n = base.size
    ts, ys, bs = [], [], []
    for j in DET_FIDS:
        d = F[j] - F["up"]
        dd = np.where(np.isfinite(d), d, np.nanmedian(d))
        off = median_filter(dd, size=61, mode="nearest")
        ts.append(F[j])
        ys.append(F[j] - base - off)
        bs.append(np.arange(n))
    t = np.concatenate(ts)
    y = np.concatenate(ys)
    b = np.concatenate(bs)
    ok = np.isfinite(t) & np.isfinite(y)
    o = np.argsort(t[ok], kind="stable")
    return t[ok][o], y[ok][o], b[ok][o]


def combined(F: dict, base: np.ndarray) -> np.ndarray:
    """Per beat, the median over foot, upstroke and peak of (fiducial - base): the detection series (down50 is
    left out: the dicrotic notch makes it much noisier)."""
    return np.nanmedian(np.vstack([F[j] - base for j in ("foot", "up", "peak")]), axis=0)


PPB = 3              # detection points per beat (foot, upstroke, peak)
GLR_W = 15 * PPB     # points (15 beats) each side in the step test
RAMP_HALF = 3 * PPB  # points (3 beats) either side over which the "ramp" alternative spreads the change
LLR_STRONG = 25.0    # likelihood ratio (log) for a slip found on its own
LLR_GRID = 8.0       # ... for a slip where the grid predicts one
SHARP_MIN = 4.0      # log ratio, sharp step against a ramp


def clean(y: np.ndarray, sd: float) -> np.ndarray:
    """Replace beats further than 5 noise SD from the 7-beat running median by that median (isolated
    mis-detections), so one bad beat cannot fake or hide a step."""
    from scipy.ndimage import median_filter
    y = np.where(np.isfinite(y), y, np.nanmedian(y))
    m = median_filter(y, size=7, mode="nearest")
    return np.where(np.abs(y - m) > 5 * sd, m, y)


def step_scores(t: np.ndarray, y: np.ndarray, sd: float, sign: int, frame: float = FRAME) -> dict:
    """At every boundary b (between beats b-1 and b), with GLR_W beats each side and one straight line through
    both sides:

    * llr: log likelihood ratio of a sharp step of exactly sign x frame at b against no step;
    * fhat: the step size estimated freely (s, in the sign's direction);
    * sharp: log likelihood ratio of the sharp step against the same change spread linearly over
      b-RAMP_HALF .. b+RAMP_HALF (physiology changes over many beats; a slip between two)."""
    n = y.size
    W = GLR_W
    out = {k: np.full(n, np.nan) for k in ("llr", "fhat", "sharp", "llr_free")}
    if n < 2 * W + 2:
        return out
    ys = sign * y
    b = np.arange(W, n - W + 1)
    idx = b[:, None] + np.arange(-W, W)[None, :]
    Y = ys[idx]
    T = t[idx] - t[b][:, None]
    T = T / max(np.median(np.abs(T)), 1e-9)
    H = (np.arange(-W, W) >= 0).astype(float)[None, :].repeat(b.size, 0)
    ramp = np.clip((np.arange(-W, W) + RAMP_HALF + 0.5) / (2 * RAMP_HALF + 1), 0, 1)[None, :].repeat(b.size, 0)
    one = np.ones_like(Y)

    def rss(X: np.ndarray, Yr: np.ndarray) -> np.ndarray:
        XtX = np.einsum("nki,nkj->nij", X, X)
        XtY = np.einsum("nki,nk->ni", X, Yr)
        beta = np.linalg.solve(XtX + 1e-12 * np.eye(X.shape[2])[None], XtY[..., None])[..., 0]
        r = Yr - np.einsum("nki,ni->nk", X, beta)
        return np.einsum("nk,nk->n", r, r), beta

    X0 = np.stack([one, T], axis=2)
    rss0, _ = rss(X0, Y)
    rss1, _ = rss(X0, Y - frame * H)
    rssr, _ = rss(X0, Y - frame * ramp)
    rss2, beta2 = rss(np.stack([one, T, H], axis=2), Y)
    s2 = 2 * sd ** 2
    out["llr"][b] = (rss0 - rss1) / s2
    out["sharp"][b] = (rssr - rss1) / s2
    out["fhat"][b] = beta2[:, 2]
    out["llr_free"][b] = (rss0 - rss2) / s2
    return out


PERSIST = (15 * PPB, 75 * PPB)   # points (15-75 beats) after and before the boundary over which the new level holds
NOISE_GATE = 2.5     # no detection where the local scatter exceeds this multiple of the recording's


PERSIST_SHORT = (15 * PPB, 40 * PPB)   # the same over 15-40 beats: a disturbance a minute away cannot veto a slip


def long_step(y: np.ndarray, b: int, sign: int, win: tuple[int, int] | None = None) -> float:
    """Persistence at boundary b: median of points b+15 .. b+75 beats minus median of points b-75 .. b-15 beats,
    in the sign's direction (a transient bump that returns within 15 beats does not change it). Near the ends of
    the stretch the windows are cut short, down to 15 beats of points."""
    a_, b_ = win or PERSIST
    lo, hi = max(b - b_, 0), min(b + b_, y.size)
    if b - a_ - lo < 15 * PPB or hi - (b + a_) < 15 * PPB:
        return float("nan")
    return float(sign * (np.nanmedian(y[b + a_:hi]) - np.nanmedian(y[lo:b - a_])))


def local_noise(y: np.ndarray, half: int = 30 * PPB) -> np.ndarray:
    """Robust beat-to-beat scatter in +/- ``half`` beats of every beat."""
    from numpy.lib.stride_tricks import sliding_window_view
    d = np.abs(np.diff(y, prepend=y[0]))
    w = 2 * half + 1
    if d.size < w:
        return np.full(d.size, np.nanmedian(d) * 1.4826 / np.sqrt(2))
    med = np.nanmedian(sliding_window_view(d, w), axis=1) * 1.4826 / np.sqrt(2)
    return np.r_[np.full(half, med[0]), med, np.full(d.size - med.size - half, med[-1])]


def detect(t: np.ndarray, y: np.ndarray, sd: float, sign: int, frame: float = FRAME) -> tuple[list[int], dict]:
    """Strong slips: llr above LLR_STRONG, step 0.6-1.4 frames, sharp, persistent (0.5-1.5 frames over 15-75
    beats), in a stretch whose local scatter is at most NOISE_GATE times the recording's; the best within
    +/- 30 beats and 60 s."""
    sc = step_scores(t, y, sd, sign, frame)
    sc["quiet"] = local_noise(y) <= NOISE_GATE * sd
    llr, f, sh = sc["llr"], sc["fhat"], sc["sharp"]
    pe = np.full(y.size, np.nan)
    pe2 = np.full(y.size, np.nan)
    pre = np.flatnonzero((llr > min(LLR_STRONG, LLR_GRID)) & (f > 0.4 * frame) & (f < 1.7 * frame))
    for b in pre:
        pe[b] = long_step(y, int(b), sign)
        pe2[b] = long_step(y, int(b), sign, PERSIST_SHORT)
    sc["persist"] = pe
    sc["persist_short"] = pe2
    holds = ((pe > 0.5 * frame) & (pe < 1.5 * frame)) | ((pe2 > 0.5 * frame) & (pe2 < 1.5 * frame))
    ok = (llr > LLR_STRONG) & (f > 0.6 * frame) & (f < 1.4 * frame) & (sh > SHARP_MIN) & holds & sc["quiet"]
    out = []
    for b in np.flatnonzero(ok):
        lo, hi = max(b - 30 * PPB, 0), min(b + 30 * PPB + 1, llr.size)
        if llr[b] < np.nanmax(llr[lo:hi]):
            continue
        if out and t[b] - t[out[-1]] < 60:
            if llr[b] > llr[out[-1]]:
                out[-1] = int(b)
            continue
        out.append(int(b))
    return out, sc


def grid_fill(t: np.ndarray, sc: dict, found: list[int], L: float, t0: float, frame: float = FRAME,
              span: tuple[float, float] | None = None, win: float | None = None) -> list[int]:
    """Slips where the grid t0 + k L predicts one but none was found strongly: the best boundary within
    max(2% of L, 10 s) of the prediction, kept if llr > LLR_GRID, not clearly a ramp (sharp > -10), the stretch
    quiet, and either the step 0.5-1.6 frames with the new level holding over 15-40 or 15-75 beats (0.4-1.6
    frames), or, too near an end of the stretch to test that, the step 0.5-1.8 frames. The grid (or the current
    fit) is the prior, so the tests are looser than for a slip found on its own. Predictions only inside ``span``
    (the stretch this run's grid covers, up to the neighbouring runs)."""
    llr, f, sh = sc["llr"], sc["fhat"], sc["sharp"]
    have = t[found] if found else np.empty(0)
    lo_t, hi_t = span if span else (t[0], t[-1])
    add = []
    k0 = int(np.ceil((lo_t - t0) / L))
    k1 = int(np.floor((hi_t - t0) / L))
    for k in range(k0, k1 + 1):
        p = t0 + k * L
        if have.size and np.min(np.abs(have - p)) < 0.1 * L:
            continue
        w = win if win else max(0.02 * L, 10.0)
        sel = np.flatnonzero((np.abs(t - p) < w) & np.isfinite(llr))
        if not sel.size:
            continue
        b = int(sel[np.argmax(llr[sel])])
        if not (llr[b] > LLR_GRID and sh[b] > -10 and sc["quiet"][b]):
            continue
        pl = long_step(sc["_y"], b, sc["_sign"])
        ps = long_step(sc["_y"], b, sc["_sign"], PERSIST_SHORT)
        held = (0.4 * frame < pl < 1.6 * frame) or (0.4 * frame < ps < 1.6 * frame)
        edge = not np.isfinite(pl) and not np.isfinite(ps)       # too near an end to test persistence
        if (held and 0.5 * frame < f[b] < 1.6 * frame) or (edge and 0.5 * frame < f[b] < 1.8 * frame):
            add.append(b)
    return add


OUTLIER = 0.05       # share of fiducials allowed to be outliers in the placement likelihood
OUT_SPAN = 0.2       # s; the outliers' uniform density spans this


def robust_ll(res: np.ndarray, mu: np.ndarray, s: np.ndarray) -> np.ndarray:
    g = np.exp(-0.5 * ((res - mu) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
    return np.log((1 - OUTLIER) * g + OUTLIER / OUT_SPAN)


def place(b: int, F: dict, r: np.ndarray, sd: dict, sign: int = 1) -> dict:
    """Place the slip at boundary b (between beats b-1 and b) between two fiducials.

    Each fiducial type j has its own level before (beats b-GUARD-STEP_W .. b-GUARD) and after; every fiducial of
    beats b-LOC_W .. b+LOC_W-1, in time order, is pre (residual ~ N(0, sd_j)) or post (~ N(step_j, sd_j)). The
    posterior over the split point gives the most likely position and a 95% bracket. The likelihood allows 5% of
    fiducials to be outliers (uniform over 0.2 s), so one disturbed fiducial cannot move the slip."""
    n = r.size
    a0, a1 = b - GUARD - STEP_W, b - GUARD
    c0, c1 = b + GUARD, b + GUARD + STEP_W
    if a0 < 0 or c1 > n:
        return {}
    pts, steps = [], {}
    for j in FIDS:
        y = sign * (F[j] - r)
        bef, aft = np.nanmedian(y[a0:a1]), np.nanmedian(y[c0:c1])
        steps[j] = float(aft - bef)
        for i in range(b - LOC_W, b + LOC_W):
            if np.isfinite(F[j][i]):
                pts.append((float(F[j][i]), j, float(y[i] - bef), i))
    if len(pts) < 8:
        return {}
    pts.sort()
    tt = np.array([p[0] for p in pts])
    res = np.array([p[2] for p in pts])
    st = np.array([steps[p[1]] for p in pts])
    s = np.array([max(sd[p[1]], 1e-4) for p in pts])
    ll_pre = robust_ll(res, 0.0, s)
    ll_post = robust_ll(res, st, s)
    # split m: points [0, m) pre, [m, N) post; m from 1 to N-1
    cp = np.cumsum(ll_pre)
    cq = np.cumsum(ll_post[::-1])[::-1]
    N = len(pts)
    ll = np.array([cp[m - 1] + cq[m] for m in range(1, N)])
    post = np.exp(ll - ll.max())
    post /= post.sum()
    m = 1 + int(np.argmax(post))
    order = np.argsort(post)[::-1]
    cum, sel = 0.0, []
    for k in order:
        sel.append(k + 1)
        cum += post[k]
        if cum >= 0.95:
            break
    lo_m, hi_m = min(sel), max(sel)
    frac = res / np.where(np.abs(st) > 1e-4, st, np.nan)
    return {"t": float(0.5 * (tt[m - 1] + tt[m])), "lo": float(tt[lo_m - 1]), "hi": float(tt[hi_m]),
            "best_lo": float(tt[m - 1]), "best_hi": float(tt[m]), "p_best": float(post.max()),
            "between": [pts[m - 1][1], pts[m][1]], "steps_ms": {k: round(v * 1000, 2) for k, v in steps.items()},
            "partial": int(np.sum((frac > 0.25) & (frac < 0.75))), "n_pts": N}


def ref_step(b: int, y: np.ndarray) -> float:
    """Step of a reference series (e.g. ECG->ART) across boundary b, levels as in :func:`place`."""
    a0, a1, c0, c1 = b - GUARD - STEP_W, b - GUARD, b + GUARD, b + GUARD + STEP_W
    if a0 < 0 or c1 > y.size or np.isfinite(y[a0:a1]).sum() < 5 or np.isfinite(y[c0:c1]).sum() < 5:
        return float("nan")
    return float(np.nanmedian(y[c0:c1]) - np.nanmedian(y[a0:a1]))


def frame_ticks_in(lo: float, hi: float, fc: dict, ppg_t0: float, pa: int) -> int | None:
    if not fc or fc.get("sharpness", 0) < 10:
        return None
    P = FRAME_SAMPLES * (1 + fc["ppm"] * 1e-6)
    n_lo = (lo - ppg_t0) * FS + pa
    n_hi = (hi - ppg_t0) * FS + pa
    k_lo = np.ceil((n_lo - fc["phase_samples"]) / P)
    k_hi = np.floor((n_hi - fc["phase_samples"]) / P)
    return int(max(k_hi - k_lo + 1, 0))


# ------------------------------------------------------------------------------------------------ period models

GRID_TOL = 0.03      # a slip continues a run if within this share of a tooth of the run's grid


def tooth_length(mid: np.ndarray) -> float:
    """The tooth length that best explains the gaps between consecutive slips: candidates are every gap divided by
    1-4; each scores the gaps within GRID_TOL of a whole number of teeth, minus half a point for every slip that
    number implies but nobody found (so a fraction of the true tooth, which fits every gap too, loses)."""
    gaps = np.diff(mid)
    best, bestL = -np.inf, float("nan")
    for g0 in gaps:
        for m in range(1, 5):
            L = g0 / m
            if L < 60:
                break
            g = gaps / L
            k = np.round(g)
            fit = (np.abs(g - k) < GRID_TOL * np.maximum(k, 1) ** 0.5) & (k >= 1)
            score = fit.sum() - 0.5 * np.sum((k - 1)[fit])
            if score > best + 1e-9 or (abs(score - best) <= 1e-9 and L > bestL):
                best, bestL = score, L
    return bestL


def grid_runs(mid: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, list[dict]]:
    """Piecewise grid: runs of slips that lie on one grid (each within GRID_TOL of a tooth of the previous slip
    plus a whole number of teeth), one tooth length for all runs, a new phase at each break (a phase jump).

    Returns the tooth length (least squares over all runs, each with its own phase), each slip's run, its number
    within the run, and the phase jumps (time between the runs, the shift as a share of a tooth)."""
    K = mid.size
    if K < 2:
        return float("nan"), np.zeros(K, int), np.zeros(K, int), []
    L = tooth_length(mid)
    if not np.isfinite(L):           # no gap long enough to be a tooth: every slip on its own
        return float("nan"), np.arange(K), np.zeros(K, int), []
    run = np.zeros(K, int)
    num = np.zeros(K, int)
    last = [(0, 0)]                  # per run: index of its last slip, and that slip's number
    for k in range(1, K):
        for r_ in range(len(last) - 1, -1, -1):      # the most recent run first
            i_, n0 = last[r_]
            g = (mid[k] - mid[i_]) / L
            n_ = round(g)
            if n_ >= 1 and abs(g - n_) < GRID_TOL * max(n_, 1) ** 0.5:
                run[k], num[k] = r_, n0 + n_
                last[r_] = (k, n0 + n_)
                break
        else:
            run[k], num[k] = len(last), 0
            last.append((k, 0))
    # common tooth, one phase per run (runs of one slip carry no information on the tooth)
    R = run.max() + 1
    multi = [r_ for r_ in range(R) if (run == r_).sum() >= 2]
    if multi:
        sel = np.isin(run, multi)
        A = np.column_stack([num[sel]] + [(run[sel] == r_).astype(float) for r_ in multi])
        beta = np.linalg.lstsq(A, mid[sel], rcond=None)[0]
        L = float(beta[0])
    # phase jumps: in time order, where the slips of one multi-slip run give way to those of another
    jumps = []
    seq = [k for k in range(K) if (run == run[k]).sum() >= 2]
    for a_, b_ in zip(seq[:-1], seq[1:]):
        if run[a_] == run[b_]:
            continue
        g = (mid[b_] - mid[a_]) / L
        jumps.append({"after": float(mid[a_]), "before": float(mid[b_]), "teeth": round(float(g), 3),
                      "shift": round(float(g - np.floor(g)), 3)})
    return L, run, num, jumps


def grid_period(mid: np.ndarray, tol: float = 0.05) -> tuple[float, np.ndarray]:
    """For grid filling: the tooth length and which slips belong to the longest run of the piecewise grid."""
    K = mid.size
    if K < 2:
        return float("nan"), np.ones(K, bool)
    L, run, _, _ = grid_runs(mid)
    big = np.bincount(run).argmax()
    return L, run == big


def assign_numbers(mid: np.ndarray, L0: float) -> np.ndarray:
    """Slip numbers n_k (integers, n_0 = 0) from the gaps, allowing missed slips (gaps of several teeth)."""
    return np.r_[0, np.cumsum(np.maximum(np.round(np.diff(mid) / L0), 1))].astype(int)


def interval_fit(n: np.ndarray, lo: np.ndarray, hi: np.ndarray, quad: bool = False) -> dict:
    """Slip k at t0 + n_k L (+ n_k^2 q) within [lo_k, hi_k]: the feasible range of L (and q) by linear
    programming, or, when no line passes through every bracket, the least total violation (s)."""
    from scipy.optimize import linprog
    K = n.size
    nv = 3 if quad else 2            # t0, L, (q)
    A = np.column_stack([np.ones(K), n] + ([n.astype(float) ** 2] if quad else []))
    # violations e_k >= 0: lo - e <= A x <= hi + e
    c = np.r_[np.zeros(nv), np.ones(K)]
    A_ub = np.block([[A, -np.eye(K)], [-A, -np.eye(K)]])
    b_ub = np.r_[hi, -lo]
    bounds = [(None, None)] * nv + [(0, None)] * K
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not r.success:
        return {"ok": False}
    viol = float(r.fun)
    out = {"ok": True, "violation_s": viol, "n_violated": int(np.sum(r.x[nv:] > 1e-6)),
           "L": float(r.x[1]), "t0": float(r.x[0])}
    if quad:
        out["q"] = float(r.x[2])
    # range of L (and q) keeping the total violation at its minimum
    A2 = np.r_[A_ub, np.r_[np.zeros(nv), np.ones(K)][None, :]]
    b2 = np.r_[b_ub, viol + 1e-9]
    for name, idx in (("L", 1),) + ((("q", 2),) if quad else ()):
        lim = []
        for sgn in (1, -1):
            cc = np.zeros(nv + K)
            cc[idx] = sgn
            rr = linprog(cc, A_ub=A2, b_ub=b2, bounds=bounds, method="highs")
            lim.append(float(rr.x[idx]) if rr.success else float("nan"))
        out[name + "_range"] = lim
    return out


def accumulator_stats(n: np.ndarray, mid: np.ndarray) -> dict:
    """Constant-period residuals of the slip midpoints, how they grow with the slip number, and the lag-1
    correlation of the tooth lengths (about -0.5 for a phase-locked accumulator with independent placement
    error; about 0 for a renewal process whose tooth lengths vary independently)."""
    if n.size < 4:
        return {}
    k = np.polyfit(n, mid, 1)
    res = mid - np.polyval(k, n)
    full = np.flatnonzero(np.diff(n) == 1)
    lens = np.diff(mid)[full]
    rho = float(np.corrcoef(lens[:-1], lens[1:])[0, 1]) if lens.size >= 4 and np.std(lens) > 0 else float("nan")
    return {"L_lsq": float(k[0]), "resid_sd_s": float(np.std(res, ddof=2)) if n.size > 2 else float("nan"),
            "resid_max_s": float(np.max(np.abs(res))), "tooth_len_lag1": rho, "n_full_teeth": int(lens.size),
            "tooth_len_sd_s": float(np.std(lens, ddof=1)) if lens.size > 1 else float("nan")}


def creep_rates(t: np.ndarray, y: np.ndarray, slips: np.ndarray) -> list[float]:
    """Creep within each full tooth from a straight line through 15-beat medians of y (ppm, positive = y
    shrinking), leaving 20 beats next to each slip out."""
    out = []
    for a_, b_ in zip(slips[:-1], slips[1:]):
        sel = np.flatnonzero((t > a_) & (t < b_))
        if sel.size < 120:
            out.append(float("nan"))
            continue
        sel = sel[20:-20]
        m = sel.size // 15
        tt = t[sel[: 15 * m]].reshape(m, 15).mean(1)
        yy = np.nanmedian(y[sel[: 15 * m]].reshape(m, 15), 1)
        ok = np.isfinite(yy)
        out.append(float(-np.polyfit(tt[ok], yy[ok], 1)[0] * 1e6) if ok.sum() > 5 else float("nan"))
    return out


def gaps_and_pauses(n: np.ndarray, mid: np.ndarray, L: float, t: np.ndarray, y: np.ndarray) -> list[dict]:
    """Stretches between slips longer than 1.5 teeth (and the ends, when longer than 1.5 teeth): the slips the
    constant-period grid predicts there, the largest step near each prediction, the slope of y there, and
    whether the grid holds on both sides (the phase continues through the pause)."""
    out = []
    edges = [(t[0], mid[0], "start")] + [(mid[k], mid[k + 1], "middle") for k in range(mid.size - 1)] + \
        [(mid[-1], t[-1], "end")]
    s15 = boundary_steps(y, STEP_W)
    for a_, b_, kind in edges:
        if b_ - a_ < 1.5 * L:
            continue
        pred = np.arange(a_ + L, b_ - 0.25 * L, L) if kind != "start" else np.arange(b_ - L, a_ + 0.25 * L, -L)[::-1]
        steps = []
        for p in pred:
            sel = np.flatnonzero(np.abs(t - p) < 30)
            steps.append(float(np.nanmax(s15[sel])) if sel.size and np.isfinite(s15[sel]).any() else float("nan"))
        sel = np.flatnonzero((t > a_ + 30) & (t < b_ - 30))
        slope = float("nan")
        if sel.size > 120:
            m = sel.size // 15
            tt = t[sel[: 15 * m]].reshape(m, 15).mean(1)
            yy = np.nanmedian(y[sel[: 15 * m]].reshape(m, 15), 1)
            ok = np.isfinite(yy)
            if ok.sum() > 5:
                slope = float(-np.polyfit(tt[ok], yy[ok], 1)[0] * 1e6)
        out.append({"kind": kind, "start": float(a_), "end": float(b_), "teeth": float((b_ - a_) / L),
                    "predicted": int(pred.size), "pred_step_ms": [round(s * 1000, 1) for s in steps],
                    "slope_ppm": slope})
    return out


def analyse(r: np.ndarray, F: dict, art: np.ndarray | None, fc: dict, ppg_t0: float, pa: int) -> dict:
    """Everything for one set of beats (real or injected)."""
    sd = {j: noise_sd(F[j] - r) for j in FIDS}
    use_art = art is not None and np.isfinite(art).mean() > 0.5
    t_pt, y_det, beat_pt = point_series(F, r)            # detection on ECG->PPG points; ART confirms (below)
    if use_art:
        y_ap = combined(F, art)                          # ART->PPG
        y_ea = art - r                                   # ECG->ART
    out = {"beats": int(r.size), "minutes": float((r[-1] - r[0]) / 60), "noise_ms": {j: round(v * 1000, 2) for j, v in sd.items()},
           "detection_series": "ecg_ppg", "art": bool(use_art)}
    sd_det = noise_sd(y_det)
    out["noise_det_ms"] = round(sd_det * 1000, 2)
    y_det = clean(y_det, sd_det)
    for sign in SIGN_CTRL:
        key = "slips" if sign > 0 else "control"
        cands, sc = detect(t_pt, y_det, sd_det, sign)
        if sign > 0:
            out["_sc"] = {k: v for k, v in sc.items() if not k.startswith("_")}
            out["_t"] = t_pt
        how = {b: "strong" for b in cands}
        sc["_y"], sc["_sign"] = y_det, sign
        if len(cands) >= 2:
            tc = t_pt[cands]
            Lg, run_, num_, _ = grid_runs(tc)
            multi = [r_ for r_ in np.unique(run_) if (run_ == r_).sum() >= 2]
            # each run's grid covers its own stretch, out to halfway to the neighbouring runs' slips
            for r_ in multi:
                sel = run_ == r_
                ref, n_ = tc[sel], num_[sel]
                t0_ = float(np.median(ref - n_ * Lg))
                others = tc[~sel]
                before = others[others < ref[0]]
                after = others[others > ref[-1]]
                lo_ = 0.5 * (before[-1] + ref[0]) if before.size else t_pt[0]
                hi_ = 0.5 * (after[0] + ref[-1]) if after.size else t_pt[-1]
                for b in grid_fill(t_pt, sc, cands, Lg, t0_, span=(lo_, hi_)):
                    how.setdefault(b, "grid")
            cands = sorted(how)
        # the current fit's slips as candidates too (upward only; its times are in ECG time, so moved by the typical
        # delay), each tested like a grid prediction near where it says
        if sign > 0:
            fit_ = fit_summary(r, F["up"], art)
            dly = float(np.nanmedian(F["up"] - r))
            have_t = t_pt[cands] if cands else np.empty(0)
            for x in fit_.get("slips_t", []):
                p_ = x + dly
                if have_t.size and np.min(np.abs(have_t - p_)) < 30:
                    continue
                for b in grid_fill(t_pt, sc, [], 1e9, p_, span=(p_ - 1, p_ + 1), win=10.0):
                    how.setdefault(b, "physiosync")
            cands = sorted(how)
        rows = []
        for bp in cands:
            b = int(beat_pt[bp])                          # the beat of the first point after the step
            p = place(b, F, r, sd, sign)
            if not p:
                continue
            p["boundary"] = int(b)
            p["found_by"] = how[bp]
            p["llr"] = round(float(sc["llr"][bp]), 1)
            p["step_free_ms"] = round(float(sc["fhat"][bp]) * 1000, 2)
            p["sharp"] = round(float(sc["sharp"][bp]), 1)
            p["point_step_t"] = round(float(0.5 * (t_pt[bp - 1] + t_pt[bp])), 3)
            p["ecg_art_step_ms"] = round(ref_step(b, art - r) * 1000, 2) if art is not None else None
            if use_art:
                ap_ = sign * ref_step(b, y_ap)
                ea_ = ref_step(b, y_ea)
                okart = np.isfinite(y_ap[max(b - STEP_W, 0):b + STEP_W]).mean() > 0.8
                if not okart or not np.isfinite(ap_) or not np.isfinite(ea_):
                    p["art_check"] = "no clean ART"
                elif 0.5 * FRAME < ap_ < 1.5 * FRAME and abs(ea_) < 0.3 * FRAME:
                    p["art_check"] = "confirmed"
                elif abs(ea_) >= 0.3 * FRAME:
                    p["art_check"] = "ECG-ART steps too"
                else:
                    p["art_check"] = "ART-PPG does not step"
                p["art_ppg_step_ms"] = round(ap_ * 1000, 2) if np.isfinite(ap_) else None
            else:
                p["art_check"] = "no ART"
            p["persist_ms"] = round(float(sc["persist"][bp]) * 1000, 2)
            p["ticks_in_bracket"] = frame_ticks_in(p["lo"], p["hi"], fc, ppg_t0, pa)
            rows.append(p)
        out[key] = [x for x in rows if x.get("art_check") not in ("ECG-ART steps too", "ART-PPG does not step")]
        out[key + "_rejected_by_art"] = [x for x in rows if x.get("art_check") in ("ECG-ART steps too",
                                                                                  "ART-PPG does not step")]
    sl = out["slips"]
    if len(sl) >= 2:
        L0, run, num, jumps = grid_runs(np.array([s["t"] for s in sl]))
        for s_, r_, n_ in zip(sl, run, num):
            s_["run"], s_["num"] = int(r_), int(n_)
        # a phase jump: was the signal interrupted between the two runs (longest gap between matched beats)?
        for jp in jumps:
            sel = (r >= jp["after"] - 5) & (r <= jp["before"] + 5)
            rr = r[sel]
            jp["longest_beat_gap_s"] = round(float(np.max(np.diff(rr))), 1) if rr.size > 1 else None
        out["runs"] = int(run.max() + 1)
        # the step that moved the phase: the largest free step between the two runs (30 s clear of their slips)
        scp = out.get("_sc")
        for jp in jumps:
            if scp is None:
                break
            sel = np.flatnonzero((t_pt > jp["after"] + 30) & (t_pt < jp["before"] - 30) & np.isfinite(scp["llr_free"]))
            if sel.size:
                k_ = int(sel[np.argmax(scp["llr_free"][sel])])
                jp["step_t"] = round(float(t_pt[k_]), 2)
                jp["step_ms"] = round(float(scp["fhat"][k_]) * 1000, 2)
                jp["step_llr"] = round(float(scp["llr_free"][k_]), 1)
                jp["step_sharp"] = round(float(scp["sharp"][k_]), 1)
        out["phase_jumps"] = jumps
        sizes = np.bincount(run)
        single = np.isin(run, np.flatnonzero(sizes == 1))
        out["off_grid"] = int(single.sum())            # slips in a run of their own
        keep = ~single
        sl = [s_ for s_, k_ in zip(sl, keep) if k_]
        run, num = run[keep], num[keep]
    if len(sl) >= 2:
        mid = np.array([s["t"] for s in sl])
        lo = np.array([s["lo"] for s in sl])
        hi = np.array([s["hi"] for s in sl])
        # one number line across runs: each run is placed at its own phase, so number across runs is only used
        # within runs below; the interval fits take the longest run
        big = np.bincount(run).argmax()
        inrun = run == big
        n = num[inrun] - num[inrun][0]
        mid_all, lo_all, hi_all = mid, lo, hi
        mid, lo, hi, sl = mid[inrun], lo[inrun], hi[inrun], [s_ for s_, k_ in zip(sl, inrun) if k_]
        out["numbers"] = n.tolist()
        out["period"] = {"L0_s": L0}
        if len(sl) >= 3:
            out["period"]["constant"] = interval_fit(n, lo, hi)
            out["period"]["constant_best"] = interval_fit(n, np.array([s["best_lo"] for s in sl]),
                                                          np.array([s["best_hi"] for s in sl]))
        if len(sl) >= 4:
            out["period"]["quadratic"] = interval_fit(n, lo, hi, quad=True)
        out["accumulator"] = accumulator_stats(n, mid)
        y_beat = combined(F, r)
        out["creep_ppm"] = creep_rates(F["up"], y_beat, mid)
        Lc = out["period"].get("constant", {}).get("L", L0) if len(sl) >= 3 else L0
        out["gaps"] = gaps_and_pauses(None, mid_all, Lc, F["up"], y_beat)
    return out


def score_injection(res: dict, truth: dict, fit: dict, delay: float) -> dict:
    """Compare found slips with the injected ones: matched within 60 s; bracket coverage and width; the
    current physiosync placement (its slip times are in ECG time: shifted here by the typical delay)."""
    ts = np.array(truth["slips_t"])
    found = res.get("slips", [])
    rows = []
    sc = res.pop("_sc", None)
    tb = res.pop("_t", None)
    miss = []
    for s in ts:
        k = int(np.argmin([abs(f["t"] - s) for f in found])) if found else -1
        if (k < 0 or abs(found[k]["t"] - s) >= 60) and sc is not None:
            b = int(np.searchsorted(tb, s))
            if 0 < b < tb.size:
                # the best boundary within 5 s of the truth (the slip lies between two points near it)
                sel = np.flatnonzero((np.abs(tb - s) < 5) & np.isfinite(sc["llr"]))
                b = int(sel[np.argmax(sc["llr"][sel])]) if sel.size else b
                m_ = {k_: (round(float(v[b]), 3) if v.dtype != bool else bool(v[b])) for k_, v in sc.items()}
                ps_ = np.array(fit.get("slips_t", [])) + delay
                m_["physiosync_found"] = bool(ps_.size and np.min(np.abs(ps_ - s)) < 60)
                miss.append(m_)
        if k >= 0 and abs(found[k]["t"] - s) < 60:
            f = found[k]
            rows.append({"err_s": f["t"] - s, "in95": bool(f["lo"] <= s <= f["hi"]), "width_s": f["hi"] - f["lo"],
                         "in_best": bool(f["best_lo"] <= s <= f["best_hi"]), "best_width_s": f["best_hi"] - f["best_lo"]})
        else:
            rows.append(None)
    ps = np.array(fit.get("slips_t", [])) + delay
    phys = [float(ps[np.argmin(np.abs(ps - s))] - s) if ps.size and np.min(np.abs(ps - s)) < 60 else None for s in ts]
    matched = [x for x in rows if x]
    n_false = sum(1 for f in found if np.min(np.abs(ts - f["t"])) >= 60) if ts.size else len(found)
    out = {"true": int(ts.size), "found": len(matched), "false": int(n_false), "rows": rows, "physiosync_err_s": phys,
           "missed_scores": miss,
           "true_L": truth["period_s"]}
    c = res.get("period", {}).get("constant")
    if c and c.get("ok"):
        out["L_err_s"] = c["L"] - truth["period_s"]
        out["L_range_covers"] = bool(c["L_range"][0] - 1e-6 <= truth["period_s"] <= c["L_range"][1] + 1e-6)
    return out


def slips_cmd(a) -> None:
    files = [f for i, f in enumerate(sorted(Path(a.scan).glob("*.npz"))) if i % a.of == a.part]
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t0 = time.time()
    with out.open("a") as fh:
        for f in files:
            c = int(f.stem)
            if c in done:
                continue
            if time.time() - t0 > a.time_budget:
                break
            z = np.load(f)
            fc = json.loads(str(z["frame_clock"]))
            ppg_t0, pa = float(z["ppg_t0"]), int(z["ppg_first_index"])
            row = {"case": c, "port": str(z["port"]), "frame_clock": fc, "fit": json.loads(str(z["fit"])),
                   "delay_mode_s": float(z["delay_mode_s"])}
            try:
                art = z["art_t"] if np.isfinite(z["art_t"]).any() else None
                row["real"] = analyse(z["r_t"], {j: z[j] for j in FIDS}, art, fc, ppg_t0, pa)
                row["real"].pop("_sc", None)
                row["real"].pop("_t", None)
                if "inj_truth" in z.files:
                    truth = json.loads(str(z["inj_truth"]))
                    art_i = z["inj_art_t"] if np.isfinite(z["inj_art_t"]).any() else None
                    inj = analyse(z["inj_r_t"], {j: z["inj_" + j] for j in FIDS}, art_i, fc, ppg_t0, pa)
                    row["inj"] = inj
                    row["inj_truth"] = truth
                    row["inj_score"] = score_injection(inj, truth, json.loads(str(z["inj_fit"])), float(z["delay_mode_s"]))
                    # the same sawtooth as a shift of the fiducial times, the waveform untouched
                    ts_ = np.array(truth["slips_t"])
                    rate = truth["rate_ppm"] * 1e-6
                    t_ref = float(z["up"][0])

                    def shift(x):
                        k = np.searchsorted(ts_, x, side="right")
                        return x + FRAME * k - rate * (x - t_ref)
                    Fs = {j: shift(z[j]) for j in FIDS}
                    inj2 = analyse(z["r_t"], Fs, art, fc, ppg_t0, pa)
                    row["inj_shift"] = inj2
                    row["inj_shift_score"] = score_injection(inj2, truth, {}, float(z["delay_mode_s"]))
                    # more fiducial-shift injections on hosts that show no one-frame step of their own
                    if a.inj_seeds and not row["real"].get("slips") and not row["real"].get("control"):
                        multi = []
                        for k_ in range(a.inj_seeds):
                            rg = np.random.default_rng(100_000 + 100 * c + k_)
                            rr_ = float(rg.uniform(*INJ_RATE_PPM)) * 1e-6
                            L_ = FRAME / rr_
                            t_a, t_b = float(z["up"][0]), float(z["up"][-1])
                            first = t_a + float(rg.uniform(0, L_))
                            ts_k = first + L_ * np.arange(int((t_b - first) / L_) + 1)
                            ts_k = ts_k[ts_k < t_b]

                            def shift_k(x, ts_k=ts_k, rr_=rr_, t_a=t_a):
                                return x + FRAME * np.searchsorted(ts_k, x, side="right") - rr_ * (x - t_a)
                            Fk = {j: shift_k(z[j]) for j in FIDS}
                            res_k = analyse(z["r_t"], Fk, art, fc, ppg_t0, pa)
                            tr_k = {"slips_t": ts_k.tolist(), "rate_ppm": rr_ * 1e6, "period_s": L_}
                            fit_k = fit_summary(z["r_t"], Fk["up"], art)       # the current fit, for comparison
                            sc_k = score_injection(res_k, tr_k, fit_k, float(z["delay_mode_s"]))
                            edge = [bool(min(np.sum(z["up"] < x), np.sum(z["up"] > x)) >= 30) for x in ts_k]
                            ps_k = np.array(fit_k.get("slips_t", [])) + float(z["delay_mode_s"])
                            multi.append({"seed": k_, "true": sc_k["true"], "false": sc_k["false"], "rows": sc_k["rows"],
                                          "physiosync_err_s": sc_k["physiosync_err_s"],
                                          "missed_scores": sc_k.get("missed_scores", []),
                                          "physiosync_false": int(sum(1 for x in ps_k if np.min(np.abs(ts_k - x)) >= 60))
                                          if ts_k.size else int(ps_k.size),
                                          "inner": edge, "minutes": res_k["minutes"], "L_err_s": sc_k.get("L_err_s"),
                                          "L_range_covers": sc_k.get("L_range_covers")})
                        row["inj_multi"] = multi
            except Exception as ex:  # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(row) + "\n")
    print(json.dumps({"part": a.part, "done": sum(1 for _ in out.open()) if out.exists() else 0,
                      "seconds": round(time.time() - t0, 1)}))


# ------------------------------------------------------------------------------------------------ stage 3: aggregates

def q(v, ps=(5, 25, 50, 75, 95)) -> dict | None:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    return {f"p{p}": round(float(np.percentile(v, p)), 4) for p in ps} | {"n": int(v.size)} if v.size else None


def quarter_lock(pj: list[dict], tol: float = 0.03) -> dict:
    """How many phase shifts lie within ``tol`` of a whole number of quarter teeth (0, 1/4, 1/2, 3/4), and of
    eighths, against the share expected if shifts were uniform (8 tol and 16 tol); split by whether the signal
    was interrupted between the runs (a gap of more than 10 s between matched beats)."""
    def part(v):
        v = np.array(v, dtype=float)
        if not v.size:
            return {"n": 0}
        dq = np.abs(v * 4 - np.round(v * 4)) / 4
        de = np.abs(v * 8 - np.round(v * 8)) / 8
        from scipy.stats import binomtest
        k_ = int(np.sum(dq < tol))
        return {"n": int(v.size), "within_quarter": k_, "expected_quarter": round(v.size * 8 * tol, 1),
                "p_binomial": float(binomtest(k_, int(v.size), 8 * tol, alternative="greater").pvalue),
                "within_eighth_not_quarter": int(np.sum((de < tol) & (dq >= tol))),
                "expected_eighth_not_quarter": round(v.size * 8 * tol, 1)}
    gap = [(j_.get("longest_beat_gap_s") or 0) > 10 for j_ in pj]
    return {"uninterrupted": part([j_["shift"] for j_, g in zip(pj, gap) if not g]),
            "interrupted": part([j_["shift"] for j_, g in zip(pj, gap) if g]), "tol": tol}


def agg_cmd(a) -> None:
    rows = [json.loads(x) for x in Path(a.slips).read_text().splitlines() if x.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    real = [r for r in rows if "real" in r]
    inj_all = [r for r in rows if "inj_score" in r]
    # a host is clean when the un-injected beats show no one-frame step in either direction
    inj = [r for r in inj_all if not r["real"].get("slips") and not r["real"].get("control")]
    S = {"cases": len(rows), "errors": sum(1 for r in rows if "error" in r)}
    # --- injected ground truth
    tr = [x for r in inj for x in r["inj_score"]["rows"]]
    hit = [x for x in tr if x]
    phys = [x for r in inj for x in r["inj_score"]["physiosync_err_s"]]
    def injected(key: str, res_key: str) -> dict:
        tr = [x for r in inj for x in r[key]["rows"]]
        hit = [x for x in tr if x]
        phys = [x for r in inj for x in r[key]["physiosync_err_s"]]
        ms = [m for r in inj for m in r[key].get("missed_scores", [])]
        out_ = {
            "cases": len(inj), "slips": len(tr), "found": len(hit), "recall": round(len(hit) / max(len(tr), 1), 3),
            "false": sum(r[key]["false"] for r in inj),
            "false_per_hour": round(sum(r[key]["false"] for r in inj)
                                    / max(sum(r[res_key]["minutes"] for r in inj) / 60, 1e-9), 3),
            "coverage95": round(float(np.mean([x["in95"] for x in hit])), 3) if hit else None,
            "coverage_best_gap": round(float(np.mean([x["in_best"] for x in hit])), 3) if hit else None,
            "abs_err_s": q([abs(x["err_s"]) for x in hit]), "width95_s": q([x["width_s"] for x in hit]),
            "best_gap_width_s": q([x["best_width_s"] for x in hit]),
            "physiosync": {"found": sum(1 for x in phys if x is not None),
                           "abs_err_s": q([abs(x) for x in phys if x is not None])},
            "period": {"cases_fitted": sum(1 for r in inj if "L_err_s" in r[key]),
                       "abs_L_err_s": q([abs(r[key]["L_err_s"]) for r in inj if "L_err_s" in r[key]]),
                       "range_covers": sum(1 for r in inj if r[key].get("L_range_covers"))}}
        if ms:
            F_ = FRAME
            out_["missed"] = {
                "n": len(ms),
                "fails_llr": sum(1 for m in ms if not m["llr"] > LLR_STRONG),
                "fails_step": sum(1 for m in ms if not 0.6 * F_ < m["fhat"] < 1.4 * F_),
                "fails_sharp": sum(1 for m in ms if not m["sharp"] > SHARP_MIN),
                "fails_persist": sum(1 for m in ms if not 0.5 * F_ < m["persist"] < 1.5 * F_),
                "fails_quiet": sum(1 for m in ms if not m["quiet"]),
                "llr": q([m["llr"] for m in ms]), "step_ms": q([m["fhat"] * 1000 for m in ms]),
                "sharp": q([m["sharp"] for m in ms]), "persist_ms": q([m["persist"] * 1000 for m in ms])}
        return out_

    def multi(seeds):
        v = [m for r in rows if "inj_multi" in r for m in r["inj_multi"] if m["seed"] in seeds]
        if not v:
            return None
        rows_ = [(x, inn) for m in v for x, inn in zip(m["rows"], m["inner"])]
        hit = [x for x, _ in rows_ if x]
        inner = [x for x, inn in rows_ if inn]
        return {"injections": len(v), "slips": len(rows_), "found": len(hit),
                "recall": round(len(hit) / max(len(rows_), 1), 3),
                "recall_inner": round(sum(1 for x in inner if x) / max(len(inner), 1), 3),
                "inner_slips": len(inner),
                "false": sum(m["false"] for m in v),
                "false_per_hour": round(sum(m["false"] for m in v) / max(sum(m["minutes"] for m in v) / 60, 1e-9), 3),
                "coverage95": round(float(np.mean([x["in95"] for x in hit])), 3) if hit else None,
                "abs_err_s": q([abs(x["err_s"]) for x in hit]),
                "physiosync": {"found": sum(1 for m in v for x in m.get("physiosync_err_s", []) if x is not None),
                               "recall": round(sum(1 for m in v for x in m.get("physiosync_err_s", []) if x is not None)
                                               / max(len(rows_), 1), 3),
                               "recall_inner": round(sum(1 for m in v for x, inn in zip(m.get("physiosync_err_s", []), m["inner"])
                                                         if inn and x is not None) / max(len(inner), 1), 3),
                               "false": sum(m.get("physiosync_false", 0) for m in v),
                               "abs_err_s": q([abs(x) for m in v for x in m.get("physiosync_err_s", []) if x is not None])},
                "period_abs_L_err_s": q([abs(m["L_err_s"]) for m in v if m.get("L_err_s") is not None]),
                "period_range_covers": f"{sum(1 for m in v if m.get('L_range_covers'))} of "
                                       f"{sum(1 for m in v if m.get('L_err_s') is not None)}"}
    S["injected_multi"] = {"tuning_seeds_0_2": multi({0, 1, 2}), "held_out_seeds_3_plus": multi(set(range(3, 100)))}
    S["injected"] = {"hosts": len(inj_all), "contaminated_hosts": [r["case"] for r in inj_all if r not in inj],
                     "waveform": injected("inj_score", "inj"),
                     "fiducial_shift": injected("inj_shift_score", "inj_shift") if all("inj_shift_score" in r for r in inj) else None}
    # --- real: detections and the sign-flipped control
    hours = sum(r["real"]["minutes"] for r in real) / 60
    n_sl = sum(len(r["real"].get("slips", [])) for r in real)
    n_ct = sum(len(r["real"].get("control", [])) for r in real)
    n_ps = sum(len(r["fit"].get("slips_t", [])) for r in real)
    S["real"] = {"cases": len(real), "hours": round(hours, 1), "slips": n_sl, "control_slips": n_ct,
                 "physiosync_slips": n_ps, "off_grid": sum(r["real"].get("off_grid", 0) or 0 for r in real),
                 "cases_with_slips": sum(1 for r in real if r["real"].get("slips"))}
    # --- per recording: setup, correction mode, frame clock, direction and rate of the sawtooth
    key_of, mode_of, snu_rate = {}, {}, {}
    if a.ports and Path(a.ports).exists():
        CORE = ("SNUADC", "Solar8000", "Primus", "BIS")
        for x in Path(a.ports).read_text().splitlines():
            if x.strip():
                d = json.loads(x)
                p_ = d.get("ports", {})
                if all(p_.get(k) for k in CORE):
                    key_of[int(d["case"])] = " · ".join(p_[k] for k in CORE)
    if a.assigned and Path(a.assigned).exists():
        import csv
        for d in csv.DictReader(open(a.assigned)):
            key_of.setdefault(int(d["case"]), d["key"])
    if a.rates and Path(a.rates).exists():
        import csv
        for d in csv.DictReader(open(a.rates)):
            if d["track"].startswith("SNUADC/") and d["rate"]:
                snu_rate[int(d["case"])] = float(d["rate"])
        mode_of = {c: ("step" if v < -20 else "frequent") for c, v in snu_rate.items()}
    recs = []
    for r in real:
        R = r["real"]
        up, dn = R.get("slips", []), R.get("control", [])
        d_ = {"case": r["case"], "key": key_of.get(r["case"]), "snuadc_rate_ppm": snu_rate.get(r["case"]),
              "mode": mode_of.get(r["case"]), "frame_ppm": r["frame_clock"].get("ppm"),
              "frame_sharpness": r["frame_clock"].get("sharpness"), "hours": round(R["minutes"] / 60, 2),
              "up": len(up), "down": len(dn), "runs": R.get("runs"), "phase_jumps": len(R.get("phase_jumps", [])),
              "physiosync_tiers": "".join(sorted(set(r["fit"].get("tiers", []))))}
        direction, tooth = None, None
        if len(up) >= 3 and len(up) >= 2 * len(dn):
            direction = "up"
            tooth = R.get("period", {}).get("L0_s")
        elif len(dn) >= 3 and len(dn) >= 2 * len(up):
            direction = "down"
            L_, run_, _, jd_ = grid_runs(np.array([x["t"] for x in dn]))
            tooth = L_ if np.bincount(run_).max() >= 3 else None
            d_["runs"], d_["phase_jumps"] = int(run_.max() + 1), len(jd_)
        d_["direction"] = direction
        d_["tooth_s"] = round(tooth, 2) if tooth else None
        d_["rate_ppm"] = round((1 if direction == "up" else -1) * FRAME / tooth * 1e6, 2) if tooth else None
        recs.append(d_)
    import csv
    with open(out / "recordings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    by, all_ = {}, {}
    for d_ in recs:
        k_ = f"{d_['key']} | {d_['mode']}"
        all_.setdefault(k_, []).append(d_)
        if d_["rate_ppm"] is None:
            continue
        by.setdefault(k_, []).append(d_)
    S["by_setup"] = {k_: {"recordings": len(all_[k_]), "n": len(v), "up": sum(1 for x in v if x["direction"] == "up"),
                          "down": sum(1 for x in v if x["direction"] == "down"),
                          "rate_ppm": q([x["rate_ppm"] for x in v], (25, 50, 75)),
                          "frame_ppm": q([x["frame_ppm"] for x in v if (x["frame_sharpness"] or 0) >= 10], (25, 50, 75))}
                     for k_, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}
    S["directions"] = {"up": sum(1 for d_ in recs if d_["direction"] == "up"),
                       "down": sum(1 for d_ in recs if d_["direction"] == "down"),
                       "neither": sum(1 for d_ in recs if d_["direction"] is None)}
    pj = [j_ | {"case": r["case"]} for r in real for j_ in r["real"].get("phase_jumps", [])]
    S["phase_jumps"] = {"n": len(pj), "recordings": sum(1 for r in real if r["real"].get("phase_jumps")),
                        "shift": q([j_["shift"] for j_ in pj]), "teeth": q([j_["teeth"] for j_ in pj]),
                        "beat_gap_over_10s": sum(1 for j_ in pj if (j_.get("longest_beat_gap_s") or 0) > 10),
                        "shift_hist_tenths": np.histogram([j_["shift"] for j_ in pj], bins=np.linspace(0, 1, 11))[0].tolist() if pj else [],
                        "step_ms": q([j_.get("step_ms") for j_ in pj]),
                        "step_frames_hist_eighths": np.histogram([j_["step_ms"] / (FRAME * 1000) for j_ in pj
                                                                  if j_.get("step_ms") is not None],
                                                                 bins=np.linspace(-1.0625, 1.0625, 18))[0].tolist(),
                        "quarter_locked": quarter_lock(pj),
                        "rows": pj}
    S["off_grid"] = {"slips": sum(r["real"].get("off_grid", 0) or 0 for r in real),
                     "of": sum(len(r["real"].get("slips", [])) for r in real)}
    # recordings with a reversed sawtooth: at least 3 downward one-frame steps, on one grid (constant tooth)
    rev = []
    for r in real:
        dn = [x["t"] for x in r["real"].get("control", [])]
        if len(dn) >= 3:
            L, on = grid_period(np.array(dn))
            if on.sum() >= 3:
                tiers = sorted(set(r["fit"].get("tiers", [])))
                rev.append({"case": r["case"], "down": len(dn), "on_grid": int(on.sum()), "tooth_s": round(L, 1),
                            "rate_ppm": round(FRAME / L * 1e6, 2), "up": len(r["real"].get("slips", [])),
                            "physiosync_tiers": tiers})
    S["reversed"] = {"cases": len(rev), "rows": rev}
    # score distributions: real slips (strong / grid), the control, and on the injected cases the true and false
    def scores(rs, key, by=None):
        v = [x for r in rs for x in r.get(key, []) if by is None or x.get("found_by") == by]
        return {"n": len(v), "llr": q([x["llr"] for x in v]), "step_free_ms": q([x["step_free_ms"] for x in v]),
                "sharp": q([x["sharp"] for x in v]), "width95_s": q([x["hi"] - x["lo"] for x in v])}
    rr = [r["real"] for r in real]
    S["scores"] = {"real_strong": scores(rr, "slips", "strong"), "real_grid": scores(rr, "slips", "grid"),
                   "control": scores(rr, "control")}
    tru, fal = [], []
    for r in inj:
        ts = np.array(r["inj_truth"]["slips_t"])
        for x in r["inj"].get("slips", []):
            (tru if ts.size and np.min(np.abs(ts - x["t"])) < 60 else fal).append(x)
    S["scores"]["inj_true"] = scores([{"s": tru}], "s")
    S["scores"]["inj_false"] = scores([{"s": fal}], "s")
    # step sizes of the control: a cluster at one frame would be real downward slips, not noise
    cs = np.array([x["step_free_ms"] for r in rr for x in r.get("control", [])])
    S["control_step_hist_ms"] = np.histogram(cs, bins=np.arange(15, 52, 2.5))[0].tolist() if cs.size else []
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


# ------------------------------------------------------------------------------------------------ diagnostic figure

def plot_cmd(a) -> None:
    """Per case: the detection series (15-beat medians and every beat), upward and downward one-frame steps
    marked, and zooms of +/- ``--zoom`` s around chosen steps. This machine only (case-level figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = {json.loads(x)["case"]: json.loads(x) for x in Path(a.slips).read_text().splitlines() if x.strip()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in a.cases:
        z = np.load(Path(a.scan) / f"{c:04d}.npz")
        r = z["r_t"]
        F = {j: z[j] for j in FIDS}
        art = z["art_t"] if np.isfinite(z["art_t"]).mean() > 0.5 else None
        y = combined(F, art) if art is not None else combined(F, r)
        y = np.where(np.isfinite(y), y, combined(F, r))
        t = F["up"]
        t0 = t[0]
        row = rows[c]["real"]
        ev = [(x["t"], +1) for x in row.get("slips", [])] + [(x["t"], -1) for x in row.get("control", [])]
        ev.sort()
        nz = min(len(ev), a.max_zooms)
        fig = plt.figure(figsize=(12, 3.2 + 2.2 * ((nz + 3) // 4)))
        gs = fig.add_gridspec(1 + (nz + 3) // 4, 4, height_ratios=[1.5] + [1] * ((nz + 3) // 4))
        ax = fig.add_subplot(gs[0, :])
        ax.plot((t - t0) / 60, y * 1000, ".", ms=1, color="#bbbbbb")
        m = t.size // 15
        ax.plot((t[: 15 * m].reshape(m, 15).mean(1) - t0) / 60, np.nanmedian(y[: 15 * m].reshape(m, 15), 1) * 1000,
                "-", lw=1, color="#2f6db5")
        lo, hi = np.nanpercentile(y * 1000, [1, 99])
        ax.set_ylim(lo - 20, hi + 20)
        for tt, sg in ev:
            ax.axvline((tt - t0) / 60, color="#e65100" if sg > 0 else "#2e7d32", lw=0.8, ls="--")
        ax.set_xlabel("minutes")
        ax.set_ylabel(("ART" if art is not None else "ECG") + " to PPG (ms)")
        ax.set_title(f"case {c}: {sum(1 for _, s_ in ev if s_ > 0)} steps up a frame (orange), "
                     f"{sum(1 for _, s_ in ev if s_ < 0)} down (green)", fontsize=9)
        for k, (tt, sg) in enumerate(ev[:nz]):
            axz = fig.add_subplot(gs[1 + k // 4, k % 4])
            sel = np.abs(t - tt) < a.zoom
            axz.plot(t[sel] - tt, y[sel] * 1000, ".-", ms=3, lw=0.5, color="#2f6db5")
            axz.axvline(0, color="#e65100" if sg > 0 else "#2e7d32", lw=0.8, ls="--")
            axz.set_title(f"{(tt - t0) / 60:.1f} min ({'up' if sg > 0 else 'down'})", fontsize=8)
            axz.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(out / f"case_{c}.png", dpi=90)
        plt.close(fig)
    print(json.dumps({"figures": len(a.cases)}))


# ------------------------------------------------------------------------------------------------ stage 4: mechanism

SUB_RANGE = (0.12, 0.88)   # sub-frame steps: |size| in frames
NIBP = ("Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP")
NIBP_WIN = (-45.0, 5.0)    # s around a reading: the cuff inflates and measures in the ~40 s before it is published


def free_scores(t: np.ndarray, y: np.ndarray, sd: float) -> dict:
    """At every point boundary (GLR_W points each side, one straight line through both): the log likelihood ratio
    of a step of free size against none, the size, and the log ratio of a sharp step against a ramp of the same
    free size over 7 beats."""
    n = y.size
    W = GLR_W
    out = {k: np.full(n, np.nan) for k in ("llr", "fhat", "sharp")}
    if n < 2 * W + 2:
        return out
    b = np.arange(W, n - W + 1)
    idx = b[:, None] + np.arange(-W, W)[None, :]
    Y = y[idx]
    T = t[idx] - t[b][:, None]
    T = T / max(np.median(np.abs(T)), 1e-9)
    H = (np.arange(-W, W) >= 0).astype(float)[None, :].repeat(b.size, 0)
    ramp = np.clip((np.arange(-W, W) + RAMP_HALF + 0.5) / (2 * RAMP_HALF + 1), 0, 1)[None, :].repeat(b.size, 0)
    one = np.ones_like(Y)

    def fit(X):
        XtX = np.einsum("nki,nkj->nij", X, X)
        XtY = np.einsum("nki,nk->ni", X, Y)
        beta = np.linalg.solve(XtX + 1e-12 * np.eye(X.shape[2])[None], XtY[..., None])[..., 0]
        r = Y - np.einsum("nki,ni->nk", X, beta)
        return np.einsum("nk,nk->n", r, r), beta

    r0, _ = fit(np.stack([one, T], axis=2))
    rs, bs = fit(np.stack([one, T, H], axis=2))
    rr, _ = fit(np.stack([one, T, ramp], axis=2))
    s2 = 2 * sd ** 2
    out["llr"][b] = (r0 - rs) / s2
    out["sharp"][b] = (rr - rs) / s2
    out["fhat"][b] = bs[:, 2]
    return out


def sub_frame_steps(t, y, sd, avoid_t, art_pair=None) -> list[dict]:
    """Sharp, persistent steps of a fraction of a frame (|size| 0.12-0.88 frame), 60 s clear of any one-frame
    step; with a clean arterial line, whether the step is in the PPG chain (ART->PPG steps the same, ECG->ART does
    not) or before it (ECG->ART steps too: physiology or the ECG fiducial)."""
    sc = free_scores(t, y, sd)
    quiet = local_noise(y) <= NOISE_GATE * sd
    llr, f, sh = sc["llr"], sc["fhat"], sc["sharp"]
    af = np.abs(f) / FRAME
    ok = (llr > LLR_STRONG) & (sh > SHARP_MIN) & (af > SUB_RANGE[0]) & (af < SUB_RANGE[1]) & quiet
    avoid = np.asarray(avoid_t, dtype=float)
    out = []
    for b in np.flatnonzero(ok):
        lo, hi = max(b - 30 * PPB, 0), min(b + 30 * PPB + 1, llr.size)
        if llr[b] < np.nanmax(llr[lo:hi]):
            continue
        if avoid.size and np.min(np.abs(avoid - t[b])) < 60:
            continue
        pe = long_step(y, int(b), 1)
        if not np.isfinite(pe) or np.sign(pe) != np.sign(f[b]) or abs(pe - f[b]) > 0.5 * abs(f[b]):
            continue
        if out and t[b] - out[-1]["t"] < 60:
            if llr[b] > out[-1]["llr"]:
                out.pop()
            else:
                continue
        row = {"t": float(t[b]), "size_ms": round(float(f[b]) * 1000, 2), "frac": round(float(f[b]) / FRAME, 4),
               "llr": round(float(llr[b]), 1), "sharp": round(float(sh[b]), 1), "chain": None}
        if art_pair is not None:
            tb, ap, ea = art_pair
            k = int(np.searchsorted(tb, t[b]))
            s_ap, s_ea = ref_step(k, ap), ref_step(k, ea)
            if np.isfinite(s_ap) and np.isfinite(s_ea):
                if abs(s_ea) < 0.3 * abs(f[b]) and abs(s_ap - f[b]) < 0.5 * abs(f[b]):
                    row["chain"] = "ppg"
                elif abs(s_ea) >= 0.5 * abs(f[b]):
                    row["chain"] = "before_ppg"
                else:
                    row["chain"] = "unclear"
                row["art_ppg_ms"], row["ecg_art_ms"] = round(s_ap * 1000, 2), round(s_ea * 1000, 2)
        out.append(row)
    return out


def staircase(t, y, sd, t0, half_beats: int = 12) -> dict:
    """Is the step at t0 one step, or two? Around t0 (+/- half_beats beats of points, levels piecewise constant):
    the best single boundary against the best pair of boundaries at least 3 points apart; the log ratio, the two
    parts (frames) and their separation (s)."""
    k = int(np.searchsorted(t, t0))
    a_, b_ = max(k - half_beats * PPB, 0), min(k + half_beats * PPB, y.size)
    yy, tt = y[a_:b_], t[a_:b_]
    n = yy.size
    if n < 4 * PPB:
        return {}
    cs = np.r_[0, np.cumsum(yy)]
    cq = np.r_[0, np.cumsum(yy ** 2)]

    def sse(i, j):            # sum of squares of yy[i:j] about its mean
        m = j - i
        return cq[j] - cq[i] - (cs[j] - cs[i]) ** 2 / m

    one = min((sse(0, i) + sse(i, n), i) for i in range(3, n - 2))
    two = min((sse(0, i) + sse(i, j) + sse(j, n), i, j) for i in range(3, n - 5) for j in range(i + 3, n - 2))
    i, j = two[1], two[2]
    m0, m1, m2 = (cs[i] - cs[0]) / i, (cs[j] - cs[i]) / (j - i), (cs[n] - cs[j]) / (n - j)
    return {"llr21": round(float((one[0] - two[0]) / (2 * sd ** 2)), 2),
            "part1": round(float((m1 - m0) / FRAME), 3), "part2": round(float((m2 - m1) / FRAME), 3),
            "sep_s": round(float(tt[j] - tt[i]), 2), "total": round(float((m2 - m0) / FRAME), 3)}


def nibp_times(root: str, case: int) -> np.ndarray:
    """Times of new NIBP readings: the numeric tracks repeat the last reading every 2 s, so a reading is where
    the systolic or diastolic value changes."""
    try:
        v = read_vital(Path(root) / "vital_files" / f"{case:04d}.vital", track_names=set(NIBP))
    except Exception:  # noqa: BLE001
        return np.empty(0)
    out = []
    for k in NIBP:
        tr = v.tracks.get(k)
        if tr is None or not tr.record_times.size:
            continue
        t = np.asarray(tr.record_times, dtype=float)
        x = np.asarray(tr.values, dtype=float)
        ok = np.isfinite(x) & (x > 0)
        t, x = t[ok], x[ok]
        if t.size:
            out.append(t[np.r_[True, np.diff(x) != 0]])
    if not out:
        return np.empty(0)
    ev = np.unique(np.round(np.concatenate(out), 0))
    return ev[np.r_[True, np.diff(ev) > 20]] if ev.size else ev


def near(ts, ev, lo=NIBP_WIN[0], hi=NIBP_WIN[1]) -> list[bool]:
    """For each event time in ts: is there an NIBP reading published between -hi and -lo s after it (the event lies
    in the 45 s before a reading, while the cuff inflates and measures, or up to 5 s after)?"""
    ev = np.asarray(ev)
    return [bool(ev.size and np.any((ev - x >= -hi) & (ev - x <= -lo))) for x in ts]


def per_run(slips: list[dict]) -> list[dict]:
    """Within each run of 4 or more slips: interval fits (constant and slowly changing tooth), least-squares
    residuals against the bracket widths, and the deviations of consecutive tooth lengths."""
    runs = {}
    for s_ in slips:
        if "run" in s_:
            runs.setdefault(s_["run"], []).append(s_)
    out = []
    for r_, v in runs.items():
        if len(v) < 4:
            continue
        v.sort(key=lambda x: x["t"])
        n = np.array([x["num"] for x in v]) - v[0]["num"]
        mid = np.array([x["t"] for x in v])
        lo = np.array([x["lo"] for x in v])
        hi = np.array([x["hi"] for x in v])
        c = interval_fit(n, lo, hi)
        qd = interval_fit(n, lo, hi, quad=True)
        k = np.polyfit(n, mid, 1)
        res = mid - np.polyval(k, n)
        full = np.flatnonzero(np.diff(n) == 1)
        lens = np.diff(mid)[full]
        out.append({"run": int(r_), "slips": len(v), "teeth": int(n[-1]), "L": float(k[0]),
                    "resid_sd_s": float(np.std(res, ddof=2)), "bracket_sd_s": float(np.median(hi - lo) / np.sqrt(12)),
                    "const_ok": bool(c.get("ok") and c["violation_s"] < 1e-6), "const_violation_s": c.get("violation_s"),
                    "L_range": c.get("L_range"), "q_range": qd.get("q_range"),
                    "len_dev": (lens - k[0]).tolist(), "consecutive": [int(x) for x in np.diff(full) == 1]})
    return out


PRED_WIN = 10.0     # s either side of a grid prediction searched for the step


def step_at(t_pt, y, sc, t0, win=PRED_WIN) -> dict:
    """The best one-frame step within ``win`` s of t0: its likelihood ratio, free size, sharpness, persistence (two
    windows) and quiet flag, and its offset from t0; and whether it passes the tests a grid-supported slip must pass
    (ratio > 8, not clearly a ramp, quiet, about a frame and holding, or too near an end to test)."""
    sel = np.flatnonzero((np.abs(t_pt - t0) < win) & np.isfinite(sc["llr"]))
    if not sel.size:
        return {}
    b = int(sel[np.argmax(sc["llr"][sel])])
    pl, ps = long_step(y, b, 1), long_step(y, b, 1, PERSIST_SHORT)
    f = float(sc["fhat"][b])
    held = (0.4 * FRAME < pl < 1.6 * FRAME) or (0.4 * FRAME < ps < 1.6 * FRAME)
    edge = not np.isfinite(pl) and not np.isfinite(ps)
    tests = {"llr": bool(sc["llr"][b] > LLR_GRID), "ramp": bool(sc["sharp"][b] > -10), "quiet": bool(sc["quiet"][b]),
             "size_held": bool((held and 0.5 * FRAME < f < 1.6 * FRAME) or (edge and 0.5 * FRAME < f < 1.8 * FRAME))}
    return {"offset_s": round(float(t_pt[b] - t0), 2), "llr": round(float(sc["llr"][b]), 1), "step_frames": round(f / FRAME, 3),
            "sharp": round(float(sc["sharp"][b]), 1),
            "persist_frames": round(pl / FRAME, 3) if np.isfinite(pl) else None,
            "persist_short_frames": round(ps / FRAME, 3) if np.isfinite(ps) else None,
            "passes": all(tests.values()), "fails": [k for k, v in tests.items() if not v]}


def pause_analysis(R: dict, z, F: dict, r: np.ndarray, setup_rate_ppm: float | None,
                   pts: tuple | None = None, case: int = 0) -> dict:
    """Stretches longer than 1.5 teeth without a slip, in a recording with an upward sawtooth (3 or more slips):

    * slope of the delay there (ECG->PPG, beat median of foot, upstroke and peak; 30 s left out at each end) as a
      share of the sawtooth's creep (1: creeping as usual; 0: flat);
    * physiosync.sawtooth_evidence at the recording's rate: share of variance a one-frame sawtooth explains, and the
      same with the sawtooth mirrored (the control);
    * the frame clock in the stretch (raw PLETH maxima): its rate against the whole recording's, and whether the
      recording's frame comb (rate and phase) still holds there (resultant);
    * for a stretch between two slips: do the slips on either side lie on one grid run (the phase carried on through
      the stretch), or not (the shift); and how far the delay fell from just after the first slip to just before the
      second, in frames.

    Recordings without any one-frame step: the slope of the delay over the whole stretch and the sawtooth evidence
    at the setup's rate."""
    from physiosync import sawtooth_evidence
    y = combined(F, r)
    t = F["up"]
    up = sorted(R.get("slips", []), key=lambda x: x["t"])
    fc = json.loads(str(z["frame_clock"]))
    rawmax = z["rawmax"]
    out = {"pauses": []}

    def slope_ppm(a_, b_):
        sel = np.flatnonzero((t > a_) & (t < b_) & np.isfinite(y))
        if sel.size < 120:
            return float("nan")
        m = sel.size // 15
        tt = t[sel[: 15 * m]].reshape(m, 15).mean(1)
        yy = np.nanmedian(y[sel[: 15 * m]].reshape(m, 15), 1)
        return float(np.polyfit(tt, yy, 1)[0] * 1e6)

    def frames_in(a_, b_):
        sel = (t > a_) & (t < b_) & (rawmax >= 0)
        idx = rawmax[sel].astype(float)
        if idx.size < 200 or not fc:
            return {}
        f2 = frame_clock(idx, FRAME_SAMPLES)
        P = FRAME_SAMPLES * (1 + fc["ppm"] * 1e-6)
        comb = float(np.abs(np.exp(2j * np.pi * (idx - fc["phase_samples"]) / P).mean()))
        whole = float(np.abs(np.exp(2j * np.pi * (rawmax[rawmax >= 0] - fc["phase_samples"]) / P).mean()))
        return {"ppm": f2.get("ppm"), "sharpness": f2.get("sharpness"), "ppm_diff": None if f2.get("ppm") is None
                else round(f2["ppm"] - fc["ppm"], 1), "comb_resultant": round(comb, 4), "comb_resultant_whole": round(whole, 4)}

    if len(up) >= 3 and R.get("period", {}).get("L0_s"):
        L = float(R["period"]["L0_s"])
        rate = FRAME / L
        st = [x["t"] for x in up]
        segs = [(t[0], st[0], "start", None, up[0])] + \
            [(st[k], st[k + 1], "middle", up[k], up[k + 1]) for k in range(len(st) - 1)] + [(st[-1], t[-1], "end", up[-1], None)]
        for a_, b_, kind, s0, s1 in segs:
            if b_ - a_ < 1.5 * L:
                continue
            p_ = {"kind": kind, "start": float(a_), "end": float(b_), "teeth": round(float((b_ - a_) / L), 2),
                  "hours": round(float((b_ - a_) / 3600), 3)}
            sl = slope_ppm(a_ + 30, b_ - 30)
            p_["slope_ppm"] = round(sl, 2) if np.isfinite(sl) else None
            p_["creep_share"] = round(-sl / (rate * 1e6), 3) if np.isfinite(sl) else None
            sel = (t > a_ + 30) & (t < b_ - 30) & np.isfinite(y)
            if sel.sum() > 300:
                ev = sawtooth_evidence(t[sel], y[sel], rate, FRAME)
                p_["evidence"] = round(ev["score"], 3) if np.isfinite(ev["score"]) else None
                p_["mirror"] = round(ev["mirror"], 3) if np.isfinite(ev["mirror"]) else None
            p_["frames"] = frames_in(a_ + 30, b_ - 30)
            if kind == "middle":
                p_["same_run"] = bool(s0.get("run") is not None and s0.get("run") == s1.get("run"))
                # the slips the grid predicts inside the stretch: what is there?
                if pts is not None:
                    t_pt, y_pt, sc_pt = pts
                    n_in = int(round((b_ - a_) / L)) - 1
                    p_["predicted"] = [step_at(t_pt, y_pt, sc_pt, a_ + k * (b_ - a_) / (n_in + 1)) | {"k": k}
                                       for k in range(1, n_in + 1)] if n_in >= 1 else []
                g = (b_ - a_) / L
                p_["shift"] = round(float(g - np.floor(g)), 3)
                a1 = np.flatnonzero((t > a_ + 5) & (t < a_ + 60))
                b1 = np.flatnonzero((t > b_ - 60) & (t < b_ - 5))
                if a1.size > 10 and b1.size > 10:
                    p_["drop_frames"] = round(float((np.nanmedian(y[a1]) - np.nanmedian(y[b1])) / FRAME), 3)
            out["pauses"].append(p_)
        # ordinary teeth for comparison: the drop over a tooth of normal length
        drops = []
        for k in range(len(st) - 1):
            if 0.8 * L < st[k + 1] - st[k] < 1.2 * L:
                a1 = np.flatnonzero((t > st[k] + 5) & (t < st[k] + 60))
                b1 = np.flatnonzero((t > st[k + 1] - 60) & (t < st[k + 1] - 5))
                if a1.size > 10 and b1.size > 10:
                    drops.append(float((np.nanmedian(y[a1]) - np.nanmedian(y[b1])) / FRAME))
        out["normal_tooth_drop_frames"] = drops
        out["tooth_s"] = L
        # controls: the same search half way through ordinary teeth, where no slip should be
        if pts is not None:
            t_pt, y_pt, sc_pt = pts
            rg = np.random.default_rng(7_000 + case)
            ctl = []
            for k in range(len(st) - 1):
                if 0.8 * L < st[k + 1] - st[k] < 1.2 * L:
                    x = st[k] + rg.uniform(0.35, 0.65) * (st[k + 1] - st[k])
                    c_ = step_at(t_pt, y_pt, sc_pt, x)
                    if c_:
                        ctl.append(c_)
            out["controls"] = ctl
    elif not R.get("slips") and not R.get("control"):
        sl = slope_ppm(t[0] + 30, t[-1] - 30)
        w = {"hours": round(float((t[-1] - t[0]) / 3600), 3), "slope_ppm": round(sl, 2) if np.isfinite(sl) else None,
             "setup_rate_ppm": setup_rate_ppm}
        sel = np.isfinite(y)
        rr = abs(setup_rate_ppm) * 1e-6 if setup_rate_ppm else 30e-6
        ev = sawtooth_evidence(t[sel], y[sel], rr, FRAME)
        w["evidence"] = round(ev["score"], 3) if np.isfinite(ev["score"]) else None
        w["mirror"] = round(ev["mirror"], 3) if np.isfinite(ev["mirror"]) else None
        w["frames"] = frames_in(t[0], t[-1])
        out["whole"] = w
    return out


def mech_cmd(a) -> None:
    rows = [json.loads(x) for x in Path(a.slips).read_text().splitlines() if x.strip()]
    # each recording's setup rate (median over the setup's recordings with an upward or downward sawtooth)
    setup_rate = {}
    if a.recordings and Path(a.recordings).exists():
        import csv
        recs = list(csv.DictReader(open(a.recordings)))
        by = {}
        for d in recs:
            if d["rate_ppm"]:
                by.setdefault(d["key"], []).append(float(d["rate_ppm"]))
        for d in recs:
            v = by.get(d["key"])
            if v:
                setup_rate[int(d["case"])] = float(np.median(v))
    files = [r for i, r in enumerate(sorted(rows, key=lambda r: r["case"])) if i % a.of == a.part]
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t_start = time.time()
    with out.open("a") as fh:
        for row in files:
            c = row["case"]
            if c in done or "real" not in row:
                continue
            if time.time() - t_start > a.time_budget:
                break
            res = {"case": c}
            try:
                z = np.load(Path(a.scan) / f"{c:04d}.npz")
                r = z["r_t"]
                F = {j: z[j] for j in FIDS}
                art = z["art_t"] if np.isfinite(z["art_t"]).mean() > 0.5 else None
                t_pt, y, _ = point_series(F, r)
                sd = noise_sd(y)
                y = clean(y, sd)
                R = row["real"]
                one_frame = [x["t"] for k_ in ("slips", "control", "slips_rejected_by_art", "control_rejected_by_art")
                             for x in R.get(k_, [])]
                art_pair = None
                if art is not None:
                    tb = F["up"]
                    art_pair = (tb, combined(F, art), art - r)
                subs = sub_frame_steps(t_pt, y, sd, one_frame, art_pair)
                # where each sub-frame step lies: between two runs (a phase jump), inside a run, or elsewhere
                jumps = R.get("phase_jumps", [])
                for s_ in subs:
                    s_["where"] = "jump" if any(j_["after"] < s_["t"] < j_["before"] for j_ in jumps) else (
                        "run" if any(x.get("run") is not None and x["t"] < s_["t"] for x in R.get("slips", [])) and
                        any(x["t"] > s_["t"] for x in R.get("slips", [])) else "elsewhere")
                res["sub_frame"] = subs
                # does a sub-frame step of the jump's size lie between its runs? (either sign: shift or shift - 1)
                for j_ in jumps:
                    inside = [s_ for s_ in subs if j_["after"] < s_["t"] < j_["before"]]
                    j_["sub_steps"] = [s_["frac"] for s_ in inside]
                    j_["matches_shift"] = any(min(abs(s_["frac"] - j_["shift"]), abs(s_["frac"] - (j_["shift"] - 1)),
                                                  abs(-s_["frac"] - j_["shift"]), abs(-s_["frac"] - (j_["shift"] - 1)))
                                              < 0.06 for s_ in inside)
                res["jumps"] = jumps
                # staircase test at every upward slip, and at the injected (single-step) slips as the control
                res["stairs"] = [staircase(t_pt, y, sd, x.get("point_step_t", x["t"])) | {"found_by": x.get("found_by")}
                                 for x in R.get("slips", [])]
                if "inj_truth" in z.files and not R.get("slips") and not R.get("control"):
                    truth = json.loads(str(z["inj_truth"]))
                    ts_ = np.array(truth["slips_t"])
                    rate = truth["rate_ppm"] * 1e-6
                    t_ref = float(z["up"][0])
                    Fs = {j: z[j] + FRAME * np.searchsorted(ts_, z[j], side="right") - rate * (z[j] - t_ref) for j in FIDS}
                    t2, y2, _ = point_series(Fs, r)
                    sd2 = noise_sd(y2)
                    y2 = clean(y2, sd2)
                    res["stairs_injected"] = [staircase(t2, y2, sd2, x) for x in ts_]
                # NIBP: events near a cuff reading, against the recording's own share of time near one
                ev = nibp_times(a.root, c) if a.root else np.empty(0)
                t_lo, t_hi = float(t_pt[0]), float(t_pt[-1])
                ev = ev[(ev > t_lo - 120) & (ev < t_hi + 120)]
                res["nibp_n"] = int(ev.size)
                grid = np.linspace(t_lo, t_hi, 2000)
                res["nibp_time_share"] = float(np.mean(near(grid, ev))) if ev.size else 0.0
                res["nibp_near_sub"] = near([s_["t"] for s_ in subs], ev)
                res["nibp_near_slip"] = near([x["t"] for x in R.get("slips", [])], ev)
                res["nibp_near_jump_step"] = near([j_["step_t"] for j_ in jumps if "step_t" in j_], ev)
                res["nibp_in_jump"] = [bool(np.any((ev > j_["after"]) & (ev < j_["before"]))) for j_ in jumps]
                res["jump_len_s"] = [j_["before"] - j_["after"] for j_ in jumps]
                res["hours"] = (t_hi - t_lo) / 3600
                res["runs"] = per_run(R.get("slips", []))
                sc_pt = step_scores(t_pt, y, sd, 1)
                sc_pt["quiet"] = local_noise(y) <= NOISE_GATE * sd
                res["pause"] = pause_analysis(R, z, F, r, setup_rate.get(c), (t_pt, y, sc_pt), c)
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(res) + "\n")
    print(json.dumps({"part": a.part, "done": sum(1 for _ in out.open()), "seconds": round(time.time() - t_start, 1)}))


def _read_jsonl_parts(paths: list[str]) -> tuple[list[dict], dict]:
    """Rows from one or more JSON-lines files (e.g. the parts of a run), one per case. A case seen more than once
    (a part run twice) is kept once; identical copies are counted, differing copies stop the aggregation."""
    seen, dup = {}, 0
    for pth in paths:
        for x in Path(pth).read_text().splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            c = r.get("case")
            if c in seen:
                if json.dumps(seen[c], sort_keys=True) != json.dumps(r, sort_keys=True):
                    raise SystemExit(f"case {c} appears twice with different results ({pth})")
                dup += 1
                continue
            seen[c] = r
    return [seen[c] for c in sorted(seen, key=lambda v: (v is None, v))], {"files": len(paths), "duplicates_dropped": dup}


def mech_agg_cmd(a) -> None:
    import csv
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows, read = _read_jsonl_parts(a.mech)
    S = {"cases": len(rows), "errors": sum(1 for r in rows if "error" in r), "input": read}
    ok = [r for r in rows if "error" not in r]
    # 1. sub-frame steps
    subs = [s_ | {"case": r["case"]} for r in ok for s_ in r.get("sub_frame", [])]
    def lock(v, tol=0.03):
        v = np.abs(np.array(v, dtype=float))
        if not v.size:
            return {"n": 0}
        dq = np.abs(v * 4 - np.round(v * 4)) / 4
        on_q = (dq < tol) & (np.round(v * 4) >= 1) & (np.round(v * 4) <= 3)
        share = 3 * 2 * tol / (SUB_RANGE[1] - SUB_RANGE[0])
        from scipy.stats import binomtest
        return {"n": int(v.size), "on_quarter": int(on_q.sum()), "expected": round(v.size * share, 1),
                "p_binomial": float(binomtest(int(on_q.sum()), int(v.size), share, alternative="greater").pvalue),
                "hist_frac_twentieths": np.histogram(v, bins=np.linspace(0.1, 0.9, 17))[0].tolist()}
    S["sub_frame"] = {"n": len(subs), "hours": round(sum(r.get("hours", 0) for r in ok), 1),
                      "per_hour": round(len(subs) / max(sum(r.get("hours", 0) for r in ok), 1e-9), 3),
                      "all": lock([s_["frac"] for s_ in subs]),
                      "ppg_chain": lock([s_["frac"] for s_ in subs if s_["chain"] == "ppg"]),
                      "before_ppg": lock([s_["frac"] for s_ in subs if s_["chain"] == "before_ppg"]),
                      "no_art": lock([s_["frac"] for s_ in subs if s_["chain"] is None]),
                      "by_where": {w: lock([s_["frac"] for s_ in subs if s_["where"] == w]) for w in ("jump", "run", "elsewhere")},
                      "sign_ppg_chain": {"up": sum(1 for s_ in subs if s_["chain"] == "ppg" and s_["frac"] > 0),
                                         "down": sum(1 for s_ in subs if s_["chain"] == "ppg" and s_["frac"] < 0)}}
    jumps = [j_ | {"case": r["case"]} for r in ok for j_ in r.get("jumps", [])]
    S["jumps"] = {"n": len(jumps), "with_sub_step": sum(1 for j_ in jumps if j_.get("sub_steps")),
                  "matches_shift": sum(1 for j_ in jumps if j_.get("matches_shift")),
                  "rows": [{k: j_.get(k) for k in ("case", "shift", "sub_steps", "matches_shift", "longest_beat_gap_s")}
                           for j_ in jumps]}
    # 2. staircase
    def stairs(v, thr=10.0):
        v = [x for x in v if x and "llr21" in x]
        two = [x for x in v if x["llr21"] > thr and min(abs(x["part1"]), abs(x["part2"])) >= 0.15 and x["sep_s"] >= 1.5]
        return {"n": len(v), "two_steps": len(two), "share": round(len(two) / max(len(v), 1), 3),
                "part1_frac": q([x["part1"] / x["total"] for x in two if x["total"]]),
                "part1_hist_eighths": np.histogram([x["part1"] / x["total"] for x in two if x["total"]],
                                                   bins=np.linspace(0, 1, 9))[0].tolist(),
                "sep_s": q([x["sep_s"] for x in two])}
    S["staircase"] = {"real": stairs([x for r in ok for x in r.get("stairs", [])]),
                      "real_strong": stairs([x for r in ok for x in r.get("stairs", []) if x.get("found_by") == "strong"]),
                      "injected_single": stairs([x for r in ok for x in r.get("stairs_injected", [])])}
    # 3. NIBP
    def nib(key):
        v = [(b_, r["nibp_time_share"]) for r in ok if r.get("nibp_n") for b_ in r.get(key, [])]
        if not v:
            return {"n": 0}
        obs = sum(b_ for b_, _ in v)
        exp = sum(p_ for _, p_ in v)
        return {"n": len(v), "near": int(obs), "expected": round(exp, 1)}
    S["nibp"] = {"recordings_with_nibp": sum(1 for r in ok if r.get("nibp_n")),
                 "sub_frame_steps": nib("nibp_near_sub"), "one_frame_slips": nib("nibp_near_slip"),
                 "jump_largest_step": nib("nibp_near_jump_step")}
    jr = [(b_, L_, r["nibp_n"] / max(r["hours"] * 3600, 1)) for r in ok if r.get("nibp_n") is not None
          for b_, L_ in zip(r.get("nibp_in_jump", []), r.get("jump_len_s", []))]
    S["nibp"]["jumps_with_reading_inside"] = {"n": len(jr), "observed": sum(1 for b_, _, _ in jr if b_),
                                              "expected": round(sum(1 - np.exp(-rate * L_) for _, L_, rate in jr), 1)}
    # 4. per run
    runs = [x | {"case": r["case"]} for r in ok for x in r.get("runs", [])]
    dev = []
    for x in runs:
        d = np.array(x["len_dev"])
        cons = np.array(x["consecutive"], bool)
        for i in np.flatnonzero(cons):
            dev.append((d[i], d[i + 1]))
    dv = np.array(dev)
    rho = float(np.corrcoef(dv[:, 0], dv[:, 1])[0, 1]) if dv.size > 6 else float("nan")
    boot = []
    rng = np.random.default_rng(0)
    if dv.size > 6:
        for _ in range(2000):
            k_ = rng.integers(0, len(dv), len(dv))
            boot.append(np.corrcoef(dv[k_, 0], dv[k_, 1])[0, 1])
    rate_prec = [abs(x["L_range"][1] - x["L_range"][0]) / x["L"] * (FRAME / x["L"] * 1e6)
                 for x in runs if x.get("L_range") and all(np.isfinite(x["L_range"]))]
    S["runs"] = {"n": len(runs), "slips": sum(x["slips"] for x in runs),
                 "constant_fits_every_bracket": sum(1 for x in runs if x["const_ok"]),
                 "resid_sd_s": q([x["resid_sd_s"] for x in runs]), "bracket_sd_s": q([x["bracket_sd_s"] for x in runs]),
                 "resid_over_bracket": q([x["resid_sd_s"] / x["bracket_sd_s"] for x in runs if x["bracket_sd_s"] > 0]),
                 "rate_range_ppm_if_constant": q(rate_prec),
                 "tooth_len_lag1": {"pairs": int(len(dv)), "rho": round(rho, 3),
                                    "ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)] if boot else None},
                 "quad_q_range_s": q([x["q_range"][1] - x["q_range"][0] for x in runs if x.get("q_range") and all(np.isfinite(x["q_range"]))])}
    # pauses
    P = [p_ | {"case": r["case"]} for r in ok for p_ in r.get("pause", {}).get("pauses", [])]
    def pz(v):
        if not v:
            return {"n": 0}
        cs = [x["creep_share"] for x in v if x.get("creep_share") is not None]
        fr = [x["frames"] for x in v if x.get("frames")]
        return {"n": len(v), "hours": round(sum(x["hours"] for x in v), 1), "recordings": len({x["case"] for x in v}),
                "teeth": q([x["teeth"] for x in v]),
                "creep_share": q(cs), "flat_under_0_3": sum(1 for x in cs if abs(x) < 0.3),
                "creeping_over_0_7": sum(1 for x in cs if x > 0.7),
                "evidence": q([x.get("evidence") for x in v]), "mirror": q([x.get("mirror") for x in v]),
                "evidence_over_0_5": sum(1 for x in v if (x.get("evidence") or -9) > 0.5),
                "frame_clock_found": sum(1 for f_ in fr if (f_.get("sharpness") or 0) >= 10),
                "frame_clock_within_2ppm": sum(1 for f_ in fr if (f_.get("sharpness") or 0) >= 10 and f_.get("ppm_diff") is not None
                                               and abs(f_["ppm_diff"]) <= 2),
                "comb_resultant": q([f_.get("comb_resultant") for f_ in fr]),
                "comb_resultant_whole": q([f_.get("comb_resultant_whole") for f_ in fr]),
                "with_frames": len(fr)}
    mid = [p_ for p_ in P if p_["kind"] == "middle"]
    S["pauses"] = {"all": pz(P), "middle": pz(mid), "start": pz([p_ for p_ in P if p_["kind"] == "start"]),
                   "end": pz([p_ for p_ in P if p_["kind"] == "end"]),
                   "middle_same_run": sum(1 for p_ in mid if p_.get("same_run")),
                   "middle_new_phase": sum(1 for p_ in mid if not p_.get("same_run")),
                   "middle_flat_same_run": sum(1 for p_ in mid if p_.get("same_run") and p_.get("creep_share") is not None
                                               and abs(p_["creep_share"]) < 0.3),
                   "middle_flat_new_phase": sum(1 for p_ in mid if not p_.get("same_run") and p_.get("creep_share") is not None
                                                and abs(p_["creep_share"]) < 0.3),
                   "new_phase_shift_quarter_lock": quarter_lock([{"shift": p_["shift"], "longest_beat_gap_s": 0}
                                                                 for p_ in mid if not p_.get("same_run")]),
                   "drop_frames_pause": q([p_.get("drop_frames") for p_ in mid]),
                   "drop_frames_normal_tooth": q([d_ for r in ok for d_ in r.get("pause", {}).get("normal_tooth_drop_frames", [])]),
                   "rows": [{k: p_.get(k) for k in ("case", "kind", "teeth", "creep_share", "evidence", "mirror", "same_run",
                                                     "shift", "drop_frames")} for p_ in P]}
    def stepz(v):
        v = [x for x in v if x and "passes" in x]          # a prediction with no points within 10 s carries no step
        from collections import Counter
        return {"n": len(v), "passes": sum(1 for x in v if x["passes"]),
                "one_frame": sum(1 for x in v if 0.6 < x["step_frames"] < 1.4 and
                                 any(p is not None and 0.6 < p < 1.4 for p in (x["persist_frames"], x["persist_short_frames"]))),
                "llr": q([x["llr"] for x in v]), "step_frames": q([x["step_frames"] for x in v]),
                "persist_frames": q([x["persist_frames"] for x in v]),
                "offset_s": q([abs(x["offset_s"]) for x in v]),
                "fails": dict(Counter(tuple(x["fails"]) for x in v if not x["passes"]).most_common()) if v else {},
                "step_hist_tenths": np.histogram([x["step_frames"] for x in v], bins=np.linspace(-0.5, 1.5, 21))[0].tolist()}
    pred_same = [x for p_ in mid if p_.get("same_run") for x in p_.get("predicted", [])]
    pred_new = [x for p_ in mid if not p_.get("same_run") for x in p_.get("predicted", [])]
    S["pauses"]["predicted_slips"] = {"same_run": stepz(pred_same), "new_phase": stepz(pred_new),
                                      "controls_mid_tooth": stepz([c_ for r in ok for c_ in r.get("pause", {}).get("controls", [])])}
    for k_ in ("same_run", "new_phase", "controls_mid_tooth"):
        S["pauses"]["predicted_slips"][k_]["fails"] = {"+".join(f) or "none": n for f, n in
                                                      S["pauses"]["predicted_slips"][k_]["fails"].items()}
    W_ = [r["pause"]["whole"] | {"case": r["case"]} for r in ok if r.get("pause", {}).get("whole")]
    S["whole_without_slips"] = {"n": len(W_), "hours": round(sum(w["hours"] for w in W_), 1),
                                "slope_ppm": q([w["slope_ppm"] for w in W_]),
                                "abs_slope_over_5ppm": sum(1 for w in W_ if w["slope_ppm"] is not None and abs(w["slope_ppm"]) > 5),
                                "evidence": q([w["evidence"] for w in W_]), "mirror": q([w["mirror"] for w in W_]),
                                "evidence_over_0_5": sum(1 for w in W_ if (w["evidence"] or -9) > 0.5),
                                "frame_clock_found": sum(1 for w in W_ if (w["frames"].get("sharpness") or 0) >= 10),
                                "rows": W_}
    # 5. clocks: frame clock, sawtooth rate and the SNUADC against the PC, per recording
    recs = list(csv.DictReader(open(a.recordings)))
    snu_pc = {}
    if a.clocks and Path(a.clocks).exists():
        for d in csv.DictReader(open(a.clocks)):
            if d["clock"] == "snuadc" and d["rate_ppm"]:
                snu_pc[int(d["recording"])] = float(d["rate_ppm"])
    tab = []
    for d in recs:
        if not d["rate_ppm"] or not d["frame_ppm"] or float(d["frame_sharpness"] or 0) < 10:
            continue
        fr, rt = float(d["frame_ppm"]), float(d["rate_ppm"])
        e = {"case": int(d["case"]), "key": d["key"], "mode": d["mode"], "frame": fr, "rate": rt,
             "frame_plus_rate": fr + rt, "frame_minus_rate": fr - rt, "snuadc_pc": snu_pc.get(int(d["case"]))}
        tab.append(e)
    with open(Path(a.out) / "clocks_by_recording.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tab[0].keys()))
        w.writeheader()
        w.writerows(tab)
    by = {}
    for e in tab:
        by.setdefault(f"{e['key']} | {e['mode']}", []).append(e)
    def med(v):
        v = [x for x in v if x is not None]
        return round(float(np.median(v)), 2) if v else None
    setups = {k: {"n": len(v), "frame": med([e["frame"] for e in v]), "rate": med([e["rate"] for e in v]),
                  "frame_plus_rate": med([e["frame_plus_rate"] for e in v]),
                  "frame_minus_rate": med([e["frame_minus_rate"] for e in v]),
                  "snuadc_pc": med([e["snuadc_pc"] for e in v])} for k, v in by.items() if len(v) >= 2}
    def spread(key):
        v = [x[key] for x in setups.values() if x[key] is not None]
        return {"setups": len(v), "sd": round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else None,
                "range": [min(v), max(v)] if v else None}
    # one oximeter chain in both SNUADC modes: what is a property of the oximeter (the same in both) and what
    # follows the SNUADC's drift against the PC (differs by it)?
    for e in tab:
        sp = e["snuadc_pc"]
        e["frame_plus_snu"] = None if sp is None else e["frame"] + sp
        e["frame_minus_snu"] = None if sp is None else e["frame"] - sp
    keys = {}
    for e in tab:
        keys.setdefault(e["key"], {}).setdefault(e["mode"], []).append(e)
    modes = {}
    for k_, d_ in keys.items():
        if "step" in d_ and "frequent" in d_:
            modes[k_] = {m: {"n": len(v), **{f: med([e[f] for e in v]) for f in
                                             ("frame", "rate", "snuadc_pc", "frame_plus_snu", "frame_minus_snu")}}
                         for m, v in d_.items() if m in ("step", "frequent")}
            modes[k_]["diff_step_minus_frequent"] = {f: (None if modes[k_]["step"][f] is None or modes[k_]["frequent"][f] is None
                                                        else round(modes[k_]["step"][f] - modes[k_]["frequent"][f], 2))
                                                    for f in ("frame", "rate", "snuadc_pc", "frame_plus_snu", "frame_minus_snu")}
    S["clocks_by_mode"] = modes
    S["clocks"] = {"recordings": len(tab), "setups_with_2": len(setups),
                   "spread_across_setups": {k: spread(k) for k in ("frame", "rate", "frame_plus_rate", "frame_minus_rate")},
                   "setups": setups}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "mech_summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: S[k] for k in ("cases", "errors")}))


def pause_plot_cmd(a) -> None:
    """Figure for the pauses study: (a) a stretch of two teeth whose slips lie on one grid, with the slip the grid
    predicts in it and a zoom on what is there (the strongest one-frame step at a prediction); (b) the longest flat
    stretch between two slips (slope under 0.15 of the creep, no sawtooth evidence, 1.8-6 teeth, in a recording whose
    tooth is 5-50 minutes), with the creep the sawtooth would have shown. Picked from a mech run unless --cases gives
    them (a's case first)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mech = {json.loads(x)["case"]: json.loads(x) for x in Path(a.mech).read_text().splitlines() if x.strip()}
    slips = {json.loads(x)["case"]: json.loads(x) for x in Path(a.slips).read_text().splitlines() if x.strip()}
    pick_a = pick_b = None
    for c, m in mech.items():
        for p_ in m.get("pause", {}).get("pauses", []):
            if p_["kind"] == "middle" and p_.get("same_run") and p_.get("predicted"):
                x = p_["predicted"][0]
                if x and "step_frames" in x and 0.7 < x["step_frames"] < 1.3 and (pick_a is None or x["llr"] > pick_a[2]["llr"]):
                    pick_a = (c, p_, x)
            L_ = m.get("pause", {}).get("tooth_s") or 0
            if p_["kind"] == "middle" and 300 <= L_ <= 3000 and p_.get("creep_share") is not None \
                    and abs(p_["creep_share"]) < 0.15 and (p_.get("evidence") or 0) < 0.3 and 1.8 <= p_["teeth"] <= 6 \
                    and (pick_b is None or p_["teeth"] > pick_b[1]["teeth"]):
                pick_b = (c, p_, None)
    if a.cases:
        def find(c, cond):
            for p_ in mech[c].get("pause", {}).get("pauses", []):
                if cond(p_):
                    return (c, p_, (p_.get("predicted") or [None])[0])
            return None
        pick_a = find(a.cases[0], lambda p_: p_["kind"] == "middle" and p_.get("predicted"))
        if len(a.cases) > 1:
            pick_b = find(a.cases[1], lambda p_: p_.get("creep_share") is not None)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), gridspec_kw={"width_ratios": [2.4, 1]})
    for row, pick, label in ((0, pick_a, "a"), (1, pick_b, "b")):
        if pick is None:
            continue
        c, p_, x = pick
        z = np.load(Path(a.scan) / f"{c:04d}.npz")
        r = z["r_t"]
        F = {j: z[j] for j in FIDS}
        y = combined(F, r)
        t = F["up"]
        t0 = t[0]
        m_ = t.size // 15
        tm = (t[: 15 * m_].reshape(m_, 15).mean(1) - t0) / 60
        ym = np.nanmedian(y[: 15 * m_].reshape(m_, 15), 1) * 1000
        ax = axes[row, 0]
        ax.plot(tm, ym, "-", lw=1, color="#2f6db5")
        lo, hi = np.nanpercentile(ym, [1, 99])
        ax.set_ylim(lo - 15, hi + 15)
        for s_ in slips[c]["real"].get("slips", []):
            ax.axvline((s_["t"] - t0) / 60, color="#e65100", lw=0.8, ls="--")
        ax.axvspan((p_["start"] - t0) / 60, (p_["end"] - t0) / 60, color="#9e9e9e", alpha=0.18, lw=0)
        ax.set_ylabel("ECG to PPG (ms)")
        ax.set_xlabel("minutes")
        az = axes[row, 1]
        if label == "a" and x:
            tp = p_["start"] + (p_["end"] - p_["start"]) / 2
            ax.axvline((tp - t0) / 60, color="k", lw=1, ls=":")
            ax.set_title(f"(a) a stretch of {p_['teeth']:.1f} teeth on one grid (grey); dotted: the slip the grid predicts",
                         fontsize=9, loc="left")
            sel = np.abs(t - tp) < 90
            az.plot(t[sel] - tp, y[sel] * 1000, ".", ms=3, color="#2f6db5")
            az.axvline(x["offset_s"], color="k", lw=1, ls=":")
            az.set_title(f"zoom: a step of {x['step_frames']:.2f} frame, {x['offset_s']:+.1f} s from the prediction",
                         fontsize=8)
            az.set_xlabel("s from the prediction")
        else:
            ax.set_title(f"(b) a flat stretch of {p_['teeth']:.1f} teeth (grey): slope {p_['creep_share']:.2f} of the creep",
                         fontsize=9, loc="left")
            sel = (t > p_["start"]) & (t < p_["end"])
            tt = (t[sel] - p_["start"]) / 60
            az.plot(tt, y[sel] * 1000, ".", ms=1.5, color="#bbbbbb")
            mm = sel.sum() // 15
            if mm > 3:
                ti = (t[sel][: 15 * mm].reshape(mm, 15).mean(1) - p_["start"]) / 60
                yi = np.nanmedian(y[sel][: 15 * mm].reshape(mm, 15), 1) * 1000
                az.plot(ti, yi, "-", color="#2f6db5", lw=1)
                L = mech[c]["pause"].get("tooth_s")
                if L:
                    creep = -(FRAME / L) * 1e3 * 60      # ms per minute
                    az.plot(ti, yi[0] + creep * (ti - ti[0]), "--", color="#e65100", lw=1)
            lo2, hi2 = np.nanpercentile(y[sel] * 1000, [2, 98])
            az.set_ylim(lo2 - 10, hi2 + 10)
            az.set_title("the stretch; dashed: the creep the sawtooth would have shown", fontsize=8)
            az.set_xlabel("minutes into the stretch")
        az.tick_params(labelsize=7)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(json.dumps({"a": pick_a[0] if pick_a else None, "b": pick_b[0] if pick_b else None, "out": str(out)}))


# ------------------------------------------------------------------------------------------------ manual labels

LABEL_STRATA = {          # stratum: (target count, what it is)
    "both": (25, "an upward slip both detectors find (within 10 s)"),
    "new_only": (30, "an upward slip only the new detector finds (no physiosync slip within 60 s)"),
    "physiosync_only": (30, "a physiosync slip (moved to PPG time) with no step of either kind from the new detector within 60 s"),
    "grid_predicted": (25, "the middle of a two-tooth stretch whose slips lie on one grid run (a slip the grid predicts, not found)"),
    "down": (25, "a downward one-frame step"),
    "mid_tooth": (25, "a random point 35-65% through an ordinary tooth (no slip expected)"),
    "no_slip_recording": (15, "a random point in a recording with no one-frame step either way"),
}


def _event_traces(z, t0: float) -> dict:
    """What the labeller sees around time t0 (PPG time): beat by beat over +/- 60 s, ECG->PPG (median of foot,
    upstroke and peak minus R), ART->PPG and ECG->ART where there is a line, in ms relative to the ECG->PPG median
    of the window; and 15-beat medians of ECG->PPG over +/- 8 min. No case id, no detector output."""
    r = z["r_t"]
    F = {j: z[j] for j in FIDS}
    t = F["up"]
    y = combined(F, r)
    art = z["art_t"] if np.isfinite(z["art_t"]).mean() > 0.3 else None
    sel = np.flatnonzero(np.abs(t - t0) <= 60)
    ref = float(np.nanmedian(y[sel])) if sel.size else 0.0
    out = {"zoom_t": np.round(t[sel] - t0, 2).tolist(), "ecg_ppg": np.round((y[sel] - ref) * 1000, 1).tolist()}
    if art is not None:
        ap = combined(F, art)[sel]
        ea = (art - r)[sel]
        if np.isfinite(ap).mean() > 0.5:
            out["art_ppg"] = [None if not np.isfinite(v) else round(float((v - np.nanmedian(ap)) * 1000), 1) for v in ap]
            out["ecg_art"] = [None if not np.isfinite(v) else round(float((v - np.nanmedian(ea)) * 1000), 1) for v in ea]
    sel2 = np.flatnonzero(np.abs(t - t0) <= 480)
    m = sel2.size // 15
    if m > 2:
        tm = t[sel2[: 15 * m]].reshape(m, 15).mean(1)
        ym = np.nanmedian(y[sel2[: 15 * m]].reshape(m, 15), 1)
        out["ov_t"] = np.round((tm - t0) / 60, 3).tolist()
        out["ov_y"] = np.round((ym - ref) * 1000, 1).tolist()
    return out


def label_sample_cmd(a) -> None:
    """Stratified, blinded sample of real events for manual labelling (from a finished slips run): writes
    OUT/events.json (what the page shows: id and traces only) and OUT/key.json (each id's stratum, case, time and
    detector output, and each stratum's size in the run, for weighting). Seeded."""
    rng = np.random.default_rng(a.seed)
    rows = [json.loads(x) for x in Path(a.slips).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "real" in r and "error" not in r]
    pools = {k: [] for k in LABEL_STRATA}
    for r in rows:
        c, R = r["case"], r["real"]
        up = R.get("slips", [])
        dn = R.get("control", [])
        allstep = np.array([x["t"] for x in up + dn + R.get("slips_rejected_by_art", []) + R.get("control_rejected_by_art", [])])
        ps = np.array(r["fit"].get("slips_t", [])) + r.get("delay_mode_s", 0.0)
        for x in up:
            near_ps = ps.size and np.min(np.abs(ps - x["t"]))
            if ps.size and near_ps < 10:
                pools["both"].append((c, x["t"], {"found_by": x.get("found_by"), "lo": x["lo"], "hi": x["hi"]}))
            elif not ps.size or near_ps >= 60:
                pools["new_only"].append((c, x["t"], {"found_by": x.get("found_by"), "lo": x["lo"], "hi": x["hi"]}))
        for p_ in ps:
            if not allstep.size or np.min(np.abs(allstep - p_)) >= 60:
                pools["physiosync_only"].append((c, float(p_), {}))
        for x in dn:
            pools["down"].append((c, x["t"], {"lo": x["lo"], "hi": x["hi"]}))
        srt = sorted(up, key=lambda x: x["t"])
        for x0, x1 in zip(srt[:-1], srt[1:]):
            if x0.get("run") is not None and x0.get("run") == x1.get("run"):
                dnum = x1.get("num", 0) - x0.get("num", 0)
                if dnum == 2:
                    pools["grid_predicted"].append((c, 0.5 * (x0["t"] + x1["t"]), {"gap_s": x1["t"] - x0["t"]}))
                elif dnum == 1:
                    pools["mid_tooth"].append((c, x0["t"] + float(rng.uniform(0.35, 0.65)) * (x1["t"] - x0["t"]), {}))
        if not up and not dn:
            z_path = Path(a.scan) / f"{c:04d}.npz"
            if z_path.exists():
                pools["no_slip_recording"].append((c, None, {}))
    events, key = [], {"strata": {k: {"target": v[0], "what": v[1], "pool": len(pools[k])} for k, v in LABEL_STRATA.items()},
                       "events": {}, "seed": a.seed, "slips": str(a.slips)}
    picked = []
    for k, (n, _) in LABEL_STRATA.items():
        pool = pools[k]
        idx = rng.permutation(len(pool))[:n]
        picked += [(k,) + pool[i] for i in idx]
    rng.shuffle(picked)
    for i, (k, c, t0, info) in enumerate(picked):
        z = np.load(Path(a.scan) / f"{c:04d}.npz")
        t = z["up"]
        if t0 is None:
            t0 = float(rng.uniform(t[0] + 120, t[-1] - 120))
        if t0 - t[0] < 60 or t[-1] - t0 < 60:
            continue
        eid = f"E{i + 1:03d}"
        events.append({"id": eid} | _event_traces(z, t0))
        key["events"][eid] = {"stratum": k, "case": int(c), "t": float(t0)} | info
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.json").write_text(json.dumps({"events": events}, separators=(",", ":")))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(json.dumps({"events": len(events), "pools": {k: len(v) for k, v in pools.items()}}))


def label_examples_cmd(a) -> None:
    """Training examples for the labelling page: SPEC is a JSON file {"examples": [{case, t, title, caption, label,
    flags}]}; writes OUT (examples.json) with the same traces as the labelled events, plus the title, caption and
    suggested label. Refuses a case that is in the labelling sample (KEY), so the examples never show an event the
    labeller will see."""
    spec = json.loads(Path(a.spec).read_text())
    insample = {v["case"] for v in json.loads(Path(a.key).read_text())["events"].values()} if a.key else set()
    out = []
    for ex in spec["examples"]:
        if ex["case"] in insample:
            raise SystemExit(f"case {ex['case']} is in the labelling sample")
        z = np.load(Path(a.scan) / f"{ex['case']:04d}.npz")
        out.append({k: ex[k] for k in ("title", "caption", "label", "flags")} | _event_traces(z, float(ex["t"])))
    Path(a.out).write_text(json.dumps({"examples": out}, separators=(",", ":")))
    print(json.dumps({"examples": len(out)}))


LABEL_VERDICTS = ("ppg_slip_up", "ppg_slip_down", "ppg_step_up_unconfirmed", "ppg_step_down_unconfirmed", "ecg_art_step",
                  "art_timing_step", "step_not_one_frame", "steps_apart", "none", "inconsistent", "unclear", "artefact")
LABEL_APART_S = 3.0  # step marks in ECG->PPG and ART->PPG further apart than this are two events, not one slip


def _mark_span(m) -> tuple[float, float] | None:
    """A step mark from the page: a time (s from the event) or an interval [t0, t1] it lies within."""
    if isinstance(m, (int, float)):
        return float(m), float(m)
    if isinstance(m, (list, tuple)) and len(m) == 2:
        return float(min(m)), float(max(m))
    return None


def _mark_mid(m) -> float | None:
    sp = _mark_span(m)
    return None if sp is None else (sp[0] + sp[1]) / 2


def label_verdict(d: dict) -> str:
    """The verdict from one labelled event (page schema 2; each trace is up, down, none, unclear or artefact, ECG->ART
    flat, step, unclear or artefact). With an arterial line, a PPG slip is ECG->PPG and ART->PPG stepping the same way
    with ECG->ART flat; ECG->PPG stepping with ART->PPG flat and ECG->ART stepping is ecg_art_step (physiology, or ECG
    or arterial timing); ART->PPG stepping with ECG->PPG flat and ECG->ART stepping is art_timing_step. Where only one
    PPG delay can be read (no arterial line, or ART->PPG or ECG->ART marked an artefact, or ECG->PPG marked an artefact
    while ART->PPG steps), its step cannot be checked against the other trace: ppg_step_*_unconfirmed. A step whose
    size is marked clearly not one frame, in either PPG delay, is step_not_one_frame. Where both PPG delays step the
    same way but their step marks (points, or intervals the step lies within) are more than LABEL_APART_S apart, the steps are two events: steps_apart. Every
    readable trace an artefact: artefact. Answers that break ECG->PPG = ECG->ART + ART->PPG are inconsistent."""
    if d.get("artefact") == "yes":  # an early page version had one artefact answer for the whole event
        return "artefact"
    ep, ap_, ea = d.get("ecg_ppg"), d.get("art_ppg"), d.get("ecg_art")
    sz = d.get("size") or {}
    sz = sz if isinstance(sz, dict) else {"ecg_ppg": sz, "art_ppg": sz, "ecg_art": sz}
    steps = ("up", "down")
    has_art = d.get("has_art", ap_ is not None)

    def single(v, tr):
        if v == "artefact":
            return "artefact"
        if v in (None, "unclear"):
            return "unclear"
        if v == "none":
            return "none"
        return "step_not_one_frame" if sz.get(tr) == "other" else f"ppg_step_{v}_unconfirmed"

    if not has_art:
        return single(ep, "ecg_ppg")
    if ep == "artefact" and ap_ == "artefact":
        return "artefact"
    if ep == "artefact":
        return single(ap_, "art_ppg") if ap_ in steps else ("unclear" if ap_ in (None, "unclear") else "artefact")
    if ap_ == "artefact" or ea == "artefact":
        return single(ep, "ecg_ppg")
    if "unclear" in (ep, ap_, ea) or None in (ep, ap_, ea):
        return "unclear"
    if ep == "none" and ap_ == "none":
        return "none" if ea == "flat" else "inconsistent"
    cl = d.get("clicks") or {}
    a_, b_ = _mark_span(cl.get("ecg_ppg")), _mark_span(cl.get("art_ppg"))
    if ep in steps and ap_ == ep and a_ and b_ and max(a_[0] - b_[1], b_[0] - a_[1]) > LABEL_APART_S:
        return "steps_apart"
    if ep in steps and ap_ == ep and ea == "flat":
        return "step_not_one_frame" if "other" in (sz.get("ecg_ppg"), sz.get("art_ppg")) else f"ppg_slip_{ep}"
    if ep in steps and ap_ == "none" and ea == "step":
        return "ecg_art_step"
    if ep == "none" and ap_ in steps and ea == "step":
        return "art_timing_step"
    return "inconsistent"


def label_eval_cmd(a) -> None:
    """Score the detectors against manual labels. LABELS is a JSON list (or {"rows": [...]}) of the page's documents
    (schema 2: ecg_ppg, art_ppg, ecg_art, size: per trace that steps, frame, other or unclear, artefact, has_art, flags: per trace, a list of disturbed, gradual, two_steps, recurring_dips; trace_notes: per trace; note, clicks: the step mark per trace, a time or an interval [t0, t1] it lies within, features: other marks, each
    {trace, t, kind: dip, artefact or other}, with t0 and t1 for an interval); each becomes a verdict
    (label_verdict). A real upward slip is ppg_slip_up, or ppg_step_up_unconfirmed where only one PPG delay can be read;
    both are also given apart. Per stratum: the verdicts; and, weighting each stratum by its pool, how many of each
    detector's slips are real (precision) and how many real slips each misses. Unclear verdicts are left out of the
    shares; artefacts and inconsistent answers are counted as not slips."""
    key = json.loads(Path(a.key).read_text())
    raw = json.loads(Path(a.labels).read_text())
    raw = raw.get("rows", raw) if isinstance(raw, dict) else raw
    lab, skipped = {}, 0
    for d in raw:
        d = d.get("data", d)
        eid = d.get("id") or d.get("event")
        if not eid:
            continue
        if d.get("schema") != 2:
            skipped += 1
            continue
        lab[eid] = d | {"verdict": label_verdict(d)}
    from collections import Counter
    UP = ("ppg_slip_up", "ppg_step_up_unconfirmed")
    S = {"labelled": len(lab), "skipped_old_schema": skipped, "events": len(key["events"]),
         "verdicts": dict(Counter(v["verdict"] for v in lab.values())), "strata": {}}
    for k, st in key["strata"].items():
        ids = [e for e, v in key["events"].items() if v["stratum"] == k]
        got = [lab[e] for e in ids if e in lab]
        cnt = Counter(d["verdict"] for d in got)
        n = len(got)
        decided = n - cnt.get("unclear", 0)
        art = [d for d in got if d.get("has_art")]
        art_dec = sum(1 for d in art if d["verdict"] != "unclear")
        real_up = sum(cnt.get(v, 0) for v in UP)
        S["strata"][k] = {
            "what": st["what"], "pool": st["pool"], "shown": len(ids), "labelled": n, "verdicts": dict(cnt),
            "share_up": round(real_up / decided, 3) if decided else None,
            "share_down": round((cnt.get("ppg_slip_down", 0) + cnt.get("ppg_step_down_unconfirmed", 0)) / decided, 3) if decided else None,
            "with_art": {"labelled": len(art), "share_up": round(sum(1 for d in art if d["verdict"] == "ppg_slip_up") / art_dec, 3) if art_dec else None,
                         "share_ecg_art_step": round(sum(1 for d in art if d["verdict"] == "ecg_art_step") / art_dec, 3) if art_dec else None},
            "est_up_in_pool": round(real_up / decided * st["pool"], 1) if decided else None}
        for tr in ("ecg_ppg", "art_ppg"):
            ms = [d["clicks"][tr] for d in got if (d.get("clicks") or {}).get(tr) is not None and d["verdict"].startswith(("ppg_slip", "ppg_step_"))]
            if ms:
                S["strata"][k][f"click_offset_s_{tr}"] = q([abs(_mark_mid(m)) for m in ms])
                wid = [_mark_span(m)[1] - _mark_span(m)[0] for m in ms if isinstance(m, (list, tuple))]
                if wid:
                    S["strata"][k][f"click_interval_width_s_{tr}"] = q(wid)
        fl = Counter((tr, f) for d in got if isinstance(d.get("flags"), dict) for tr, fs in d["flags"].items() for f in fs)
        if fl:
            S["strata"][k]["flags"] = {f"{tr}:{f}": n for (tr, f), n in sorted(fl.items())}
        fk = Counter((f.get("trace"), f.get("kind")) for d in got for f in d.get("features") or [])
        if fk:
            S["strata"][k]["features"] = {f"{tr}:{kd}": n for (tr, kd), n in sorted(fk.items())}
            S["strata"][k]["events_with_features"] = sum(1 for d in got if d.get("features"))
        both = [abs(_mark_mid(d["clicks"]["ecg_ppg"]) - _mark_mid(d["clicks"]["art_ppg"])) for d in got
                if (d.get("clicks") or {}).get("ecg_ppg") is not None and (d.get("clicks") or {}).get("art_ppg") is not None]
        if both:
            S["strata"][k]["click_gap_ecg_art_s"] = q(both)
    st = S["strata"]
    def est(k):
        return st[k]["est_up_in_pool"] or 0.0
    new_pool = st["both"]["pool"] + st["new_only"]["pool"]
    ps_pool = st["both"]["pool"] + st["physiosync_only"]["pool"]
    real = est("both") + est("new_only") + est("physiosync_only") + est("grid_predicted")
    S["estimate"] = {
        "new_detector_slips": new_pool, "new_detector_real": round(est("both") + est("new_only"), 1),
        "new_detector_precision": round((est("both") + est("new_only")) / new_pool, 3) if new_pool else None,
        "physiosync_slips": ps_pool, "physiosync_real": round(est("both") + est("physiosync_only"), 1),
        "physiosync_precision": round((est("both") + est("physiosync_only")) / ps_pool, 3) if ps_pool else None,
        "real_slips_in_these_strata": round(real, 1),
        "new_detector_recall": round((est("both") + est("new_only")) / real, 3) if real else None,
        "physiosync_recall": round((est("both") + est("physiosync_only")) / real, 3) if real else None,
        "note": "a real slip is ppg_slip_up, or ppg_step_up_unconfirmed where only one PPG delay can be read (an ECG-side "
                "step cannot be ruled out there; with_art gives the arterial-line events alone); recall counts only real slips in "
                "the sampled strata (found by either, or predicted by the grid)"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_eval.json").write_text(json.dumps(S, indent=1) + "\n")
    (out / "label_verdicts.json").write_text(json.dumps(
        {e: {"stratum": key["events"][e]["stratum"], **{k: d.get(k) for k in ("verdict", "ecg_ppg", "art_ppg", "ecg_art", "size", "flags", "trace_notes", "note", "clicks", "features")}}
         for e, d in sorted(lab.items()) if e in key["events"]}, indent=1) + "\n")
    print(json.dumps(S["estimate"], indent=1))


def _wave_channel(t: np.ndarray, x: np.ndarray, t0: float, half: float, step: int) -> dict | None:
    """One waveform channel for the labelling page: samples within t0 +/- half, every step-th, as int16 with a
    scale and offset (base64, little-endian); times relative to t0."""
    import base64
    sel = np.flatnonzero((t >= t0 - half) & (t <= t0 + half))[::step]
    if sel.size < 10:
        return None
    v = np.asarray(x[sel], dtype=np.float64)
    ok = np.isfinite(v)
    if ok.mean() < 0.5:
        return None
    v = np.where(ok, v, np.nanmedian(v))
    lo, hi = float(np.min(v)), float(np.max(v))
    scale = (hi - lo) / 60000.0 or 1.0
    q_ = np.round((v - lo) / scale - 30000).astype("<i2")
    gaps = np.flatnonzero(np.diff(t[sel]) > 1.5 * step / FS)
    return {"t0": round(float(t[sel[0]] - t0), 4), "fs": FS / step, "offset": lo + 30000 * scale, "scale": scale,
            "n": int(sel.size), "gaps": [round(float(t[sel[g]] - t0), 3) for g in gaps], "b64": base64.b64encode(q_.tobytes()).decode()}


def label_waves_cmd(a) -> None:
    """The waveforms behind each labelling event (or worked example): ECG lead II (250 Hz), PLETH and ART (125 Hz)
    over +/- HALF s, every R peak the detector of the scan finds in the recording's stretch (marked whether it was
    matched to a pulse and so used in the delays), and the matched beats' PPG fiducials and arterial upstrokes.
    One file per event, OUT/<id>.json, no case id. Resumable; stops before TIME_BUDGET s."""
    import base64  # noqa: F401
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    if a.key:
        key = json.loads(Path(a.key).read_text())
        items += [(eid, v["case"], v["t"]) for eid, v in sorted(key["events"].items())]
    if a.spec:
        spec = json.loads(Path(a.spec).read_text())
        items += [(f"X{i + 1}", ex["case"], float(ex["t"])) for i, ex in enumerate(spec["examples"])]
    t_start, slowest, n = time.time(), 0.0, 0
    for eid, c, t0 in items:
        f = out / f"{eid}.json"
        if f.exists():
            continue
        if time.time() - t_start + max(1.5 * slowest, 5) > a.time_budget:
            break
        t1 = time.time()
        z = np.load(Path(a.scan) / f"{c:04d}.npz")
        case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG, ART})
        e, p = case.tracks[ECG], case.tracks[PPG]
        te, tp = sample_times(e), sample_times(p)
        # the scan's stretch, found the same way, so the R peaks are those the scan found
        gap = np.flatnonzero(np.abs(np.diff(te) - 1 / FS) > 0.01 / FS)
        be = np.r_[0, gap + 1, te.size]
        k = int(np.argmax(np.diff(be)))
        ea, eb = be[k], min(be[k + 1], be[k] + int(a.minutes * 60 * FS))
        xe_all = e.physical()
        xe = xe_all[ea:eb].astype(np.float64)
        rp = r_peaks(xe, FS)
        r_all = te[ea] + ecg_subsample(xe, rp, FS) / FS
        r_m = z["r_t"]
        win = lambda v: (v >= t0 - a.half) & (v <= t0 + a.half)  # noqa: E731
        rr = r_all[win(r_all)]
        used = np.isin(np.round(rr, 4), np.round(r_m, 4))
        rec = {"id": eid, "half_s": a.half,
               "ecg": _wave_channel(te, xe_all, t0, a.half, 2),
               "ppg": _wave_channel(tp, p.physical(), t0, a.half, 4),
               "r": [[round(float(x - t0), 4), int(u)] for x, u in zip(rr, used)],
               "beats": []}
        tr = case.tracks.get(ART)
        if tr is not None and tr.record_times.size and float(tr.srate) == FS:
            rec["art"] = _wave_channel(sample_times(tr), tr.physical(), t0, a.half, 4)
        for i in np.flatnonzero(win(r_m)):
            b = {"r": round(float(r_m[i] - t0), 4)}
            for j in ("foot", "up", "peak"):
                if np.isfinite(z[j][i]):
                    b[j] = round(float(z[j][i] - t0), 4)
            if np.isfinite(z["art_t"][i]):
                b["art"] = round(float(z["art_t"][i] - t0), 4)
            rec["beats"].append(b)
        f.write_text(json.dumps(rec, separators=(",", ":")))
        slowest = max(slowest, time.time() - t1)
        n += 1
    print(json.dumps({"new": n, "done": len(list(out.glob("*.json"))), "of": len(items),
                      "seconds": round(time.time() - t_start, 1)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--root", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--minutes", type=float, default=90.0)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--time-budget", type=float, default=150)
    p = sub.add_parser("slips")
    p.add_argument("--scan", required=True, help="the scan folder")
    p.add_argument("--inj-seeds", type=int, default=0,
                   help="extra fiducial-shift injections per clean host (seeds 0-2 tune, the rest are held out)")
    p.add_argument("--out", required=True, help="JSON lines, one per case (resumable)")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--time-budget", type=float, default=150)
    p = sub.add_parser("agg")
    p.add_argument("--slips", required=True, help="the slips JSON lines (all parts concatenated)")
    p.add_argument("--out", required=True)
    p.add_argument("--ports", help="setups/ports.jsonl: device ports per case (the PC key)")
    p.add_argument("--assigned", help="setups/assign/assigned.csv: keys assigned where a port is missing")
    p.add_argument("--rates", help="correction_census/census_rates.csv: SNUADC rate per case (mode: step < -20 ppm)")
    p = sub.add_parser("plot")
    p.add_argument("--scan", required=True)
    p.add_argument("--slips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cases", type=int, nargs="+", required=True)
    p.add_argument("--zoom", type=float, default=60.0, help="half width of the zooms (s)")
    p.add_argument("--max-zooms", type=int, default=12)
    p = sub.add_parser("mech")
    p.add_argument("--scan", required=True)
    p.add_argument("--recordings", help="recordings.csv from agg (each setup's rate, for recordings without slips)")
    p.add_argument("--slips", required=True, help="the slips JSON lines of a finished run")
    p.add_argument("--root", help="the VitalDB release (for the NIBP readings)")
    p.add_argument("--out", required=True, help="JSON lines, one per case (resumable)")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--time-budget", type=float, default=150)
    p = sub.add_parser("mech-agg")
    p.add_argument("--mech", required=True, nargs="+", help="the mech JSON lines: one file, or every part of a run")
    p.add_argument("--recordings", required=True, help="recordings.csv from agg")
    p.add_argument("--clocks", help="clock network clocks.csv (the SNUADC against the PC per recording)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("pause-plot")
    p.add_argument("--scan", required=True)
    p.add_argument("--slips", required=True)
    p.add_argument("--mech", required=True)
    p.add_argument("--out", required=True, help="the figure file")
    p.add_argument("--cases", type=int, nargs="*", help="the two recordings (a, b); picked from the mech run if not given")
    p = sub.add_parser("label-sample")
    p.add_argument("--scan", required=True)
    p.add_argument("--slips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261102)
    p = sub.add_parser("label-examples")
    p.add_argument("--scan", required=True)
    p.add_argument("--spec", required=True, help="examples_spec.json")
    p.add_argument("--key", help="the labelling sample's key.json (examples from its cases are refused)")
    p.add_argument("--out", required=True, help="examples.json")
    p = sub.add_parser("label-waves")
    p.add_argument("--scan", required=True)
    p.add_argument("--root", required=True, help="the VitalDB release")
    p.add_argument("--key", help="the labelling sample's key.json")
    p.add_argument("--spec", help="examples_spec.json (the worked examples, ids X1, X2, ...)")
    p.add_argument("--out", required=True, help="a folder: one <id>.json per event")
    p.add_argument("--half", type=float, default=62.0, help="half width of the window (s)")
    p.add_argument("--minutes", type=float, default=90.0, help="as in scan: the stretch's maximum length")
    p.add_argument("--time-budget", type=float, default=150)
    p = sub.add_parser("label-eval")
    p.add_argument("--key", required=True)
    p.add_argument("--labels", required=True, help="the labels exported from the labelling page (JSON)")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"scan": scan, "slips": slips_cmd, "agg": agg_cmd, "plot": plot_cmd, "mech": mech_cmd,
     "mech-agg": mech_agg_cmd, "pause-plot": pause_plot_cmd, "label-sample": label_sample_cmd,
     "label-examples": label_examples_cmd, "label-eval": label_eval_cmd, "label-waves": label_waves_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
