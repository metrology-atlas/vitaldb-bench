# SPDX-License-Identifier: MIT
"""Confidence tiers of the rebuilt sawtooth correction, per tooth, with a gallery for looking at (metrology-atlas
study ppg-sawtooth). Reads a fit of tools/slip_rebuild.py and the scan it was made from.

    python tools/slip_tiers.py --fit $W/slip_rebuild/fit_v16.jsonl --scan $W/sawtooth_char/scan_full \
        --out $W/slip_tiers [--per-tier 8 --seed 0]

The slips of a recording (real beats) are its events: confirmed slips (the shape test agrees, on the grid, off it,
or the other way), one-frame steps whose shape cannot be judged or is unclear ("steps"), and the fallback at
predicted slips that could not be placed: one slip within the 5-95% interval with probability p ("uncertain"; an
event when p >= 0.5). A tooth runs from one event to the next (the first and last from the ends of the stretch).

Tiers of a tooth:

* **A**: both edges confirmed slips.
* **B**: both edges confirmed slips or steps, at least one a step.
* **C**: an edge is an uncertain slip (not corrected by default; with --correct-uncertain, as a ramp across its
  interval weighted by p).
* **U**: unresolved: the tooth holds a predicted slip with p < 0.5 (a slip may or may not be there), a step the
  other way, or is longer than 1.5 teeth (a missed slip or a pause; at the ends too: the sawtooth stopped).
* **I**: steps only: isolated slips in a recording without a sawtooth, or a sawtooth whose tooth rests on two
  confirmed slips (the tooth and its creep reported, not corrected; slip_rebuild --anchor-check). Each slip is
  corrected as a step, with no creep. The drift slip_rebuild
  reports for the recording (the delay's slope with a free step at each slip) is given in the reason and summarised,
  not corrected.
* **N**: no slip at all.

A partial tooth at either end takes the tier of its inner edge. The correction: D(t) = sign x frame x (N(t) - (t -
t_ref) / L) in a sawtooth recording (the expected slip count N, ramps for uncertain slips; the creep at the fitted
tooth L, from a tooth before the first slip to a tooth after the last), sign x frame x N(t) for isolated slips; the corrected delay is the measured one minus D.

Writes teeth.csv, summary.json and tier-<X>.png (--per-tier teeth from different recordings, at random).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

FRAME = 1 / 30
TIERS = ("A", "B", "C", "U", "I", "N")
COLOUR = {"A": "#2e7d32", "B": "#9e9d24", "C": "#ef6c00", "U": "#c62828", "I": "#5e35b1", "N": "#607d8b"}
CONFIRMED = ("slip", "off grid", "other way", "isolated")
STEP = ("step, cannot judge", "step, shape unclear")


def events(R: dict) -> list[dict]:
    """The recording's slip events in time order: t, sign, kind (confirmed, step, uncertain) and, for uncertain ones,
    p and the interval."""
    ev = []
    for s in R.get("slips", []):
        ev.append({"t": s["t"], "sign": s.get("sign", R.get("sign", 1)),
                   "kind": "confirmed" if s["kind"] in CONFIRMED else "step", "what": s["kind"]})
    for u in R.get("uncertain", []):
        if any(abs(e["t"] - u["q50"]) < 30 for e in ev):
            continue
        ev.append({"t": u["q50"], "sign": R.get("sign", 1), "kind": "uncertain", "p": u["p_slip"], "q05": u["q05"],
                   "q95": u["q95"], "what": u["outcome"]})
    return sorted(ev, key=lambda e: e["t"])


def teeth(R: dict) -> list[dict]:
    g0, g1 = R["start"], R["end"]
    saw = bool(R.get("sawtooth"))
    reported = saw and (R.get("fit") or {}).get("creep") == "reported"
    ev = events(R)
    L = (R.get("fit") or {}).get("L")
    sign = R.get("sign", 1)
    if not ev:
        return [{"t0": g0, "t1": g1, "tier": "N", "edges": "", "reason": "no slip"}]
    real = [e for e in ev if e["kind"] != "uncertain" or e["p"] >= 0.5]
    weak = [e for e in ev if e["kind"] == "uncertain" and e["p"] < 0.5]
    edges = [{"t": g0, "kind": "end"}] + real + [{"t": g1, "kind": "end"}]
    out = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        kinds = [e["kind"] for e in (a_, b_) if e["kind"] != "end"]
        if reported:
            tier, reason = "I", "two confirmed slips: tooth %.1f min (%.1f ppm) reported, creep not corrected" % (
                L / 60, FRAME / L * 1e6)
        elif not saw:
            dr = R.get("drift") or {}
            tier, reason = "I", ("isolated slips, no sawtooth" if dr.get("ppm") is None else
                                 "isolated slips, no sawtooth; drift %+.1f ppm (sd %.1f), not corrected" % (dr["ppm"], dr["sd_ppm"]))
        elif "uncertain" in kinds:
            tier, reason = "C", "edge uncertain (p %.2f)" % min(e.get("p", 1) for e in (a_, b_) if e["kind"] == "uncertain")
        elif "step" in kinds:
            tier, reason = "B", "edge a step the shape cannot judge"
        else:
            tier, reason = "A", "edges confirmed" if len(kinds) == 2 else "edge confirmed (end of stretch)"
        inside_weak = [e for e in weak if a_["t"] < e["t"] < b_["t"]]
        other = [e for e in (a_, b_) if e["kind"] != "end" and e.get("sign", sign) != sign]
        if reported:
            pass
        elif saw and inside_weak:
            tier, reason = "U", "a slip with p %.2f inside" % max(e["p"] for e in inside_weak)
        elif saw and other:
            tier, reason = "U", "edge a step the other way"
        elif saw and L and b_["t"] - a_["t"] > 1.5 * L:
            tier, reason = "U", ("%.1f teeth long" if len(kinds) == 2 else "%.1f teeth without a slip at the end") % (
                (b_["t"] - a_["t"]) / L)
        out.append({"t0": a_["t"], "t1": b_["t"], "tier": tier, "reason": reason,
                    "edges": "-".join(e.get("what", "end") for e in (a_, b_))})
    return out


def correction(t: np.ndarray, R: dict, uncertain: bool = False) -> np.ndarray:
    """D(t), the modelled sawtooth delay (s). Slips placed only within an interval (tier C) are left out unless
    ``uncertain`` (the review of the corrections: 3 right, 1 wrong, 5 can't tell)."""
    sign = R.get("sign", 1)
    N = np.zeros(t.size)
    for e in events(R):
        if e["kind"] == "uncertain":
            if e["p"] < 0.5 or not uncertain:
                continue
            ramp = np.clip((t - e["q05"]) / max(e["q95"] - e["q05"], 1e-6), 0, 1)
            N += e["p"] * ramp
        else:
            N += (t >= e["t"]) * (e["sign"] * sign)
    D = sign * FRAME * N
    L = (R.get("fit") or {}).get("L")
    ev = [e for e in events(R) if e["kind"] != "uncertain" or e["p"] >= 0.5]
    if R.get("sawtooth") and L and ev and (R.get("fit") or {}).get("creep") != "reported":
        # the creep only where the sawtooth runs: from a tooth before the first slip to a tooth after the last
        lo, hi = ev[0]["t"] - L, ev[-1]["t"] + L
        D -= sign * FRAME * (np.clip(t, lo, hi) - lo) / L
    return D


def series(z, block: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """The measured ECG-PPG delay, per beat the median of foot, upstroke and peak minus R, in 15-beat medians."""
    r = z["r_t"]
    d = np.nanmedian(np.vstack([z[j] - r for j in ("foot", "up", "peak")]), axis=0)
    t = z["up"]
    m = d.size // block
    tt = np.nanmedian(t[: m * block].reshape(m, block), 1)
    yy = np.nanmedian(d[: m * block].reshape(m, block), 1)
    ok = np.isfinite(tt) & np.isfinite(yy)
    return tt[ok], yy[ok]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-tier", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--correct-uncertain", action="store_true", help="also correct tier C slips (as ramps)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fits = {}
    for line in open(a.fit):
        r = json.loads(line)
        if "real" in r:
            fits[r["case"]] = r
    rows = []
    for c, r in sorted(fits.items()):
        R = r["real"]
        T = teeth(R)
        for k, x in enumerate(T):
            rows.append({"case": c, "port": r.get("port"), "mode": r.get("mode"), "direction": R.get("direction"),
                         "sawtooth": R.get("sawtooth"), "k": k, "n_teeth": len(T), "tier": x["tier"],
                         "t0": round(x["t0"], 2), "t1": round(x["t1"], 2), "minutes": round((x["t1"] - x["t0"]) / 60, 2),
                         "rate_ppm": round((R.get("fit") or {}).get("rate_ppm") or 0, 2) if R.get("sawtooth") else None,
                         "edges": x["edges"], "reason": x["reason"]})
    with open(out / "teeth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    S = {"fit": a.fit, "recordings": len(fits), "teeth": dict(Counter(x["tier"] for x in rows)),
         "minutes": {t: round(sum(x["minutes"] for x in rows if x["tier"] == t), 1) for t in TIERS},
         "reasons": {t: dict(Counter(_generic(x["reason"]) for x in rows if x["tier"] == t).most_common()) for t in TIERS},
         "tier_i_drift_ppm": _drift_summary(fits),
         "recordings_by_worst_tier": dict(Counter(max((x["tier"] for x in rows if x["case"] == c),
                                                     key=lambda t: "AIBCNU".index(t)) for c in fits)),
         "examples": {}}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(a.seed)
    for tier in TIERS:
        pool = [x for x in rows if x["tier"] == tier]
        chosen, seen = [], set()
        for i in rng.permutation(len(pool)):
            x = pool[i]
            if x["case"] in seen:
                continue
            chosen.append(x)
            seen.add(x["case"])
            if len(chosen) == a.per_tier:
                break
        S["examples"][tier] = [{k: x[k] for k in ("case", "k", "t0", "t1", "reason", "edges")} for x in chosen]
        if not chosen:
            continue
        nrow = (len(chosen) + 1) // 2
        fig, axs = plt.subplots(nrow, 2, figsize=(15, 2.9 * nrow), squeeze=False)
        for ax, x in zip(axs.ravel(), chosen):
            R = fits[x["case"]]["real"]
            z = np.load(Path(a.scan) / f"{x['case']:04d}.npz")
            tt, yy = series(z)
            g0 = R["start"]
            D = correction(tt, R, a.correct_uncertain)
            cor = yy - D
            L = x["t1"] - x["t0"]
            pad = max(0.25 * L, 300.0)
            sel = (tt >= x["t0"] - pad) & (tt <= x["t1"] + pad)
            if sel.sum() < 3:
                sel = np.ones(tt.size, bool)
            inside = (tt >= x["t0"]) & (tt <= x["t1"])
            ref = np.median(yy[inside]) if inside.any() else np.median(yy[sel])
            cor_s = cor - np.median(cor[inside] if inside.any() else cor[sel]) + ref
            m = lambda v: (np.asarray(v) - g0) / 60
            ax.axvspan(m(x["t0"]), m(x["t1"]), color=COLOUR[tier], alpha=0.12, lw=0)
            ax.scatter(m(tt[sel]), yy[sel] * 1000, s=5, color="C0", label="measured ECG-PPG delay (15-beat medians)")
            ax.scatter(m(tt[sel]), cor_s[sel] * 1000, s=5, color="C3", label="after correction (same median)")
            lo, hi = tt[sel].min(), tt[sel].max()
            for e in events(R):
                if not lo <= e["t"] <= hi:
                    continue
                if e["kind"] == "uncertain":
                    ax.axvspan(m(e["q05"]), m(e["q95"]), color="0.5", alpha=0.25 if e["p"] >= 0.5 else 0.1, lw=0)
                    ax.text(m(e["t"]), 0.97, f"p {e['p']:.2f}", transform=ax.get_xaxis_transform(), fontsize=7,
                            ha="center", va="top")
                else:
                    ax.axvline(m(e["t"]), color="0.45", lw=0.8, ls="-" if e["kind"] == "confirmed" else "--")
            y = yy[sel] * 1000
            mid = np.median(y)
            span = max(np.percentile(y, 98) - np.percentile(y, 2), 40.0)
            ax.set_ylim(mid - 0.75 * span, mid + 0.75 * span)
            rate = f"{x['rate_ppm']:.1f} ppm" if x["rate_ppm"] else "no sawtooth"
            ax.set_title(f"case {x['case']}, tooth {x['k'] + 1} of {x['n_teeth']}: tier {tier} - {x['reason']}\n"
                         f"{R.get('direction')}, {rate}; {x['minutes']:.0f} min; edges {x['edges']}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("minutes into the analysed stretch", fontsize=7)
            ax.set_ylabel("delay (ms)", fontsize=7)
        for ax in axs.ravel()[len(chosen):]:
            ax.set_visible(False)
        axs.ravel()[0].legend(fontsize=7, loc="upper left", markerscale=2)
        fig.suptitle(f"Tier {tier}: {len(pool)} teeth in all; {len(chosen)} examples from different recordings (shaded: "
                     f"the tooth; lines: confirmed slips, dashed = steps the shape cannot judge; grey bands: uncertain "
                     f"slips, 5-95%)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"tier-{tier}.png", dpi=90)
        plt.close(fig)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: S[k] for k in ("recordings", "teeth", "minutes", "reasons", "recordings_by_worst_tier")}, indent=1))


def _drift_summary(fits: dict) -> dict | None:
    v = [r["real"]["drift"]["ppm"] for r in fits.values() if (r["real"].get("drift") or {}).get("ppm") is not None]
    if not v:
        return None
    return {"recordings": len(v), "quartiles": [round(float(x), 1) for x in np.percentile(v, [25, 50, 75])],
            "over_2sd": sum(1 for x in v if abs(x) > 13.0)}


def _generic(reason: str) -> str:
    import re
    return re.sub(r"\s*\(?(sd )?[-+]?\d+(\.\d+)?( ppm)?\)?", "", reason).replace("  ", " ").strip()


if __name__ == "__main__":
    main()
