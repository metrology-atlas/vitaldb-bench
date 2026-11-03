# SPDX-License-Identifier: MIT
"""Example teeth of each confidence tier of physiosync's ECG-to-PPG sawtooth fit, for looking at (metrology-atlas
study ppg-sawtooth). Reads the per-case output of tools/ppg_fix_check.py (run with --apply-min-tier C, so every
fitted tooth is drawn corrected, and with tooth_edges_t and tier_reasons in its output).

    python tools/tier_gallery.py --scan $W/ppg_correction_check/check_tierC_v2.jsonl --per-tier 8 --seed 0 \
        --out $W/tier_gallery

Per tier (A, B, C, N, U), --per-tier teeth from different recordings, drawn at random (--seed). Each panel: the
measured PAT (15-beat medians, blue) around the tooth, the PAT after the sawtooth correction (red, shifted to the
same median), the tooth shaded, detected slips as grey lines and inserted slips as dashed lines; the title gives
the recording, the tooth's tier and reason, its rate and length. Writes tier-<X>.png and summary.json (teeth per
tier, reasons per tier, the examples drawn).
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np

TIERS = ("A", "B", "C", "N", "U")
COLOUR = {"A": "#2e7d32", "B": "#9e9d24", "C": "#ef6c00", "N": "#607d8b", "U": "#c62828"}


def teeth(rows):
    for r in rows:
        e = r.get("tooth_edges_t") or []
        for k, t in enumerate(r.get("tiers") or []):
            if k + 1 < len(e):
                yield {"case": r["case"], "k": k, "tier": t, "t0": e[k], "t1": e[k + 1],
                       "reason": (r.get("tier_reasons") or [""] * (k + 1))[k] if k < len(r.get("tier_reasons") or []) else "",
                       "rate": (r.get("rates_ppm") or [None] * (k + 1))[k] if k < len(r.get("rates_ppm") or []) else None,
                       "n_teeth": len(r["tiers"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--per-tier", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in Path(a.scan).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "series" in r and r.get("tooth_edges_t")]
    by = {r["case"]: r for r in rows}
    all_teeth = list(teeth(rows))
    rng = np.random.default_rng(a.seed)
    S = {"scan": a.scan, "recordings": len(rows), "teeth": dict(Counter(t["tier"] for t in all_teeth)),
         "reasons": {}, "examples": {}}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for tier in TIERS:
        pool = [t for t in all_teeth if t["tier"] == tier]
        S["reasons"][tier] = dict(Counter(_generic(t["reason"]) for t in pool).most_common())
        order = rng.permutation(len(pool))
        chosen, seen = [], set()
        for i in order:
            t = pool[i]
            if t["case"] in seen:
                continue
            chosen.append(t)
            seen.add(t["case"])
            if len(chosen) == a.per_tier:
                break
        S["examples"][tier] = [{k: t[k] for k in ("case", "k", "t0", "t1", "reason", "rate")} for t in chosen]
        if not chosen:
            continue
        nrow = (len(chosen) + 1) // 2
        fig, axs = plt.subplots(nrow, 2, figsize=(15, 3.2 * nrow), squeeze=False)
        for ax, t in zip(axs.ravel(), chosen):
            r = by[t["case"]]
            tt, raw, cor = (np.array(v, float) for v in r["series"])
            L = t["t1"] - t["t0"]
            pad = max(0.25 * L, 300.0)
            sel = (tt >= t["t0"] - pad) & (tt <= t["t1"] + pad)
            if sel.sum() < 3:
                sel = np.ones(tt.size, bool)
            inside = (tt >= t["t0"]) & (tt <= t["t1"])
            ref = np.median(raw[inside]) if inside.any() else np.median(raw[sel])
            cor_s = cor - np.median(cor[inside] if inside.any() else cor[sel]) + ref
            ax.axvspan(t["t0"] / 60, t["t1"] / 60, color=COLOUR[tier], alpha=0.12, lw=0)
            ax.scatter(tt[sel] / 60, raw[sel], s=5, color="C0", label="measured PAT (15-beat medians)")
            ax.scatter(tt[sel] / 60, cor_s[sel], s=5, color="C3", label="after correction (same median)")
            lo, hi = (tt[sel].min(), tt[sel].max())
            for x in r.get("slips_t") or []:
                if lo <= x <= hi:
                    ax.axvline(x / 60, color="0.6", lw=0.8, ls="--" if x in (r.get("inserted_t") or []) else "-")
            y = raw[sel]
            mid = np.median(y)
            span = max(np.percentile(y, 98) - np.percentile(y, 2), 40.0)
            ax.set_ylim(mid - 0.75 * span, mid + 0.75 * span)
            rate = f"{t['rate']:.1f} ppm" if t["rate"] is not None else "no rate"
            head = f"case {t['case']}, tooth {t['k'] + 1} of {t['n_teeth']}, tier {tier}: {rate}, {L / 60:.0f} min"
            ax.set_title(head + "\n" + textwrap.fill(t["reason"], 95), fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("minutes into the analysed stretch", fontsize=7)
            ax.set_ylabel("PAT (ms)", fontsize=7)
        for ax in axs.ravel()[len(chosen):]:
            ax.set_visible(False)
        axs.ravel()[0].legend(fontsize=7, loc="upper left", markerscale=2)
        fig.suptitle(f"Tier {tier}: {len(pool)} teeth in all; {len(chosen)} examples from different recordings "
                     f"(shaded: the tooth; grey lines: slips, dashed = inserted)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"tier-{tier}.png", dpi=90)
        plt.close(fig)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: S[k] for k in ("recordings", "teeth", "reasons")}, indent=1))


def _generic(reason: str) -> str:
    """The reason without its numbers, for counting ('rate 50.7 ppm corroborated' -> 'rate corroborated')."""
    return re.sub(r"\s+", " ", re.sub(r"[-+]?\d+(\.\d+)?\s*(ppm|ms|min)?", "#", reason)).strip()


if __name__ == "__main__":
    main()
