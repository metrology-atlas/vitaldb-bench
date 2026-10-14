# SPDX-License-Identifier: MIT
"""Conceptual figure: VitalDB devices and the timing relationships measured between them (aggregates only).

    python tools/device_map_figure.py out.svg [out.png]

Numbers are the aggregate results recorded in the "Device clock drift in VitalDB" doc (observations 22-28);
edit NODES/EDGES to change the layout or text.
"""

from __future__ import annotations

import sys
from xml.sax.saxutils import escape

W, H = 1700, 1160
FONT = "Helvetica, Arial, sans-serif"
CONF = {"high": ("#1b7f4b", "#e6f4ec"), "moderate": ("#b26b00", "#fdf1dd"), "low": ("#6b6b6b", "#f0f0f0"),
        "none": ("#9a9a9a", "#ffffff")}

# name: (x, y, w, h, title, subtitle, lines)
NODES = {
    "snuadc": (30, 110, 530, 470, "SNUADC", "500 Hz converter: ECG II/V5, PLETH, ART, CVP, FEM", [
        ("b", "One clock: all channels share identical stamps"),
        ("", "Stamps follow the sample count between corrections"),
        ("", "Rate vs recorder: \u221262 ppm median (p10\u2013p90 \u221271 to +7)"),
        ("", "Step mode: corrections in clusters (\u221280/\u2212120 ms) \u2248 \u2212480 ms"),
        ("", "Small mode: frequent small corrections, stamps \u2248 recorder"),
    ]),
    "bis": (1230, 110, 440, 330, "BIS", "128 Hz: EEG1, EEG2", [
        ("b", "One clock: EEG1 and EEG2 identical"),
        ("", "Stamps follow the sample count between corrections"),
        ("", "Rate vs recorder: +23 ppm median (p10–p90 +17 to +47)"),
        ("", "Corrections: single forward steps +155 ms"),
        ("", "(p10–p90 +89 to +296 ms)"),
        ("b", "Averaged cardiac artefact in the EEG"),
        ("", "(strong in 13/30 cases; single beats too weak for IBI)"),
    ]),
    "primus": (700, 890, 440, 240, "Primus ventilator", "62.5 Hz: AWP, CO2", [
        ("b", "No drift corrections in the stamps"),
        ("", "Re-anchored only at gaps (AWP: ~20 s gap every ~2 h)"),
        ("", "Stretches between gaps: 39–120 min (median 93)"),
        ("", "Drift visible only against SNUADC (CVP)"),
        ("", "CO2: sidestream delay ≥ 1 s, not a timing reference"),
    ]),
    "solar": (30, 960, 380, 170, "Solar8000 monitor", "Numerics (HR, NIBP, SpO2, ...)", [
        ("", "Single values, no sample count:"),
        ("", "stamp model not applicable"),
        ("", "Update interval ~1–2 s"),
    ]),
}

# nested sub-nodes: (parent, x, y, w, conf, title, lines)
SUBNODES = [
    ("snuadc", 50, 300, 490, "high", "PLETH sub-clock:  ECG vs PLETH (same converter)", [
        ("Mechanism", "oximeter output updated in 30 Hz frames, then sampled at 500 Hz"),
        ("Relative drift", "frame creep 18\u201352 ppm per setup; ~constant within a"),
        ("", "recording (CV 0.3%), \u00b10.3\u20131.3% between cases of a setup"),
        ("Correction period", "one-frame (33.3 ms) slip every ~11\u201331 min (a tooth)"),
        ("Offset", "raw PAT (foot) median 652 ms (IQR 640\u2013667); sawtooth 0\u201333 ms"),
        ("", "hardware delay 368\u2013399 ms per setup (adopted PAT open)"),
        ("Detection method", "R peak \u2192 PPG foot PAT, sawtooth fit and slip detection"),
        ("Evidence", "ABP reference: 97\u2013100% of detected slips real; mid-tooth"),
        ("", "drift 1.9/3.6/5.0 ms (tiers A/B/C) vs ~15 ms uncorrected"),
        ("Confidence", "HIGH for A/B teeth; no slips (pauses) in 32 recordings"),
    ]),
]

# (a, b, conf, style, label anchor (x, y), box width, title, lines)
EDGES = [
    ("snuadc", "bis", "high", "solid", (900, 300), 640, "SNUADC vs BIS   (EEG minus ECG stamps)", [
        ("Relative drift", "−72 ppm median (IQR −82 to −54) ≈ −0.26 s/h"),
        ("", "SNUADC step mode ≈ −80 ppm, small mode ≈ −50 ppm"),
        ("Correction period", "irregular, at stamp corrections; teeth 10–107 min (typically 30–90)"),
        ("Offset, raw stamps", "mean −0.79 s; p5–p95 −1.05 to −0.53 s; resets +0.07 to +0.44 s"),
        ("After stamp model", "constant −0.80 s (step) / −0.69 s (small); within-case SD 25 ms"),
        ("Detection method", "shared electrocautery/EMG bursts, ECG 30–60 Hz × EEG 30–55 Hz"),
        ("Evidence", "direct shared artefact; ramps match clock rates (r 0.88);"),
        ("", "79% of jumps at stamp steps (12% by chance); EEG cardiac artefact agrees (−16 ms)"),
        ("Confidence", "HIGH: 167/191 cases; ±11 ms per window; 33 ms without bursts"),
    ]),
    ("snuadc", "primus", "moderate", "solid", (860, 690), 600, "SNUADC vs Primus   (CVP response minus AWP breath onset)", [
        ("Relative drift", "+141 ppm median (31 to 200); not constant within a case"),
        ("Correction period", "none between Primus gaps; stretches 39–120 min"),
        ("Offset", "median 1.52 s (p5–p95 0.43–4.0 s; min −3.8, max 4.9 s)"),
        ("Note", "offset includes the physiological AWP→CVP delay"),
        ("Detection method", "breath-onset-triggered AWP/CVP averages + breath matching"),
        ("Evidence", "physiological coupling (ventilation → CVP); SBP and R amplitude agree"),
        ("Confidence", "MODERATE: pilot, 20 cases; 53% of windows breath-confirmed; ±49 ms (p90 165)"),
    ]),
    ("bis", "primus", "none", "dash", (1400, 660), 440, "BIS vs Primus", [
        ("Measured", "no: no shared signal found"),
        ("Possible", "chain through SNUADC (both measured against it)"),
        ("Confidence", "NONE (inferred only)"),
    ]),
    ("snuadc", "solar", "low", "dot", (220, 820), 380, "SNUADC vs Solar8000", [
        ("Anchors", "NIBP cuff occlusions seen in ART;"),
        ("", "NIBP result recorded 20–40 s later"),
        ("Resolution", "seconds; not quantified"),
        ("Confidence", "LOW"),
    ]),
]


def centre(n):
    x, y, w, h = NODES[n][:4]
    return x + w / 2, y + h / 2


def text(x, y, s, size=13, weight="normal", fill="#222", anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')


def build() -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           text(W / 2, 40, "VitalDB devices and their measured timing relationships", 24, "bold", anchor="middle"),
           text(W / 2, 66, "Offsets are 'event stamp in the second device minus in the first'. Rates are relative drift "
                "of the stamps (ppm). Recorder-clock behaviour is inside each box.", 13, fill="#555", anchor="middle")]
    dash = {"solid": "", "dash": ' stroke-dasharray="12 8"', "dot": ' stroke-dasharray="3 6"'}
    for a, b, conf, style, _, _, _, _ in EDGES:
        (x1, y1), (x2, y2) = centre(a), centre(b)
        col = CONF[conf][0]
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="5"{dash[style]}/>')
    for n, (x, y, w, h, title, sub, lines) in NODES.items():
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#f6f8fb" stroke="#33415c" stroke-width="2"/>')
        out.append(text(x + 18, y + 34, title, 22, "bold", "#1d2b4f"))
        out.append(text(x + 18, y + 56, sub, 13, fill="#555"))
        yy = y + 86
        for kind, s in lines:
            out.append(text(x + 18, yy, s, 13.5, "bold" if kind == "b" else "normal"))
            yy += 24 if kind == "b" else 21
    for parent, x0, y0, bw, conf, title, lines in SUBNODES:
        col, bg = CONF[conf]
        bh = 40 + 21 * len(lines)
        out.append(f'<rect x="{x0}" y="{y0}" width="{bw}" height="{bh}" rx="10" fill="{bg}" stroke="{col}" stroke-width="2" stroke-dasharray="6 4"/>')
        out.append(text(x0 + 14, y0 + 24, title, 14.5, "bold", col))
        yy = y0 + 48
        for k, v in lines:
            out.append(f'<text x="{x0 + 14:.1f}" y="{yy:.1f}" font-family="{FONT}" font-size="12.5" fill="#222">'
                       + (f'<tspan font-weight="bold">{escape(k)}: </tspan>' if k else '') + f'{escape(v)}</text>')
            yy += 21
    for a, b, conf, style, (cx, cy), bw, title, lines in EDGES:
        col, bg = CONF[conf]
        bh = 40 + 21 * len(lines)
        x0, y0 = cx - bw / 2, cy - bh / 2
        out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw}" height="{bh}" rx="10" fill="{bg}" stroke="{col}" stroke-width="2"/>')
        out.append(text(x0 + 14, y0 + 24, title, 14.5, "bold", col))
        yy = y0 + 48
        for k, v in lines:
            out.append(f'<text x="{x0 + 14:.1f}" y="{yy:.1f}" font-family="{FONT}" font-size="12.5" fill="#222">'
                       + (f'<tspan font-weight="bold">{escape(k)}: </tspan>' if k else '') + f'{escape(v)}</text>')
            yy += 21
    # legend
    lx, ly = 1260, 950
    out.append(text(lx, ly, "Confidence", 14, "bold"))
    for i, (k, st) in enumerate((("high", "solid"), ("moderate", "solid"), ("low", "dot"), ("none", "dash"))):
        y = ly + 26 + i * 26
        out.append(f'<line x1="{lx}" y1="{y - 5}" x2="{lx + 60}" y2="{y - 5}" stroke="{CONF[k][0]}" stroke-width="5"{dash[st]}/>')
        out.append(text(lx + 72, y, k.upper() if k != "none" else "NONE (not measured)", 13))
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    svg = build()
    open(sys.argv[1], "w").write(svg)
    if len(sys.argv) > 2:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode(), write_to=sys.argv[2], output_width=W)
