"""The HR delay parameters of the atlas (study hr-delay-groups) from a set of delay labels (11 Nov 2026).

Until now these were read off the label files by hand. From tools/numeric_delay_labels.py's numeric_delay_labels.csv
(Solar8000/HR rows) and hr_delay_group_labels.csv:

  hr-label-delay       median, p5, p95 and n of the per-case HR delay (all HR rows)
  hr-label-se          median, p10, p90 of the standard errors of the HR rows
  hr-early/late-label-delay   median, p5, p10, p90, p95 of the delay in each group
  hr-early/late-delay-rsd     robust SD (IQR / 1.349) of the delays in each group, and n
  hr-delay-early/late  the two-component mixture's means and weights (the groups' file, its "mixture" column),
                       and each group's share of the classified (early + late) cases

    python examples/hr_label_params.py LABELS_DIR [--out params.json]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = Path(a.labels)
    hr = [r for r in csv.DictReader(open(d / "numeric_delay_labels.csv")) if r["track"] == "Solar8000/HR"]
    x = np.array([float(r["delay_from_window_end_s"]) for r in hr])
    se = np.array([float(r["se_s"]) for r in hr])
    G = list(csv.DictReader(open(d / "hr_delay_group_labels.csv")))
    mix = json.loads(G[0]["mixture"])
    out = {"hr-label-delay": {"median": round(float(np.median(x)), 3), "p5": round(float(np.percentile(x, 5)), 3),
                              "p95": round(float(np.percentile(x, 95)), 3), "n": int(x.size)},
           "hr-label-se": {"median": round(float(np.median(se)), 3), "p10": round(float(np.percentile(se, 10)), 3),
                           "p90": round(float(np.percentile(se, 90)), 3), "n": int(se.size)},
           "mixture": mix}
    n_class = sum(r["group"] in ("early", "late") for r in G)
    for g in ("early", "late"):
        v = np.array([float(r["delay_from_window_end_s"]) for r in G if r["group"] == g])
        q1, q3 = np.percentile(v, [25, 75])
        out[g] = {"n": int(v.size), "label_delay_median": round(float(np.median(v)), 3),
                  "label_delay_p5_p10_p90_p95": [round(float(np.percentile(v, q)), 3) for q in (5, 10, 90, 95)],
                  "rsd": round(float((q3 - q1) / 1.349), 3), "share_of_classified": round(v.size / n_class, 4)}
    out["early"]["mixture_mu"], out["late"]["mixture_mu"] = round(mix["mu"][0], 3), round(mix["mu"][1], 3)
    out["early"]["mixture_w"], out["late"]["mixture_w"] = round(mix["w"][0], 3), round(mix["w"][1], 3)
    out["classified"] = n_class
    out["uncertain"] = sum(r["group"] == "uncertain" for r in G)
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
