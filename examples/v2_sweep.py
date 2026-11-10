"""v2 sweep (10 Nov 2026): for every atlas item, the saved outputs behind it and the run that made them.

After the stage-3 rate changes (v4 SNUADC rates, model5), list each parameter, observation, link and study with the
outputs it cites (metrology-atlas tools/output_manifest.py), and for each output: the folder, the tool it is
presumably from, a version tag in its path (_v3, _v4, _m5 ...), and the newest modification time of its files.
A parameter that cites no output of its own gets its study's outputs (marked "via study"). This is a worklist for a
manual review, not a verdict: which tool made a folder, and which rates it used, are read off names and dates.

Each folder also gets the vitaldb-bench scripts that name it (a "$W/<folder>" or "work/<folder>" path, or the
folder name quoted), and whether those scripts use the SNUADC/BIS fallback rates (a rates file, fallback_rate,
port rates, a stamp-model variant) or the stamp model only (stamp_clock, corrected_times, sa_corrected), directly
or through an import of another vitaldb-bench script. Writes items_outputs.csv and folders.csv.

Per study, its code list (vitaldb-bench scripts) is checked the same way, and every parameter, observation and link
inherits its study's flags (studies.csv, items.csv): the candidates for a rerun are the items whose study uses the
fallback rates and whose date is before the v4 rates (9 Nov 2026 evening). Scripts named anywhere in a study's or
item's text (tools/NAME.py, examples/NAME.py, or a bare NAME.py) count as well as the code list, since many
studies and parameters name their script only in the text (text_fallback, text_stamp_only).

  python examples/v2_sweep.py --atlas ../../metrology-atlas/datasets/vitaldb --manifest OUTPUTS.json --work $W \
      --bench . --out $W/v2_sweep_2026-11-10
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
from pathlib import Path

import yaml

TAG = re.compile(r"_(v\d+[a-z]?)\b|_(m\d+[a-z]?)\b|(model\d+)|_(r\d+)\b|_(t\d+)\b")
# Tools that take the SNUADC (and BIS) fallback rates (runbooks/2026-11-09_setup_fallback_rerun.md, stage 3 list)
RATE_TOOLS = {
    "hr_derivation", "numeric_recompute_lag", "numeric_delay_variability", "numeric_derivation_pilot",
    "numeric_clock_check", "spo2_event_pilot", "primus_sync_accuracy", "primus_breath_id", "primus_vent_events",
    "primus_pause_steps", "primus_ptt_arm", "primus_awp_anchor_figs", "primus_drift_probe", "airway_heartbeat_pilot",
    "vent_sync_pilot", "vent_match_check", "ppv_by_breath", "loop_closure_art", "solar_snuadc_clock",
    "rare_device_sync", "vigilance_window", "eeg_cardiac_recheck", "clock_model_fit",
}
FOLDER_TOOL = {"overnight/spo2": "spo2_event_pilot", "overnight/numeric": "numeric_derivation_pilot",
               "overnight_v4/spo2": "spo2_event_pilot", "overnight_v4/numeric": "numeric_derivation_pilot"}


def newest(p: Path):
    if p.is_file():
        return p.stat().st_mtime
    ts = [f.stat().st_mtime for f in p.rglob("*") if f.is_file()] if p.is_dir() else []
    return max(ts) if ts else None


def folder_of(path: str) -> str:
    parts = path.split("/")
    if parts[0] in ("examples", "results", "overnight", "overnight_v4", "overnight_v4b") and len(parts) > 2:
        return "/".join(parts[:2])
    return parts[0] if len(parts) > 1 else path


def tool_of(folder: str) -> str:
    if folder in FOLDER_TOOL:
        return FOLDER_TOOL[folder]
    base = folder.split("/")[-1]
    base = TAG.sub("", base)
    return base


FALLBACK = re.compile(r"step_rates|step-rates|load_step_rates|fallback_rate|port_rates|bis_rates|bis-rates|--variant|model[345]")
STAMP = re.compile(r"stamp_clock|corrected_times|sa_corrected|snu_times|clock_model")


def script_flags(bench: Path):
    files = sorted(glob.glob(str(bench / "tools" / "*.py")) + glob.glob(str(bench / "examples" / "*.py")))
    src = {Path(f).stem: Path(f).read_text(encoding="utf-8", errors="replace") for f in files}
    src.pop("v2_sweep", None)
    imp = {m: set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", t, re.M)) & set(src) for m, t in src.items()}

    def closure(seed):
        S, changed = set(seed), True
        while changed:
            changed = False
            for m, i in imp.items():
                if m not in S and i & S:
                    S.add(m)
                    changed = True
        return S
    fb = closure({m for m, t in src.items() if FALLBACK.search(t)})
    st = closure({m for m, t in src.items() if STAMP.search(t)})
    return src, fb, st


def producers(folder: str, src: dict):
    pats = [f"$W/{folder}", f"work/{folder}", f'"{folder}"', f"'{folder}'"]
    return sorted(m for m, t in src.items() if any(p in t for p in pats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--bench", required=True, help="vitaldb-bench checkout")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, FB, ST = script_flags(Path(a.bench))
    A, W, O = Path(a.atlas), Path(a.work), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    M = json.load(open(a.manifest))
    cites = {}
    for e in M["files"] + M["folders"]:
        for item in e["cited_by"]:
            cites.setdefault(item, set()).add(e["path"])
    items = []
    for kind in ("parameters", "observations", "links", "studies"):
        for f in sorted((A / kind).glob("*.yaml")):
            d = yaml.safe_load(f.read_text(encoding="utf-8"))
            items.append((kind, d))
    study_of = {f"parameters/{d['id']}": d.get("study") for k, d in items if k == "parameters"}
    rows = []
    for kind, d in items:
        key = f"{kind}/{d['id']}"
        own = sorted(cites.get(key, []))
        via = []
        if not own and kind == "parameters" and d.get("study"):
            via = sorted(cites.get(f"studies/{d['study']}", []))
        for path, how in [(p, "own") for p in own] + [(p, "via study") for p in via] or [("", "none")]:
            fold = folder_of(path) if path else ""
            t = newest(W / path) if path and (W / path).exists() else None
            m = TAG.search(path) if path else None
            rows.append({
                "item": key, "status": d.get("status"), "date": str(d.get("date", "")), "study": d.get("study", ""),
                "value": d.get("value", ""), "cites": how, "output": path, "folder": fold,
                "tool": tool_of(fold) if fold else "", "uses_rates": (tool_of(fold) in RATE_TOOLS) if fold else "",
                "tag": next((g for g in m.groups() if g), "") if m else "",
                "output_mtime": dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else "",
            })
    with open(O / "items_outputs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    folders = {}
    for r in rows:
        if r["folder"]:
            f = folders.setdefault(r["folder"], {"folder": r["folder"], "tag": r["tag"], "newest": "", "items": set()})
            f["newest"] = max(f["newest"], r["output_mtime"])
            f["items"].add(r["item"])
    with open(O / "folders.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["folder", "tag", "newest", "n_items", "producers", "fallback", "stamp_only", "items"])
        for k in sorted(folders):
            f = folders[k]
            pr = producers(k, src)
            w.writerow([k, f["tag"], f["newest"], len(f["items"]), " ".join(pr),
                        " ".join(p for p in pr if p in FB), " ".join(p for p in pr if p in ST and p not in FB),
                        " ".join(sorted(f["items"]))])
    print(len(folders), "folders ->", O / "folders.csv")
    # studies: code lists against the script flags
    S = {d["id"]: d for k, d in items if k == "studies"}
    srow = {}
    for sid, d in S.items():
        mods = [Path(c).stem for c in d.get("code") or [] if str(c).startswith("vitaldb-bench/")]
        srow[sid] = {"study": sid, "status": d.get("status"), "date": str(d.get("date", "")),
                     "fallback": " ".join(m for m in mods if m in FB),
                     "stamp_only": " ".join(m for m in mods if m in ST and m not in FB),
                     "other_code": " ".join(str(c) for c in d.get("code") or [] if not str(c).startswith("vitaldb-bench/"))[:200]}
    with open(O / "studies.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(srow.values()))))
        w.writeheader()
        w.writerows(srow[k] for k in sorted(srow))
    NAMED = re.compile(r"(?:tools|examples)/(\w+)\.py|\b(\w+)\.py\b")

    def named(d):
        t = yaml.safe_dump(d, allow_unicode=True)
        return {a or b for a, b in NAMED.findall(t)} & set(src)
    text_mods = {sid: named(d) for sid, d in S.items()}
    irows = []
    for kind, d in items:
        if kind == "studies":
            continue
        st = d.get("study") or ""
        sts = [st] if isinstance(st, str) else list(st)
        fb = " ".join(srow[x]["fallback"] for x in sts if x in srow).strip()
        so = " ".join(srow[x]["stamp_only"] for x in sts if x in srow).strip()
        tm = named(d).union(*[text_mods.get(x, set()) for x in sts])
        tfb = " ".join(sorted(m for m in tm if m in FB))
        tso = " ".join(sorted(m for m in tm if m in ST and m not in FB))
        irows.append({"item": f"{kind}/{d['id']}", "status": d.get("status"), "date": str(d.get("date", "")),
                      "study": " ".join(sts), "value": d.get("value", ""), "study_fallback": fb, "study_stamp_only": so,
                      "text_fallback": tfb, "text_stamp_only": tso,
                      "candidate": bool(fb or tfb) and str(d.get("date", "")) < "2026-11-10"})
    with open(O / "items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(irows[0]))
        w.writeheader()
        w.writerows(irows)
    print(sum(r["candidate"] for r in irows), "candidate items of", len(irows), "->", O / "items.csv")
    n = {}
    for r in rows:
        n.setdefault(r["cites"], set()).add(r["item"])
    print({k: len(v) for k, v in n.items()}, "rows", len(rows), "->", O / "items_outputs.csv")


if __name__ == "__main__":
    main()
