# Ad hoc helpers

Short aggregation and plotting scripts written interactively during the VitalDB timing audit. They are
kept because numbers or figures in the drift document came from them, not because they are maintained
tools. Each has a one-line docstring saying what it read and what it produced.

- They read per-case JSONL files (e.g. `zero206.jsonl`, `sa191.jsonl`, `vent20.jsonl`) written by the
  main tools in `tools/`. Those files hold per-case results and stay on the analysis machine; run a
  script from the folder that holds them.
- Paths to the VitalDB release and the work folder come from `VITALDB_ROOT` and `VITALDB_WORK`
  (see `scripts/local.env.example`).
- `hr_group_ports/` is the analysis behind the HR delay-group split by recording PC
  (`results/pat/hr_port_split_summary.json`): `fs.py` and `hr_port.py` build `rows.json`; `an*.py` query it.
