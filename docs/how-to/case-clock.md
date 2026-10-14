# Recover a case's clock time of day

VitalDB moves every recording to a fictional date (2100-01-01) and stores times relative to `casestart`.
The time of day is still recoverable from `clinical_data.csv`, and some questions need it (shift patterns,
night cases, time since midnight). `vitaldb_bench.case_clock` recovers it; the date stays hidden.

## How it works

1. **The minute.** `adm` (the admission date, midnight) is stored relative to `casestart` truncated to the
   minute, so the case start's clock minute is `(-adm) mod 86400`.
2. **The seconds.** The EMR times `anestart`, `aneend`, `opstart` and `opend` are on whole minutes of the
   EMR clock. The seconds past the minute at `casestart` are `(-t) mod 60` for the phase that at least three
   of the four share.

A case gets no clock when `adm` is not on a whole minute or fewer than three EMR times agree.

## In Python

```python
from vitaldb_bench.case_clock import shift_seconds, time_of_day

# a case that started at 07:43:12, admitted two days earlier (the values clinical_data.csv would hold)
adm = -(2 * 86400 + 7 * 3600 + 43 * 60)
row = {"adm": str(adm), "anestart": "-552", "aneend": "10848", "opstart": "1668", "opend": "10368"}

r = time_of_day(row)
assert r["status"] == "ok" and r["events_agree"] == 4
h, rem = divmod(r["tod_s"], 3600)
print(f"{h:02d}:{rem // 60:02d}:{rem % 60:02d}")        # 07:43:12

# seconds to add to the case's stored times so they read as clock time (Korea, UTC+9)
print(shift_seconds(r["tod_s"], 9))
```

For a whole release, `clock_table("clinical_data.csv")` returns the result for every case.

## In an AtriumDB build

The AtriumDB builder can anchor each case at its clock time of day. This is **off by default**:

```bash
python -m vitaldb_bench_atriumdb.build atriumdb --root "$VITALDB_ROOT" --cases cases.json --out build/adb-clock \
    --wall-clock --tz-hours 9
```

The shifts are written to `dataset/case_clock.json`, and readers and the label pass apply them. The choice is
fixed per build: start a new `--out` to switch.

!!! note "Which clock"
    The recovered time is on the EMR clock (local time, KST). Its offset from the recorder's clock is not
    known.
