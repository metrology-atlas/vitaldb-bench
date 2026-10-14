# Availability: when could a value have been known?

A model that predicts an event at time `t` may only use information that existed before `t`. In a
recorded dataset this is easy to break by accident, because a value's time stamp is not always the time
it became known, nor the time it describes. The **availability rule**: *no value may be used before it
could have been known.*

## Numerics describe a window that ended before the stamp

A monitor numeric (HR, SpO2, ETCO2, the BIS index, ...) is computed from a window of signal. VitalDB stamps
it when the monitor reports it, which is after that window ended. For example, the BIS index with its 15 s
smoothing is a trailing mean over about 16 s of EEG ending about 3 s before the stamp, and Solar8000 HR
falls in one of two delay groups about one 2 s record apart.

So each numeric has two times: the end of the window it describes, and its stamp. The stamp is when it was
known; the window is what it means. Aligning a numeric with a waveform by its stamp misplaces it by the
delay; using it before its stamp leaks information from the future. This audit stamps every derived value
at the **end of its window** (window-end stamping) and records the delay separately.

## `available_at`

Every label written into a build carries `available_at`, the earliest point its information could have been
known:

| Value | Known at | Used for |
|---|---|---|
| `span_end` | The end of the label's own span. | Stretches (e.g. a value at a range limit) and clinical-time intervals. |
| `track_end` | The end of the track. | Whole-track statistics such as `value_resolution`. |
| `case_end` | The end of the case. | Anything from a whole-recording scan or a per-case fit: every derived label family except `lab_time` (line accesses, delay groups, sawtooth fits, zeroing, ...). |
| `timing_model` | Before the case: population constants. | `numeric_lag`, the measured delay of each numeric. |
| `lab_time_plus_turnaround` | The lab timestamp plus an unrecorded turnaround. | `lab_time`, VitalDB's lab timestamps. |

A label with `available_at = case_end` describes the case; it may be used to *select* or *stratify* cases
offline, but not as an input to a model running during the case.

## The lab timestamp is not the result time

VitalDB's lab table gives each result a time `dt`. That time marks the specimen, not the result:

- **Blood gases** (point-of-care): `dt` falls about 3–12 min after the blood draw found in the arterial
  pressure waveform, with modes near 4 and 9 min.
- **Central-lab panels**: `dt` is when the specimen was logged at the lab, 10 min or more after the draw.

The result itself is known only after a turnaround that the dataset does not record. Using a lab value at
`dt` therefore **leaks information from the future**. The `lab_time` labels carry
`available_at = lab_time_plus_turnaround`; an analysis must add an assumed turnaround (and say what it
assumed) before using a lab value as an input. The earlier `lab_result` labels, which treated `dt` as the
result time, are retired, and the label pass deletes them from older builds.

Where the draw itself matters, use the `lab_draw` labels, which mark the arterial-line access estimated to
be the draw.
