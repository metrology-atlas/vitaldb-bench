# SPDX-License-Identifier: MIT
"""Write the derived label families into a TimeNet build, as record annotations.

Each label becomes one :class:`~timenet.types.Annotation` on its case's record:

- ``key`` = the family (``ppg_timing``, ``lab_time``, ...), ``value`` = the label name
  (``ppg_timing:A``), ``source`` = the library that produced it;
- ``span`` = the label's interval, or a time point where it is an instant (``lab_time``), on the
  record's own timeline (microseconds from the case origin the connector kept in the record's
  ``vital_origin_s`` metadata);
- ``metadata`` = every parameter column of the family's CSV, plus ``measure_tag`` (the track the label
  is about, empty for the whole case), ``label_source``, ``source_version`` and ``available_at``.

The labels are the same as the AtriumDB label pass writes (:mod:`vitaldb_bench.label_families`), with the
same ``available_at``; a label used as model input before that point carries information from the
future.

TimeNet versions are immutable, so labelling an existing build writes a new version of each partition
(copy-on-write, :func:`label_version`): ``1.0.0`` stays, the labelled copy is ``1.1.0``, and
:class:`~vitaldb_bench.stores.TimeNetStore` reads the newest version. A new build can be labelled as it
is written instead (``python -m vitaldb_bench.build timenet --results DIR``).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from vitaldb_bench.label_families import FAMILIES, RETIRED, Family, read_family, versions_from

US = 10**6


def case_of(record_id: str) -> int:
    """VitalDB case id of a TimeNet record (``vitaldb-0003`` -> 3)."""
    return int(record_id.rsplit("-", 1)[1])


def origin_of(record) -> Fraction:
    """The case origin (s, the file's own time stamps) that the record's timeline starts at."""
    return Fraction(float.fromhex(record.metadata["vital_origin_s"]))


def family_annotations(fam: Family, rows: list[dict], origin: Fraction, record_id: str,
                       versions: dict[str, str] | None = None) -> tuple[list, int]:
    """Annotations for one family's rows of one case.

    Args:
        fam: The family.
        rows: The case's CSV rows.
        origin: The case origin in seconds (see :func:`origin_of`).
        record_id: The record's id, used to give each annotation a stable id.
        versions: Version string per label source.

    Returns:
        ``(annotations, skipped)``: rows whose end is before their start are skipped.
    """
    from timenet.types import Annotation, TimeInterval, TimePoint  # noqa: PLC0415

    versions = versions or {}
    out, skipped = [], 0
    for i, r in enumerate(rows):
        a = round((Fraction(str(r[fam.start])) - origin) * US)
        b = round((Fraction(str(r[fam.end])) - origin) * US)
        if b < a:
            skipped += 1
            continue
        span = TimePoint.micros(a) if a == b else TimeInterval.micros(a, b)
        measure = r.get(fam.measure_col) if fam.measure_col else fam.measure
        meta = {"label_source": fam.source, "source_version": versions.get(fam.source, ""),
                "measure_tag": measure or "", "available_at": fam.available_at,
                **{k: v for k, v in r.items() if k != "case" and v not in ("", None)}}
        out.append(Annotation(key=fam.name, value=fam.label(r), span=span, source=fam.source, metadata=meta,
                              id=f"{record_id}-{fam.name}-{i}"))
    return out, skipped


def annotate_record(record, labels: dict[str, list[dict]], versions: dict[str, str] | None = None) -> dict:
    """Replace a record's derived labels with ``labels``.

    Existing annotations of every family (and of retired families) are removed first, so a rerun
    replaces rather than duplicates.

    Args:
        record: A TimeNet record of a VitalDB case.
        labels: ``{family name: the case's CSV rows}``.
        versions: Version string per label source.

    Returns:
        Counts per family, and how many rows were skipped.
    """
    drop = set(FAMILIES) | set(RETIRED)
    record.annotations = tuple(an for an in record.annotations if an.key not in drop)
    origin = origin_of(record)
    counts, skipped, new = {}, 0, []
    for name, rows in labels.items():
        anns, s = family_annotations(FAMILIES[name], rows, origin, record.record_id, versions)
        new += anns
        skipped += s
        counts[name] = len(anns)
    if new:
        record.add_annotations(new, warn_when_outside=True)
    return {"labels": counts, "skipped": skipped}


def load_labels(results: str | Path, families: list[str]) -> tuple[dict[int, dict[str, list[dict]]], dict]:
    """Read the label CSVs once: ``{case: {family: rows}}``, and the label versions."""
    results = Path(results)
    by_case: dict[int, dict[str, list[dict]]] = {}
    for name in families:
        for case, rows in read_family(results, FAMILIES[name]).items():
            by_case.setdefault(case, {})[name] = rows
    return by_case, versions_from(results)


def label_dataset(dataset, by_case: dict[int, dict[str, list[dict]]], families: list[str],
                  versions: dict[str, str]) -> dict:
    """Annotate every record of an in-memory TimeNet dataset; cases without rows lose old labels only."""
    total: dict[str, int] = dict.fromkeys(families, 0)
    skipped = 0
    for record in dataset.records:
        case_rows = by_case.get(case_of(record.record_id), {})
        res = annotate_record(record, {f: case_rows.get(f, []) for f in families}, versions)
        skipped += res["skipped"]
        for f, n in res["labels"].items():
            total[f] += n
    return {"records": len(dataset.records), "labels": total, "skipped": skipped}


def label_version(base_dir: str | Path, out_root: str | Path, results: str | Path, families: list[str],
                  new_version: str = "1.1.0", labels: tuple | None = None) -> dict:
    """Copy-on-write: read one committed partition version, label it, and write ``new_version``.

    Args:
        base_dir: The committed version directory (``.../part-0000/vitaldb/vitaldb/1.0.0``).
        out_root: Where the new version goes (the partition root: ``.../part-0000``).
        results: Folder with the label CSVs.
        families: Families to write.
        new_version: The new version's number; must differ from the base's.
        labels: ``load_labels`` output, to read the CSVs once for many partitions.

    Returns:
        Counts, and the new version's directory.
    """
    from timenet.dataset import TimeFDataset  # noqa: PLC0415
    from timenet.reader import TimeFReader  # noqa: PLC0415
    from timenet.registry.version import DatasetVersion  # noqa: PLC0415
    from timenet.types import DatasetSchema, Version  # noqa: PLC0415
    from timenet.writer import TimeFWriter  # noqa: PLC0415

    by_case, versions = labels if labels is not None else load_labels(results, families)
    with TimeFReader(DatasetVersion.open_local(Path(base_dir))) as reader:
        base = reader.read()
        stats = label_dataset(base, by_case, families, versions)
        edited = TimeFDataset.from_parts(
            metadata=base.metadata.model_copy(update={"dataset_version": Version.parse(new_version)}),
            records=base.records, tasks=base.tasks, schema=DatasetSchema(),
            registered_annotations=base.registered_annotations, annotations=base.annotations)
        edited.derive_schema()
        with TimeFWriter(Path(out_root), edited, values_backend=reader.values_backend) as writer:
            writer.write()
    stats["version_dir"] = str(Path(out_root) / edited.metadata.dataset_id / new_version)
    return stats
