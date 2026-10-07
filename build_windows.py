"""Validierte Annotationen in 1-s- und 2-s-Fensterdatensätze überführen."""

from __future__ import annotations

import argparse
from pathlib import Path

from trinkerkennung_analysis.window_dataset import (
    generate_windows_for_sessions,
    write_window_dataset_csv,
)


CONFIGURATIONS = (
    ("windows_1s_stride1s.csv", 1.0, 1.0),
    ("windows_2s_stride2s.csv", 2.0, 2.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", action="append", type=Path,
        help="Datenroot mit Rohdateien, Video und Sync-Berichten; mehrfach möglich.",
    )
    parser.add_argument(
        "--annotations-root", action="append", type=Path,
        help="Bei Angabe ein Annotationsordner je Datenroot, in derselben Reihenfolge.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("ML/01_Window_Datasets"),
    )
    args = parser.parse_args()
    data_roots = args.data_root or [Path("data")]
    annotation_roots = args.annotations_root
    if annotation_roots is not None and len(annotation_roots) != len(data_roots):
        parser.error("Ein --annotations-root je --data-root ist erforderlich.")
    if annotation_roots is None:
        annotation_roots = [root / "annotations" for root in data_roots]

    groups = []
    seen_sessions: set[str] = set()
    for root, annotations in zip(data_roots, annotation_roots, strict=True):
        sessions = [
            path.stem.removeprefix("annotation_")
            for path in sorted(annotations.glob("annotation_*.json"))
        ]
        if not sessions:
            parser.error(f"Keine finalen Annotationen in {annotations} gefunden.")
        if seen_sessions.intersection(sessions):
            parser.error("Session-UUID in mehreren Datenroots enthalten.")
        seen_sessions.update(sessions)
        groups.append((root, annotations, sessions))

    datasets = []
    for filename, length, stride in CONFIGURATIONS:
        records = []
        for root, annotations, sessions in groups:
            records.extend(generate_windows_for_sessions(
                data_root=root,
                annotations_root=annotations,
                session_ids=sessions,
                window_length_seconds=length,
                stride_seconds=stride,
            ))
        if not records:
            parser.error(f"Keine vollständig passenden Fenster für {filename}.")
        datasets.append((filename, records))

    for filename, records in datasets:
        path = write_window_dataset_csv(records, args.output_root / filename)
        print(f"{path}: {len(records)} Fenster")


if __name__ == "__main__":
    main()
