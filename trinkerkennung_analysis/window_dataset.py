from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .annotation import validate_annotation_file
from .video_alignment import AUDIO_TIME_REFERENCE


WINDOW_DATASET_SCHEMA_VERSION = 1

DRINK_LABEL = "DRINK"
NON_DRINK_LABEL = "NON_DRINK"

DRINK_SOURCE_CATEGORY = "DRINK_EVENT"
NEGATIVE_SOURCE_CATEGORY = "NEGATIVE_INTERVAL"

_TIME_EPSILON_SECONDS = 1e-9


class WindowDatasetError(ValueError):
    """Fehler bei der reproduzierbaren ML-Fensterbildung."""


@dataclass(frozen=True)
class WindowRecord:
    schema_version: int
    participant_id: str
    session_id: str
    session_mode: str
    time_reference: str
    window_id: str
    label: str
    source_category: str
    source_id: str
    scenario: str
    container_type: str | None
    source_start_seconds: float
    source_end_seconds: float
    window_index: int
    window_start_seconds: float
    window_end_seconds: float
    window_length_seconds: float
    stride_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SourceInterval:
    category: str
    source_id: str
    label: str
    scenario: str
    container_type: str | None
    start_seconds: float
    end_seconds: float


WINDOW_DATASET_FIELDNAMES = (
    "schema_version",
    "participant_id",
    "session_id",
    "session_mode",
    "time_reference",
    "window_id",
    "label",
    "source_category",
    "source_id",
    "scenario",
    "container_type",
    "source_start_seconds",
    "source_end_seconds",
    "window_index",
    "window_start_seconds",
    "window_end_seconds",
    "window_length_seconds",
    "stride_seconds",
)


def generate_windows_from_validated_annotation(
    annotation: dict[str, Any],
    *,
    window_length_seconds: float,
    stride_seconds: float,
) -> list[WindowRecord]:
    """
    Erzeugt ML-Fenster ausschließlich aus expliziten DRINK- und
    NEGATIVE-Intervallen einer bereits validierten Annotation.

    Implizit unbeschriftete Zeit, EXCLUDED- und UNCERTAIN-Intervalle
    werden bewusst nicht als ML-Beispiele verwendet.
    """
    window_length = _positive_finite_number(
        window_length_seconds,
        "window_length_seconds",
    )
    stride = _positive_finite_number(
        stride_seconds,
        "stride_seconds",
    )

    participant_id = _required_string(
        annotation.get("participant_id"),
        "participant_id",
    )
    session_id = _required_string(
        annotation.get("session_id"),
        "session_id",
    )
    session_mode = _required_string(
        annotation.get("session_mode"),
        "session_mode",
    )

    if annotation.get("time_reference") != AUDIO_TIME_REFERENCE:
        raise WindowDatasetError(
            "Die Annotation muss AUDIO_SECONDS_FROM_WAV_START verwenden."
        )

    source_intervals = _collect_source_intervals(annotation)
    source_intervals.sort(
        key=lambda interval: (
            interval.start_seconds,
            interval.end_seconds,
            interval.source_id,
        )
    )

    windows: list[WindowRecord] = []

    for interval in source_intervals:
        window_count = _full_window_count(
            interval_start=interval.start_seconds,
            interval_end=interval.end_seconds,
            window_length=window_length,
            stride=stride,
        )

        for zero_based_index in range(window_count):
            window_index = zero_based_index + 1

            window_start = (
                interval.start_seconds
                + zero_based_index * stride
            )
            window_end = window_start + window_length

            window_start = _stable_time(window_start)
            window_end = _stable_time(window_end)

            if (
                window_start
                < interval.start_seconds - _TIME_EPSILON_SECONDS
                or window_end
                > interval.end_seconds + _TIME_EPSILON_SECONDS
            ):
                raise WindowDatasetError(
                    "Intern erzeugtes Fenster liegt außerhalb "
                    "seines Quellintervalls."
                )

            window_id = (
                f"{session_id}__{interval.source_id}"
                f"__W{window_index:03d}"
            )

            windows.append(
                WindowRecord(
                    schema_version=WINDOW_DATASET_SCHEMA_VERSION,
                    participant_id=participant_id,
                    session_id=session_id,
                    session_mode=session_mode,
                    time_reference=AUDIO_TIME_REFERENCE,
                    window_id=window_id,
                    label=interval.label,
                    source_category=interval.category,
                    source_id=interval.source_id,
                    scenario=interval.scenario,
                    container_type=interval.container_type,
                    source_start_seconds=interval.start_seconds,
                    source_end_seconds=interval.end_seconds,
                    window_index=window_index,
                    window_start_seconds=window_start,
                    window_end_seconds=window_end,
                    window_length_seconds=window_length,
                    stride_seconds=stride,
                )
            )

    return windows


def generate_windows_from_annotation_file(
    *,
    data_root: str | Path,
    annotation_path: str | Path,
    window_length_seconds: float,
    stride_seconds: float,
) -> list[WindowRecord]:
    """
    Validiert zuerst die Ground-Truth-Annotation und erzeugt erst danach
    ML-Fenster.
    """
    root = Path(data_root).expanduser().resolve()
    path = Path(annotation_path).expanduser().resolve()

    validate_annotation_file(
        data_root=root,
        annotation_path=path,
    )

    try:
        annotation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise WindowDatasetError(
            f"Annotationsdatei nicht lesbar: {path}"
        ) from exception

    if not isinstance(annotation, dict):
        raise WindowDatasetError(
            "Annotationsdatei muss ein JSON-Objekt enthalten."
        )

    return generate_windows_from_validated_annotation(
        annotation,
        window_length_seconds=window_length_seconds,
        stride_seconds=stride_seconds,
    )


def generate_windows_for_sessions(
    *,
    data_root: str | Path,
    session_ids: Sequence[str],
    window_length_seconds: float,
    stride_seconds: float,
    annotations_root: str | Path | None = None,
) -> list[WindowRecord]:
    """
    Erzeugt einen gemeinsamen Fensterdatensatz für explizit angegebene
    Sessions.

    data_root wird für die sitzungsübergreifende Validierung verwendet.
    Der kanonische Annotationsordner kann davon getrennt über
    annotations_root angegeben werden.

    Wird annotations_root nicht angegeben, bleibt das bisherige Verhalten
    erhalten und <data_root>/annotations wird verwendet.
    """
    root = Path(data_root).expanduser().resolve()

    if annotations_root is None:
        resolved_annotations_root = root / "annotations"
    else:
        resolved_annotations_root = (
            Path(annotations_root).expanduser().resolve()
        )

    if not resolved_annotations_root.is_dir():
        raise WindowDatasetError(
            "Annotationsordner nicht gefunden: "
            f"{resolved_annotations_root}"
        )

    seen_session_ids: set[str] = set()
    records: list[WindowRecord] = []

    for session_id in session_ids:
        if session_id in seen_session_ids:
            raise WindowDatasetError(
                f"Doppelte Session-ID: {session_id}"
            )
        seen_session_ids.add(session_id)

        annotation_path = (
            resolved_annotations_root
            / f"annotation_{session_id}.json"
        )

        if not annotation_path.is_file():
            raise WindowDatasetError(
                f"Annotationsdatei nicht gefunden: {annotation_path}"
            )

        records.extend(
            generate_windows_from_annotation_file(
                data_root=root,
                annotation_path=annotation_path,
                window_length_seconds=window_length_seconds,
                stride_seconds=stride_seconds,
            )
        )

    return records


def write_window_dataset_csv(
    records: Iterable[WindowRecord],
    output_path: str | Path,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=WINDOW_DATASET_FIELDNAMES,
            delimiter=";",
        )
        writer.writeheader()

        for record in records:
            writer.writerow(record.to_dict())

    return path


def _collect_source_intervals(
    annotation: dict[str, Any],
) -> list[_SourceInterval]:
    intervals: list[_SourceInterval] = []

    drink_events = annotation.get("drink_events", [])
    negative_intervals = annotation.get("negative_intervals", [])

    if not isinstance(drink_events, list):
        raise WindowDatasetError(
            "drink_events muss ein Array sein."
        )
    if not isinstance(negative_intervals, list):
        raise WindowDatasetError(
            "negative_intervals muss ein Array sein."
        )

    for event in drink_events:
        if not isinstance(event, dict):
            raise WindowDatasetError(
                "Ein drink_event ist kein JSON-Objekt."
            )

        intervals.append(
            _SourceInterval(
                category=DRINK_SOURCE_CATEGORY,
                source_id=_required_string(
                    event.get("event_id"),
                    "event_id",
                ),
                label=DRINK_LABEL,
                scenario=_required_string(
                    event.get("scenario"),
                    "drink_event.scenario",
                ),
                container_type=_optional_string(
                    event.get("container_type")
                ),
                start_seconds=_finite_number(
                    event.get("event_start_seconds"),
                    "event_start_seconds",
                ),
                end_seconds=_finite_number(
                    event.get("event_end_seconds"),
                    "event_end_seconds",
                ),
            )
        )

    for interval in negative_intervals:
        if not isinstance(interval, dict):
            raise WindowDatasetError(
                "Ein negative_interval ist kein JSON-Objekt."
            )

        intervals.append(
            _SourceInterval(
                category=NEGATIVE_SOURCE_CATEGORY,
                source_id=_required_string(
                    interval.get("interval_id"),
                    "interval_id",
                ),
                label=NON_DRINK_LABEL,
                scenario=_required_string(
                    interval.get("scenario"),
                    "negative_interval.scenario",
                ),
                container_type=_optional_string(
                    interval.get("container_type")
                ),
                start_seconds=_finite_number(
                    interval.get("start_seconds"),
                    "negative_interval.start_seconds",
                ),
                end_seconds=_finite_number(
                    interval.get("end_seconds"),
                    "negative_interval.end_seconds",
                ),
            )
        )

    for interval in intervals:
        if interval.start_seconds >= interval.end_seconds:
            raise WindowDatasetError(
                f"Ungültiges Quellintervall: {interval.source_id}"
            )

    return intervals


def _full_window_count(
    *,
    interval_start: float,
    interval_end: float,
    window_length: float,
    stride: float,
) -> int:
    duration = interval_end - interval_start

    if duration + _TIME_EPSILON_SECONDS < window_length:
        return 0

    remaining_after_first = duration - window_length

    return (
        math.floor(
            (
                remaining_after_first
                + _TIME_EPSILON_SECONDS
            )
            / stride
        )
        + 1
    )


def _positive_finite_number(
    value: Any,
    name: str,
) -> float:
    numeric = _finite_number(value, name)

    if numeric <= 0:
        raise WindowDatasetError(
            f"{name} muss größer als 0 sein."
        )

    return numeric


def _finite_number(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise WindowDatasetError(
            f"{name} muss numerisch sein."
        )

    numeric = float(value)

    if not math.isfinite(numeric):
        raise WindowDatasetError(
            f"{name} muss endlich sein."
        )

    return numeric


def _required_string(
    value: Any,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WindowDatasetError(
            f"{name} muss eine nichtleere Zeichenkette sein."
        )

    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise WindowDatasetError(
            "Optionaler Zeichenkettenwert besitzt einen falschen Typ."
        )

    stripped = value.strip()
    return stripped if stripped else None


def _stable_time(value: float) -> float:
    return round(float(value), 9)