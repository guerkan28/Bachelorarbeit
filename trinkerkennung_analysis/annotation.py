from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .session_loader import (
    SessionValidationError,
    load_and_validate_session,
)
from .video_alignment import (
    AUDIO_TIME_REFERENCE,
    VIDEO_TIME_REFERENCE,
)


SCHEMA_VERSION = 1
GROUND_TRUTH_SOURCE = "SYNCHRONIZED_VIDEO"
MARKER_TOLERANCE_SECONDS = 0.050

RECORDING_PROTOCOL_VERSIONS = frozenset({"1.0"})
SESSION_MODES = frozenset({
    "CONTROLLED_SINGLE_ACTIVITY",
    "CONTROLLED_MIXED_ACTIVITY",
    "CONTINUOUS_MIXED_ACTIVITY",
    "EDGE_CASE",
})
ANNOTATION_COVERAGES = frozenset({"PARTIAL", "EXHAUSTIVE"})

DRINK_SCENARIOS = frozenset({
    "DRINK_FROM_GLASS",
    "DRINK_FROM_CUP",
    "DRINK_FROM_BOTTLE",
})
CONTAINER_TYPES = frozenset({"GLASS", "CUP", "BOTTLE"})
NEGATIVE_SCENARIOS = frozenset({
    "REST",
    "LIFT_CONTAINER_WITHOUT_DRINKING",
    "MOVE_CONTAINER_WITHOUT_DRINKING",
    "HAND_TO_FACE_WITHOUT_CONTAINER",
    "TOUCH_FACE",
    "SMARTPHONE_USE",
    "SPEAKING",
    "EATING",
    "COUGHING",
    "ARM_MOVEMENT_WITHOUT_CONTAINER",
})
EXCLUDED_REASONS = frozenset({
    "INITIAL_POSITIONING",
    "SYNCHRONIZATION_MARKER",
    "RECORDING_START_INTERACTION",
    "RECORDING_STOP_INTERACTION",
    "EXPERIMENT_INSTRUCTION",
    "INVALID_SIGNAL",
    "OTHER_TECHNICAL_ARTIFACT",
})
UNCERTAIN_REASONS = frozenset({
    "MOUTH_CONTACT_NOT_VISIBLE",
    "EVENT_BOUNDARY_NOT_VISIBLE",
    "POSSIBLE_DRINKING_WITHOUT_SUFFICIENT_EVIDENCE",
    "ABORTED_ACTIVITY",
    "OVERLAPPING_ACTIVITIES",
    "OTHER_SEMANTIC_UNCERTAINTY",
})

FORBIDDEN_EXPERIMENT_KEYS = frozenset({
    "window_length_seconds",
    "window_overlap",
    "window_stride",
    "feature_set",
    "classifier",
    "fusion_strategy",
    "window_label_threshold",
    "sip_count",
    "boundary_confidence",
    "container_state_at_start",
    "container_state_at_end",
})

PARTICIPANT_ID_PATTERN = re.compile(r"^P[0-9]{3,}$")
ANNOTATOR_ID_PATTERN = re.compile(r"^A[0-9]{3,}$")


class AnnotationValidationError(ValueError):
    """Enthält alle bei einer Annotationsprüfung gefundenen Fehler."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        message = "Annotationsvalidierung fehlgeschlagen:\n" + "\n".join(
            f"- {error}" for error in self.errors
        )
        super().__init__(message)


@dataclass(frozen=True)
class AnnotationValidationReport:
    session_id: str
    participant_id: str
    annotation_file: str
    session_metadata_file: str
    synchronization_report_file: str
    reference_video_file: str
    wav_duration_seconds: float
    annotation_coverage: str
    drink_event_count: int
    mouth_contact_interval_count: int
    negative_interval_count: int
    excluded_interval_count: int
    uncertain_interval_count: int
    reviewed_interval_count: int
    explicit_intervals_are_disjoint: bool
    implicit_unlabeled_time_is_not_negative: bool
    marker_tolerance_seconds: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "annotation_file": self.annotation_file,
            "session_metadata_file": self.session_metadata_file,
            "synchronization_report_file": (
                self.synchronization_report_file
            ),
            "reference_video_file": self.reference_video_file,
            "wav_duration_seconds": self.wav_duration_seconds,
            "annotation_coverage": self.annotation_coverage,
            "drink_event_count": self.drink_event_count,
            "mouth_contact_interval_count": (
                self.mouth_contact_interval_count
            ),
            "negative_interval_count": self.negative_interval_count,
            "excluded_interval_count": self.excluded_interval_count,
            "uncertain_interval_count": self.uncertain_interval_count,
            "reviewed_interval_count": self.reviewed_interval_count,
            "explicit_intervals_are_disjoint": (
                self.explicit_intervals_are_disjoint
            ),
            "implicit_unlabeled_time_is_not_negative": (
                self.implicit_unlabeled_time_is_not_negative
            ),
            "marker_tolerance_seconds": self.marker_tolerance_seconds,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _Interval:
    category: str
    identifier: str
    start: float
    end: float

    def overlaps(self, other: "_Interval") -> bool:
        return self.start < other.end and other.start < self.end

    def contains(
        self,
        value: float,
        tolerance: float = 0.0,
    ) -> bool:
        return (
            self.start - tolerance
            <= value
            < self.end + tolerance
        )

    def covers(
        self,
        start: float,
        end: float,
        tolerance: float,
    ) -> bool:
        return (
            self.start <= start + tolerance
            and self.end >= end - tolerance
        )


class _AnnotationValidator:
    def __init__(
        self,
        *,
        data_root: Path,
        annotation_path: Path,
        marker_tolerance_seconds: float,
    ) -> None:
        self.data_root = data_root
        self.annotation_path = annotation_path
        self.marker_tolerance_seconds = marker_tolerance_seconds
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.identifiers: set[str] = set()
        self.classification_intervals: list[_Interval] = []
        self.marker_intervals: list[_Interval] = []
        self.mouth_contact_count = 0
        self.annotation: dict[str, Any] = {}
        self.validation_report: dict[str, Any] = {}
        self.synchronization_report: dict[str, Any] = {}
        self.reference_video_path: Path | None = None
        self.synchronization_report_path: Path | None = None

    def validate(self) -> AnnotationValidationReport:
        if self.marker_tolerance_seconds <= 0:
            raise AnnotationValidationError([
                "Die Markertoleranz muss positiv sein."
            ])

        self.annotation = self._read_json(self.annotation_path, "Annotation")
        self._reject_forbidden_keys(self.annotation, "$" )
        self._validate_top_level_shape()

        session_id = self.annotation.get("session_id")
        if isinstance(session_id, str):
            self._validate_uuid(session_id)
            expected_name = f"annotation_{session_id}.json"
            if self.annotation_path.name != expected_name:
                self.errors.append(
                    "Der Annotationsdateiname muss exakt "
                    f"'{expected_name}' lauten."
                )
        else:
            session_id = ""
            self.errors.append("session_id muss eine Zeichenkette sein.")

        if session_id:
            try:
                self.validation_report = load_and_validate_session(
                    self.data_root,
                    session_id,
                )
            except SessionValidationError as exception:
                self.errors.append(str(exception))

        wav_duration = self._wav_duration()
        self._validate_scalar_metadata(session_id)
        self._validate_ground_truth(session_id)
        self._validate_reviewed_intervals(wav_duration)
        self._validate_drink_events(wav_duration)
        self._validate_negative_intervals(wav_duration)
        self._validate_excluded_intervals(wav_duration)
        self._validate_uncertain_intervals(wav_duration)
        self._validate_top_level_interval_disjointness()
        self._validate_synchronization_report(session_id)
        self._validate_marker_coverage()
        self._validate_partial_review_coverage()

        if self.errors:
            raise AnnotationValidationError(self.errors)

        ground_truth = self.annotation["ground_truth"]
        return AnnotationValidationReport(
            session_id=session_id,
            participant_id=self.annotation["participant_id"],
            annotation_file=str(self.annotation_path),
            session_metadata_file=str(
                self.validation_report["metadata_path"]
            ),
            synchronization_report_file=str(
                self.synchronization_report_path
            ),
            reference_video_file=str(self.reference_video_path),
            wav_duration_seconds=wav_duration,
            annotation_coverage=ground_truth["annotation_coverage"],
            drink_event_count=len(self.annotation["drink_events"]),
            mouth_contact_interval_count=self.mouth_contact_count,
            negative_interval_count=len(
                self.annotation["negative_intervals"]
            ),
            excluded_interval_count=len(
                self.annotation["excluded_intervals"]
            ),
            uncertain_interval_count=len(
                self.annotation["uncertain_intervals"]
            ),
            reviewed_interval_count=len(
                self.annotation.get("reviewed_intervals", [])
            ),
            explicit_intervals_are_disjoint=True,
            implicit_unlabeled_time_is_not_negative=True,
            marker_tolerance_seconds=self.marker_tolerance_seconds,
            warnings=tuple(self.warnings),
        )

    def _validate_top_level_shape(self) -> None:
        required = {
            "schema_version",
            "session_id",
            "participant_id",
            "recording_protocol_version",
            "session_mode",
            "time_reference",
            "synchronization_report_file",
            "ground_truth",
            "drink_events",
            "negative_intervals",
            "excluded_intervals",
            "uncertain_intervals",
        }
        allowed = required | {"reviewed_intervals"}
        self._validate_object_keys(
            self.annotation,
            required=required,
            allowed=allowed,
            path="$",
        )

        for key in (
            "drink_events",
            "negative_intervals",
            "excluded_intervals",
            "uncertain_intervals",
        ):
            if not isinstance(self.annotation.get(key), list):
                self.errors.append(f"{key} muss ein Array sein.")
                self.annotation[key] = []
        if "reviewed_intervals" in self.annotation and not isinstance(
            self.annotation["reviewed_intervals"],
            list,
        ):
            self.errors.append("reviewed_intervals muss ein Array sein.")
            self.annotation["reviewed_intervals"] = []

    def _validate_scalar_metadata(self, session_id: str) -> None:
        if self.annotation.get("schema_version") != SCHEMA_VERSION:
            self.errors.append("schema_version muss 1 sein.")

        participant_id = self.annotation.get("participant_id")
        if (
            not isinstance(participant_id, str)
            or not PARTICIPANT_ID_PATTERN.fullmatch(participant_id)
        ):
            self.errors.append(
                "participant_id muss pseudonymisiert dem Muster P001 entsprechen."
            )

        protocol = self.annotation.get("recording_protocol_version")
        if protocol not in RECORDING_PROTOCOL_VERSIONS:
            self.errors.append(
                "recording_protocol_version ist unbekannt. Zulässig: "
                + ", ".join(sorted(RECORDING_PROTOCOL_VERSIONS))
            )

        if self.annotation.get("session_mode") not in SESSION_MODES:
            self.errors.append(
                "session_mode ist unbekannt."
            )
        if self.annotation.get("time_reference") != AUDIO_TIME_REFERENCE:
            self.errors.append(
                f"time_reference muss exakt {AUDIO_TIME_REFERENCE} sein."
            )

        report_file = self.annotation.get("synchronization_report_file")
        if not isinstance(report_file, str) or not report_file.strip():
            self.errors.append(
                "synchronization_report_file fehlt oder ist leer."
            )
        elif session_id and session_id not in report_file:
            self.errors.append(
                "synchronization_report_file enthält nicht die Session-ID."
            )

    def _validate_ground_truth(self, session_id: str) -> None:
        value = self.annotation.get("ground_truth")
        if not isinstance(value, dict):
            self.errors.append("ground_truth muss ein JSON-Objekt sein.")
            self.annotation["ground_truth"] = {
                "source": None,
                "reference_video_file": None,
                "annotator_id": None,
                "annotation_coverage": None,
            }
            return

        required = {
            "source",
            "reference_video_file",
            "annotator_id",
            "annotation_coverage",
        }
        self._validate_object_keys(
            value,
            required=required,
            allowed=required,
            path="$.ground_truth",
        )

        if value.get("source") != GROUND_TRUTH_SOURCE:
            self.errors.append(
                f"ground_truth.source muss {GROUND_TRUTH_SOURCE} sein."
            )

        video_file = value.get("reference_video_file")
        if not isinstance(video_file, str) or not video_file.strip():
            self.errors.append(
                "reference_video_file ist für SYNCHRONIZED_VIDEO erforderlich."
            )
        else:
            video_file = video_file.strip()
            if session_id and session_id not in video_file:
                self.errors.append(
                    "reference_video_file enthält nicht die Session-ID."
                )
            if session_id and not video_file.startswith(
                f"reference_video_{session_id}_"
            ):
                self.errors.append(
                    "reference_video_file muss dem festgelegten Dateinamensschema entsprechen."
                )
            if not Path(video_file).suffix:
                self.errors.append(
                    "reference_video_file benötigt eine Dateiendung."
                )
            matches = self._find_files(Path(video_file).name)
            if len(matches) == 1:
                self.reference_video_path = matches[0]
            elif len(matches) == 0:
                self.errors.append(
                    f"Referenzvideo nicht gefunden: {video_file}"
                )
            else:
                self.errors.append(
                    f"Referenzvideo mehrfach gefunden: {video_file}"
                )

        annotator_id = value.get("annotator_id")
        if (
            not isinstance(annotator_id, str)
            or not ANNOTATOR_ID_PATTERN.fullmatch(annotator_id)
        ):
            self.errors.append(
                "annotator_id muss pseudonymisiert dem Muster A001 entsprechen."
            )
        if value.get("annotation_coverage") not in ANNOTATION_COVERAGES:
            self.errors.append(
                "annotation_coverage muss PARTIAL oder EXHAUSTIVE sein."
            )

    def _validate_reviewed_intervals(self, wav_duration: float) -> None:
        intervals = self.annotation.get("reviewed_intervals", [])
        previous: _Interval | None = None
        for index, item in enumerate(intervals):
            path = f"$.reviewed_intervals[{index}]"
            interval = self._simple_interval(
                item,
                path=path,
                category="REVIEWED",
                identifier=f"R{index + 1:03d}",
                wav_duration=wav_duration,
                extra_required=set(),
                extra_allowed=set(),
            )
            if interval is None:
                continue
            if previous is not None and previous.overlaps(interval):
                self.errors.append(
                    f"{path} überlappt ein vorheriges reviewed_interval."
                )
            if previous is not None and interval.start < previous.start:
                self.errors.append(
                    "reviewed_intervals müssen aufsteigend sortiert sein."
                )
            previous = interval

    def _validate_drink_events(self, wav_duration: float) -> None:
        for index, event in enumerate(self.annotation["drink_events"]):
            path = f"$.drink_events[{index}]"
            if not isinstance(event, dict):
                self.errors.append(f"{path} muss ein JSON-Objekt sein.")
                continue
            required = {
                "event_id",
                "scenario",
                "container_type",
                "event_start_seconds",
                "event_end_seconds",
                "mouth_contact_intervals",
                "notes",
            }
            self._validate_object_keys(
                event,
                required=required,
                allowed=required,
                path=path,
            )
            event_id = self._identifier(event.get("event_id"), path)
            scenario = event.get("scenario")
            if scenario not in DRINK_SCENARIOS:
                self.errors.append(f"{path}.scenario ist unbekannt.")
            container_type = event.get("container_type")
            if container_type not in CONTAINER_TYPES:
                self.errors.append(
                    f"{path}.container_type ist unbekannt."
                )
            self._validate_notes(event.get("notes"), f"{path}.notes")

            interval = self._interval_from_values(
                category="DRINK_EVENT",
                identifier=event_id,
                start_value=event.get("event_start_seconds"),
                end_value=event.get("event_end_seconds"),
                path=path,
                wav_duration=wav_duration,
            )
            if interval is not None:
                self.classification_intervals.append(interval)

            contacts = event.get("mouth_contact_intervals")
            if not isinstance(contacts, list) or not contacts:
                self.errors.append(
                    f"{path}.mouth_contact_intervals muss mindestens ein Intervall enthalten."
                )
                continue

            previous_contact: _Interval | None = None
            for contact_index, contact in enumerate(contacts):
                contact_path = (
                    f"{path}.mouth_contact_intervals[{contact_index}]"
                )
                contact_interval = self._simple_interval(
                    contact,
                    path=contact_path,
                    category="MOUTH_CONTACT",
                    identifier=f"{event_id}:C{contact_index + 1}",
                    wav_duration=wav_duration,
                    extra_required=set(),
                    extra_allowed=set(),
                )
                if contact_interval is None:
                    continue
                self.mouth_contact_count += 1
                if interval is not None and not (
                    interval.start <= contact_interval.start
                    and contact_interval.end <= interval.end
                ):
                    self.errors.append(
                        f"{contact_path} liegt nicht vollständig innerhalb des DRINK_EVENT."
                    )
                if previous_contact is not None:
                    if contact_interval.start < previous_contact.start:
                        self.errors.append(
                            f"{path}.mouth_contact_intervals müssen sortiert sein."
                        )
                    if previous_contact.overlaps(contact_interval):
                        self.errors.append(
                            f"{path}.mouth_contact_intervals überlappen sich."
                        )
                previous_contact = contact_interval

    def _validate_negative_intervals(self, wav_duration: float) -> None:
        for index, item in enumerate(self.annotation["negative_intervals"]):
            path = f"$.negative_intervals[{index}]"
            interval = self._simple_interval(
                item,
                path=path,
                category="NEGATIVE",
                identifier_key="interval_id",
                wav_duration=wav_duration,
                extra_required={"scenario", "notes"},
                extra_allowed={"scenario", "container_type", "notes"},
            )
            if isinstance(item, dict):
                if item.get("scenario") not in NEGATIVE_SCENARIOS:
                    self.errors.append(f"{path}.scenario ist unbekannt.")

                container_type = item.get("container_type")
                if (
                    container_type is not None
                    and container_type not in CONTAINER_TYPES
                ):
                    self.errors.append(
                        f"{path}.container_type ist unbekannt."
                    )

                self._validate_notes(item.get("notes"), f"{path}.notes")
                if "mouth_contact_intervals" in item:
                    self.errors.append(
                        f"{path} darf keine mouth_contact_intervals enthalten."
                    )
            if interval is not None:
                self.classification_intervals.append(interval)

    def _validate_excluded_intervals(self, wav_duration: float) -> None:
        for index, item in enumerate(self.annotation["excluded_intervals"]):
            path = f"$.excluded_intervals[{index}]"
            interval = self._simple_interval(
                item,
                path=path,
                category="EXCLUDED",
                identifier_key="interval_id",
                wav_duration=wav_duration,
                extra_required={"reason"},
                extra_allowed={"reason"},
            )
            if isinstance(item, dict):
                if item.get("reason") not in EXCLUDED_REASONS:
                    self.errors.append(f"{path}.reason ist unbekannt.")
            if interval is not None:
                self.classification_intervals.append(interval)
                if isinstance(item, dict) and item.get("reason") == (
                    "SYNCHRONIZATION_MARKER"
                ):
                    self.marker_intervals.append(interval)

    def _validate_uncertain_intervals(self, wav_duration: float) -> None:
        for index, item in enumerate(self.annotation["uncertain_intervals"]):
            path = f"$.uncertain_intervals[{index}]"
            interval = self._simple_interval(
                item,
                path=path,
                category="UNCERTAIN",
                identifier_key="interval_id",
                wav_duration=wav_duration,
                extra_required={"reason"},
                extra_allowed={"reason", "notes"},
            )
            if isinstance(item, dict):
                if item.get("reason") not in UNCERTAIN_REASONS:
                    self.errors.append(f"{path}.reason ist unbekannt.")
                if "notes" in item:
                    self._validate_notes(item.get("notes"), f"{path}.notes")
            if interval is not None:
                self.classification_intervals.append(interval)

    def _validate_top_level_interval_disjointness(self) -> None:
        ordered = sorted(
            self.classification_intervals,
            key=lambda interval: (interval.start, interval.end),
        )
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1:]:
                if right.start >= left.end:
                    break
                if left.overlaps(right):
                    self.errors.append(
                        "Explizite fachliche Intervalle müssen disjunkt sein: "
                        f"{left.category}:{left.identifier} überlappt "
                        f"{right.category}:{right.identifier}."
                    )

    def _validate_synchronization_report(self, session_id: str) -> None:
        report_file = self.annotation.get("synchronization_report_file")
        if not isinstance(report_file, str) or not report_file:
            return
        matches = self._find_files(report_file)
        if len(matches) == 0:
            self.errors.append(
                f"Synchronisationsbericht nicht gefunden: {report_file}"
            )
            return
        if len(matches) > 1:
            self.errors.append(
                f"Synchronisationsbericht mehrfach gefunden: {report_file}"
            )
            return
        self.synchronization_report_path = matches[0]
        self.synchronization_report = self._read_json(
            matches[0],
            "Synchronisationsbericht",
        )
        if self.synchronization_report.get("session_id") != session_id:
            self.errors.append(
                "Session-ID im Synchronisationsbericht stimmt nicht überein."
            )

        mappings = self.synchronization_report.get("time_mappings")
        if not isinstance(mappings, dict):
            self.errors.append(
                "Der Synchronisationsbericht enthält keine time_mappings-Struktur."
            )
            return
        video_mapping = mappings.get("video_to_audio")
        if not isinstance(video_mapping, dict):
            self.errors.append(
                "Für SYNCHRONIZED_VIDEO fehlt die Video-zu-Audio-Abbildung."
            )
            return
        if video_mapping.get("source_time_reference") != VIDEO_TIME_REFERENCE:
            self.errors.append(
                "Die Videoabbildung besitzt eine falsche Quellzeitreferenz."
            )
        if video_mapping.get("target_time_reference") != AUDIO_TIME_REFERENCE:
            self.errors.append(
                "Die Videoabbildung besitzt eine falsche Zielzeitreferenz."
            )
        if video_mapping.get("method") != "LINEAR_TWO_POINT":
            self.errors.append(
                "Die Videoabbildung muss LINEAR_TWO_POINT verwenden."
            )
        scale = self._number(
            video_mapping.get("scale"),
            "time_mappings.video_to_audio.scale",
        )
        self._number(
            video_mapping.get("offset_seconds"),
            "time_mappings.video_to_audio.offset_seconds",
        )
        if scale is not None and scale <= 0:
            self.errors.append(
                "Die Video-zu-Audio-Skalierung muss positiv sein."
            )
        ground_truth = self.annotation.get("ground_truth", {})
        expected_video = (
            ground_truth.get("reference_video_file")
            if isinstance(ground_truth, dict)
            else None
        )

        mapped_video = video_mapping.get("reference_video_file")

        if (
            not isinstance(expected_video, str)
            or not isinstance(mapped_video, str)
            or Path(expected_video).name != Path(mapped_video).name
        ):
            self.errors.append(
                "Referenzvideo in Annotation und Videoabbildung stimmt nicht überein."
            )

    def _validate_marker_coverage(self) -> None:
        if not self.synchronization_report or not self.marker_intervals:
            if self.synchronization_report and not self.marker_intervals:
                self.errors.append(
                    "Es fehlt mindestens ein EXCLUDED-Intervall mit reason "
                    "SYNCHRONIZATION_MARKER."
                )
            return

        marker_times: list[float] = []
        for key in ("start_audio_marker", "end_audio_marker"):
            marker = self.synchronization_report.get(key)
            if not isinstance(marker, dict):
                self.errors.append(
                    f"{key} fehlt im Synchronisationsbericht."
                )
                continue
            times = marker.get("peak_times_seconds")
            if not isinstance(times, list):
                self.errors.append(
                    f"{key}.peak_times_seconds fehlt."
                )
                continue
            for index, value in enumerate(times):
                numeric = self._number(
                    value,
                    f"{key}.peak_times_seconds[{index}]",
                )
                if numeric is not None:
                    marker_times.append(numeric)

        for marker_time in marker_times:
            if not any(
                interval.contains(
                    marker_time,
                    self.marker_tolerance_seconds,
                )
                for interval in self.marker_intervals
            ):
                self.errors.append(
                    "Ein Audiomarker liegt außerhalb aller "
                    "SYNCHRONIZATION_MARKER-Ausschlussintervalle: "
                    f"{marker_time:.6f} s."
                )

        report_intervals = self.synchronization_report.get(
            "audio_marker_exclusion_intervals_seconds"
        )
        if isinstance(report_intervals, list):
            for index, pair in enumerate(report_intervals):
                if not isinstance(pair, list) or len(pair) != 2:
                    self.errors.append(
                        "audio_marker_exclusion_intervals_seconds enthält "
                        f"ein ungültiges Element an Index {index}."
                    )
                    continue
                start = self._number(pair[0], "Markerintervall-Start")
                end = self._number(pair[1], "Markerintervall-Ende")
                if start is None or end is None:
                    continue
                if not any(
                    interval.covers(
                        start,
                        end,
                        self.marker_tolerance_seconds,
                    )
                    for interval in self.marker_intervals
                ):
                    self.errors.append(
                        "Ein im Synchronisationsbericht dokumentiertes "
                        "Markerintervall wird nicht toleranzbasiert durch "
                        "die Annotation abgedeckt."
                    )

    def _validate_partial_review_coverage(self) -> None:
        ground_truth = self.annotation.get("ground_truth")
        if not isinstance(ground_truth, dict):
            return
        if ground_truth.get("annotation_coverage") != "PARTIAL":
            return
        reviewed_items = self.annotation.get("reviewed_intervals", [])
        if not reviewed_items:
            return

        reviewed: list[_Interval] = []
        for index, item in enumerate(reviewed_items):
            if not isinstance(item, dict):
                continue
            start = self._safe_number(item.get("start_seconds"))
            end = self._safe_number(item.get("end_seconds"))
            if start is not None and end is not None and start < end:
                reviewed.append(
                    _Interval("REVIEWED", str(index), start, end)
                )

        for interval in self.classification_intervals:
            if interval.category == "EXCLUDED":
                continue
            if not any(
                candidate.start <= interval.start
                and interval.end <= candidate.end
                for candidate in reviewed
            ):
                self.errors.append(
                    "Bei PARTIAL liegt ein fachliches Intervall außerhalb "
                    "der dokumentierten reviewed_intervals: "
                    f"{interval.identifier}."
                )

    def _simple_interval(
        self,
        item: Any,
        *,
        path: str,
        category: str,
        wav_duration: float,
        extra_required: set[str],
        extra_allowed: set[str],
        identifier: str | None = None,
        identifier_key: str | None = None,
    ) -> _Interval | None:
        if not isinstance(item, dict):
            self.errors.append(f"{path} muss ein JSON-Objekt sein.")
            return None
        required = {"start_seconds", "end_seconds"} | extra_required
        allowed = required | extra_allowed
        if identifier_key is not None:
            required.add(identifier_key)
            allowed.add(identifier_key)
        self._validate_object_keys(
            item,
            required=required,
            allowed=allowed,
            path=path,
        )
        resolved_id = identifier
        if identifier_key is not None:
            resolved_id = self._identifier(
                item.get(identifier_key),
                path,
            )
        assert resolved_id is not None
        return self._interval_from_values(
            category=category,
            identifier=resolved_id,
            start_value=item.get("start_seconds"),
            end_value=item.get("end_seconds"),
            path=path,
            wav_duration=wav_duration,
        )

    def _interval_from_values(
        self,
        *,
        category: str,
        identifier: str,
        start_value: Any,
        end_value: Any,
        path: str,
        wav_duration: float,
    ) -> _Interval | None:
        start = self._number(start_value, f"{path}.start_seconds")
        end = self._number(end_value, f"{path}.end_seconds")
        if start is None or end is None:
            return None
        if start < 0 or end < 0:
            self.errors.append(f"{path} enthält eine negative Zeit.")
        if start >= end:
            self.errors.append(
                f"{path} verletzt start_seconds < end_seconds."
            )
        if end > wav_duration:
            self.errors.append(
                f"{path} liegt außerhalb der WAV-Dauer von "
                f"{wav_duration:.6f} s."
            )
        return _Interval(category, identifier, start, end)

    def _identifier(self, value: Any, path: str) -> str:
        if not isinstance(value, str) or not value.strip():
            self.errors.append(f"{path} enthält keine gültige ID.")
            return f"INVALID-{len(self.identifiers)}"
        identifier = value.strip()
        if identifier in self.identifiers:
            self.errors.append(
                f"Doppelte Ereignis- oder Intervall-ID: {identifier}."
            )
        self.identifiers.add(identifier)
        return identifier

    def _validate_object_keys(
        self,
        value: dict[str, Any],
        *,
        required: set[str],
        allowed: set[str],
        path: str,
    ) -> None:
        missing = required - set(value)
        unknown = set(value) - allowed
        for key in sorted(missing):
            self.errors.append(f"{path}: Pflichtfeld fehlt: {key}.")
        for key in sorted(unknown):
            self.errors.append(f"{path}: unbekanntes Feld: {key}.")

    def _reject_forbidden_keys(self, value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if key in FORBIDDEN_EXPERIMENT_KEYS:
                    self.errors.append(
                        f"Verbotenes Fenster-, Modell- oder nicht freigegebenes "
                        f"Feld: {child_path}."
                    )
                self._reject_forbidden_keys(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._reject_forbidden_keys(child, f"{path}[{index}]")

    def _wav_duration(self) -> float:
        try:
            return float(self.validation_report["wav"]["duration_seconds"])
        except (KeyError, TypeError, ValueError):
            return 0.0

    def _validate_uuid(self, value: str) -> None:
        try:
            parsed = uuid.UUID(value)
        except ValueError:
            self.errors.append("session_id ist keine gültige UUID.")
            return
        if str(parsed) != value.lower():
            self.errors.append(
                "session_id muss als kanonische UUID geschrieben sein."
            )

    def _find_files(self, file_name: str) -> list[Path]:
        return sorted(
            path
            for path in self.data_root.rglob(file_name)
            if path.is_file()
        )

    def _read_json(self, path: Path, description: str) -> dict[str, Any]:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            raise AnnotationValidationError([
                f"{description} nicht lesbar: {path}"
            ]) from exception
        if not isinstance(loaded, dict):
            raise AnnotationValidationError([
                f"{description} muss ein JSON-Objekt enthalten."
            ])
        return loaded

    def _number(self, value: Any, path: str) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.errors.append(f"{path} muss numerisch sein.")
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            self.errors.append(f"{path} muss endlich sein.")
            return None
        return numeric

    @staticmethod
    def _safe_number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    def _validate_notes(self, value: Any, path: str) -> None:
        if value is not None and not isinstance(value, str):
            self.errors.append(f"{path} muss null oder eine Zeichenkette sein.")


def validate_annotation_file(
    *,
    data_root: str | Path,
    annotation_path: str | Path,
    marker_tolerance_seconds: float = MARKER_TOLERANCE_SECONDS,
) -> AnnotationValidationReport:
    root = Path(data_root).expanduser().resolve()
    path = Path(annotation_path).expanduser().resolve()
    if not root.is_dir():
        raise AnnotationValidationError([
            f"Datenordner nicht gefunden: {root}"
        ])
    if not path.is_file():
        raise AnnotationValidationError([
            f"Annotationsdatei nicht gefunden: {path}"
        ])
    return _AnnotationValidator(
        data_root=root,
        annotation_path=path,
        marker_tolerance_seconds=marker_tolerance_seconds,
    ).validate()


def validate_annotation_for_session(
    *,
    data_root: str | Path,
    session_id: str,
    marker_tolerance_seconds: float = MARKER_TOLERANCE_SECONDS,
) -> AnnotationValidationReport:
    root = Path(data_root).expanduser().resolve()
    file_name = f"annotation_{session_id}.json"
    matches = sorted(
        path for path in root.rglob(file_name) if path.is_file()
    )
    if not matches:
        raise AnnotationValidationError([
            f"Annotationsdatei nicht gefunden: {file_name}"
        ])
    if len(matches) > 1:
        raise AnnotationValidationError([
            f"Annotationsdatei mehrfach gefunden: {file_name}"
        ])
    return validate_annotation_file(
        data_root=root,
        annotation_path=matches[0],
        marker_tolerance_seconds=marker_tolerance_seconds,
    )
