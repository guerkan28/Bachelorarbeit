from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


AUDIO_TIME_REFERENCE = "AUDIO_SECONDS_FROM_WAV_START"
WATCH_TIME_REFERENCE = "WATCH_SECONDS_FROM_SENSOR_SESSION_START"
VIDEO_TIME_REFERENCE = "VIDEO_SECONDS_FROM_REFERENCE_VIDEO_START"


class VideoAlignmentError(ValueError):
    """Wird ausgelöst, wenn die Video-zu-Audio-Abbildung ungültig ist."""


@dataclass(frozen=True)
class LinearTimeMapping:
    scale: float
    offset_seconds: float

    def apply(self, source_seconds: float) -> float:
        return self.scale * source_seconds + self.offset_seconds

    def to_dict(self) -> dict[str, float]:
        return {
            "scale": self.scale,
            "offset_seconds": self.offset_seconds,
        }


def build_watch_to_audio_mapping(
    synchronization_report: dict[str, Any],
) -> dict[str, Any]:
    """Erzeugt die verschachtelte Watch-Abbildung aus bestehenden Berichtsfeldern."""
    selected_offset = _finite_number(
        synchronization_report.get("selected_offset_seconds"),
        "selected_offset_seconds",
    )
    start_offset = _finite_number(
        synchronization_report.get("start_offset_seconds"),
        "start_offset_seconds",
    )
    end_offset = _finite_number(
        synchronization_report.get("end_offset_seconds"),
        "end_offset_seconds",
    )
    method = synchronization_report.get("selected_alignment_method")
    if not isinstance(method, str) or not method:
        raise VideoAlignmentError(
            "selected_alignment_method fehlt im Synchronisationsbericht."
        )

    return {
        "source_time_reference": WATCH_TIME_REFERENCE,
        "target_time_reference": AUDIO_TIME_REFERENCE,
        "method": method,
        "scale": 1.0,
        "offset_seconds": selected_offset,
        "formula": "audio_seconds = scale * watch_seconds + offset_seconds",
        "start_offset_seconds": start_offset,
        "end_offset_seconds": end_offset,
        "offset_consistent": bool(
            synchronization_report.get("offset_consistent")
        ),
    }


def estimate_video_to_audio_mapping(
    *,
    audio_start_marker_times_seconds: Iterable[float],
    video_start_marker_times_seconds: Iterable[float],
    audio_end_marker_times_seconds: Iterable[float],
    video_end_marker_times_seconds: Iterable[float],
    maximum_marker_residual_seconds: float = 0.100,
) -> dict[str, Any]:
    """Schätzt eine lineare Video-zu-Audio-Abbildung aus Anfangs- und Endmarker."""
    audio_start = _validated_marker_group(
        audio_start_marker_times_seconds,
        "Audio-Anfangsmarker",
    )
    video_start = _validated_marker_group(
        video_start_marker_times_seconds,
        "Video-Anfangsmarker",
    )
    audio_end = _validated_marker_group(
        audio_end_marker_times_seconds,
        "Audio-Endmarker",
    )
    video_end = _validated_marker_group(
        video_end_marker_times_seconds,
        "Video-Endmarker",
    )

    if len(audio_start) != len(video_start):
        raise VideoAlignmentError(
            "Audio- und Video-Anfangsmarker müssen gleich viele Impulse enthalten."
        )
    if len(audio_end) != len(video_end):
        raise VideoAlignmentError(
            "Audio- und Video-Endmarker müssen gleich viele Impulse enthalten."
        )
    if maximum_marker_residual_seconds <= 0:
        raise VideoAlignmentError(
            "Die Markertoleranz muss positiv sein."
        )

    audio_start_center = float(statistics.median(audio_start))
    video_start_center = float(statistics.median(video_start))
    audio_end_center = float(statistics.median(audio_end))
    video_end_center = float(statistics.median(video_end))

    video_span = video_end_center - video_start_center
    audio_span = audio_end_center - audio_start_center
    if video_span <= 0 or audio_span <= 0:
        raise VideoAlignmentError(
            "Anfangs- und Endmarker müssen eine positive Sitzungsdauer aufspannen."
        )

    mapping = LinearTimeMapping(
        scale=audio_span / video_span,
        offset_seconds=(
            audio_start_center
            - (audio_span / video_span) * video_start_center
        ),
    )
    if not math.isfinite(mapping.scale) or mapping.scale <= 0:
        raise VideoAlignmentError(
            "Die geschätzte lineare Skalierung ist nicht positiv und endlich."
        )
    if not math.isfinite(mapping.offset_seconds):
        raise VideoAlignmentError(
            "Der geschätzte Videooffset ist nicht endlich."
        )

    start_residuals = [
        mapping.apply(video_time) - audio_time
        for video_time, audio_time in zip(video_start, audio_start)
    ]
    end_residuals = [
        mapping.apply(video_time) - audio_time
        for video_time, audio_time in zip(video_end, audio_end)
    ]
    all_residuals = start_residuals + end_residuals
    maximum_absolute_residual = max(
        abs(value) for value in all_residuals
    )

    warnings: list[str] = []
    if maximum_absolute_residual > maximum_marker_residual_seconds:
        warnings.append(
            "Mindestens ein Video-Markerresiduum überschreitet die "
            "konfigurierte Toleranz."
        )

    return {
        "source_time_reference": VIDEO_TIME_REFERENCE,
        "target_time_reference": AUDIO_TIME_REFERENCE,
        "method": "LINEAR_TWO_POINT",
        "scale": mapping.scale,
        "offset_seconds": mapping.offset_seconds,
        "formula": "audio_seconds = scale * video_seconds + offset_seconds",
        "reference_video_file": None,
        "start_anchor": {
            "video_marker_times_seconds": video_start,
            "audio_marker_times_seconds": audio_start,
            "video_center_seconds": video_start_center,
            "audio_center_seconds": audio_start_center,
            "paired_residuals_seconds": start_residuals,
        },
        "end_anchor": {
            "video_marker_times_seconds": video_end,
            "audio_marker_times_seconds": audio_end,
            "video_center_seconds": video_end_center,
            "audio_center_seconds": audio_end_center,
            "paired_residuals_seconds": end_residuals,
        },
        "maximum_absolute_marker_residual_seconds": (
            maximum_absolute_residual
        ),
        "marker_residual_tolerance_seconds": (
            maximum_marker_residual_seconds
        ),
        "warnings": warnings,
    }


def add_video_alignment_to_report(
    *,
    synchronization_report_path: str | Path,
    reference_video_file: str,
    video_start_marker_times_seconds: Iterable[float],
    video_end_marker_times_seconds: Iterable[float],
    output_path: str | Path | None = None,
    maximum_marker_residual_seconds: float = 0.100,
) -> dict[str, Any]:
    """Ergänzt einen bestehenden Synchronisationsbericht rückwärtskompatibel."""
    report_path = Path(
        synchronization_report_path
    ).expanduser().resolve()
    report = _read_json_object(report_path)

    session_id = report.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise VideoAlignmentError(
            "Der Synchronisationsbericht enthält keine Session-ID."
        )
    if (
        not isinstance(reference_video_file, str)
        or not reference_video_file.strip()
    ):
        raise VideoAlignmentError(
            "Der Dateiname des Referenzvideos fehlt."
        )
    reference_video_file = reference_video_file.strip()
    if session_id not in reference_video_file:
        raise VideoAlignmentError(
            "Der Referenzvideodateiname enthält nicht die Session-ID."
        )

    start_audio = _marker_times_from_report(
        report,
        "start_audio_marker",
    )
    end_audio = _marker_times_from_report(
        report,
        "end_audio_marker",
    )

    video_mapping = estimate_video_to_audio_mapping(
        audio_start_marker_times_seconds=start_audio,
        video_start_marker_times_seconds=(
            video_start_marker_times_seconds
        ),
        audio_end_marker_times_seconds=end_audio,
        video_end_marker_times_seconds=(
            video_end_marker_times_seconds
        ),
        maximum_marker_residual_seconds=(
            maximum_marker_residual_seconds
        ),
    )
    video_mapping["reference_video_file"] = reference_video_file

    time_mappings = report.get("time_mappings")
    if time_mappings is None:
        time_mappings = {}
    if not isinstance(time_mappings, dict):
        raise VideoAlignmentError(
            "time_mappings muss ein JSON-Objekt sein."
        )
    if "watch_to_audio" not in time_mappings:
        time_mappings["watch_to_audio"] = (
            build_watch_to_audio_mapping(report)
        )
    time_mappings["video_to_audio"] = video_mapping

    report["schema_version"] = max(
        int(report.get("schema_version", 1)),
        2,
    )
    report["time_mappings"] = time_mappings

    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else report_path
    )
    temporary_path = resolved_output.with_suffix(
        resolved_output.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(resolved_output)

    return report


def apply_linear_mapping(
    source_seconds: float,
    mapping: dict[str, Any],
) -> float:
    scale = _finite_number(mapping.get("scale"), "scale")
    offset = _finite_number(
        mapping.get("offset_seconds"),
        "offset_seconds",
    )
    return scale * source_seconds + offset


def _validated_marker_group(
    values: Iterable[float],
    description: str,
) -> list[float]:
    result = [
        _finite_number(value, description)
        for value in values
    ]
    if not result:
        raise VideoAlignmentError(
            f"{description} enthält keine Markerzeiten."
        )
    if result != sorted(result):
        raise VideoAlignmentError(
            f"{description} muss aufsteigend sortiert sein."
        )
    if len(set(result)) != len(result):
        raise VideoAlignmentError(
            f"{description} enthält doppelte Markerzeiten."
        )
    return result


def _marker_times_from_report(
    report: dict[str, Any],
    key: str,
) -> list[float]:
    marker = report.get(key)
    if not isinstance(marker, dict):
        raise VideoAlignmentError(
            f"{key} fehlt im Synchronisationsbericht."
        )
    times = marker.get("peak_times_seconds")
    if not isinstance(times, list):
        raise VideoAlignmentError(
            f"{key}.peak_times_seconds fehlt."
        )
    return _validated_marker_group(times, key)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise VideoAlignmentError(
            f"Synchronisationsbericht nicht lesbar: {path}"
        ) from exception
    if not isinstance(loaded, dict):
        raise VideoAlignmentError(
            "Der Synchronisationsbericht muss ein JSON-Objekt enthalten."
        )
    return loaded


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoAlignmentError(
            f"{field_name} muss numerisch sein."
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise VideoAlignmentError(
            f"{field_name} muss endlich sein."
        )
    return numeric
