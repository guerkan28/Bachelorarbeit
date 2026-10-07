from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


VIDEO_TIME_REFERENCE = "VIDEO_SECONDS_FROM_REFERENCE_VIDEO_START"
AUDIO_TIME_REFERENCE = "AUDIO_SECONDS_FROM_WAV_START"


class VideoTimeConversionError(ValueError):
    """Fehler bei der sicheren Umrechnung von Video- auf Audiozeit."""


def load_video_to_audio_mapping(
    synchronization_report_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lädt und validiert die Video-zu-Audio-Abbildung aus einem Synchronisationsbericht."""
    path = Path(synchronization_report_path).expanduser().resolve()

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exception:
        raise VideoTimeConversionError(
            f"Synchronisationsbericht konnte nicht gelesen werden: {path}"
        ) from exception
    except json.JSONDecodeError as exception:
        raise VideoTimeConversionError(
            f"Synchronisationsbericht enthält kein gültiges JSON: {path}"
        ) from exception

    if not isinstance(report, dict):
        raise VideoTimeConversionError(
            "Der Synchronisationsbericht muss ein JSON-Objekt sein."
        )

    mappings = report.get("time_mappings")
    if not isinstance(mappings, dict):
        raise VideoTimeConversionError(
            "Im Synchronisationsbericht fehlt 'time_mappings'."
        )

    mapping = mappings.get("video_to_audio")
    if not isinstance(mapping, dict):
        raise VideoTimeConversionError(
            "Im Synchronisationsbericht ist noch keine Video-zu-Audio-Abbildung vorhanden."
        )

    if mapping.get("source_time_reference") != VIDEO_TIME_REFERENCE:
        raise VideoTimeConversionError(
            "Die Quellzeitreferenz der Videoabbildung ist nicht unterstützt."
        )
    if mapping.get("target_time_reference") != AUDIO_TIME_REFERENCE:
        raise VideoTimeConversionError(
            "Die Zielzeitreferenz der Videoabbildung ist nicht AUDIO_SECONDS_FROM_WAV_START."
        )

    scale = _finite_number(mapping.get("scale"), "scale")
    offset = _finite_number(mapping.get("offset_seconds"), "offset_seconds")
    if scale <= 0.0:
        raise VideoTimeConversionError(
            "Die Skalierung der Videoabbildung muss größer als 0 sein."
        )

    # Speichert normalisierte numerische Werte mit einheitlicher float-Semantik.
    mapping = dict(mapping)
    mapping["scale"] = scale
    mapping["offset_seconds"] = offset

    return report, mapping


def convert_video_times(
    video_times_seconds: Sequence[float],
    *,
    scale: float,
    offset_seconds: float,
    audio_duration_seconds: float | None = None,
    labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Überführt Videozeitpunkte in die kanonische Audiozeitreferenz."""
    if not video_times_seconds:
        raise VideoTimeConversionError(
            "Mindestens ein Videozeitpunkt muss angegeben werden."
        )

    scale = _finite_number(scale, "scale")
    offset_seconds = _finite_number(offset_seconds, "offset_seconds")
    if scale <= 0.0:
        raise VideoTimeConversionError(
            "Die Skalierung muss größer als 0 sein."
        )

    if audio_duration_seconds is not None:
        audio_duration_seconds = _finite_number(
            audio_duration_seconds,
            "audio_duration_seconds",
        )
        if audio_duration_seconds <= 0.0:
            raise VideoTimeConversionError(
                "Die Audiodauer muss größer als 0 sein."
            )

    if labels is not None and len(labels) != len(video_times_seconds):
        raise VideoTimeConversionError(
            "Die Anzahl der Labels muss der Anzahl der Videozeiten entsprechen."
        )

    results: list[dict[str, Any]] = []
    for index, raw_video_seconds in enumerate(video_times_seconds):
        video_seconds = _finite_number(
            raw_video_seconds,
            f"video_times_seconds[{index}]",
        )
        if video_seconds < 0.0:
            raise VideoTimeConversionError(
                "Videozeitpunkte dürfen nicht negativ sein."
            )

        audio_seconds = scale * video_seconds + offset_seconds
        result: dict[str, Any] = {
            "index": index + 1,
            "video_seconds": video_seconds,
            "audio_seconds": audio_seconds,
        }
        if labels is not None:
            result["label"] = str(labels[index])
        if audio_duration_seconds is not None:
            result["within_audio_recording"] = (
                0.0 <= audio_seconds < audio_duration_seconds
            )
        results.append(result)

    return results


def create_conversion_report(
    synchronization_report_path: str | Path,
    video_times_seconds: Sequence[float],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Erzeugt einen serialisierbaren Konvertierungsbericht aus einem Synchronisationsbericht."""
    report, mapping = load_video_to_audio_mapping(
        synchronization_report_path
    )

    audio_duration = report.get("audio_duration_seconds")
    if audio_duration is not None:
        audio_duration = _finite_number(
            audio_duration,
            "audio_duration_seconds",
        )

    results = convert_video_times(
        video_times_seconds,
        scale=mapping["scale"],
        offset_seconds=mapping["offset_seconds"],
        audio_duration_seconds=audio_duration,
        labels=labels,
    )

    return {
        "session_id": report.get("session_id"),
        "source_time_reference": VIDEO_TIME_REFERENCE,
        "target_time_reference": AUDIO_TIME_REFERENCE,
        "formula": mapping.get(
            "formula",
            "audio_seconds = scale * video_seconds + offset_seconds",
        ),
        "scale": mapping["scale"],
        "offset_seconds": mapping["offset_seconds"],
        "audio_duration_seconds": audio_duration,
        "results": results,
    }


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise VideoTimeConversionError(
            f"{field_name} muss numerisch sein."
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exception:
        raise VideoTimeConversionError(
            f"{field_name} muss numerisch sein."
        ) from exception
    if not math.isfinite(number):
        raise VideoTimeConversionError(
            f"{field_name} muss endlich sein."
        )
    return number


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rechnet manuell aus dem Referenzvideo abgelesene Zeitpunkte "
            "mithilfe des Synchronisationsberichts in "
            "AUDIO_SECONDS_FROM_WAV_START um."
        )
    )
    parser.add_argument(
        "--synchronization-report",
        required=True,
        help="Pfad zu synchronization_report_<UUID>.json",
    )
    parser.add_argument(
        "--video-times",
        nargs="+",
        type=float,
        required=True,
        metavar="SEKUNDE",
        help="Ein oder mehrere Zeitpunkte ab Beginn des Referenzvideos.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help=(
            "Optionale Bezeichnungen in derselben Reihenfolge wie --video-times."
        ),
    )
    arguments = parser.parse_args()

    try:
        conversion_report = create_conversion_report(
            synchronization_report_path=arguments.synchronization_report,
            video_times_seconds=arguments.video_times,
            labels=arguments.labels,
        )
    except VideoTimeConversionError as exception:
        print(
            f"Konvertierungsfehler: {exception}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            conversion_report,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
