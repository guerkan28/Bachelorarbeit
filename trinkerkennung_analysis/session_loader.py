from __future__ import annotations

import csv
import json
import wave
from collections import Counter
from pathlib import Path
from typing import Any


class SessionValidationError(ValueError):
    pass


def load_and_validate_session(
    data_root: str | Path,
    session_id: str,
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    session_id = session_id.strip()

    if not root.is_dir():
        raise SessionValidationError(
            f"Datenordner nicht gefunden: {root}"
        )
    if not session_id:
        raise SessionValidationError(
            "Die Session-ID darf nicht leer sein."
        )

    metadata_path = _find_one(
        root,
        f"session_{session_id}.json",
        "Metadatendatei",
    )
    metadata = _read_json(metadata_path)

    if metadata.get("session_id") != session_id:
        raise SessionValidationError(
            "Session-ID in JSON stimmt nicht überein."
        )

    wav_name = _required_text(
        metadata,
        "phone_audio_file_name",
    )
    csv_name = _required_text(
        metadata,
        "watch_sensor_file_name",
    )

    wav_path = _find_one(
        root,
        wav_name,
        "WAV-Datei",
    )
    csv_path = _find_one(
        root,
        csv_name,
        "CSV-Datei",
    )

    wav_summary = _read_wav(wav_path)
    csv_summary = _read_csv(
        csv_path,
        session_id,
    )

    warnings: list[str] = []
    if metadata.get("status") != "COMPLETED":
        warnings.append(
            "Metadatenstatus ist nicht COMPLETED."
        )
    if metadata.get("phone_audio_success") is not True:
        warnings.append(
            "phone_audio_success ist nicht True."
        )
    if metadata.get("watch_recording_success") is not True:
        warnings.append(
            "watch_recording_success ist nicht True."
        )

    phone_duration = _duration_seconds(
        metadata.get("phone_audio_start_epoch_ms"),
        metadata.get("phone_audio_stop_epoch_ms"),
    )
    watch_duration = _duration_seconds(
        metadata.get("watch_start_epoch_ms"),
        metadata.get("watch_stop_epoch_ms"),
    )

    if (
        phone_duration is not None and
        abs(phone_duration - wav_summary["duration_seconds"]) > 0.25
    ):
        warnings.append(
            "WAV-Dauer weicht um mehr als 0,25 s "
            "von den Smartphone-Metadaten ab."
        )

    if (
        watch_duration is not None and
        abs(watch_duration - csv_summary["duration_seconds"]) > 0.25
    ):
        warnings.append(
            "CSV-Dauer weicht um mehr als 0,25 s "
            "von den Watch-Metadaten ab."
        )

    return {
        "session_id": session_id,
        "metadata_path": str(metadata_path),
        "metadata_status": metadata.get("status"),
        "phone_audio_success": metadata.get(
            "phone_audio_success"
        ),
        "watch_recording_success": metadata.get(
            "watch_recording_success"
        ),
        "wav": wav_summary,
        "csv": csv_summary,
        "phone_audio_duration_from_metadata_seconds":
            phone_duration,
        "watch_duration_from_metadata_seconds":
            watch_duration,
        "phone_start_round_trip_seconds":
            _duration_seconds(
                metadata.get(
                    "phone_start_command_epoch_ms"
                ),
                metadata.get(
                    "phone_started_received_epoch_ms"
                ),
            ),
        "phone_stop_round_trip_seconds":
            _duration_seconds(
                metadata.get(
                    "phone_stop_command_epoch_ms"
                ),
                metadata.get(
                    "phone_stopped_received_epoch_ms"
                ),
            ),
        "warnings": warnings,
    }


def _find_one(
    root: Path,
    file_name: str,
    description: str,
) -> Path:
    matches = [
        path for path in root.rglob(file_name)
        if path.is_file()
    ]
    if not matches:
        raise SessionValidationError(
            f"{description} nicht gefunden: {file_name}"
        )
    if len(matches) > 1:
        paths = "\n".join(
            f"- {path}" for path in matches
        )
        raise SessionValidationError(
            f"{description} mehrfach gefunden:\n{paths}"
        )
    return matches[0]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exception:
        raise SessionValidationError(
            f"JSON nicht lesbar: {path}"
        ) from exception

    if not isinstance(loaded, dict):
        raise SessionValidationError(
            "JSON muss ein Objekt enthalten."
        )
    return loaded


def _required_text(
    metadata: dict[str, Any],
    key: str,
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SessionValidationError(
            f"Metadatenfeld fehlt oder ist leer: {key}"
        )
    return value.strip()


def _read_wav(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
    except (OSError, EOFError, wave.Error) as exception:
        raise SessionValidationError(
            f"WAV ungültig: {path}"
        ) from exception

    if sample_rate <= 0 or frame_count <= 0:
        raise SessionValidationError(
            "WAV enthält keine gültigen Audiodaten."
        )

    return {
        "path": str(path),
        "file_name": path.name,
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "duration_seconds": round(
            frame_count / sample_rate,
            6,
        ),
    }


def _read_csv(
    path: Path,
    expected_session_id: str,
) -> dict[str, Any]:
    required = {
        "session_id",
        "sensor",
        "relative_time_ns",
    }
    sensor_counts: Counter[str] = Counter()
    session_ids: set[str] = set()
    relative_times: list[int] = []

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise SessionValidationError(
                    "CSV besitzt keine Kopfzeile."
                )

            missing = required - set(reader.fieldnames)
            if missing:
                raise SessionValidationError(
                    "CSV-Spalten fehlen: " +
                    ", ".join(sorted(missing))
                )

            for line_number, row in enumerate(
                reader,
                start=2,
            ):
                session_value = (
                    row.get("session_id") or ""
                ).strip()
                sensor_value = (
                    row.get("sensor") or ""
                ).strip()

                try:
                    relative_time = int(
                        (
                            row.get("relative_time_ns")
                            or ""
                        ).strip()
                    )
                except ValueError as exception:
                    raise SessionValidationError(
                        "Ungültiger relativer Zeitwert "
                        f"in Zeile {line_number}."
                    ) from exception

                if relative_time < 0:
                    raise SessionValidationError(
                        "Negativer relativer Zeitwert "
                        f"in Zeile {line_number}."
                    )
                if not session_value or not sensor_value:
                    raise SessionValidationError(
                        f"Leeres Pflichtfeld in Zeile {line_number}."
                    )

                session_ids.add(session_value)
                sensor_counts[sensor_value] += 1
                relative_times.append(relative_time)
    except OSError as exception:
        raise SessionValidationError(
            f"CSV nicht lesbar: {path}"
        ) from exception

    if not relative_times:
        raise SessionValidationError(
            "CSV enthält keine Sensordaten."
        )
    if session_ids != {expected_session_id}:
        raise SessionValidationError(
            "Session-ID im CSV-Inhalt stimmt nicht überein."
        )

    minimum = min(relative_times)
    maximum = max(relative_times)

    return {
        "path": str(path),
        "file_name": path.name,
        "row_count": len(relative_times),
        "session_ids": sorted(session_ids),
        "sensor_counts": dict(
            sorted(sensor_counts.items())
        ),
        "minimum_relative_time_ns": minimum,
        "maximum_relative_time_ns": maximum,
        "duration_seconds": round(
            (maximum - minimum) / 1_000_000_000.0,
            6,
        ),
    }


def _duration_seconds(
    start: Any,
    stop: Any,
) -> float | None:
    if not isinstance(start, int) or not isinstance(stop, int):
        return None
    if stop < start:
        return None
    return round((stop - start) / 1000.0, 6)
