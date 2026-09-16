from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .signal_loader import (
    AudioSignal,
    SessionSignals,
    load_session_signals,
)
from .synchronization import apply_watch_offset
from .video_alignment import AUDIO_TIME_REFERENCE
from .window_dataset import WindowRecord


WATCH_TIME_REFERENCE = "WATCH_SECONDS_FROM_SENSOR_SESSION_START"
WATCH_ALIGNMENT_METHOD = "CONSTANT_OFFSET"

_TIME_EPSILON_SECONDS = 1e-9


class WindowSignalError(ValueError):
    """Fehler beim Erzeugen synchronisierter ML-Signalfenster."""


@dataclass(frozen=True)
class WatchToAudioMapping:
    method: str
    scale: float
    offset_seconds: float
    source_time_reference: str
    target_time_reference: str
    synchronization_report_path: Path


@dataclass(frozen=True)
class AlignedSessionSignals:
    session_id: str
    signals: SessionSignals
    watch_to_audio_mapping: WatchToAudioMapping
    accelerometer: pd.DataFrame
    gyroscope: pd.DataFrame


@dataclass(frozen=True)
class SynchronizedWindowSignals:
    window: WindowRecord
    audio_samples: np.ndarray
    audio_time_seconds: np.ndarray
    audio_sample_rate_hz: int
    accelerometer: pd.DataFrame
    gyroscope: pd.DataFrame


def load_aligned_session_signals(
    *,
    data_root: str | Path,
    session_id: str,
) -> AlignedSessionSignals:
    """
    Lädt Audio- und Watch-Signale und überführt die Watch-Zeit
    mit dem kanonischen CONSTANT_OFFSET-Mapping auf die Audiozeitachse.
    """
    root = Path(data_root).expanduser().resolve()

    signals = load_session_signals(
        data_root=root,
        session_id=session_id,
    )

    mapping = load_watch_to_audio_mapping(
        data_root=root,
        session_id=session_id,
    )

    accelerometer = _align_sensor_frame(
        signals.accelerometer,
        mapping,
    )
    gyroscope = _align_sensor_frame(
        signals.gyroscope,
        mapping,
    )

    return AlignedSessionSignals(
        session_id=session_id,
        signals=signals,
        watch_to_audio_mapping=mapping,
        accelerometer=accelerometer,
        gyroscope=gyroscope,
    )


def load_watch_to_audio_mapping(
    *,
    data_root: str | Path,
    session_id: str,
) -> WatchToAudioMapping:
    root = Path(data_root).expanduser().resolve()

    report_name = (
        f"synchronization_report_{session_id}.json"
    )

    matches = sorted(
        path
        for path in root.rglob(report_name)
        if path.is_file()
    )

    if not matches:
        raise WindowSignalError(
            f"Synchronisationsbericht nicht gefunden: {report_name}"
        )

    if len(matches) > 1:
        raise WindowSignalError(
            f"Synchronisationsbericht mehrfach gefunden: {report_name}"
        )

    report_path = matches[0]

    try:
        report = json.loads(
            report_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exception:
        raise WindowSignalError(
            f"Synchronisationsbericht nicht lesbar: {report_path}"
        ) from exception

    if not isinstance(report, dict):
        raise WindowSignalError(
            "Synchronisationsbericht muss ein JSON-Objekt enthalten."
        )

    if report.get("session_id") != session_id:
        raise WindowSignalError(
            "Session-ID im Synchronisationsbericht stimmt nicht überein."
        )

    mappings = report.get("time_mappings")

    if not isinstance(mappings, dict):
        raise WindowSignalError(
            "Synchronisationsbericht enthält keine time_mappings."
        )

    mapping = mappings.get("watch_to_audio")

    if not isinstance(mapping, dict):
        raise WindowSignalError(
            "watch_to_audio-Mapping fehlt."
        )

    source_reference = mapping.get(
        "source_time_reference"
    )
    target_reference = mapping.get(
        "target_time_reference"
    )
    method = mapping.get("method")

    if source_reference != WATCH_TIME_REFERENCE:
        raise WindowSignalError(
            "watch_to_audio besitzt eine falsche Quellzeitreferenz."
        )

    if target_reference != AUDIO_TIME_REFERENCE:
        raise WindowSignalError(
            "watch_to_audio besitzt eine falsche Zielzeitreferenz."
        )

    if method != WATCH_ALIGNMENT_METHOD:
        raise WindowSignalError(
            "watch_to_audio muss CONSTANT_OFFSET verwenden."
        )

    scale = _finite_number(
        mapping.get("scale"),
        "watch_to_audio.scale",
    )

    offset = _finite_number(
        mapping.get("offset_seconds"),
        "watch_to_audio.offset_seconds",
    )

    if not math.isclose(
        scale,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise WindowSignalError(
            "CONSTANT_OFFSET erfordert scale = 1.0."
        )

    return WatchToAudioMapping(
        method=method,
        scale=scale,
        offset_seconds=offset,
        source_time_reference=source_reference,
        target_time_reference=target_reference,
        synchronization_report_path=report_path,
    )


def extract_synchronized_window(
    *,
    aligned_session: AlignedSessionSignals,
    window: WindowRecord,
) -> SynchronizedWindowSignals:
    """
    Schneidet für ein ML-Fenster Audio, Accelerometer und Gyroskop
    auf derselben AUDIO_SECONDS_FROM_WAV_START-Zeitachse aus.
    """
    if window.session_id != aligned_session.session_id:
        raise WindowSignalError(
            "Fenster und geladene Sitzung besitzen unterschiedliche Session-IDs."
        )

    if window.time_reference != AUDIO_TIME_REFERENCE:
        raise WindowSignalError(
            f"Fenster muss {AUDIO_TIME_REFERENCE} verwenden."
        )

    start = _finite_number(
        window.window_start_seconds,
        "window_start_seconds",
    )
    end = _finite_number(
        window.window_end_seconds,
        "window_end_seconds",
    )

    if start < 0 or end <= start:
        raise WindowSignalError(
            "Fenstergrenzen sind ungültig."
        )

    audio = aligned_session.signals.audio

    if (
        end
        > audio.duration_seconds
        + _TIME_EPSILON_SECONDS
    ):
        raise WindowSignalError(
            "Fenster liegt außerhalb der Audiodauer."
        )

    audio_start_index = int(
        np.searchsorted(
            audio.time_seconds,
            start,
            side="left",
        )
    )

    audio_end_index = int(
        np.searchsorted(
            audio.time_seconds,
            end,
            side="left",
        )
    )

    audio_samples = audio.samples[
        audio_start_index:audio_end_index
    ].copy()

    audio_times = audio.time_seconds[
        audio_start_index:audio_end_index
    ].copy()

    if audio_samples.size == 0:
        raise WindowSignalError(
            "Das Fenster enthält keine Audiosamples."
        )

    accelerometer = _slice_sensor_frame(
        aligned_session.accelerometer,
        start_seconds=start,
        end_seconds=end,
        sensor_name="ACCELEROMETER",
    )

    gyroscope = _slice_sensor_frame(
        aligned_session.gyroscope,
        start_seconds=start,
        end_seconds=end,
        sensor_name="GYROSCOPE",
    )

    return SynchronizedWindowSignals(
        window=window,
        audio_samples=audio_samples,
        audio_time_seconds=audio_times,
        audio_sample_rate_hz=audio.sample_rate_hz,
        accelerometer=accelerometer,
        gyroscope=gyroscope,
    )


def _align_sensor_frame(
    frame: pd.DataFrame,
    mapping: WatchToAudioMapping,
) -> pd.DataFrame:
    if "time_seconds" not in frame.columns:
        raise WindowSignalError(
            "Sensorframe enthält keine time_seconds-Spalte."
        )

    aligned = frame.copy()

    aligned["audio_time_seconds"] = apply_watch_offset(
        aligned["time_seconds"].to_numpy(
            dtype=np.float64
        ),
        mapping.offset_seconds,
    )

    values = aligned[
        "audio_time_seconds"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise WindowSignalError(
            "Ausgerichtete Watch-Zeit enthält ungültige Werte."
        )

    if (
        values.size > 1
        and (np.diff(values) < 0).any()
    ):
        raise WindowSignalError(
            "Ausgerichtete Watch-Zeit ist nicht aufsteigend sortiert."
        )

    return aligned


def _slice_sensor_frame(
    frame: pd.DataFrame,
    *,
    start_seconds: float,
    end_seconds: float,
    sensor_name: str,
) -> pd.DataFrame:
    if "audio_time_seconds" not in frame.columns:
        raise WindowSignalError(
            f"{sensor_name}: audio_time_seconds fehlt."
        )

    mask = (
        (
            frame["audio_time_seconds"]
            >= start_seconds
        )
        & (
            frame["audio_time_seconds"]
            < end_seconds
        )
    )

    sliced = (
        frame.loc[mask]
        .reset_index(drop=True)
        .copy()
    )

    if sliced.empty:
        raise WindowSignalError(
            f"{sensor_name}: Fenster enthält keine Messwerte."
        )

    return sliced


def _finite_number(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise WindowSignalError(
            f"{name} muss numerisch sein."
        )

    numeric = float(value)

    if not math.isfinite(numeric):
        raise WindowSignalError(
            f"{name} muss endlich sein."
        )

    return numeric