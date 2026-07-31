from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .session_loader import load_and_validate_session


class SignalLoadingError(ValueError):
    """Wird ausgelöst, wenn validierte Dateien nicht als Signale ladbar sind."""


@dataclass(frozen=True)
class AudioSignal:
    samples: np.ndarray
    time_seconds: np.ndarray
    sample_rate_hz: int
    duration_seconds: float
    file_path: Path


@dataclass(frozen=True)
class SessionSignals:
    session_id: str
    validation_report: dict[str, Any]
    audio: AudioSignal
    accelerometer: pd.DataFrame
    gyroscope: pd.DataFrame


def load_session_signals(
    data_root: str | Path,
    session_id: str,
) -> SessionSignals:
    """Validiert und lädt Audio- sowie Watch-Signale einer Sitzung."""
    validation_report = load_and_validate_session(
        data_root=data_root,
        session_id=session_id,
    )

    wav_path = Path(
        validation_report["wav"]["path"]
    )
    csv_path = Path(
        validation_report["csv"]["path"]
    )

    audio = load_pcm16_mono_wav(wav_path)
    sensor_frame = load_sensor_csv(
        csv_path=csv_path,
        expected_session_id=session_id,
    )

    accelerometer = _prepare_sensor(
        sensor_frame,
        sensor_name="ACCELEROMETER",
    )
    gyroscope = _prepare_sensor(
        sensor_frame,
        sensor_name="GYROSCOPE",
    )

    return SessionSignals(
        session_id=session_id,
        validation_report=validation_report,
        audio=audio,
        accelerometer=accelerometer,
        gyroscope=gyroscope,
    )


def load_pcm16_mono_wav(
    wav_path: str | Path,
) -> AudioSignal:
    """Lädt eine unkomprimierte Mono-PCM-WAV-Datei mit 16 Bit."""
    path = Path(wav_path).expanduser().resolve()

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width_bytes = wav_file.getsampwidth()
            sample_rate_hz = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            compression_type = wav_file.getcomptype()
            raw_frames = wav_file.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as exception:
        raise SignalLoadingError(
            f"WAV-Datei konnte nicht als Audiosignal geladen werden: {path}"
        ) from exception

    if compression_type != "NONE":
        raise SignalLoadingError(
            "Es werden ausschließlich unkomprimierte PCM-WAV-Dateien unterstützt."
        )
    if channels != 1:
        raise SignalLoadingError(
            f"Erwartet wurde Mono-Audio, gefunden wurden {channels} Kanäle."
        )
    if sample_width_bytes != 2:
        raise SignalLoadingError(
            "Erwartet wurde PCM-Audio mit 16 Bit pro Sample."
        )
    if sample_rate_hz <= 0 or frame_count <= 0:
        raise SignalLoadingError(
            "Die WAV-Datei enthält keine gültigen Audiodaten."
        )

    integer_samples = np.frombuffer(
        raw_frames,
        dtype="<i2",
    )

    if integer_samples.size != frame_count:
        raise SignalLoadingError(
            "Die gelesene Sample-Anzahl stimmt nicht mit dem WAV-Header überein."
        )

    samples = (
        integer_samples.astype(np.float32) /
        np.float32(32768.0)
    )
    time_seconds = (
        np.arange(
            frame_count,
            dtype=np.float64,
        ) /
        float(sample_rate_hz)
    )

    return AudioSignal(
        samples=samples,
        time_seconds=time_seconds,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=frame_count / float(sample_rate_hz),
        file_path=path,
    )


def load_sensor_csv(
    csv_path: str | Path,
    expected_session_id: str,
) -> pd.DataFrame:
    """Lädt und typisiert die für die Signalanalyse benötigten CSV-Spalten."""
    path = Path(csv_path).expanduser().resolve()

    try:
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )
    except (OSError, pd.errors.ParserError) as exception:
        raise SignalLoadingError(
            f"Sensor-CSV konnte nicht geladen werden: {path}"
        ) from exception

    required_columns = {
        "session_id",
        "sensor",
        "relative_time_ns",
        "x",
        "y",
        "z",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise SignalLoadingError(
            "In der Sensor-CSV fehlen Spalten: "
            + ", ".join(sorted(missing_columns))
        )

    if frame.empty:
        raise SignalLoadingError(
            "Die Sensor-CSV enthält keine Messwerte."
        )

    session_ids = set(
        frame["session_id"]
        .astype(str)
        .str.strip()
        .unique()
    )
    if session_ids != {expected_session_id}:
        raise SignalLoadingError(
            "Die Session-ID der Sensor-CSV stimmt nicht eindeutig überein."
        )

    numeric_columns = [
        "relative_time_ns",
        "x",
        "y",
        "z",
    ]
    try:
        for column in numeric_columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="raise",
            )
    except (ValueError, TypeError) as exception:
        raise SignalLoadingError(
            "Mindestens eine Sensor- oder Zeitspalte enthält ungültige Werte."
        ) from exception

    if (frame["relative_time_ns"] < 0).any():
        raise SignalLoadingError(
            "Die Sensor-CSV enthält negative relative Zeitwerte."
        )

    values = frame[["x", "y", "z"]].to_numpy(
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise SignalLoadingError(
            "Die Sensorachsen enthalten NaN oder unendliche Werte."
        )

    return frame


def _prepare_sensor(
    frame: pd.DataFrame,
    sensor_name: str,
) -> pd.DataFrame:
    sensor_frame = (
        frame.loc[
            frame["sensor"] == sensor_name,
            [
                "session_id",
                "sensor",
                "relative_time_ns",
                "x",
                "y",
                "z",
            ],
        ]
        .sort_values(
            "relative_time_ns",
            kind="stable",
        )
        .reset_index(drop=True)
        .copy()
    )

    if sensor_frame.empty:
        raise SignalLoadingError(
            f"Die Sensor-CSV enthält keine Werte für {sensor_name}."
        )

    sensor_frame["time_seconds"] = (
        sensor_frame["relative_time_ns"]
        .astype(np.float64) /
        1_000_000_000.0
    )
    sensor_frame["magnitude"] = np.sqrt(
        np.square(sensor_frame["x"].to_numpy(dtype=np.float64))
        + np.square(sensor_frame["y"].to_numpy(dtype=np.float64))
        + np.square(sensor_frame["z"].to_numpy(dtype=np.float64))
    )

    return sensor_frame


def calculate_rms_envelope(
    samples: np.ndarray,
    sample_rate_hz: int,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Berechnet eine RMS-Hüllkurve mit Mittelpunktzeit je Fenster."""
    if samples.ndim != 1 or samples.size == 0:
        raise SignalLoadingError(
            "Für die RMS-Berechnung wird ein eindimensionales Audiosignal benötigt."
        )
    if sample_rate_hz <= 0:
        raise SignalLoadingError(
            "Die Abtastrate muss positiv sein."
        )
    if window_ms <= 0 or hop_ms <= 0:
        raise SignalLoadingError(
            "Fenster- und Schrittweite müssen positiv sein."
        )

    window_samples = max(
        1,
        int(round(sample_rate_hz * window_ms / 1000.0)),
    )
    hop_samples = max(
        1,
        int(round(sample_rate_hz * hop_ms / 1000.0)),
    )

    if samples.size < window_samples:
        rms = np.array(
            [
                np.sqrt(
                    np.mean(
                        np.square(
                            samples.astype(np.float64)
                        )
                    )
                )
            ],
            dtype=np.float64,
        )
        times = np.array(
            [
                (samples.size - 1) /
                (2.0 * sample_rate_hz)
            ],
            dtype=np.float64,
        )
        return times, rms

    starts = np.arange(
        0,
        samples.size - window_samples + 1,
        hop_samples,
        dtype=np.int64,
    )
    squared = np.square(
        samples.astype(np.float64)
    )
    cumulative = np.concatenate(
        (
            np.array([0.0], dtype=np.float64),
            np.cumsum(squared, dtype=np.float64),
        )
    )
    window_sums = (
        cumulative[starts + window_samples]
        - cumulative[starts]
    )
    rms = np.sqrt(
        window_sums / float(window_samples)
    )
    times = (
        starts + (window_samples - 1) / 2.0
    ) / float(sample_rate_hz)

    return times, rms


def downsample_waveform_min_max(
    time_seconds: np.ndarray,
    samples: np.ndarray,
    maximum_points: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduziert eine Wellenform, ohne lokale Minima und Maxima zu verlieren."""
    if (
        time_seconds.ndim != 1
        or samples.ndim != 1
        or time_seconds.size != samples.size
        or samples.size == 0
    ):
        raise SignalLoadingError(
            "Zeit- und Sample-Array müssen eindimensional und gleich lang sein."
        )
    if maximum_points < 2:
        raise SignalLoadingError(
            "maximum_points muss mindestens 2 betragen."
        )
    if samples.size <= maximum_points:
        return time_seconds, samples

    target_bin_count = max(
        1,
        maximum_points // 2,
    )
    bin_size = int(
        np.ceil(
            samples.size /
            target_bin_count
        )
    )
    bin_count = int(
        np.ceil(
            samples.size /
            bin_size
        )
    )
    padded_size = bin_count * bin_size
    padding = padded_size - samples.size

    sample_values = samples.astype(
        np.float64,
        copy=False,
    )
    time_values = time_seconds.astype(
        np.float64,
        copy=False,
    )

    if padding:
        sample_values = np.pad(
            sample_values,
            (0, padding),
            mode="constant",
            constant_values=np.nan,
        )
        time_values = np.pad(
            time_values,
            (0, padding),
            mode="constant",
            constant_values=np.nan,
        )

    sample_bins = sample_values.reshape(
        bin_count,
        bin_size,
    )
    time_bins = time_values.reshape(
        bin_count,
        bin_size,
    )

    minima = np.nanmin(sample_bins, axis=1)
    maxima = np.nanmax(sample_bins, axis=1)
    centers = np.nanmean(time_bins, axis=1)

    reduced_time = np.repeat(centers, 2)
    reduced_samples = np.empty(
        bin_count * 2,
        dtype=np.float64,
    )
    reduced_samples[0::2] = minima
    reduced_samples[1::2] = maxima

    valid = (
        np.isfinite(reduced_time)
        & np.isfinite(reduced_samples)
    )
    return reduced_time[valid], reduced_samples[valid]
