from __future__ import annotations

import math
from typing import Any

import librosa
import numpy as np
import pandas as pd

from .window_signals import SynchronizedWindowSignals


AUDIO_FRAME_MS = 25.0
AUDIO_HOP_MS = 10.0
AUDIO_N_MFCC = 13
AUDIO_N_MELS = 40
AUDIO_ROLLOFF_PERCENT = 0.85

AUDIO_BASE_FEATURES = (
    "rms",
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
)

AUDIO_SUMMARY_STATISTICS = (
    "mean",
    "std",
)

IMU_SIGNALS = (
    "x",
    "y",
    "z",
    "magnitude",
)

IMU_STATISTICS = (
    "mean",
    "std",
    "median",
    "min",
    "max",
    "rms",
)


class FeatureExtractionError(ValueError):
    """Fehler bei der reproduzierbaren Feature-Extraktion."""


def extract_audio_features(
    *,
    samples: np.ndarray,
    sample_rate_hz: int,
) -> dict[str, float]:
    """
    Extrahiert kompakte klassische Audiofeatures aus einem ML-Fenster.

    Kurzzeitfeatures werden auf überlappenden Analyseframes berechnet
    und anschließend durch Mittelwert und Standardabweichung über das
    vollständige ML-Fenster zusammengefasst.
    """
    y = np.asarray(
        samples,
        dtype=np.float64,
    )

    if y.ndim != 1 or y.size == 0:
        raise FeatureExtractionError(
            "Audiosamples müssen eindimensional und nicht leer sein."
        )

    if not np.isfinite(y).all():
        raise FeatureExtractionError(
            "Audiosamples enthalten NaN oder unendliche Werte."
        )

    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, (int, np.integer))
        or int(sample_rate_hz) <= 0
    ):
        raise FeatureExtractionError(
            "Die Audio-Abtastrate muss eine positive Ganzzahl sein."
        )

    sample_rate = int(sample_rate_hz)

    frame_length = max(
        1,
        int(
            round(
                sample_rate
                * AUDIO_FRAME_MS
                / 1000.0
            )
        ),
    )

    hop_length = max(
        1,
        int(
            round(
                sample_rate
                * AUDIO_HOP_MS
                / 1000.0
            )
        ),
    )

    n_fft = _next_power_of_two(
        frame_length
    )

    if y.size < n_fft:
        raise FeatureExtractionError(
            "Das Audiosignal ist zu kurz für die konfigurierte "
            "Kurzzeitanalyse."
        )

    stft = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        window="hann",
        center=False,
    )

    magnitude = np.abs(
        stft
    )

    power = np.square(
        magnitude
    )

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]

    centroid = librosa.feature.spectral_centroid(
        S=magnitude,
        sr=sample_rate,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude,
        sr=sample_rate,
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude,
        sr=sample_rate,
        roll_percent=AUDIO_ROLLOFF_PERCENT,
    )[0]

    flatness = librosa.feature.spectral_flatness(
        S=magnitude,
        power=2.0,
    )[0]

    mel_power = librosa.feature.melspectrogram(
        S=power,
        sr=sample_rate,
        n_mels=AUDIO_N_MELS,
    )

    mel_db = librosa.power_to_db(
        mel_power,
        ref=np.max,
    )

    mfcc = librosa.feature.mfcc(
        S=mel_db,
        n_mfcc=AUDIO_N_MFCC,
    )

    feature_series = {
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
    }

    features: dict[str, float] = {}

    for feature_name in AUDIO_BASE_FEATURES:
        values = _validated_feature_series(
            feature_series[feature_name],
            f"audio_{feature_name}",
        )

        features[
            f"audio_{feature_name}_mean"
        ] = float(
            np.mean(values)
        )

        features[
            f"audio_{feature_name}_std"
        ] = float(
            np.std(values)
        )

    if (
        mfcc.ndim != 2
        or mfcc.shape[0] != AUDIO_N_MFCC
        or mfcc.shape[1] == 0
        or not np.isfinite(mfcc).all()
    ):
        raise FeatureExtractionError(
            "MFCC-Berechnung lieferte eine ungültige Matrix."
        )

    for coefficient_index in range(
        AUDIO_N_MFCC
    ):
        values = mfcc[
            coefficient_index
        ]

        coefficient_number = (
            coefficient_index + 1
        )

        features[
            f"audio_mfcc_{coefficient_number:02d}_mean"
        ] = float(
            np.mean(values)
        )

        features[
            f"audio_mfcc_{coefficient_number:02d}_std"
        ] = float(
            np.std(values)
        )

    _validate_feature_dictionary(
        features,
        expected_count=38,
        description="Audio",
    )

    return features


def extract_watch_features(
    *,
    accelerometer: pd.DataFrame,
    gyroscope: pd.DataFrame,
) -> dict[str, float]:
    """
    Extrahiert kompakte Zeitbereichsfeatures aus Accelerometer und
    Gyroskop. x/y/z erhalten richtungsabhängige Information, während
    magnitude eine orientierungsrobustere Gesamtstärke beschreibt.
    """
    features: dict[str, float] = {}

    for prefix, frame in (
        ("acc", accelerometer),
        ("gyro", gyroscope),
    ):
        _validate_sensor_frame(
            frame,
            sensor_name=prefix,
        )

        for signal_name in IMU_SIGNALS:
            values = frame[
                signal_name
            ].to_numpy(
                dtype=np.float64
            )

            statistics = (
                _summary_statistics(
                    values
                )
            )

            for statistic_name in IMU_STATISTICS:
                features[
                    f"{prefix}_{signal_name}_{statistic_name}"
                ] = statistics[
                    statistic_name
                ]

    _validate_feature_dictionary(
        features,
        expected_count=48,
        description="Watch",
    )

    return features


def extract_multimodal_features(
    window_signals: SynchronizedWindowSignals,
) -> dict[str, float]:
    """
    Führt Audio- und Watch-Features desselben synchronisierten
    ML-Fensters auf Feature-Ebene zusammen.
    """
    audio_features = extract_audio_features(
        samples=window_signals.audio_samples,
        sample_rate_hz=(
            window_signals.audio_sample_rate_hz
        ),
    )

    watch_features = extract_watch_features(
        accelerometer=window_signals.accelerometer,
        gyroscope=window_signals.gyroscope,
    )

    duplicate_names = (
        set(audio_features)
        & set(watch_features)
    )

    if duplicate_names:
        raise FeatureExtractionError(
            "Audio- und Watch-Features besitzen doppelte Namen: "
            + ", ".join(
                sorted(
                    duplicate_names
                )
            )
        )

    combined = {
        **audio_features,
        **watch_features,
    }

    _validate_feature_dictionary(
        combined,
        expected_count=86,
        description="Multimodal",
    )

    return combined


def _summary_statistics(
    values: np.ndarray,
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        array.ndim != 1
        or array.size == 0
        or not np.isfinite(array).all()
    ):
        raise FeatureExtractionError(
            "IMU-Signal enthält keine gültigen endlichen Werte."
        )

    return {
        "mean": float(
            np.mean(array)
        ),
        "std": float(
            np.std(array)
        ),
        "median": float(
            np.median(array)
        ),
        "min": float(
            np.min(array)
        ),
        "max": float(
            np.max(array)
        ),
        "rms": float(
            np.sqrt(
                np.mean(
                    np.square(array)
                )
            )
        ),
    }


def _validate_sensor_frame(
    frame: pd.DataFrame,
    *,
    sensor_name: str,
) -> None:
    if not isinstance(
        frame,
        pd.DataFrame,
    ):
        raise FeatureExtractionError(
            f"{sensor_name} muss als pandas.DataFrame vorliegen."
        )

    if frame.empty:
        raise FeatureExtractionError(
            f"{sensor_name} enthält keine Messwerte."
        )

    missing = (
        set(IMU_SIGNALS)
        - set(frame.columns)
    )

    if missing:
        raise FeatureExtractionError(
            f"{sensor_name} enthält nicht alle benötigten Spalten: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    values = frame[
        list(IMU_SIGNALS)
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():
        raise FeatureExtractionError(
            f"{sensor_name} enthält NaN oder unendliche Werte."
        )


def _validated_feature_series(
    values: Any,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        array.ndim != 1
        or array.size == 0
        or not np.isfinite(array).all()
    ):
        raise FeatureExtractionError(
            f"{name} enthält keine gültigen Featurewerte."
        )

    return array


def _validate_feature_dictionary(
    features: dict[str, float],
    *,
    expected_count: int,
    description: str,
) -> None:
    if len(features) != expected_count:
        raise FeatureExtractionError(
            f"{description}-Featurezahl ist unerwartet: "
            f"{len(features)} statt {expected_count}."
        )

    values = np.asarray(
        list(
            features.values()
        ),
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise FeatureExtractionError(
            f"{description}-Features enthalten NaN oder unendliche Werte."
        )


def _next_power_of_two(
    value: int,
) -> int:
    if value <= 0:
        raise FeatureExtractionError(
            "FFT-Mindestlänge muss positiv sein."
        )

    return int(
        2 ** math.ceil(
            math.log2(value)
        )
    )