from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .signal_loader import (
    SessionSignals,
    calculate_rms_envelope,
    load_session_signals,
)


class SynchronizationError(ValueError):
    """Wird ausgelöst, wenn Marker nicht belastbar bestimmt werden können."""


@dataclass(frozen=True)
class MarkerGroup:
    peak_times_seconds: np.ndarray
    peak_values: np.ndarray
    window_start_seconds: float
    window_end_seconds: float
    detection_threshold: float
    interval_coefficient_of_variation: float
    detection_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_times_seconds": [
                float(value)
                for value in self.peak_times_seconds
            ],
            "peak_values": [
                float(value)
                for value in self.peak_values
            ],
            "window_start_seconds":
                self.window_start_seconds,
            "window_end_seconds":
                self.window_end_seconds,
            "detection_threshold":
                self.detection_threshold,
            "interval_coefficient_of_variation":
                self.interval_coefficient_of_variation,
            "detection_score":
                self.detection_score,
        }


@dataclass(frozen=True)
class SynchronizationResult:
    session_id: str
    start_audio_marker: MarkerGroup
    start_watch_marker: MarkerGroup
    end_audio_marker: MarkerGroup
    end_watch_marker: MarkerGroup
    start_pair_offsets_seconds: np.ndarray
    end_pair_offsets_seconds: np.ndarray
    start_offset_seconds: float
    end_offset_seconds: float
    selected_offset_seconds: float
    offset_change_seconds: float
    watch_sampling_interval_seconds: float
    offset_consistency_tolerance_seconds: float
    offset_consistent: bool
    selected_alignment_method: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "start_audio_marker":
                self.start_audio_marker.to_dict(),
            "start_watch_marker":
                self.start_watch_marker.to_dict(),
            "end_audio_marker":
                self.end_audio_marker.to_dict(),
            "end_watch_marker":
                self.end_watch_marker.to_dict(),
            "start_pair_offsets_seconds": [
                float(value)
                for value in self.start_pair_offsets_seconds
            ],
            "end_pair_offsets_seconds": [
                float(value)
                for value in self.end_pair_offsets_seconds
            ],
            "start_offset_seconds":
                self.start_offset_seconds,
            "end_offset_seconds":
                self.end_offset_seconds,
            "selected_offset_seconds":
                self.selected_offset_seconds,
            "offset_change_seconds":
                self.offset_change_seconds,
            "absolute_offset_change_seconds":
                abs(self.offset_change_seconds),
            "watch_sampling_interval_seconds":
                self.watch_sampling_interval_seconds,
            "offset_consistency_tolerance_seconds":
                self.offset_consistency_tolerance_seconds,
            "offset_consistent":
                self.offset_consistent,
            "selected_alignment_method":
                self.selected_alignment_method,
            "warnings": list(self.warnings),
        }


def analyze_session_synchronization(
    data_root: str | Path,
    session_id: str,
    start_window_seconds: tuple[float, float] | None = None,
    end_window_seconds: tuple[float, float] | None = None,
    expected_impulses: int = 3,
) -> tuple[
    SessionSignals,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    SynchronizationResult,
]:
    """Lädt eine Sitzung und schätzt den konstanten Watch-zu-Audio-Offset."""
    signals = load_session_signals(
        data_root=data_root,
        session_id=session_id,
    )

    rms_time, rms_values = calculate_rms_envelope(
        signals.audio.samples,
        signals.audio.sample_rate_hz,
    )
    acceleration_score = (
        calculate_acceleration_marker_score(
            signals.accelerometer
        )
    )

    common_duration = min(
        signals.audio.duration_seconds,
        float(
            signals.accelerometer[
                "time_seconds"
            ].iloc[-1]
        ),
        float(
            signals.gyroscope[
                "time_seconds"
            ].iloc[-1]
        ),
    )
    resolved_start, resolved_end = (
        resolve_marker_windows(
            common_duration_seconds=common_duration,
            start_window_seconds=start_window_seconds,
            end_window_seconds=end_window_seconds,
        )
    )

    start_audio = detect_regular_peak_group(
        time_seconds=rms_time,
        values=rms_values,
        window_seconds=resolved_start,
        expected_peak_count=expected_impulses,
    )
    end_audio = detect_regular_peak_group(
        time_seconds=rms_time,
        values=rms_values,
        window_seconds=resolved_end,
        expected_peak_count=expected_impulses,
    )
    start_watch = detect_regular_peak_group(
        time_seconds=signals.accelerometer[
            "time_seconds"
        ].to_numpy(dtype=np.float64),
        values=acceleration_score,
        window_seconds=resolved_start,
        expected_peak_count=expected_impulses,
    )
    end_watch = detect_regular_peak_group(
        time_seconds=signals.accelerometer[
            "time_seconds"
        ].to_numpy(dtype=np.float64),
        values=acceleration_score,
        window_seconds=resolved_end,
        expected_peak_count=expected_impulses,
    )

    result = estimate_constant_offset(
        session_id=session_id,
        start_audio_marker=start_audio,
        start_watch_marker=start_watch,
        end_audio_marker=end_audio,
        end_watch_marker=end_watch,
        watch_time_seconds=signals.accelerometer[
            "time_seconds"
        ].to_numpy(dtype=np.float64),
    )

    return (
        signals,
        rms_time,
        rms_values,
        acceleration_score,
        result,
    )


def resolve_marker_windows(
    common_duration_seconds: float,
    start_window_seconds: tuple[float, float] | None,
    end_window_seconds: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Erzeugt nicht überlappende Suchfenster für Anfangs- und Endmarker."""
    if common_duration_seconds <= 8.0:
        raise SynchronizationError(
            "Die Sitzung ist zu kurz für getrennte Anfangs- und Endmarker."
        )

    if start_window_seconds is None:
        start_window_seconds = (
            2.0,
            min(
                15.0,
                common_duration_seconds * 0.35,
            ),
        )
    if end_window_seconds is None:
        end_window_seconds = (
            max(
                common_duration_seconds - 15.0,
                common_duration_seconds * 0.65,
            ),
            common_duration_seconds - 2.0,
        )

    start_window = _validate_window(
        start_window_seconds,
        common_duration_seconds,
        "Anfangsmarker",
    )
    end_window = _validate_window(
        end_window_seconds,
        common_duration_seconds,
        "Endmarker",
    )

    if start_window[1] >= end_window[0]:
        raise SynchronizationError(
            "Die Suchfenster für Anfangs- und Endmarker überlappen."
        )

    return start_window, end_window


def calculate_acceleration_marker_score(
    accelerometer: pd.DataFrame,
    baseline_window_seconds: float = 1.0,
) -> np.ndarray:
    """Berechnet die Abweichung der Magnitude von einer lokalen Ruhelage."""
    required_columns = {
        "time_seconds",
        "magnitude",
    }
    if not required_columns.issubset(
        accelerometer.columns
    ):
        raise SynchronizationError(
            "Für die Markerbewertung fehlen Zeit oder Magnitude."
        )
    if len(accelerometer) < 3:
        raise SynchronizationError(
            "Zu wenige Beschleunigungswerte für die Markerbewertung."
        )
    if baseline_window_seconds <= 0:
        raise SynchronizationError(
            "Das Baseline-Fenster muss positiv sein."
        )

    time_values = accelerometer[
        "time_seconds"
    ].to_numpy(dtype=np.float64)
    magnitude = accelerometer[
        "magnitude"
    ].to_numpy(dtype=np.float64)

    intervals = np.diff(time_values)
    positive_intervals = intervals[
        intervals > 0
    ]
    if positive_intervals.size == 0:
        raise SynchronizationError(
            "Die Beschleunigungszeitachse ist nicht streng ansteigend."
        )

    median_interval = float(
        np.median(positive_intervals)
    )
    window_size = max(
        3,
        int(
            round(
                baseline_window_seconds /
                median_interval
            )
        ),
    )
    if window_size % 2 == 0:
        window_size += 1

    baseline = (
        pd.Series(magnitude)
        .rolling(
            window=window_size,
            center=True,
            min_periods=1,
        )
        .median()
        .to_numpy(dtype=np.float64)
    )
    score = np.abs(
        magnitude - baseline
    )

    if not np.isfinite(score).all():
        raise SynchronizationError(
            "Der Beschleunigungs-Markerscore enthält ungültige Werte."
        )

    return score


def detect_regular_peak_group(
    time_seconds: np.ndarray,
    values: np.ndarray,
    window_seconds: tuple[float, float],
    expected_peak_count: int = 3,
    minimum_peak_distance_seconds: float = 0.25,
    minimum_interval_seconds: float = 0.30,
    maximum_interval_seconds: float = 1.20,
    maximum_interval_cv: float = 0.35,
) -> MarkerGroup:
    """Erkennt eine regelmäßige Gruppe deutlich ausgeprägter Impulse."""
    time_values = np.asarray(
        time_seconds,
        dtype=np.float64,
    )
    signal_values = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        time_values.ndim != 1
        or signal_values.ndim != 1
        or time_values.size != signal_values.size
        or time_values.size < 3
    ):
        raise SynchronizationError(
            "Zeit- und Markersignal müssen eindimensional und gleich lang sein."
        )
    if expected_peak_count < 2:
        raise SynchronizationError(
            "Es werden mindestens zwei Markerimpulse benötigt."
        )
    if not (
        np.isfinite(time_values).all()
        and np.isfinite(signal_values).all()
    ):
        raise SynchronizationError(
            "Zeit- oder Markersignal enthält ungültige Werte."
        )

    window_start, window_end = _validate_window(
        window_seconds,
        float(time_values[-1]),
        "Markersuche",
    )
    mask = (
        (time_values >= window_start)
        & (time_values <= window_end)
    )
    window_time = time_values[mask]
    window_values = signal_values[mask]

    if window_time.size < expected_peak_count + 2:
        raise SynchronizationError(
            "Das Markerfenster enthält zu wenige Werte."
        )

    center = float(
        np.median(window_values)
    )
    mad = float(
        np.median(
            np.abs(window_values - center)
        )
    )
    robust_scale = max(
        1.4826 * mad,
        float(np.std(window_values)) * 0.05,
        np.finfo(np.float64).eps,
    )
    robust_z = (
        window_values - center
    ) / robust_scale
    threshold = center + 5.0 * robust_scale

    local_maxima = (
        np.where(
            (window_values[1:-1] > window_values[:-2])
            & (
                window_values[1:-1]
                >= window_values[2:]
            )
        )[0]
        + 1
    )
    if local_maxima.size < expected_peak_count:
        raise SynchronizationError(
            "Im Markerfenster wurden zu wenige lokale Maxima gefunden."
        )

    ranked = local_maxima[
        np.argsort(
            window_values[local_maxima]
        )[::-1]
    ]
    separated: list[int] = []
    for index in ranked:
        if all(
            abs(
                window_time[index]
                - window_time[selected]
            )
            >= minimum_peak_distance_seconds
            for selected in separated
        ):
            separated.append(int(index))
        if len(separated) >= 20:
            break

    significant = [
        index
        for index in separated
        if window_values[index] >= threshold
    ]
    if len(significant) >= expected_peak_count:
        candidates = significant
    else:
        candidates = separated[
            :max(expected_peak_count, 12)
        ]

    candidates = sorted(
        candidates,
        key=lambda index: window_time[index],
    )

    best: tuple[
        float,
        tuple[int, ...],
        float,
    ] | None = None

    for combination in itertools.combinations(
        candidates,
        expected_peak_count,
    ):
        peak_times = window_time[
            list(combination)
        ]
        intervals = np.diff(peak_times)

        if (
            (intervals < minimum_interval_seconds).any()
            or (
                intervals
                > maximum_interval_seconds
            ).any()
        ):
            continue

        interval_cv = float(
            np.std(intervals)
            / np.mean(intervals)
        )
        if interval_cv > maximum_interval_cv:
            continue

        score = float(
            np.sum(
                robust_z[
                    list(combination)
                ]
            )
            - 5.0 * interval_cv
        )

        if best is None or score > best[0]:
            best = (
                score,
                combination,
                interval_cv,
            )

    if best is None:
        raise SynchronizationError(
            "Es wurde keine ausreichend regelmäßige Markergruppe gefunden."
        )

    score, selected, interval_cv = best
    selected_indices = list(selected)

    return MarkerGroup(
        peak_times_seconds=window_time[
            selected_indices
        ].copy(),
        peak_values=window_values[
            selected_indices
        ].copy(),
        window_start_seconds=window_start,
        window_end_seconds=window_end,
        detection_threshold=float(threshold),
        interval_coefficient_of_variation=
            interval_cv,
        detection_score=score,
    )


def estimate_constant_offset(
    session_id: str,
    start_audio_marker: MarkerGroup,
    start_watch_marker: MarkerGroup,
    end_audio_marker: MarkerGroup,
    end_watch_marker: MarkerGroup,
    watch_time_seconds: np.ndarray,
) -> SynchronizationResult:
    """Bestimmt robuste Start-, End- und Gesamtoffsets aus gepaarten Impulsen."""
    _validate_matching_peak_counts(
        start_audio_marker,
        start_watch_marker,
        "Anfangsmarker",
    )
    _validate_matching_peak_counts(
        end_audio_marker,
        end_watch_marker,
        "Endmarker",
    )

    start_offsets = (
        start_audio_marker.peak_times_seconds
        - start_watch_marker.peak_times_seconds
    )
    end_offsets = (
        end_audio_marker.peak_times_seconds
        - end_watch_marker.peak_times_seconds
    )

    start_offset = float(
        np.median(start_offsets)
    )
    end_offset = float(
        np.median(end_offsets)
    )
    selected_offset = float(
        np.median(
            np.concatenate(
                (start_offsets, end_offsets)
            )
        )
    )
    offset_change = (
        end_offset - start_offset
    )

    watch_times = np.asarray(
        watch_time_seconds,
        dtype=np.float64,
    )
    positive_intervals = np.diff(
        watch_times
    )
    positive_intervals = positive_intervals[
        positive_intervals > 0
    ]
    if positive_intervals.size == 0:
        raise SynchronizationError(
            "Die Watch-Zeitachse besitzt keine positiven Abstände."
        )

    sampling_interval = float(
        np.median(positive_intervals)
    )
    tolerance = max(
        0.040,
        2.0 * sampling_interval,
    )
    offset_consistent = (
        abs(offset_change) <= tolerance
    )

    warnings: list[str] = []
    _append_interval_match_warning(
        warnings,
        start_audio_marker,
        start_watch_marker,
        "Anfangsmarker",
        tolerance,
    )
    _append_interval_match_warning(
        warnings,
        end_audio_marker,
        end_watch_marker,
        "Endmarker",
        tolerance,
    )
    if not offset_consistent:
        warnings.append(
            "Anfangs- und Endoffset unterscheiden sich stärker als "
            "die aus der Watch-Abtastrate abgeleitete Toleranz. "
            "Eine mögliche Uhrdrift oder fehlerhafte Markerzuordnung "
            "muss untersucht werden."
        )

    return SynchronizationResult(
        session_id=session_id,
        start_audio_marker=start_audio_marker,
        start_watch_marker=start_watch_marker,
        end_audio_marker=end_audio_marker,
        end_watch_marker=end_watch_marker,
        start_pair_offsets_seconds=
            start_offsets,
        end_pair_offsets_seconds=
            end_offsets,
        start_offset_seconds=start_offset,
        end_offset_seconds=end_offset,
        selected_offset_seconds=
            selected_offset,
        offset_change_seconds=
            offset_change,
        watch_sampling_interval_seconds=
            sampling_interval,
        offset_consistency_tolerance_seconds=
            tolerance,
        offset_consistent=offset_consistent,
        selected_alignment_method=(
            "CONSTANT_OFFSET"
            if offset_consistent
            else "CONSTANT_OFFSET_WITH_DRIFT_WARNING"
        ),
        warnings=tuple(warnings),
    )


def apply_watch_offset(
    watch_time_seconds: np.ndarray | pd.Series,
    offset_seconds: float,
) -> np.ndarray:
    """Überführt lokale Watch-Zeiten auf die Audiozeitachse."""
    values = np.asarray(
        watch_time_seconds,
        dtype=np.float64,
    )
    if values.ndim != 1:
        raise SynchronizationError(
            "Die Watch-Zeitachse muss eindimensional sein."
        )
    if not np.isfinite(values).all():
        raise SynchronizationError(
            "Die Watch-Zeitachse enthält ungültige Werte."
        )
    if not np.isfinite(offset_seconds):
        raise SynchronizationError(
            "Der Offset muss endlich sein."
        )

    return values + float(offset_seconds)


def marker_exclusion_interval(
    marker_group: MarkerGroup,
    margin_seconds: float = 0.50,
) -> tuple[float, float]:
    """Erzeugt einen auszuschließenden Zeitbereich um eine Markergruppe."""
    if margin_seconds < 0:
        raise SynchronizationError(
            "Der Markerrand darf nicht negativ sein."
        )

    return (
        max(
            0.0,
            float(
                marker_group
                .peak_times_seconds[0]
                - margin_seconds
            ),
        ),
        float(
            marker_group
            .peak_times_seconds[-1]
            + margin_seconds
        ),
    )


def _validate_window(
    window_seconds: tuple[float, float],
    maximum_time_seconds: float,
    description: str,
) -> tuple[float, float]:
    try:
        start = float(window_seconds[0])
        end = float(window_seconds[1])
    except (
        TypeError,
        ValueError,
        IndexError,
    ) as exception:
        raise SynchronizationError(
            f"Ungültiges Zeitfenster für {description}."
        ) from exception

    if (
        not np.isfinite(start)
        or not np.isfinite(end)
        or start < 0
        or end <= start
        or end > maximum_time_seconds
    ):
        raise SynchronizationError(
            f"Ungültiges Zeitfenster für {description}: "
            f"{start} bis {end} Sekunden."
        )

    return start, end


def _validate_matching_peak_counts(
    audio_marker: MarkerGroup,
    watch_marker: MarkerGroup,
    description: str,
) -> None:
    if (
        audio_marker.peak_times_seconds.size
        != watch_marker.peak_times_seconds.size
    ):
        raise SynchronizationError(
            f"{description}: Audio- und Watch-Marker "
            "besitzen unterschiedlich viele Impulse."
        )


def _append_interval_match_warning(
    warnings: list[str],
    audio_marker: MarkerGroup,
    watch_marker: MarkerGroup,
    description: str,
    tolerance_seconds: float,
) -> None:
    audio_intervals = np.diff(
        audio_marker.peak_times_seconds
    )
    watch_intervals = np.diff(
        watch_marker.peak_times_seconds
    )
    maximum_difference = float(
        np.max(
            np.abs(
                audio_intervals
                - watch_intervals
            )
        )
    )

    if maximum_difference > tolerance_seconds:
        warnings.append(
            f"{description}: Die Impulsabstände zwischen Audio "
            "und Watch unterscheiden sich stärker als die "
            "Abtastratentoleranz."
        )
