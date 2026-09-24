from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .session_loader import SessionValidationError
from .signal_loader import SignalLoadingError
from .synchronization import (
    SynchronizationError,
    analyze_session_synchronization,
    apply_watch_offset,
    marker_exclusion_interval,
)
from .video_alignment import build_watch_to_audio_mapping


DATA_ENV_NAME = "TRINKERKENNUNG_DATA_ROOT"


def create_synchronization_outputs(
    data_root: str | Path,
    session_id: str,
    output_root: str | Path,
    start_window_seconds: tuple[float, float] | None = None,
    end_window_seconds: tuple[float, float] | None = None,
    expected_impulses: int = 3,
    start_audio_peak_windows_seconds: (
        tuple[tuple[float, float], ...] | None
    ) = None,
    start_watch_peak_windows_seconds: (
        tuple[tuple[float, float], ...] | None
    ) = None,
    end_audio_peak_windows_seconds: (
        tuple[tuple[float, float], ...] | None
    ) = None,
    end_watch_peak_windows_seconds: (
        tuple[tuple[float, float], ...] | None
    ) = None,
) -> dict[str, object]:
    """Schätzt den Offset und speichert Bericht sowie ausgerichtete Übersicht."""
    (
        signals,
        rms_time,
        rms_values,
        acceleration_score,
        result,
    ) = analyze_session_synchronization(
        data_root=data_root,
        session_id=session_id,
        start_window_seconds=
            start_window_seconds,
        end_window_seconds=
            end_window_seconds,
        expected_impulses=
            expected_impulses,
        start_audio_peak_windows_seconds=
            start_audio_peak_windows_seconds,
        start_watch_peak_windows_seconds=
            start_watch_peak_windows_seconds,
        end_audio_peak_windows_seconds=
            end_audio_peak_windows_seconds,
        end_watch_peak_windows_seconds=
            end_watch_peak_windows_seconds,
    )

    session_output = (
        Path(output_root)
        .expanduser()
        .resolve()
        / session_id
    )
    session_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    overview_path = (
        session_output
        / f"session_{session_id}_aligned_overview.png"
    )
    report_path = (
        session_output
        / f"synchronization_report_{session_id}.json"
    )

    aligned_acceleration_time = apply_watch_offset(
        signals.accelerometer[
            "time_seconds"
        ],
        result.selected_offset_seconds,
    )
    aligned_gyroscope_time = apply_watch_offset(
        signals.gyroscope[
            "time_seconds"
        ],
        result.selected_offset_seconds,
    )

    _plot_aligned_overview(
        session_id=session_id,
        rms_time_seconds=rms_time,
        rms_values=rms_values,
        aligned_acceleration_time_seconds=
            aligned_acceleration_time,
        acceleration_score=
            acceleration_score,
        aligned_gyroscope_time_seconds=
            aligned_gyroscope_time,
        gyroscope_magnitude=signals.gyroscope[
            "magnitude"
        ].to_numpy(dtype=np.float64),
        result=result,
        output_path=overview_path,
    )

    report = result.to_dict()
    report["schema_version"] = 2
    report["time_mappings"] = {
        "watch_to_audio": build_watch_to_audio_mapping(report),
        "video_to_audio": None,
    }

    confirmed_windows = {
        "start_audio": (
            [
                list(window)
                for window in start_audio_peak_windows_seconds
            ]
            if start_audio_peak_windows_seconds is not None
            else None
        ),
        "start_watch": (
            [
                list(window)
                for window in start_watch_peak_windows_seconds
            ]
            if start_watch_peak_windows_seconds is not None
            else None
        ),
        "end_audio": (
            [
                list(window)
                for window in end_audio_peak_windows_seconds
            ]
            if end_audio_peak_windows_seconds is not None
            else None
        ),
        "end_watch": (
            [
                list(window)
                for window in end_watch_peak_windows_seconds
            ]
            if end_watch_peak_windows_seconds is not None
            else None
        ),
    }

    report["marker_detection"] = {
        "group_modes": {
            key: (
                "CONFIRMED_PEAK_WINDOWS"
                if value is not None
                else "AUTOMATIC_REGULAR_GROUP"
            )
            for key, value in confirmed_windows.items()
        },
        "confirmed_peak_windows_seconds":
            confirmed_windows,
        "note": (
            "CONFIRMED_PEAK_WINDOWS verwendet fachlich best?tigte "
            "kleine Suchfenster. Die konkrete Peakposition wird "
            "innerhalb der Fenster weiterhin algorithmisch bestimmt."
        ),
    }
    report.update({
        "time_axis_note": (
            "Die lokalen Watch-Zeiten wurden mit dem geschätzten "
            "konstanten Offset auf die lokale Audiozeitachse übertragen."
        ),
        "audio_duration_seconds":
            signals.audio.duration_seconds,
        "accelerometer_row_count": int(
            len(signals.accelerometer)
        ),
        "gyroscope_row_count": int(
            len(signals.gyroscope)
        ),
        "audio_marker_exclusion_intervals_seconds": [
            list(
                marker_exclusion_interval(
                    result.start_audio_marker
                )
            ),
            list(
                marker_exclusion_interval(
                    result.end_audio_marker
                )
            ),
        ],
        "aligned_watch_marker_exclusion_intervals_seconds": [
            [
                value
                + result.selected_offset_seconds
                for value in marker_exclusion_interval(
                    result.start_watch_marker
                )
            ],
            [
                value
                + result.selected_offset_seconds
                for value in marker_exclusion_interval(
                    result.end_watch_marker
                )
            ],
        ],
        "output_directory":
            str(session_output),
        "aligned_overview":
            str(overview_path),
        "validation_warnings":
            signals.validation_report[
                "warnings"
            ],
    })

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(
        report_path
    )

    return report


def _plot_aligned_overview(
    session_id: str,
    rms_time_seconds: np.ndarray,
    rms_values: np.ndarray,
    aligned_acceleration_time_seconds: np.ndarray,
    acceleration_score: np.ndarray,
    aligned_gyroscope_time_seconds: np.ndarray,
    gyroscope_magnitude: np.ndarray,
    result,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
    )

    axes[0].plot(
        rms_time_seconds,
        rms_values,
        linewidth=0.9,
    )
    axes[0].scatter(
        np.concatenate((
            result.start_audio_marker
                .peak_times_seconds,
            result.end_audio_marker
                .peak_times_seconds,
        )),
        np.concatenate((
            result.start_audio_marker
                .peak_values,
            result.end_audio_marker
                .peak_values,
        )),
        marker="x",
        s=45,
        label="erkannte Audioimpulse",
    )
    axes[0].set_ylabel(
        "Audio-RMS"
    )
    axes[0].grid(
        True,
        alpha=0.3,
    )
    axes[0].legend()

    axes[1].plot(
        aligned_acceleration_time_seconds,
        acceleration_score,
        linewidth=0.9,
    )
    watch_peak_times = np.concatenate((
        result.start_watch_marker
            .peak_times_seconds,
        result.end_watch_marker
            .peak_times_seconds,
    ))
    watch_peak_values = np.concatenate((
        result.start_watch_marker
            .peak_values,
        result.end_watch_marker
            .peak_values,
    ))
    axes[1].scatter(
        watch_peak_times
        + result.selected_offset_seconds,
        watch_peak_values,
        marker="x",
        s=45,
        label="ausgerichtete Watch-Impulse",
    )
    axes[1].set_ylabel(
        "Dynamische\nBeschleunigung [m/s²]"
    )
    axes[1].grid(
        True,
        alpha=0.3,
    )
    axes[1].legend()

    axes[2].plot(
        aligned_gyroscope_time_seconds,
        gyroscope_magnitude,
        linewidth=0.9,
    )
    axes[2].set_xlabel(
        "Gemeinsame lokale Audiozeit [s]"
    )
    axes[2].set_ylabel(
        "Drehrate\n[rad/s]"
    )
    axes[2].grid(
        True,
        alpha=0.3,
    )

    for marker_time in np.concatenate((
        result.start_audio_marker
            .peak_times_seconds,
        result.end_audio_marker
            .peak_times_seconds,
    )):
        for axis in axes:
            axis.axvline(
                marker_time,
                linewidth=0.7,
                linestyle="--",
                alpha=0.35,
            )

    figure.suptitle(
        (
            f"Ausgerichtete Modalitätsübersicht – Session {session_id}\n"
            f"Konstanter Watch-zu-Audio-Offset: "
            f"{result.selected_offset_seconds:.6f} s"
        )
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Erkennt kontrollierte Synchronisationsmarker, "
            "schätzt den Watch-zu-Audio-Offset und erzeugt "
            "eine ausgerichtete Übersicht."
        )
    )
    parser.add_argument(
        "--session-id",
        required=True,
    )
    parser.add_argument(
        "--data-root",
    )
    parser.add_argument(
        "--output-root",
    )
    parser.add_argument(
        "--start-window",
        nargs=2,
        type=float,
        metavar=("START", "ENDE"),
    )
    parser.add_argument(
        "--end-window",
        nargs=2,
        type=float,
        metavar=("START", "ENDE"),
    )
    parser.add_argument(
        "--start-audio-peak-window",
        action="append",
        nargs=2,
        type=float,
        metavar=("START", "ENDE"),
    )
    parser.add_argument(
        "--start-watch-peak-window",
        action="append",
        nargs=2,
        type=float,
        metavar=("START", "ENDE"),
    )
    parser.add_argument(
        "--end-audio-peak-window",
        action="append",
        nargs=2,
        type=float,
        metavar=("START", "ENDE"),
    )
    parser.add_argument(
        "--end-watch-peak-window",
        action="append",
        nargs=2,
        type=float,
        metavar=("START", "ENDE"),
    )
    parser.add_argument(
        "--expected-impulses",
        type=int,
        default=3,
    )
    arguments = parser.parse_args()

    data_root = (
        arguments.data_root
        or os.environ.get(
            DATA_ENV_NAME
        )
    )
    if not data_root:
        print(
            "Fehler: --data-root fehlt und "
            f"{DATA_ENV_NAME} ist nicht gesetzt.",
            file=sys.stderr,
        )
        return 2

    output_root = (
        Path(arguments.output_root)
        if arguments.output_root
        else (
            Path(data_root)
            .expanduser()
            .resolve()
            .parent
            / "Synchronisationsergebnisse"
        )
    )

    start_window = (
        tuple(arguments.start_window)
        if arguments.start_window
        else None
    )
    end_window = (
        tuple(arguments.end_window)
        if arguments.end_window
        else None
    )

    start_audio_peak_windows = (
        tuple(
            tuple(window)
            for window in arguments.start_audio_peak_window
        )
        if arguments.start_audio_peak_window
        else None
    )

    start_watch_peak_windows = (
        tuple(
            tuple(window)
            for window in arguments.start_watch_peak_window
        )
        if arguments.start_watch_peak_window
        else None
    )

    end_audio_peak_windows = (
        tuple(
            tuple(window)
            for window in arguments.end_audio_peak_window
        )
        if arguments.end_audio_peak_window
        else None
    )

    end_watch_peak_windows = (
        tuple(
            tuple(window)
            for window in arguments.end_watch_peak_window
        )
        if arguments.end_watch_peak_window
        else None
    )

    try:
        report = create_synchronization_outputs(
            data_root=data_root,
            session_id=arguments.session_id,
            output_root=output_root,
            start_window_seconds=
                start_window,
            end_window_seconds=
                end_window,
            expected_impulses=
                arguments.expected_impulses,
            start_audio_peak_windows_seconds=
                start_audio_peak_windows,
            start_watch_peak_windows_seconds=
                start_watch_peak_windows,
            end_audio_peak_windows_seconds=
                end_audio_peak_windows,
            end_watch_peak_windows_seconds=
                end_watch_peak_windows,
        )
    except (
        SessionValidationError,
        SignalLoadingError,
        SynchronizationError,
    ) as exception:
        print(
            f"Synchronisationsfehler: {exception}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
