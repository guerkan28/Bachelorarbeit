from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .signal_loader import (
    SignalLoadingError,
    calculate_rms_envelope,
    downsample_waveform_min_max,
    load_session_signals,
)
from .session_loader import SessionValidationError


DATA_ENV_NAME = "TRINKERKENNUNG_DATA_ROOT"


def create_session_figures(
    data_root: str | Path,
    session_id: str,
    output_root: str | Path,
) -> dict[str, object]:
    """Lädt eine Sitzung und speichert reproduzierbare Signaldiagramme."""
    signals = load_session_signals(
        data_root=data_root,
        session_id=session_id,
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

    audio_time, audio_samples = (
        downsample_waveform_min_max(
            signals.audio.time_seconds,
            signals.audio.samples,
        )
    )
    rms_time, rms_values = calculate_rms_envelope(
        signals.audio.samples,
        signals.audio.sample_rate_hz,
    )

    audio_path = (
        session_output /
        f"session_{session_id}_audio_waveform.png"
    )
    accelerometer_path = (
        session_output /
        f"session_{session_id}_accelerometer.png"
    )
    gyroscope_path = (
        session_output /
        f"session_{session_id}_gyroscope.png"
    )
    overview_path = (
        session_output /
        f"session_{session_id}_overview.png"
    )

    _plot_audio(
        session_id=session_id,
        time_seconds=audio_time,
        samples=audio_samples,
        rms_time_seconds=rms_time,
        rms_values=rms_values,
        output_path=audio_path,
    )
    _plot_sensor(
        session_id=session_id,
        sensor_frame=signals.accelerometer,
        title="Beschleunigungssensor",
        unit="m/s²",
        output_path=accelerometer_path,
    )
    _plot_sensor(
        session_id=session_id,
        sensor_frame=signals.gyroscope,
        title="Gyroskop",
        unit="rad/s",
        output_path=gyroscope_path,
    )
    _plot_overview(
        session_id=session_id,
        rms_time_seconds=rms_time,
        rms_values=rms_values,
        accelerometer=signals.accelerometer,
        gyroscope=signals.gyroscope,
        output_path=overview_path,
    )

    report = {
        "session_id": session_id,
        "time_axis_note": (
            "Smartphone-Audio und Watch-Sensoren werden auf ihren "
            "jeweiligen lokalen relativen Zeitachsen dargestellt. "
            "Es wurde noch kein geräteübergreifender Offset angewendet."
        ),
        "audio_sample_count": int(
            signals.audio.samples.size
        ),
        "audio_sample_rate_hz":
            signals.audio.sample_rate_hz,
        "audio_duration_seconds":
            signals.audio.duration_seconds,
        "accelerometer_row_count": int(
            len(signals.accelerometer)
        ),
        "gyroscope_row_count": int(
            len(signals.gyroscope)
        ),
        "accelerometer_duration_seconds": float(
            signals.accelerometer[
                "time_seconds"
            ].iloc[-1]
            - signals.accelerometer[
                "time_seconds"
            ].iloc[0]
        ),
        "gyroscope_duration_seconds": float(
            signals.gyroscope[
                "time_seconds"
            ].iloc[-1]
            - signals.gyroscope[
                "time_seconds"
            ].iloc[0]
        ),
        "output_directory": str(session_output),
        "figures": {
            "audio_waveform": str(audio_path),
            "accelerometer": str(
                accelerometer_path
            ),
            "gyroscope": str(gyroscope_path),
            "overview": str(overview_path),
        },
        "validation_warnings":
            signals.validation_report["warnings"],
    }

    report_path = (
        session_output /
        f"session_{session_id}_visualization_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)

    return report


def _plot_audio(
    session_id: str,
    time_seconds,
    samples,
    rms_time_seconds,
    rms_values,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
    )

    axes[0].plot(
        time_seconds,
        samples,
        linewidth=0.5,
    )
    axes[0].set_ylabel(
        "Normierte Amplitude"
    )
    axes[0].set_title(
        "PCM-Wellenform"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        rms_time_seconds,
        rms_values,
        linewidth=0.9,
    )
    axes[1].set_xlabel(
        "Lokale Audiozeit [s]"
    )
    axes[1].set_ylabel(
        "RMS-Amplitude"
    )
    axes[1].set_title(
        "RMS-Hüllkurve (20-ms-Fenster, 10-ms-Schritt)"
    )
    axes[1].grid(True, alpha=0.3)

    figure.suptitle(
        f"Smartphone-Audio – Session {session_id}"
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_sensor(
    session_id: str,
    sensor_frame,
    title: str,
    unit: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(14, 7),
    )

    for column in (
        "x",
        "y",
        "z",
        "magnitude",
    ):
        axis.plot(
            sensor_frame["time_seconds"],
            sensor_frame[column],
            label=column,
            linewidth=0.9,
        )

    axis.set_xlabel(
        "Lokale Watch-Zeit seit Sitzungsstart [s]"
    )
    axis.set_ylabel(
        f"Sensorwert [{unit}]"
    )
    axis.set_title(
        f"{title} – Session {session_id}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_overview(
    session_id: str,
    rms_time_seconds,
    rms_values,
    accelerometer,
    gyroscope,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
    )

    axes[0].plot(
        rms_time_seconds,
        rms_values,
        linewidth=0.9,
    )
    axes[0].set_ylabel(
        "Audio-RMS"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        accelerometer["time_seconds"],
        accelerometer["magnitude"],
        linewidth=0.9,
    )
    axes[1].set_ylabel(
        "Beschleunigung\n[m/s²]"
    )
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        gyroscope["time_seconds"],
        gyroscope["magnitude"],
        linewidth=0.9,
    )
    axes[2].set_xlabel(
        "Jeweilige lokale relative Zeit [s]"
    )
    axes[2].set_ylabel(
        "Drehrate\n[rad/s]"
    )
    axes[2].grid(True, alpha=0.3)

    figure.suptitle(
        (
            f"Modalitätsübersicht – Session {session_id}\n"
            "Noch ohne geräteübergreifende Offsetkorrektur"
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
            "Lädt und visualisiert die Audio- und "
            "Watch-Signale einer validierten Sitzung."
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
    arguments = parser.parse_args()

    data_root = (
        arguments.data_root
        or os.environ.get(DATA_ENV_NAME)
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
        else Path(data_root).expanduser().resolve().parent
        / "Visualisierungen"
    )

    try:
        report = create_session_figures(
            data_root=data_root,
            session_id=arguments.session_id,
            output_root=output_root,
        )
    except (
        SessionValidationError,
        SignalLoadingError,
    ) as exception:
        print(
            f"Verarbeitungsfehler: {exception}",
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
