from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_extraction import (
    extract_audio_features,
    extract_watch_features,
)
from .window_dataset import (
    WINDOW_DATASET_FIELDNAMES,
    WindowRecord,
)
from .window_signals import (
    extract_synchronized_window,
    load_aligned_session_signals,
)


AUDIO_FEATURE_COUNT = 38
WATCH_FEATURE_COUNT = 48
FUSION_FEATURE_COUNT = 86


class FeatureDatasetError(ValueError):
    """Fehler bei der reproduzierbaren Bildung von Feature-Datensätzen."""


@dataclass(frozen=True)
class FeatureDatasetTables:
    audio: pd.DataFrame
    watch: pd.DataFrame
    fusion: pd.DataFrame


def build_feature_datasets_from_window_csv(
    *,
    data_root: str | Path,
    window_csv: str | Path,
    participant_data_roots: dict[str, str | Path] | None = None,
) -> FeatureDatasetTables:
    """
    Erzeugt Audio-, Watch- und Fusionsfeaturetabellen aus einem
    kanonischen ML-Fensterdatensatz.
    """
    root = Path(
        data_root
    ).expanduser().resolve()

    csv_path = Path(
        window_csv
    ).expanduser().resolve()

    if not root.is_dir():
        raise FeatureDatasetError(
            f"Datenordner nicht gefunden: {root}"
        )

    resolved_participant_data_roots: dict[str, Path] = {}

    if participant_data_roots is not None:
        for participant_id, participant_root in (
            participant_data_roots.items()
        ):
            resolved_root = Path(
                participant_root
            ).expanduser().resolve()

            if not resolved_root.is_dir():
                raise FeatureDatasetError(
                    "Datenordner f?r Teilnehmer "
                    f"{participant_id} nicht gefunden: "
                    f"{resolved_root}"
                )

            resolved_participant_data_roots[
                str(participant_id)
            ] = resolved_root

    if not csv_path.is_file():
        raise FeatureDatasetError(
            f"Fensterdatensatz nicht gefunden: {csv_path}"
        )

    try:
        frame = pd.read_csv(
            csv_path,
            sep=";",
            encoding="utf-8",
        )
    except (
        OSError,
        pd.errors.ParserError,
    ) as exception:
        raise FeatureDatasetError(
            f"Fensterdatensatz nicht lesbar: {csv_path}"
        ) from exception

    if frame.empty:
        raise FeatureDatasetError(
            "Der Fensterdatensatz ist leer."
        )

    missing_columns = (
        set(WINDOW_DATASET_FIELDNAMES)
        - set(frame.columns)
    )

    if missing_columns:
        raise FeatureDatasetError(
            "Im Fensterdatensatz fehlen Spalten: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    if frame["window_id"].duplicated().any():
        raise FeatureDatasetError(
            "Der Fensterdatensatz enthält doppelte window_id."
        )

    session_cache: dict[str, Any] = {}

    audio_rows: list[dict[str, Any]] = []
    watch_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []

    for raw_row in frame.to_dict(
        orient="records"
    ):
        window = _row_to_window_record(
            raw_row
        )

        if window.session_id not in session_cache:
            session_root = (
                resolved_participant_data_roots.get(
                    window.participant_id,
                    root,
                )
            )

            session_cache[
                window.session_id
            ] = load_aligned_session_signals(
                data_root=session_root,
                session_id=window.session_id,
            )

        synchronized = (
            extract_synchronized_window(
                aligned_session=session_cache[
                    window.session_id
                ],
                window=window,
            )
        )

        audio_features = (
            extract_audio_features(
                samples=(
                    synchronized.audio_samples
                ),
                sample_rate_hz=(
                    synchronized.audio_sample_rate_hz
                ),
            )
        )

        watch_features = (
            extract_watch_features(
                accelerometer=(
                    synchronized.accelerometer
                ),
                gyroscope=(
                    synchronized.gyroscope
                ),
            )
        )

        metadata = _window_metadata(
            window
        )

        audio_rows.append({
            **metadata,
            **audio_features,
        })

        watch_rows.append({
            **metadata,
            **watch_features,
        })

        fusion_rows.append({
            **metadata,
            **audio_features,
            **watch_features,
        })

    audio = pd.DataFrame(
        audio_rows
    )

    watch = pd.DataFrame(
        watch_rows
    )

    fusion = pd.DataFrame(
        fusion_rows
    )

    _validate_feature_table(
        frame=audio,
        feature_prefixes=(
            "audio_",
        ),
        expected_feature_count=(
            AUDIO_FEATURE_COUNT
        ),
        description="Audio",
    )

    _validate_feature_table(
        frame=watch,
        feature_prefixes=(
            "acc_",
            "gyro_",
        ),
        expected_feature_count=(
            WATCH_FEATURE_COUNT
        ),
        description="Watch",
    )

    _validate_feature_table(
        frame=fusion,
        feature_prefixes=(
            "audio_",
            "acc_",
            "gyro_",
        ),
        expected_feature_count=(
            FUSION_FEATURE_COUNT
        ),
        description="Fusion",
    )

    expected_ids = (
        frame["window_id"]
        .astype(str)
        .tolist()
    )

    for description, table in (
        ("Audio", audio),
        ("Watch", watch),
        ("Fusion", fusion),
    ):
        actual_ids = (
            table["window_id"]
            .astype(str)
            .tolist()
        )

        if actual_ids != expected_ids:
            raise FeatureDatasetError(
                f"{description}: Reihenfolge oder "
                "window_id stimmt nicht mit dem "
                "Fensterdatensatz überein."
            )

    return FeatureDatasetTables(
        audio=audio,
        watch=watch,
        fusion=fusion,
    )


def write_feature_datasets(
    *,
    tables: FeatureDatasetTables,
    output_root: str | Path,
    file_prefix: str,
    configuration_name: str,
) -> dict[str, Path]:
    """
    Schreibt Audio-, Watch- und Fusionsfeaturetabellen reproduzierbar
    als Semikolon-getrennte UTF-8-CSV-Dateien.
    """
    root = Path(
        output_root
    ).expanduser().resolve()

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = _safe_file_token(
        file_prefix,
        "file_prefix",
    )

    configuration = _safe_file_token(
        configuration_name,
        "configuration_name",
    )

    paths = {
        "audio": (
            root
            / (
                f"{prefix}_audio_features_"
                f"{configuration}.csv"
            )
        ),
        "watch": (
            root
            / (
                f"{prefix}_watch_features_"
                f"{configuration}.csv"
            )
        ),
        "fusion": (
            root
            / (
                f"{prefix}_fusion_features_"
                f"{configuration}.csv"
            )
        ),
    }

    for key, path in paths.items():
        table = getattr(
            tables,
            key,
        )

        table.to_csv(
            path,
            sep=";",
            index=False,
            encoding="utf-8",
        )

    return paths


def _row_to_window_record(
    row: dict[str, Any],
) -> WindowRecord:
    try:
        return WindowRecord(
            schema_version=int(
                row["schema_version"]
            ),
            participant_id=str(
                row["participant_id"]
            ),
            session_id=str(
                row["session_id"]
            ),
            session_mode=str(
                row["session_mode"]
            ),
            time_reference=str(
                row["time_reference"]
            ),
            window_id=str(
                row["window_id"]
            ),
            label=str(
                row["label"]
            ),
            source_category=str(
                row["source_category"]
            ),
            source_id=str(
                row["source_id"]
            ),
            scenario=str(
                row["scenario"]
            ),
            container_type=(
                _optional_string(
                    row["container_type"]
                )
            ),
            source_start_seconds=float(
                row["source_start_seconds"]
            ),
            source_end_seconds=float(
                row["source_end_seconds"]
            ),
            window_index=int(
                row["window_index"]
            ),
            window_start_seconds=float(
                row["window_start_seconds"]
            ),
            window_end_seconds=float(
                row["window_end_seconds"]
            ),
            window_length_seconds=float(
                row["window_length_seconds"]
            ),
            stride_seconds=float(
                row["stride_seconds"]
            ),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exception:
        raise FeatureDatasetError(
            "Mindestens eine Fensterzeile enthält "
            "ungültige Werte."
        ) from exception


def _window_metadata(
    window: WindowRecord,
) -> dict[str, Any]:
    return {
        "schema_version":
            window.schema_version,
        "participant_id":
            window.participant_id,
        "session_id":
            window.session_id,
        "session_mode":
            window.session_mode,
        "time_reference":
            window.time_reference,
        "window_id":
            window.window_id,
        "label":
            window.label,
        "source_category":
            window.source_category,
        "source_id":
            window.source_id,
        "scenario":
            window.scenario,
        "container_type":
            window.container_type,
        "source_start_seconds":
            window.source_start_seconds,
        "source_end_seconds":
            window.source_end_seconds,
        "window_index":
            window.window_index,
        "window_start_seconds":
            window.window_start_seconds,
        "window_end_seconds":
            window.window_end_seconds,
        "window_length_seconds":
            window.window_length_seconds,
        "stride_seconds":
            window.stride_seconds,
    }


def _validate_feature_table(
    *,
    frame: pd.DataFrame,
    feature_prefixes: tuple[str, ...],
    expected_feature_count: int,
    description: str,
) -> None:
    if frame.empty:
        raise FeatureDatasetError(
            f"{description}-Featuretabelle ist leer."
        )

    if frame["window_id"].duplicated().any():
        raise FeatureDatasetError(
            f"{description}-Featuretabelle enthält "
            "doppelte window_id."
        )

    feature_columns = [
        column
        for column in frame.columns
        if column.startswith(
            feature_prefixes
        )
    ]

    if (
        len(feature_columns)
        != expected_feature_count
    ):
        raise FeatureDatasetError(
            f"{description}: erwartet wurden "
            f"{expected_feature_count} Features, "
            f"gefunden wurden {len(feature_columns)}."
        )

    values = frame[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        values
    ).all():
        raise FeatureDatasetError(
            f"{description}-Featuretabelle enthält "
            "NaN oder unendliche Werte."
        )


def _optional_string(
    value: Any,
) -> str | None:
    if value is None or pd.isna(
        value
    ):
        return None

    text = str(
        value
    ).strip()

    return text if text else None


def _safe_file_token(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(
            value,
            str,
        )
        or not value
        or not all(
            character.isalnum()
            or character in {
                "-",
                "_",
            }
            for character in value
        )
    ):
        raise FeatureDatasetError(
            f"{name} darf nur Buchstaben, Zahlen, "
            "Bindestrich und Unterstrich enthalten."
        )

    return value