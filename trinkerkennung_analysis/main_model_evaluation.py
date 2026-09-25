from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trinkerkennung_analysis.model_evaluation import (
    MODEL_DUMMY,
    MODEL_LOGISTIC,
    MODEL_RANDOM_FOREST,
    MODALITY_AUDIO,
    MODALITY_FUSION,
    MODALITY_WATCH,
    evaluate_loso,
    summarize_fold_results,
    validate_comparable_feature_tables,
)


WINDOW_CONFIGURATIONS = (
    "1s_stride1s",
    "2s_stride2s",
)

FINAL_TEST_PARTICIPANTS = (
    "P104",
    "P105",
    "P106",
    "P107",
    "P108",
)

LEARNING_RUNS = (
    (MODALITY_AUDIO, MODEL_LOGISTIC),
    (MODALITY_AUDIO, MODEL_RANDOM_FOREST),
    (MODALITY_WATCH, MODEL_LOGISTIC),
    (MODALITY_WATCH, MODEL_RANDOM_FOREST),
    (MODALITY_FUSION, MODEL_LOGISTIC),
    (MODALITY_FUSION, MODEL_RANDOM_FOREST),
)

FEATURE_FILENAMES = {
    "1s_stride1s": {
        MODALITY_AUDIO: (
            "primary_P101_P108_"
            "audio_features_1s_stride1s.csv"
        ),
        MODALITY_WATCH: (
            "primary_P101_P108_"
            "watch_features_1s_stride1s.csv"
        ),
        MODALITY_FUSION: (
            "primary_P101_P108_"
            "fusion_features_1s_stride1s.csv"
        ),
    },
    "2s_stride2s": {
        MODALITY_AUDIO: (
            "sensitivity_P101_P108_"
            "audio_features_2s_stride2s.csv"
        ),
        MODALITY_WATCH: (
            "sensitivity_P101_P108_"
            "watch_features_2s_stride2s.csv"
        ),
        MODALITY_FUSION: (
            "sensitivity_P101_P108_"
            "fusion_features_2s_stride2s.csv"
        ),
    },
}


def load_feature_table(
    *,
    feature_root: Path,
    modality: str,
    configuration: str,
) -> pd.DataFrame:
    try:
        filename = FEATURE_FILENAMES[
            configuration
        ][modality]
    except KeyError as exception:
        raise ValueError(
            "Unbekannte Feature-Konfiguration: "
            f"configuration={configuration}, "
            f"modality={modality}"
        ) from exception

    path = feature_root / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Featuredatei fehlt: {path}"
        )

    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
    )


def prediction_outcome(
    *,
    label: str,
    predicted_label: str,
) -> str:
    if label == "DRINK":
        if predicted_label == "DRINK":
            return "TP"
        return "FN"

    if predicted_label == "DRINK":
        return "FP"

    return "TN"


def run_main_evaluation(
    *,
    feature_root: Path,
    output_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_rows: list[
        dict[str, object]
    ] = []

    summary_rows: list[
        dict[str, object]
    ] = []

    prediction_tables: list[
        pd.DataFrame
    ] = []

    for configuration in (
        WINDOW_CONFIGURATIONS
    ):
        audio = load_feature_table(
            feature_root=feature_root,
            modality=MODALITY_AUDIO,
            configuration=configuration,
        )

        watch = load_feature_table(
            feature_root=feature_root,
            modality=MODALITY_WATCH,
            configuration=configuration,
        )

        fusion = load_feature_table(
            feature_root=feature_root,
            modality=MODALITY_FUSION,
            configuration=configuration,
        )

        validate_comparable_feature_tables(
            audio=audio,
            watch=watch,
            fusion=fusion,
        )

        frames = {
            MODALITY_AUDIO: audio,
            MODALITY_WATCH: watch,
            MODALITY_FUSION: fusion,
        }

        runs: list[
            tuple[
                str,
                str,
                str,
                pd.DataFrame,
            ]
        ] = [
            (
                "BASELINE",
                MODALITY_AUDIO,
                MODEL_DUMMY,
                audio,
            )
        ]

        for (
            modality,
            model_name,
        ) in LEARNING_RUNS:
            runs.append(
                (
                    modality,
                    modality,
                    model_name,
                    frames[modality],
                )
            )

        for (
            output_modality,
            evaluation_modality,
            model_name,
            frame,
        ) in runs:
            (
                fold_results,
                predictions,
            ) = evaluate_loso(
                frame=frame,
                modality=(
                    evaluation_modality
                ),
                model_name=model_name,
                return_predictions=True,
                test_participants=(
                    FINAL_TEST_PARTICIPANTS
                ),
            )

            for result in fold_results:
                row = result.to_dict()

                row[
                    "window_configuration"
                ] = configuration

                row[
                    "modality"
                ] = output_modality

                fold_rows.append(
                    row
                )

            summary = (
                summarize_fold_results(
                    fold_results
                )
            )

            summary_rows.append(
                {
                    "window_configuration":
                        configuration,
                    "modality":
                        output_modality,
                    "model_name":
                        model_name,
                    **summary,
                    "accuracy_mean":
                        sum(
                            result.accuracy
                            for result
                            in fold_results
                        )
                        / len(
                            fold_results
                        ),
                    "true_negative_total":
                        sum(
                            result.true_negative
                            for result
                            in fold_results
                        ),
                    "false_positive_total":
                        sum(
                            result.false_positive
                            for result
                            in fold_results
                        ),
                    "false_negative_total":
                        sum(
                            result.false_negative
                            for result
                            in fold_results
                        ),
                    "true_positive_total":
                        sum(
                            result.true_positive
                            for result
                            in fold_results
                        ),
                }
            )

            predictions = (
                predictions.copy()
            )

            predictions[
                "window_configuration"
            ] = configuration

            predictions[
                "modality"
            ] = output_modality

            predictions[
                "outcome"
            ] = [
                prediction_outcome(
                    label=str(label),
                    predicted_label=str(
                        predicted_label
                    ),
                )
                for (
                    label,
                    predicted_label,
                )
                in zip(
                    predictions[
                        "label"
                    ],
                    predictions[
                        "predicted_label"
                    ],
                    strict=True,
                )
            ]

            prediction_tables.append(
                predictions
            )

    fold_table = pd.DataFrame(
        fold_rows
    )

    summary_table = pd.DataFrame(
        summary_rows
    )

    prediction_table = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    error_table = (
        prediction_table.loc[
            ~prediction_table[
                "is_correct"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    fold_path = (
        output_root
        / "main_fold_results.csv"
    )

    summary_path = (
        output_root
        / "main_summary_results.csv"
    )

    prediction_path = (
        output_root
        / "main_window_predictions.csv"
    )

    error_path = (
        output_root
        / "main_window_errors.csv"
    )

    fold_table.to_csv(
        fold_path,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    summary_table.to_csv(
        summary_path,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    prediction_table.to_csv(
        prediction_path,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    error_table.to_csv(
        error_path,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    return (
        fold_table,
        summary_table,
        prediction_table,
        error_table,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finale personenseparierte "
            "Modellbewertung der Hauptstudie."
        )
    )

    parser.add_argument(
        "--feature-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )

    arguments = parser.parse_args()

    run_main_evaluation(
        feature_root=(
            arguments.feature_root
        ),
        output_root=(
            arguments.output_root
        ),
    )


if __name__ == "__main__":
    main()
