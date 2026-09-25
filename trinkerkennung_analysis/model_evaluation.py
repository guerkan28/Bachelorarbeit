from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


POSITIVE_LABEL = "DRINK"
NEGATIVE_LABEL = "NON_DRINK"

MODEL_DUMMY = "DUMMY_MOST_FREQUENT"
MODEL_LOGISTIC = "LOGISTIC_REGRESSION"
MODEL_RANDOM_FOREST = "RANDOM_FOREST"

MODALITY_AUDIO = "AUDIO"
MODALITY_WATCH = "WATCH"
MODALITY_FUSION = "FUSION"

EXPECTED_FEATURE_COUNTS = {
    MODALITY_AUDIO: 38,
    MODALITY_WATCH: 48,
    MODALITY_FUSION: 86,
}


class ModelEvaluationError(ValueError):
    """Fehler bei der personenseparierten ML-Evaluation."""


@dataclass(frozen=True)
class FoldEvaluationResult:
    model_name: str
    modality: str
    test_participant: str
    train_participants: tuple[str, ...]
    train_rows: int
    test_rows: int
    test_drink_rows: int
    test_non_drink_rows: int
    precision_drink: float
    recall_drink: float
    f1_drink: float
    balanced_accuracy: float
    accuracy: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["train_participants"] = ",".join(
            self.train_participants
        )

        return result


def build_model(
    model_name: str,
) -> ClassifierMixin:
    if model_name == MODEL_DUMMY:
        return DummyClassifier(
            strategy="most_frequent",
        )

    if model_name == MODEL_LOGISTIC:
        return Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ])

    if model_name == MODEL_RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    raise ModelEvaluationError(
        f"Unbekanntes Modell: {model_name}"
    )


def select_feature_columns(
    frame: pd.DataFrame,
    modality: str,
) -> list[str]:
    if modality == MODALITY_AUDIO:
        prefixes = (
            "audio_",
        )

    elif modality == MODALITY_WATCH:
        prefixes = (
            "acc_",
            "gyro_",
        )

    elif modality == MODALITY_FUSION:
        prefixes = (
            "audio_",
            "acc_",
            "gyro_",
        )

    else:
        raise ModelEvaluationError(
            f"Unbekannte ModalitÃ¤t: {modality}"
        )

    columns = [
        column
        for column in frame.columns
        if column.startswith(
            prefixes
        )
    ]

    expected = EXPECTED_FEATURE_COUNTS[
        modality
    ]

    if len(columns) != expected:
        raise ModelEvaluationError(
            f"{modality}: erwartet wurden "
            f"{expected} Feature-Spalten, "
            f"gefunden wurden {len(columns)}."
        )

    return columns

def validate_comparable_feature_tables(
    *,
    audio: pd.DataFrame,
    watch: pd.DataFrame,
    fusion: pd.DataFrame,
) -> None:
    required_columns = (
        "participant_id",
        "window_id",
        "label",
    )

    reference = audio

    for description, frame in (
        ("Audio", audio),
        ("Watch", watch),
        ("Fusion", fusion),
    ):
        missing_columns = [
            column
            for column in required_columns
            if column not in frame.columns
        ]

        if missing_columns:
            raise ModelEvaluationError(
                f"{description}: Pflichtspalten fehlen: "
                + ", ".join(
                    missing_columns
                )
            )

        if len(frame) != len(reference):
            raise ModelEvaluationError(
                f"{description}: Zeilenanzahl stimmt "
                "nicht mit Audio überein."
            )

        for column in required_columns:
            if (
                frame[column]
                .astype(str)
                .tolist()
                != reference[column]
                .astype(str)
                .tolist()
            ):
                raise ModelEvaluationError(
                    f"{description}: Spalte {column} "
                    "stimmt nicht mit Audio überein."
                )

def evaluate_loso(
    *,
    frame: pd.DataFrame,
    modality: str,
    model_name: str,
    return_predictions: bool = False,
    test_participants: (
        tuple[str, ...]
        | list[str]
        | set[str]
        | None
    ) = None,
) -> (
    list[FoldEvaluationResult]
    | tuple[list[FoldEvaluationResult], pd.DataFrame]
):
    required_columns = (
        "participant_id",
        "label",
    )
    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]
    if missing_columns:
        raise ModelEvaluationError(
            "Pflichtspalten fehlen: "
            + ", ".join(missing_columns)
        )

    if frame.empty:
        raise ModelEvaluationError(
            "Der Feature-Datensatz ist leer."
        )

    feature_columns = select_feature_columns(
        frame=frame,
        modality=modality,
    )

    feature_matrix = frame[
        feature_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(feature_matrix).all():
        raise ModelEvaluationError(
            "Der Feature-Datensatz enthält "
            "NaN- oder Inf-Werte."
        )

    labels = (
        frame["label"]
        .astype(str)
        .to_numpy()
    )
    groups = (
        frame["participant_id"]
        .astype(str)
        .to_numpy()
    )

    unique_labels = set(labels)
    expected_labels = {
        NEGATIVE_LABEL,
        POSITIVE_LABEL,
    }
    if unique_labels != expected_labels:
        raise ModelEvaluationError(
            "Erwartet werden ausschließlich die Labels "
            f"{NEGATIVE_LABEL} und {POSITIVE_LABEL}."
        )

    participants = sorted(set(groups))
    if len(participants) < 3:
        raise ModelEvaluationError(
            "Für die LOSO-Auswertung werden mindestens "
            "drei Teilnehmende benötigt."
        )

    if test_participants is None:
        selected_test_participants = set(
            participants
        )
    else:
        selected_test_participants = {
            str(participant)
            for participant in test_participants
        }

        if not selected_test_participants:
            raise ModelEvaluationError(
                "test_participants darf nicht leer sein."
            )

        unknown_test_participants = sorted(
            selected_test_participants
            - set(participants)
        )

        if unknown_test_participants:
            raise ModelEvaluationError(
                "Unbekannte Testpersonen: "
                + ", ".join(
                    unknown_test_participants
                )
            )

    metadata_columns = [
        column
        for column in frame.columns
        if column not in feature_columns
    ]

    splitter = LeaveOneGroupOut()

    fold_results: list[FoldEvaluationResult] = []
    prediction_tables: list[pd.DataFrame] = []

    for train_indices, test_indices in splitter.split(
        feature_matrix,
        labels,
        groups,
    ):
        train_participants = tuple(
            sorted(
                set(groups[train_indices])
            )
        )
        fold_test_participants = sorted(
            set(groups[test_indices])
        )

        if len(fold_test_participants) != 1:
            raise ModelEvaluationError(
                "Ein LOSO-Testfold muss genau eine "
                "Testperson enthalten."
            )

        test_participant = fold_test_participants[0]

        if (
            test_participant
            not in selected_test_participants
        ):
            continue

        if test_participant in train_participants:
            raise ModelEvaluationError(
                "LOSO-Fehler: Die Testperson befindet "
                "sich gleichzeitig im Training."
            )

        train_features = feature_matrix[
            train_indices
        ]
        test_features = feature_matrix[
            test_indices
        ]
        train_labels = labels[
            train_indices
        ]
        test_labels = labels[
            test_indices
        ]

        model = build_model(
            model_name=model_name,
        )
        model.fit(
            train_features,
            train_labels,
        )

        predicted_labels = model.predict(
            test_features
        )

        confusion = confusion_matrix(
            test_labels,
            predicted_labels,
            labels=[
                NEGATIVE_LABEL,
                POSITIVE_LABEL,
            ],
        )
        (
            true_negative,
            false_positive,
            false_negative,
            true_positive,
        ) = confusion.ravel()

        fold_results.append(
            FoldEvaluationResult(
                model_name=model_name,
                modality=modality,
                test_participant=test_participant,
                train_participants=train_participants,
                train_rows=len(train_indices),
                test_rows=len(test_indices),
                test_drink_rows=int(
                    np.sum(
                        test_labels
                        == POSITIVE_LABEL
                    )
                ),
                test_non_drink_rows=int(
                    np.sum(
                        test_labels
                        == NEGATIVE_LABEL
                    )
                ),
                precision_drink=precision_score(
                    test_labels,
                    predicted_labels,
                    pos_label=POSITIVE_LABEL,
                    zero_division=0,
                ),
                recall_drink=recall_score(
                    test_labels,
                    predicted_labels,
                    pos_label=POSITIVE_LABEL,
                    zero_division=0,
                ),
                f1_drink=f1_score(
                    test_labels,
                    predicted_labels,
                    pos_label=POSITIVE_LABEL,
                    zero_division=0,
                ),
                balanced_accuracy=balanced_accuracy_score(
                    test_labels,
                    predicted_labels,
                ),
                accuracy=accuracy_score(
                    test_labels,
                    predicted_labels,
                ),
                true_negative=int(
                    true_negative
                ),
                false_positive=int(
                    false_positive
                ),
                false_negative=int(
                    false_negative
                ),
                true_positive=int(
                    true_positive
                ),
            )
        )

        if return_predictions:
            if not hasattr(
                model,
                "predict_proba",
            ):
                raise ModelEvaluationError(
                    f"{model_name} unterstützt keine "
                    "Wahrscheinlichkeitsausgabe."
                )

            class_labels = [
                str(value)
                for value in model.classes_
            ]

            if POSITIVE_LABEL not in class_labels:
                raise ModelEvaluationError(
                    "Das trainierte Modell enthält "
                    "keine DRINK-Klasse."
                )

            drink_class_index = (
                class_labels.index(
                    POSITIVE_LABEL
                )
            )

            drink_scores = model.predict_proba(
                test_features
            )[:, drink_class_index]

            prediction_frame = (
                frame.iloc[test_indices][
                    metadata_columns
                ]
                .copy()
            )

            prediction_frame[
                "predicted_label"
            ] = predicted_labels

            prediction_frame[
                "drink_score"
            ] = drink_scores

            prediction_frame[
                "is_correct"
            ] = (
                prediction_frame[
                    "label"
                ]
                .astype(str)
                .to_numpy()
                == predicted_labels.astype(str)
            )

            prediction_frame[
                "model_name"
            ] = model_name
            prediction_frame[
                "modality"
            ] = modality
            prediction_frame[
                "test_participant"
            ] = test_participant
            prediction_frame[
                "train_participants"
            ] = ", ".join(
                train_participants
            )

            prediction_frame[
                "_evaluation_row_order"
            ] = test_indices

            prediction_tables.append(
                prediction_frame
            )

    if not return_predictions:
        return fold_results

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    predictions = (
        predictions
        .sort_values(
            "_evaluation_row_order"
        )
        .drop(
            columns=[
                "_evaluation_row_order",
            ]
        )
        .reset_index(drop=True)
    )

    return (
        fold_results,
        predictions,
    )


def summarize_fold_results(
    results: list[
        FoldEvaluationResult
    ],
) -> dict[str, float]:
    if not results:
        raise ModelEvaluationError(
            "Keine Fold-Ergebnisse vorhanden."
        )

    f1_values = np.asarray(
        [
            result.f1_drink
            for result in results
        ],
        dtype=np.float64,
    )

    precision_values = np.asarray(
        [
            result.precision_drink
            for result in results
        ],
        dtype=np.float64,
    )

    recall_values = np.asarray(
        [
            result.recall_drink
            for result in results
        ],
        dtype=np.float64,
    )

    balanced_values = np.asarray(
        [
            result.balanced_accuracy
            for result in results
        ],
        dtype=np.float64,
    )

    return {
        "fold_count":
            float(
                len(
                    results
                )
            ),
        "f1_drink_mean":
            float(
                np.mean(
                    f1_values
                )
            ),
        "f1_drink_std":
            float(
                np.std(
                    f1_values
                )
            ),
        "precision_drink_mean":
            float(
                np.mean(
                    precision_values
                )
            ),
        "recall_drink_mean":
            float(
                np.mean(
                    recall_values
                )
            ),
        "balanced_accuracy_mean":
            float(
                np.mean(
                    balanced_values
                )
            ),
    }