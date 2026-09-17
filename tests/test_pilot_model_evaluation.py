import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trinkerkennung_analysis.model_evaluation import (
    FoldEvaluationResult,
)
from trinkerkennung_analysis.pilot_model_evaluation import (
    prediction_outcome,
    run_pilot_evaluation,
)


class PilotModelEvaluationTest(unittest.TestCase):
    def test_prediction_outcome_maps_confusion_cases(self):
        self.assertEqual(
            prediction_outcome(
                label="DRINK",
                predicted_label="DRINK",
            ),
            "TP",
        )
        self.assertEqual(
            prediction_outcome(
                label="DRINK",
                predicted_label="NON_DRINK",
            ),
            "FN",
        )
        self.assertEqual(
            prediction_outcome(
                label="NON_DRINK",
                predicted_label="DRINK",
            ),
            "FP",
        )
        self.assertEqual(
            prediction_outcome(
                label="NON_DRINK",
                predicted_label="NON_DRINK",
            ),
            "TN",
        )

    def test_runner_exports_all_pilot_configurations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            feature_root = root / "features"
            output_root = root / "results"

            feature_root.mkdir()

            base_frame = pd.DataFrame(
                {
                    "participant_id": [
                        "P101",
                        "P102",
                        "P103",
                    ],
                    "window_id": [
                        "W001",
                        "W002",
                        "W003",
                    ],
                    "label": [
                        "DRINK",
                        "DRINK",
                        "NON_DRINK",
                    ],
                }
            )

            for configuration in ("1s", "2s"):
                for modality in (
                    "audio",
                    "watch",
                    "fusion",
                ):
                    base_frame.to_csv(
                        feature_root
                        / (
                            f"pilot_{modality}_features_"
                            f"{configuration}.csv"
                        ),
                        sep=";",
                        index=False,
                        encoding="utf-8",
                    )

            def fake_evaluate_loso(
                *,
                frame,
                modality,
                model_name,
                return_predictions=False,
            ):
                participants = (
                    "P101",
                    "P102",
                    "P103",
                )

                fold_results = []

                for test_participant in participants:
                    train_participants = tuple(
                        participant
                        for participant in participants
                        if participant != test_participant
                    )

                    if test_participant == "P101":
                        result = FoldEvaluationResult(
                            model_name=model_name,
                            modality=modality,
                            test_participant=test_participant,
                            train_participants=train_participants,
                            train_rows=2,
                            test_rows=1,
                            test_drink_rows=1,
                            test_non_drink_rows=0,
                            precision_drink=0.0,
                            recall_drink=0.0,
                            f1_drink=0.0,
                            balanced_accuracy=0.0,
                            accuracy=0.0,
                            true_negative=0,
                            false_positive=0,
                            false_negative=1,
                            true_positive=0,
                        )
                    elif test_participant == "P102":
                        result = FoldEvaluationResult(
                            model_name=model_name,
                            modality=modality,
                            test_participant=test_participant,
                            train_participants=train_participants,
                            train_rows=2,
                            test_rows=1,
                            test_drink_rows=1,
                            test_non_drink_rows=0,
                            precision_drink=1.0,
                            recall_drink=1.0,
                            f1_drink=1.0,
                            balanced_accuracy=1.0,
                            accuracy=1.0,
                            true_negative=0,
                            false_positive=0,
                            false_negative=0,
                            true_positive=1,
                        )
                    else:
                        result = FoldEvaluationResult(
                            model_name=model_name,
                            modality=modality,
                            test_participant=test_participant,
                            train_participants=train_participants,
                            train_rows=2,
                            test_rows=1,
                            test_drink_rows=0,
                            test_non_drink_rows=1,
                            precision_drink=0.0,
                            recall_drink=0.0,
                            f1_drink=0.0,
                            balanced_accuracy=1.0,
                            accuracy=1.0,
                            true_negative=1,
                            false_positive=0,
                            false_negative=0,
                            true_positive=0,
                        )

                    fold_results.append(result)

                predictions = pd.DataFrame(
                    {
                        "participant_id": [
                            "P101",
                            "P102",
                            "P103",
                        ],
                        "window_id": [
                            "W001",
                            "W002",
                            "W003",
                        ],
                        "label": [
                            "DRINK",
                            "DRINK",
                            "NON_DRINK",
                        ],
                        "predicted_label": [
                            "NON_DRINK",
                            "DRINK",
                            "NON_DRINK",
                        ],
                        "drink_score": [
                            0.1,
                            0.9,
                            0.2,
                        ],
                        "is_correct": [
                            False,
                            True,
                            True,
                        ],
                        "model_name": [
                            model_name,
                            model_name,
                            model_name,
                        ],
                        "modality": [
                            modality,
                            modality,
                            modality,
                        ],
                        "test_participant": [
                            "P101",
                            "P102",
                            "P103",
                        ],
                        "train_participants": [
                            "P102, P103",
                            "P101, P103",
                            "P101, P102",
                        ],
                    }
                )

                if return_predictions:
                    return (
                        fold_results,
                        predictions,
                    )

                return fold_results

            with patch(
                "trinkerkennung_analysis."
                "pilot_model_evaluation.evaluate_loso",
                side_effect=fake_evaluate_loso,
            ):
                (
                    fold_table,
                    summary_table,
                    prediction_table,
                    error_table,
                ) = run_pilot_evaluation(
                    feature_root=feature_root,
                    output_root=output_root,
                )

            self.assertEqual(
                len(fold_table),
                42,
            )
            self.assertEqual(
                len(summary_table),
                14,
            )
            self.assertEqual(
                len(prediction_table),
                42,
            )
            self.assertEqual(
                len(error_table),
                14,
            )

            self.assertEqual(
                set(
                    summary_table[
                        "window_configuration"
                    ]
                ),
                {"1s", "2s"},
            )

            for configuration in ("1s", "2s"):
                configuration_rows = (
                    summary_table.loc[
                        summary_table[
                            "window_configuration"
                        ]
                        == configuration
                    ]
                )
                self.assertEqual(
                    len(configuration_rows),
                    7,
                )

            self.assertTrue(
                (
                    error_table["outcome"]
                    == "FN"
                ).all()
            )

            expected_files = (
                "pilot_fold_results.csv",
                "pilot_summary_results.csv",
                "pilot_window_predictions.csv",
                "pilot_window_errors.csv",
            )

            for filename in expected_files:
                self.assertTrue(
                    (
                        output_root
                        / filename
                    ).is_file()
                )


if __name__ == "__main__":
    unittest.main()
