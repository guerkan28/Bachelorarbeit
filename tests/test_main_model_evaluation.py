from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trinkerkennung_analysis.main_model_evaluation import (
    FINAL_TEST_PARTICIPANTS,
    run_main_evaluation,
)
from trinkerkennung_analysis.model_evaluation import (
    FoldEvaluationResult,
)


class MainModelEvaluationTest(unittest.TestCase):
    def test_runner_evaluates_both_window_configurations_with_main_test_persons(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            feature_root = root / "features"
            output_root = root / "results"

            feature_root.mkdir()

            participants = (
                "P101",
                "P102",
                "P103",
                "P104",
                "P105",
                "P106",
                "P107",
                "P108",
            )

            base_frame = pd.DataFrame(
                {
                    "participant_id": participants,
                    "window_id": [
                        f"{participant}_W001"
                        for participant in participants
                    ],
                    "label": [
                        "NON_DRINK",
                        "DRINK",
                        "NON_DRINK",
                        "DRINK",
                        "NON_DRINK",
                        "DRINK",
                        "NON_DRINK",
                        "DRINK",
                    ],
                }
            )

            feature_files = {
                "1s_stride1s": {
                    "audio":
                        "primary_P101_P108_"
                        "audio_features_1s_stride1s.csv",
                    "watch":
                        "primary_P101_P108_"
                        "watch_features_1s_stride1s.csv",
                    "fusion":
                        "primary_P101_P108_"
                        "fusion_features_1s_stride1s.csv",
                },
                "2s_stride2s": {
                    "audio":
                        "sensitivity_P101_P108_"
                        "audio_features_2s_stride2s.csv",
                    "watch":
                        "sensitivity_P101_P108_"
                        "watch_features_2s_stride2s.csv",
                    "fusion":
                        "sensitivity_P101_P108_"
                        "fusion_features_2s_stride2s.csv",
                },
            }

            for filenames in feature_files.values():
                audio = base_frame.copy()
                audio["audio_test_feature"] = range(
                    len(audio)
                )

                watch = base_frame.copy()
                watch["acc_test_feature"] = range(
                    len(watch)
                )

                fusion = base_frame.copy()
                fusion["audio_test_feature"] = range(
                    len(fusion)
                )
                fusion["acc_test_feature"] = range(
                    len(fusion)
                )

                audio.to_csv(
                    feature_root / filenames["audio"],
                    sep=";",
                    index=False,
                    encoding="utf-8",
                )

                watch.to_csv(
                    feature_root / filenames["watch"],
                    sep=";",
                    index=False,
                    encoding="utf-8",
                )

                fusion.to_csv(
                    feature_root / filenames["fusion"],
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
                test_participants=None,
            ):
                self.assertEqual(
                    tuple(test_participants),
                    FINAL_TEST_PARTICIPANTS,
                )

                fold_results = []

                for test_participant in (
                    FINAL_TEST_PARTICIPANTS
                ):
                    train_participants = tuple(
                        participant
                        for participant in participants
                        if participant
                        != test_participant
                    )

                    fold_results.append(
                        FoldEvaluationResult(
                            model_name=model_name,
                            modality=modality,
                            test_participant=(
                                test_participant
                            ),
                            train_participants=(
                                train_participants
                            ),
                            train_rows=7,
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
                    )

                predictions = pd.DataFrame(
                    {
                        "participant_id":
                            list(
                                FINAL_TEST_PARTICIPANTS
                            ),
                        "window_id": [
                            f"{participant}_W001"
                            for participant
                            in FINAL_TEST_PARTICIPANTS
                        ],
                        "label": [
                            "DRINK"
                            for _ in (
                                FINAL_TEST_PARTICIPANTS
                            )
                        ],
                        "predicted_label": [
                            "NON_DRINK"
                            for _ in (
                                FINAL_TEST_PARTICIPANTS
                            )
                        ],
                        "drink_score": [
                            0.1
                            for _ in (
                                FINAL_TEST_PARTICIPANTS
                            )
                        ],
                        "is_correct": [
                            False
                            for _ in (
                                FINAL_TEST_PARTICIPANTS
                            )
                        ],
                        "model_name": [
                            model_name
                            for _ in (
                                FINAL_TEST_PARTICIPANTS
                            )
                        ],
                        "modality": [
                            modality
                            for _ in (
                                FINAL_TEST_PARTICIPANTS
                            )
                        ],
                        "test_participant":
                            list(
                                FINAL_TEST_PARTICIPANTS
                            ),
                        "train_participants": [
                            ", ".join(
                                participant
                                for participant
                                in participants
                                if participant
                                != test_participant
                            )
                            for test_participant
                            in FINAL_TEST_PARTICIPANTS
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
                "main_model_evaluation.evaluate_loso",
                side_effect=fake_evaluate_loso,
            ) as evaluate_mock:
                (
                    fold_table,
                    summary_table,
                    prediction_table,
                    error_table,
                ) = run_main_evaluation(
                    feature_root=feature_root,
                    output_root=output_root,
                )

            self.assertEqual(
                evaluate_mock.call_count,
                14,
            )

            self.assertEqual(
                len(fold_table),
                70,
            )

            self.assertEqual(
                len(summary_table),
                14,
            )

            self.assertEqual(
                len(prediction_table),
                70,
            )

            self.assertEqual(
                len(error_table),
                70,
            )

            self.assertEqual(
                set(
                    fold_table[
                        "window_configuration"
                    ]
                ),
                {
                    "1s_stride1s",
                    "2s_stride2s",
                },
            )

            self.assertEqual(
                set(
                    summary_table[
                        "window_configuration"
                    ]
                ),
                {
                    "1s_stride1s",
                    "2s_stride2s",
                },
            )

            for configuration in (
                "1s_stride1s",
                "2s_stride2s",
            ):
                configuration_folds = (
                    fold_table.loc[
                        fold_table[
                            "window_configuration"
                        ]
                        == configuration
                    ]
                )

                configuration_summary = (
                    summary_table.loc[
                        summary_table[
                            "window_configuration"
                        ]
                        == configuration
                    ]
                )

                self.assertEqual(
                    len(configuration_folds),
                    35,
                )

                self.assertEqual(
                    len(configuration_summary),
                    7,
                )

                self.assertEqual(
                    set(
                        configuration_folds[
                            "test_participant"
                        ]
                    ),
                    set(
                        FINAL_TEST_PARTICIPANTS
                    ),
                )

            expected_runs = {
                (
                    "BASELINE",
                    "DUMMY_MOST_FREQUENT",
                ),
                (
                    "AUDIO",
                    "LOGISTIC_REGRESSION",
                ),
                (
                    "AUDIO",
                    "RANDOM_FOREST",
                ),
                (
                    "WATCH",
                    "LOGISTIC_REGRESSION",
                ),
                (
                    "WATCH",
                    "RANDOM_FOREST",
                ),
                (
                    "FUSION",
                    "LOGISTIC_REGRESSION",
                ),
                (
                    "FUSION",
                    "RANDOM_FOREST",
                ),
            }

            actual_runs = set(
                zip(
                    summary_table["modality"],
                    summary_table["model_name"],
                    strict=True,
                )
            )

            self.assertEqual(
                actual_runs,
                expected_runs,
            )

            for train_participants in (
                fold_table[
                    "train_participants"
                ]
            ):
                self.assertIn(
                    "P101",
                    train_participants,
                )
                self.assertIn(
                    "P102",
                    train_participants,
                )
                self.assertIn(
                    "P103",
                    train_participants,
                )

            expected_files = (
                "main_fold_results.csv",
                "main_summary_results.csv",
                "main_window_predictions.csv",
                "main_window_errors.csv",
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
