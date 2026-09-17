from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trinkerkennung_analysis.model_evaluation import (
    MODEL_DUMMY,
    MODEL_LOGISTIC,
    MODEL_RANDOM_FOREST,
    MODALITY_AUDIO,
    MODALITY_FUSION,
    MODALITY_WATCH,
    ModelEvaluationError,
    build_model,
    evaluate_loso,
    select_feature_columns,
    summarize_fold_results,
    validate_comparable_feature_tables,
)


class ModelEvaluationTest(unittest.TestCase):
    def test_feature_selection_uses_only_allowed_prefixes(
        self,
    ) -> None:
        frame = self._fusion_frame()

        audio_columns = select_feature_columns(
            frame,
            MODALITY_AUDIO,
        )

        watch_columns = select_feature_columns(
            frame,
            MODALITY_WATCH,
        )

        fusion_columns = select_feature_columns(
            frame,
            MODALITY_FUSION,
        )

        self.assertEqual(
            len(audio_columns),
            38,
        )

        self.assertEqual(
            len(watch_columns),
            48,
        )

        self.assertEqual(
            len(fusion_columns),
            86,
        )

        self.assertTrue(
            all(
                column.startswith("audio_")
                for column in audio_columns
            )
        )

        self.assertTrue(
            all(
                column.startswith(
                    (
                        "acc_",
                        "gyro_",
                    )
                )
                for column in watch_columns
            )
        )

        forbidden_metadata = {
            "participant_id",
            "session_id",
            "window_id",
            "label",
            "scenario",
            "container_type",
        }

        self.assertTrue(
            forbidden_metadata.isdisjoint(
                fusion_columns
            )
        )

    def test_logistic_regression_uses_scaler_pipeline(
        self,
    ) -> None:
        model = build_model(
            MODEL_LOGISTIC
        )

        self.assertIsInstance(
            model,
            Pipeline,
        )

        self.assertIsInstance(
            model.named_steps["scaler"],
            StandardScaler,
        )

        classifier = (
            model.named_steps["classifier"]
        )

        self.assertEqual(
            classifier.C,
            1.0,
        )

        self.assertEqual(
            classifier.max_iter,
            2000,
        )

        self.assertEqual(
            classifier.class_weight,
            "balanced",
        )

    def test_random_forest_is_reproducible_and_balanced(
        self,
    ) -> None:
        model = build_model(
            MODEL_RANDOM_FOREST
        )

        self.assertIsInstance(
            model,
            RandomForestClassifier,
        )

        self.assertEqual(
            model.n_estimators,
            500,
        )

        self.assertEqual(
            model.random_state,
            42,
        )

        self.assertEqual(
            model.class_weight,
            "balanced",
        )

    def test_window_predictions_match_fold_confusion_matrix(self):
        frame = self._audio_frame()

        results, predictions = evaluate_loso(
            frame=frame,
            modality=MODALITY_AUDIO,
            model_name=MODEL_DUMMY,
            return_predictions=True,
        )

        results_by_participant = {
            result.test_participant: result
            for result in results
        }

        for test_participant, group in predictions.groupby(
            "test_participant"
        ):
            result = results_by_participant[
                str(test_participant)
            ]

            true_negative = int(
                (
                    (group["label"] == "NON_DRINK")
                    & (
                        group["predicted_label"]
                        == "NON_DRINK"
                    )
                ).sum()
            )

            false_positive = int(
                (
                    (group["label"] == "NON_DRINK")
                    & (
                        group["predicted_label"]
                        == "DRINK"
                    )
                ).sum()
            )

            false_negative = int(
                (
                    (group["label"] == "DRINK")
                    & (
                        group["predicted_label"]
                        == "NON_DRINK"
                    )
                ).sum()
            )

            true_positive = int(
                (
                    (group["label"] == "DRINK")
                    & (
                        group["predicted_label"]
                        == "DRINK"
                    )
                ).sum()
            )

            self.assertEqual(
                true_negative,
                result.true_negative,
            )
            self.assertEqual(
                false_positive,
                result.false_positive,
            )
            self.assertEqual(
                false_negative,
                result.false_negative,
            )
            self.assertEqual(
                true_positive,
                result.true_positive,
            )

    def test_loso_can_return_window_level_predictions(self):
        frame = self._audio_frame()

        results, predictions = evaluate_loso(
            frame=frame,
            modality=MODALITY_AUDIO,
            model_name=MODEL_DUMMY,
            return_predictions=True,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(len(predictions), len(frame))

        self.assertEqual(
            set(predictions["window_id"].astype(str)),
            set(frame["window_id"].astype(str)),
        )

        required_columns = {
            "window_id",
            "participant_id",
            "label",
            "predicted_label",
            "drink_score",
            "is_correct",
            "model_name",
            "modality",
            "test_participant",
            "train_participants",
        }
        self.assertTrue(
            required_columns.issubset(predictions.columns)
        )

        self.assertTrue(
            predictions["drink_score"]
            .between(0.0, 1.0)
            .all()
        )

        expected_correct = (
            predictions["label"].astype(str)
            == predictions["predicted_label"].astype(str)
        )

        self.assertTrue(
            (
                predictions["is_correct"]
                == expected_correct
            ).all()
        )

    def test_loso_holds_out_each_participant_once(
        self,
    ) -> None:
        frame = self._audio_frame()

        results = evaluate_loso(
            frame=frame,
            modality=MODALITY_AUDIO,
            model_name=MODEL_DUMMY,
        )

        self.assertEqual(
            len(results),
            3,
        )

        self.assertEqual(
            {
                result.test_participant
                for result in results
            },
            {
                "P101",
                "P102",
                "P103",
            },
        )

        for result in results:
            self.assertNotIn(
                result.test_participant,
                result.train_participants,
            )

            self.assertEqual(
                len(
                    result.train_participants
                ),
                2,
            )

            self.assertEqual(
                result.train_rows,
                8,
            )

            self.assertEqual(
                result.test_rows,
                4,
            )

    def test_dummy_fold_metrics_are_correct(
        self,
    ) -> None:
        frame = self._audio_frame()

        results = evaluate_loso(
            frame=frame,
            modality=MODALITY_AUDIO,
            model_name=MODEL_DUMMY,
        )

        for result in results:
            self.assertEqual(
                result.test_drink_rows,
                1,
            )

            self.assertEqual(
                result.test_non_drink_rows,
                3,
            )

            self.assertEqual(
                result.true_negative,
                3,
            )

            self.assertEqual(
                result.false_positive,
                0,
            )

            self.assertEqual(
                result.false_negative,
                1,
            )

            self.assertEqual(
                result.true_positive,
                0,
            )

            self.assertAlmostEqual(
                result.precision_drink,
                0.0,
            )

            self.assertAlmostEqual(
                result.recall_drink,
                0.0,
            )

            self.assertAlmostEqual(
                result.f1_drink,
                0.0,
            )

            self.assertAlmostEqual(
                result.balanced_accuracy,
                0.5,
            )

            self.assertAlmostEqual(
                result.accuracy,
                0.75,
            )

    def test_fold_results_are_aggregated_per_participant(
        self,
    ) -> None:
        frame = self._audio_frame()

        results = evaluate_loso(
            frame=frame,
            modality=MODALITY_AUDIO,
            model_name=MODEL_DUMMY,
        )

        summary = summarize_fold_results(
            results
        )

        self.assertEqual(
            summary["fold_count"],
            3.0,
        )

        self.assertAlmostEqual(
            summary["f1_drink_mean"],
            0.0,
        )

        self.assertAlmostEqual(
            summary["precision_drink_mean"],
            0.0,
        )

        self.assertAlmostEqual(
            summary["recall_drink_mean"],
            0.0,
        )

        self.assertAlmostEqual(
            summary["balanced_accuracy_mean"],
            0.5,
        )

        self.assertAlmostEqual(
            summary["f1_drink_std"],
            0.0,
        )
    def test_feature_tables_require_identical_windows(
        self,
    ) -> None:
        audio = self._audio_frame()
        watch = self._fusion_frame().drop(
            columns=[
                column
                for column in self._fusion_frame().columns
                if column.startswith("audio_")
            ]
        )
        fusion = self._fusion_frame()

        validate_comparable_feature_tables(
            audio=audio,
            watch=watch,
            fusion=fusion,
        )

        broken_watch = watch.copy()

        broken_watch.loc[
            broken_watch.index[0],
            "window_id",
        ] = "WRONG_WINDOW"

        with self.assertRaises(
            ModelEvaluationError
        ):
            validate_comparable_feature_tables(
                audio=audio,
                watch=broken_watch,
                fusion=fusion,
            )
    def test_less_than_three_participants_is_rejected(
        self,
    ) -> None:
        frame = self._audio_frame()

        frame = frame[
            frame["participant_id"]
            != "P103"
        ].copy()

        with self.assertRaises(
            ModelEvaluationError
        ):
            evaluate_loso(
                frame=frame,
                modality=MODALITY_AUDIO,
                model_name=MODEL_DUMMY,
            )

    def _audio_frame(
        self,
    ) -> pd.DataFrame:
        rows = []

        for participant_index, participant in enumerate(
            (
                "P101",
                "P102",
                "P103",
            )
        ):
            labels = (
                "DRINK",
                "NON_DRINK",
                "NON_DRINK",
                "NON_DRINK",
            )

            for row_index, label in enumerate(
                labels
            ):
                row = {
                    "participant_id":
                        participant,
                    "session_id":
                        (
                            f"session_"
                            f"{participant}"
                        ),
                    "window_id":
                        (
                            f"{participant}"
                            f"_W{row_index:03d}"
                        ),
                    "label":
                        label,
                    "scenario":
                        (
                            "DRINK_FROM_GLASS"
                            if label == "DRINK"
                            else "REST"
                        ),
                    "container_type":
                        (
                            "GLASS"
                            if label == "DRINK"
                            else np.nan
                        ),
                }

                for feature_index in range(
                    38
                ):
                    row[
                        f"audio_feature_"
                        f"{feature_index:02d}"
                    ] = float(
                        participant_index
                        + row_index
                        + feature_index
                    )

                rows.append(
                    row
                )

        return pd.DataFrame(
            rows
        )

    def _fusion_frame(
        self,
    ) -> pd.DataFrame:
        frame = self._audio_frame()

        for feature_index in range(
            24
        ):
            frame[
                f"acc_feature_"
                f"{feature_index:02d}"
            ] = float(
                feature_index
            )

        for feature_index in range(
            24
        ):
            frame[
                f"gyro_feature_"
                f"{feature_index:02d}"
            ] = float(
                feature_index
            )

        return frame


if __name__ == "__main__":
    unittest.main()