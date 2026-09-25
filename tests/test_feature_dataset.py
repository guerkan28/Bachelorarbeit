from __future__ import annotations

import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from trinkerkennung_analysis.feature_dataset import (
    FeatureDatasetError,
    build_feature_datasets_from_window_csv,
    write_feature_datasets,
)
from trinkerkennung_analysis.video_alignment import (
    AUDIO_TIME_REFERENCE,
)
from trinkerkennung_analysis.window_dataset import (
    WINDOW_DATASET_FIELDNAMES,
)


class FeatureDatasetTest(unittest.TestCase):
    SESSION_ID = (
        "11111111-2222-4333-8444-555555555555"
    )

    SECOND_SESSION_ID = (
        "66666666-7777-4888-8999-000000000000"
    )

    def setUp(self) -> None:
        self.temporary_directory = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.temporary_directory.name
        )

        self._create_session_files()

        self.window_csv = (
            self.root
            / "windows.csv"
        )

        self._write_window_csv(
            self._valid_window_rows()
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_feature_tables_have_expected_shape(
        self,
    ) -> None:
        tables = self._build()

        self.assertEqual(
            len(tables.audio),
            2,
        )

        self.assertEqual(
            len([
                column
                for column in tables.audio.columns
                if column.startswith("audio_")
            ]),
            38,
        )

        self.assertEqual(
            len([
                column
                for column in tables.watch.columns
                if column.startswith(
                    (
                        "acc_",
                        "gyro_",
                    )
                )
            ]),
            48,
        )

        self.assertEqual(
            len([
                column
                for column in tables.fusion.columns
                if column.startswith(
                    (
                        "audio_",
                        "acc_",
                        "gyro_",
                    )
                )
            ]),
            86,
        )

    def test_metadata_and_labels_are_preserved(
        self,
    ) -> None:
        tables = self._build()

        self.assertEqual(
            tables.fusion[
                "participant_id"
            ].tolist(),
            [
                "P001",
                "P001",
            ],
        )

        self.assertEqual(
            tables.fusion[
                "label"
            ].tolist(),
            [
                "DRINK",
                "NON_DRINK",
            ],
        )

        self.assertEqual(
            tables.fusion[
                "scenario"
            ].tolist(),
            [
                "DRINK_FROM_GLASS",
                (
                    "MOVE_CONTAINER_WITHOUT_DRINKING"
                ),
            ],
        )

    def test_duplicate_window_id_is_rejected(
        self,
    ) -> None:
        rows = self._valid_window_rows()

        rows[1]["window_id"] = (
            rows[0]["window_id"]
        )

        self._write_window_csv(
            rows
        )

        with self.assertRaises(
            FeatureDatasetError
        ):
            self._build()

    def test_missing_window_column_is_rejected(
        self,
    ) -> None:
        frame = pd.read_csv(
            self.window_csv,
            sep=";",
        )

        frame = frame.drop(
            columns=["label"]
        )

        frame.to_csv(
            self.window_csv,
            sep=";",
            index=False,
        )

        with self.assertRaises(
            FeatureDatasetError
        ):
            self._build()

    def test_feature_tables_are_written(
        self,
    ) -> None:
        tables = self._build()

        output_root = (
            self.root
            / "features"
        )

        paths = write_feature_datasets(
            tables=tables,
            output_root=output_root,
            file_prefix="pilot",
            configuration_name="1s",
        )

        self.assertEqual(
            set(
                paths
            ),
            {
                "audio",
                "watch",
                "fusion",
            },
        )

        for path in paths.values():
            self.assertTrue(
                path.is_file()
            )

    def test_participants_can_use_separate_data_roots(
        self,
    ) -> None:
        second_root = self.root / "second_study"
        second_root.mkdir()

        rows = self._valid_window_rows()

        rows[1]["participant_id"] = "P002"
        rows[1]["session_id"] = self.SECOND_SESSION_ID
        rows[1]["window_id"] = (
            f"{self.SECOND_SESSION_ID}"
            "__N001__W001"
        )

        self._write_window_csv(rows)

        synchronized = SimpleNamespace(
            audio_samples=np.array(
                [0.0],
                dtype=np.float64,
            ),
            audio_sample_rate_hz=8000,
            accelerometer=pd.DataFrame(),
            gyroscope=pd.DataFrame(),
        )

        audio_features = {
            f"audio_test_{index:02d}": float(index)
            for index in range(38)
        }

        watch_features = {
            **{
                f"acc_test_{index:02d}": float(index)
                for index in range(24)
            },
            **{
                f"gyro_test_{index:02d}": float(index)
                for index in range(24)
            },
        }

        with (
            patch(
                "trinkerkennung_analysis.feature_dataset."
                "load_aligned_session_signals"
            ) as load_mock,
            patch(
                "trinkerkennung_analysis.feature_dataset."
                "extract_synchronized_window",
                return_value=synchronized,
            ),
            patch(
                "trinkerkennung_analysis.feature_dataset."
                "extract_audio_features",
                return_value=audio_features,
            ),
            patch(
                "trinkerkennung_analysis.feature_dataset."
                "extract_watch_features",
                return_value=watch_features,
            ),
        ):
            load_mock.return_value = object()

            tables = build_feature_datasets_from_window_csv(
                data_root=self.root,
                participant_data_roots={
                    "P002": second_root,
                },
                window_csv=self.window_csv,
            )

        self.assertEqual(
            len(tables.fusion),
            2,
        )

        load_mock.assert_any_call(
            data_root=self.root.resolve(),
            session_id=self.SESSION_ID,
        )

        load_mock.assert_any_call(
            data_root=second_root.resolve(),
            session_id=self.SECOND_SESSION_ID,
        )

        self.assertEqual(
            load_mock.call_count,
            2,
        )

    def _build(self):
        return (
            build_feature_datasets_from_window_csv(
                data_root=self.root,
                window_csv=self.window_csv,
            )
        )

    def _valid_window_rows(
        self,
    ) -> list[dict]:
        common = {
            "schema_version": 1,
            "participant_id": "P001",
            "session_id": self.SESSION_ID,
            "session_mode":
                "CONTROLLED_MIXED_ACTIVITY",
            "time_reference":
                AUDIO_TIME_REFERENCE,
            "container_type": "GLASS",
            "window_length_seconds": 1.0,
            "stride_seconds": 1.0,
        }

        return [
            {
                **common,
                "window_id": (
                    f"{self.SESSION_ID}"
                    "__D001__W001"
                ),
                "label": "DRINK",
                "source_category":
                    "DRINK_EVENT",
                "source_id": "D001",
                "scenario":
                    "DRINK_FROM_GLASS",
                "source_start_seconds": 1.0,
                "source_end_seconds": 2.0,
                "window_index": 1,
                "window_start_seconds": 1.0,
                "window_end_seconds": 2.0,
            },
            {
                **common,
                "window_id": (
                    f"{self.SESSION_ID}"
                    "__N001__W001"
                ),
                "label": "NON_DRINK",
                "source_category":
                    "NEGATIVE_INTERVAL",
                "source_id": "N001",
                "scenario": (
                    "MOVE_CONTAINER_WITHOUT_DRINKING"
                ),
                "source_start_seconds": 2.0,
                "source_end_seconds": 3.0,
                "window_index": 1,
                "window_start_seconds": 2.0,
                "window_end_seconds": 3.0,
            },
        ]

    def _write_window_csv(
        self,
        rows: list[dict],
    ) -> None:
        with self.window_csv.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    WINDOW_DATASET_FIELDNAMES
                ),
                delimiter=";",
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

    def _create_session_files(
        self,
    ) -> None:
        wav_name = (
            f"smartphone_{self.SESSION_ID}_test.wav"
        )

        csv_name = (
            f"watch_{self.SESSION_ID}_test.csv"
        )

        sample_rate = 8_000

        time = (
            np.arange(
                sample_rate * 4,
                dtype=np.float64,
            )
            / sample_rate
        )

        samples = (
            0.25
            * np.sin(
                2.0
                * np.pi
                * 440.0
                * time
            )
        )

        integer_samples = np.clip(
            samples
            * 32767.0,
            -32768,
            32767,
        ).astype(
            "<i2"
        )

        with wave.open(
            str(
                self.root
                / wav_name
            ),
            "wb",
        ) as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(
                sample_rate
            )
            wav_file.writeframes(
                integer_samples.tobytes()
            )

        rows = []

        watch_times = np.arange(
            0.0,
            3.8,
            0.05,
        )

        for sensor in (
            "ACCELEROMETER",
            "GYROSCOPE",
        ):
            for index, time_seconds in enumerate(
                watch_times
            ):
                rows.append({
                    "session_id":
                        self.SESSION_ID,
                    "sensor":
                        sensor,
                    "relative_time_ns":
                        int(
                            round(
                                time_seconds
                                * 1_000_000_000
                            )
                        ),
                    "x":
                        float(
                            np.sin(
                                index / 5.0
                            )
                        ),
                    "y":
                        float(
                            np.cos(
                                index / 7.0
                            )
                        ),
                    "z":
                        float(
                            np.sin(
                                index / 9.0
                            )
                        ),
                })

        with (
            self.root
            / csv_name
        ).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "session_id",
                    "sensor",
                    "relative_time_ns",
                    "x",
                    "y",
                    "z",
                ],
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        metadata = {
            "schema_version": 1,
            "session_id": self.SESSION_ID,
            "status": "COMPLETED",
            "phone_audio_file_name":
                wav_name,
            "watch_sensor_file_name":
                csv_name,
            "phone_audio_start_epoch_ms":
                1000,
            "phone_audio_stop_epoch_ms":
                5000,
            "watch_start_epoch_ms":
                1000,
            "watch_stop_epoch_ms":
                4750,
            "phone_audio_success": True,
            "watch_recording_success": True,
        }

        (
            self.root
            / (
                f"session_"
                f"{self.SESSION_ID}.json"
            )
        ).write_text(
            json.dumps(
                metadata
            ),
            encoding="utf-8",
        )

        report = {
            "schema_version": 2,
            "session_id":
                self.SESSION_ID,
            "time_mappings": {
                "watch_to_audio": {
                    "source_time_reference": (
                        "WATCH_SECONDS_FROM_SENSOR_SESSION_START"
                    ),
                    "target_time_reference":
                        AUDIO_TIME_REFERENCE,
                    "method":
                        "CONSTANT_OFFSET",
                    "scale": 1.0,
                    "offset_seconds": 0.2,
                }
            },
        }

        (
            self.root
            / (
                f"synchronization_report_"
                f"{self.SESSION_ID}.json"
            )
        ).write_text(
            json.dumps(
                report
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()