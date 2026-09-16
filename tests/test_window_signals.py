from __future__ import annotations

import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from trinkerkennung_analysis.video_alignment import (
    AUDIO_TIME_REFERENCE,
)
from trinkerkennung_analysis.window_dataset import (
    WindowRecord,
)
from trinkerkennung_analysis.window_signals import (
    WindowSignalError,
    extract_synchronized_window,
    load_aligned_session_signals,
    load_watch_to_audio_mapping,
)


class WindowSignalsTest(unittest.TestCase):
    SESSION_ID = "11111111-2222-4333-8444-555555555555"

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self._create_session_files()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_constant_watch_offset_is_loaded(self) -> None:
        mapping = load_watch_to_audio_mapping(
            data_root=self.root,
            session_id=self.SESSION_ID,
        )

        self.assertEqual(
            mapping.method,
            "CONSTANT_OFFSET",
        )
        self.assertAlmostEqual(
            mapping.scale,
            1.0,
        )
        self.assertAlmostEqual(
            mapping.offset_seconds,
            0.2,
        )

    def test_watch_times_are_mapped_to_audio_reference(
        self,
    ) -> None:
        aligned = load_aligned_session_signals(
            data_root=self.root,
            session_id=self.SESSION_ID,
        )

        np.testing.assert_allclose(
            aligned.accelerometer[
                "audio_time_seconds"
            ].to_numpy(),
            np.array(
                [1.0, 1.5, 2.0, 2.5]
            ),
        )

    def test_audio_and_watch_use_same_half_open_window(
        self,
    ) -> None:
        aligned = load_aligned_session_signals(
            data_root=self.root,
            session_id=self.SESSION_ID,
        )

        window = self._window(
            start=1.0,
            end=2.0,
        )

        result = extract_synchronized_window(
            aligned_session=aligned,
            window=window,
        )

        self.assertEqual(
            result.audio_samples.size,
            10,
        )

        np.testing.assert_allclose(
            result.audio_time_seconds,
            np.arange(
                1.0,
                2.0,
                0.1,
            ),
            atol=1e-12,
        )

        np.testing.assert_allclose(
            result.accelerometer[
                "audio_time_seconds"
            ].to_numpy(),
            np.array(
                [1.0, 1.5]
            ),
        )

        np.testing.assert_allclose(
            result.gyroscope[
                "audio_time_seconds"
            ].to_numpy(),
            np.array(
                [1.0, 1.5]
            ),
        )

    def test_end_boundary_is_excluded(
        self,
    ) -> None:
        aligned = load_aligned_session_signals(
            data_root=self.root,
            session_id=self.SESSION_ID,
        )

        result = extract_synchronized_window(
            aligned_session=aligned,
            window=self._window(
                start=1.0,
                end=2.0,
            ),
        )

        self.assertNotIn(
            2.0,
            result.accelerometer[
                "audio_time_seconds"
            ].tolist(),
        )

    def test_different_session_id_is_rejected(
        self,
    ) -> None:
        aligned = load_aligned_session_signals(
            data_root=self.root,
            session_id=self.SESSION_ID,
        )

        window = self._window(
            start=1.0,
            end=2.0,
            session_id=(
                "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
            ),
        )

        with self.assertRaises(WindowSignalError):
            extract_synchronized_window(
                aligned_session=aligned,
                window=window,
            )

    def test_window_outside_audio_is_rejected(
        self,
    ) -> None:
        aligned = load_aligned_session_signals(
            data_root=self.root,
            session_id=self.SESSION_ID,
        )

        with self.assertRaises(WindowSignalError):
            extract_synchronized_window(
                aligned_session=aligned,
                window=self._window(
                    start=3.5,
                    end=4.5,
                ),
            )

    def test_wrong_alignment_method_is_rejected(
        self,
    ) -> None:
        path = self._synchronization_report_path()

        report = json.loads(
            path.read_text(encoding="utf-8")
        )

        report["time_mappings"]["watch_to_audio"][
            "method"
        ] = "LINEAR"

        path.write_text(
            json.dumps(report),
            encoding="utf-8",
        )

        with self.assertRaises(WindowSignalError):
            load_watch_to_audio_mapping(
                data_root=self.root,
                session_id=self.SESSION_ID,
            )

    def test_non_unit_scale_is_rejected(
        self,
    ) -> None:
        path = self._synchronization_report_path()

        report = json.loads(
            path.read_text(encoding="utf-8")
        )

        report["time_mappings"]["watch_to_audio"][
            "scale"
        ] = 1.01

        path.write_text(
            json.dumps(report),
            encoding="utf-8",
        )

        with self.assertRaises(WindowSignalError):
            load_watch_to_audio_mapping(
                data_root=self.root,
                session_id=self.SESSION_ID,
            )

    def _window(
        self,
        *,
        start: float,
        end: float,
        session_id: str | None = None,
    ) -> WindowRecord:
        return WindowRecord(
            schema_version=1,
            participant_id="P001",
            session_id=(
                session_id
                if session_id is not None
                else self.SESSION_ID
            ),
            session_mode="CONTROLLED_SINGLE_ACTIVITY",
            time_reference=AUDIO_TIME_REFERENCE,
            window_id="test-window",
            label="DRINK",
            source_category="DRINK_EVENT",
            source_id="D001",
            scenario="DRINK_FROM_GLASS",
            container_type="GLASS",
            source_start_seconds=start,
            source_end_seconds=end,
            window_index=1,
            window_start_seconds=start,
            window_end_seconds=end,
            window_length_seconds=end - start,
            stride_seconds=end - start,
        )

    def _synchronization_report_path(self) -> Path:
        return (
            self.root
            / f"synchronization_report_{self.SESSION_ID}.json"
        )

    def _create_session_files(self) -> None:
        wav_name = (
            f"smartphone_{self.SESSION_ID}_test.wav"
        )
        csv_name = (
            f"watch_{self.SESSION_ID}_test.csv"
        )

        with wave.open(
            str(self.root / wav_name),
            "wb",
        ) as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(10)
            wav_file.writeframes(
                np.arange(
                    40,
                    dtype="<i2",
                ).tobytes()
            )

        sensor_times_ns = [
            800_000_000,
            1_300_000_000,
            1_800_000_000,
            2_300_000_000,
        ]

        rows = []

        for sensor in (
            "ACCELEROMETER",
            "GYROSCOPE",
        ):
            for index, time_ns in enumerate(
                sensor_times_ns
            ):
                rows.append({
                    "session_id": self.SESSION_ID,
                    "sensor": sensor,
                    "relative_time_ns": time_ns,
                    "x": float(index + 1),
                    "y": float(index + 2),
                    "z": float(index + 3),
                })

        with (self.root / csv_name).open(
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
            writer.writerows(rows)

        metadata = {
            "schema_version": 1,
            "session_id": self.SESSION_ID,
            "status": "COMPLETED",
            "phone_audio_file_name": wav_name,
            "watch_sensor_file_name": csv_name,
            "phone_audio_start_epoch_ms": 1000,
            "phone_audio_stop_epoch_ms": 5000,
            "watch_start_epoch_ms": 1000,
            "watch_stop_epoch_ms": 2500,
            "phone_audio_success": True,
            "watch_recording_success": True,
        }

        (
            self.root
            / f"session_{self.SESSION_ID}.json"
        ).write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        report = {
            "schema_version": 2,
            "session_id": self.SESSION_ID,
            "time_mappings": {
                "watch_to_audio": {
                    "source_time_reference": (
                        "WATCH_SECONDS_FROM_SENSOR_SESSION_START"
                    ),
                    "target_time_reference": (
                        AUDIO_TIME_REFERENCE
                    ),
                    "method": "CONSTANT_OFFSET",
                    "scale": 1.0,
                    "offset_seconds": 0.2,
                }
            },
        }

        self._synchronization_report_path().write_text(
            json.dumps(report),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()