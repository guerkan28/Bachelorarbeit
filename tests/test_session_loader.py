import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path

from trinkerkennung_analysis.session_loader import (
    SessionValidationError,
    load_and_validate_session,
)


class SessionLoaderTest(unittest.TestCase):

    def test_valid_session(self):
        session_id = (
            "11111111-2222-3333-4444-555555555555"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_session(
                root=root,
                session_id=session_id,
                minimum_relative_time_ns=0,
            )

            result = load_and_validate_session(
                root,
                session_id,
            )

            self.assertEqual(
                result["wav"]["sample_rate_hz"],
                48_000,
            )
            self.assertEqual(
                result["csv"]["row_count"],
                2,
            )
            self.assertEqual(
                result["warnings"],
                [],
            )

    def test_negative_relative_time_is_rejected(self):
        session_id = (
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_session(
                root=root,
                session_id=session_id,
                minimum_relative_time_ns=-1,
            )

            with self.assertRaises(
                SessionValidationError
            ):
                load_and_validate_session(
                    root,
                    session_id,
                )

    @staticmethod
    def _create_session(
        root: Path,
        session_id: str,
        minimum_relative_time_ns: int,
    ) -> None:
        wav_name = (
            f"smartphone_{session_id}_test.wav"
        )
        csv_name = (
            f"watch_{session_id}_test.csv"
        )

        with wave.open(
            str(root / wav_name),
            "wb",
        ) as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(48_000)
            wav_file.writeframes(
                b"\x00\x00" * 48_000
            )

        with (root / csv_name).open(
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
                ],
            )
            writer.writeheader()
            writer.writerows([
                {
                    "session_id": session_id,
                    "sensor": "ACCELEROMETER",
                    "relative_time_ns":
                        minimum_relative_time_ns,
                },
                {
                    "session_id": session_id,
                    "sensor": "GYROSCOPE",
                    "relative_time_ns": 10_000_000,
                },
            ])

        metadata = {
            "session_id": session_id,
            "status": "COMPLETED",
            "phone_audio_file_name": wav_name,
            "watch_sensor_file_name": csv_name,
            "phone_audio_start_epoch_ms": 1000,
            "phone_audio_stop_epoch_ms": 2000,
            "watch_start_epoch_ms": 1000,
            "watch_stop_epoch_ms": 1010,
            "phone_audio_success": True,
            "watch_recording_success": True,
        }
        (
            root /
            f"session_{session_id}.json"
        ).write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
