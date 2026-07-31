from __future__ import annotations

import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from trinkerkennung_analysis.signal_loader import (
    SignalLoadingError,
    calculate_rms_envelope,
    downsample_waveform_min_max,
    load_session_signals,
)


class SignalLoaderTest(unittest.TestCase):

    def test_session_signals_are_loaded_and_prepared(
        self,
    ) -> None:
        session_id = (
            "11111111-2222-3333-4444-555555555555"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_session(
                root=root,
                session_id=session_id,
            )

            signals = load_session_signals(
                data_root=root,
                session_id=session_id,
            )

            self.assertEqual(
                signals.audio.samples.shape,
                (4,),
            )
            self.assertEqual(
                signals.audio.samples.dtype,
                np.float32,
            )
            self.assertAlmostEqual(
                float(signals.audio.samples[0]),
                -1.0,
            )
            self.assertAlmostEqual(
                float(signals.audio.samples[-1]),
                32767.0 / 32768.0,
            )
            np.testing.assert_allclose(
                signals.audio.time_seconds,
                np.array(
                    [0.0, 0.25, 0.5, 0.75]
                ),
            )

            self.assertEqual(
                len(signals.accelerometer),
                2,
            )
            self.assertEqual(
                len(signals.gyroscope),
                2,
            )
            self.assertListEqual(
                signals.accelerometer[
                    "relative_time_ns"
                ].tolist(),
                [0, 20_000_000],
            )
            self.assertAlmostEqual(
                float(
                    signals.accelerometer[
                        "magnitude"
                    ].iloc[0]
                ),
                5.0,
            )
            self.assertAlmostEqual(
                float(
                    signals.gyroscope[
                        "magnitude"
                    ].iloc[0]
                ),
                3.0,
            )

    def test_rms_envelope_has_expected_shape(
        self,
    ) -> None:
        samples = np.ones(
            100,
            dtype=np.float32,
        )

        times, rms = calculate_rms_envelope(
            samples=samples,
            sample_rate_hz=100,
            window_ms=200,
            hop_ms=100,
        )

        self.assertEqual(
            times.shape,
            rms.shape,
        )
        self.assertEqual(
            len(rms),
            9,
        )
        np.testing.assert_allclose(
            rms,
            np.ones(9),
        )

    def test_waveform_downsampling_is_finite(
        self,
    ) -> None:
        time_seconds = np.arange(
            1_001,
            dtype=np.float64,
        )
        samples = np.sin(time_seconds)

        reduced_time, reduced_samples = (
            downsample_waveform_min_max(
                time_seconds=time_seconds,
                samples=samples,
                maximum_points=100,
            )
        )

        self.assertLessEqual(
            reduced_time.size,
            100,
        )
        self.assertEqual(
            reduced_time.shape,
            reduced_samples.shape,
        )
        self.assertTrue(
            np.isfinite(reduced_time).all()
        )
        self.assertTrue(
            np.isfinite(reduced_samples).all()
        )

    def test_missing_sensor_is_rejected(
        self,
    ) -> None:
        session_id = (
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_session(
                root=root,
                session_id=session_id,
                include_gyroscope=False,
            )

            with self.assertRaises(
                SignalLoadingError
            ):
                load_session_signals(
                    data_root=root,
                    session_id=session_id,
                )

    @staticmethod
    def _create_session(
        root: Path,
        session_id: str,
        include_gyroscope: bool = True,
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
            wav_file.setframerate(4)
            wav_file.writeframes(
                np.array(
                    [-32768, -16384, 0, 32767],
                    dtype="<i2",
                ).tobytes()
            )

        rows = [
            {
                "session_id": session_id,
                "sensor": "ACCELEROMETER",
                "relative_time_ns": 20_000_000,
                "x": 0.0,
                "y": 0.0,
                "z": 1.0,
            },
            {
                "session_id": session_id,
                "sensor": "ACCELEROMETER",
                "relative_time_ns": 0,
                "x": 3.0,
                "y": 4.0,
                "z": 0.0,
            },
        ]
        if include_gyroscope:
            rows.extend([
                {
                    "session_id": session_id,
                    "sensor": "GYROSCOPE",
                    "relative_time_ns": 10_000_000,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 3.0,
                },
                {
                    "session_id": session_id,
                    "sensor": "GYROSCOPE",
                    "relative_time_ns": 30_000_000,
                    "x": 1.0,
                    "y": 2.0,
                    "z": 2.0,
                },
            ])

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
                    "x",
                    "y",
                    "z",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        metadata = {
            "schema_version": 1,
            "session_id": session_id,
            "status": "COMPLETED",
            "phone_audio_file_name": wav_name,
            "watch_sensor_file_name": csv_name,
            "phone_audio_start_epoch_ms": 1000,
            "phone_audio_stop_epoch_ms": 2000,
            "watch_start_epoch_ms": 1000,
            "watch_stop_epoch_ms": 1030,
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
