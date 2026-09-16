from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trinkerkennung_analysis.feature_extraction import (
    FeatureExtractionError,
    extract_audio_features,
    extract_multimodal_features,
    extract_watch_features,
)
from trinkerkennung_analysis.video_alignment import (
    AUDIO_TIME_REFERENCE,
)
from trinkerkennung_analysis.window_dataset import (
    WindowRecord,
)
from trinkerkennung_analysis.window_signals import (
    SynchronizedWindowSignals,
)


class FeatureExtractionTest(unittest.TestCase):

    def test_audio_feature_count_is_38(
        self,
    ) -> None:
        features = extract_audio_features(
            samples=self._sine_wave(
                amplitude=0.5
            ),
            sample_rate_hz=8_000,
        )

        self.assertEqual(
            len(features),
            38,
        )

    def test_audio_features_are_finite(
        self,
    ) -> None:
        features = extract_audio_features(
            samples=self._sine_wave(
                amplitude=0.5
            ),
            sample_rate_hz=8_000,
        )

        self.assertTrue(
            np.isfinite(
                list(
                    features.values()
                )
            ).all()
        )

    def test_audio_rms_increases_with_amplitude(
        self,
    ) -> None:
        low = extract_audio_features(
            samples=self._sine_wave(
                amplitude=0.1
            ),
            sample_rate_hz=8_000,
        )

        high = extract_audio_features(
            samples=self._sine_wave(
                amplitude=0.8
            ),
            sample_rate_hz=8_000,
        )

        self.assertLess(
            low["audio_rms_mean"],
            high["audio_rms_mean"],
        )

    def test_invalid_audio_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            FeatureExtractionError
        ):
            extract_audio_features(
                samples=np.array(
                    [0.0, np.nan]
                ),
                sample_rate_hz=8_000,
            )

    def test_watch_feature_count_is_48(
        self,
    ) -> None:
        features = extract_watch_features(
            accelerometer=self._sensor_frame(),
            gyroscope=self._sensor_frame(),
        )

        self.assertEqual(
            len(features),
            48,
        )

    def test_known_watch_statistics(
        self,
    ) -> None:
        frame = self._sensor_frame()

        features = extract_watch_features(
            accelerometer=frame,
            gyroscope=frame,
        )

        self.assertAlmostEqual(
            features["acc_x_mean"],
            2.0,
        )

        self.assertAlmostEqual(
            features["acc_x_median"],
            2.0,
        )

        self.assertAlmostEqual(
            features["acc_x_min"],
            1.0,
        )

        self.assertAlmostEqual(
            features["acc_x_max"],
            3.0,
        )

        self.assertAlmostEqual(
            features["acc_x_std"],
            float(
                np.std(
                    [1.0, 2.0, 3.0]
                )
            ),
        )

        self.assertAlmostEqual(
            features["acc_x_rms"],
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            [1.0, 2.0, 3.0]
                        )
                    )
                )
            ),
        )

    def test_missing_watch_column_is_rejected(
        self,
    ) -> None:
        frame = self._sensor_frame().drop(
            columns=["magnitude"]
        )

        with self.assertRaises(
            FeatureExtractionError
        ):
            extract_watch_features(
                accelerometer=frame,
                gyroscope=self._sensor_frame(),
            )

    def test_nonfinite_watch_value_is_rejected(
        self,
    ) -> None:
        frame = self._sensor_frame()
        frame.loc[0, "x"] = np.nan

        with self.assertRaises(
            FeatureExtractionError
        ):
            extract_watch_features(
                accelerometer=frame,
                gyroscope=self._sensor_frame(),
            )

    def test_multimodal_feature_count_is_86(
        self,
    ) -> None:
        features = extract_multimodal_features(
            self._window_signals()
        )

        self.assertEqual(
            len(features),
            86,
        )

    def test_multimodal_features_are_finite(
        self,
    ) -> None:
        features = extract_multimodal_features(
            self._window_signals()
        )

        self.assertTrue(
            np.isfinite(
                list(
                    features.values()
                )
            ).all()
        )

    @staticmethod
    def _sine_wave(
        *,
        amplitude: float,
        sample_rate: int = 8_000,
        duration_seconds: float = 1.0,
        frequency_hz: float = 440.0,
    ) -> np.ndarray:
        time = (
            np.arange(
                int(
                    sample_rate
                    * duration_seconds
                ),
                dtype=np.float64,
            )
            / sample_rate
        )

        return (
            amplitude
            * np.sin(
                2.0
                * np.pi
                * frequency_hz
                * time
            )
        ).astype(
            np.float32
        )

    @staticmethod
    def _sensor_frame() -> pd.DataFrame:
        x = np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float64,
        )

        y = np.array(
            [2.0, 3.0, 4.0],
            dtype=np.float64,
        )

        z = np.array(
            [3.0, 4.0, 5.0],
            dtype=np.float64,
        )

        magnitude = np.sqrt(
            np.square(x)
            + np.square(y)
            + np.square(z)
        )

        return pd.DataFrame({
            "x": x,
            "y": y,
            "z": z,
            "magnitude": magnitude,
        })

    def _window_signals(
        self,
    ) -> SynchronizedWindowSignals:
        session_id = (
            "11111111-2222-4333-8444-555555555555"
        )

        window = WindowRecord(
            schema_version=1,
            participant_id="P001",
            session_id=session_id,
            session_mode=(
                "CONTROLLED_SINGLE_ACTIVITY"
            ),
            time_reference=AUDIO_TIME_REFERENCE,
            window_id="test-window",
            label="DRINK",
            source_category="DRINK_EVENT",
            source_id="D001",
            scenario="DRINK_FROM_GLASS",
            container_type="GLASS",
            source_start_seconds=0.0,
            source_end_seconds=1.0,
            window_index=1,
            window_start_seconds=0.0,
            window_end_seconds=1.0,
            window_length_seconds=1.0,
            stride_seconds=1.0,
        )

        samples = self._sine_wave(
            amplitude=0.5
        )

        return SynchronizedWindowSignals(
            window=window,
            audio_samples=samples,
            audio_time_seconds=(
                np.arange(
                    samples.size,
                    dtype=np.float64,
                )
                / 8_000.0
            ),
            audio_sample_rate_hz=8_000,
            accelerometer=self._sensor_frame(),
            gyroscope=self._sensor_frame(),
        )


if __name__ == "__main__":
    unittest.main()