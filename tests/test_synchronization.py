from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trinkerkennung_analysis.synchronization import (
    MarkerGroup,
    SynchronizationError,
    apply_watch_offset,
    calculate_acceleration_marker_score,
    detect_confirmed_peak_group,
    detect_regular_peak_group,
    estimate_constant_offset,
    marker_exclusion_interval,
    resolve_marker_windows,
)


class SynchronizationTest(unittest.TestCase):

    def test_regular_peak_group_is_detected(
        self,
    ) -> None:
        time_seconds = np.arange(
            0.0,
            10.0,
            0.01,
        )
        values = np.zeros_like(
            time_seconds
        )
        for peak_time in (
            3.0,
            3.6,
            4.2,
        ):
            values += np.exp(
                -0.5
                * (
                    (
                        time_seconds
                        - peak_time
                    )
                    / 0.025
                )
                ** 2
            )

        marker = detect_regular_peak_group(
            time_seconds=time_seconds,
            values=values,
            window_seconds=(2.0, 5.0),
        )

        np.testing.assert_allclose(
            marker.peak_times_seconds,
            np.array(
                [3.0, 3.6, 4.2]
            ),
            atol=0.011,
        )

    def test_confirmed_peak_group_accepts_irregular_intervals(
        self,
    ) -> None:
        time_seconds = np.arange(
            0.0,
            10.0,
            0.01,
        )
        values = np.zeros_like(
            time_seconds
        )

        for peak_time in (
            3.0,
            4.0,
            5.4,
        ):
            values += np.exp(
                -0.5
                * (
                    (
                        time_seconds
                        - peak_time
                    )
                    / 0.025
                )
                ** 2
            )

        # Der reguläre Detektor muss wegen
        # des Abstands von 1,4 s weiterhin scheitern.
        with self.assertRaises(
            SynchronizationError
        ):
            detect_regular_peak_group(
                time_seconds=time_seconds,
                values=values,
                window_seconds=(2.5, 6.0),
            )

        marker = detect_confirmed_peak_group(
            time_seconds=time_seconds,
            values=values,
            peak_windows_seconds=(
                (2.85, 3.15),
                (3.85, 4.15),
                (5.25, 5.55),
            ),
        )

        np.testing.assert_allclose(
            marker.peak_times_seconds,
            np.array(
                [3.0, 4.0, 5.4]
            ),
            atol=0.011,
        )

    def test_constant_offset_is_estimated(
        self,
    ) -> None:
        start_audio = self._marker(
            [5.0, 5.6, 6.2]
        )
        start_watch = self._marker(
            [4.8, 5.4, 6.0]
        )
        end_audio = self._marker(
            [50.0, 50.6, 51.2]
        )
        end_watch = self._marker(
            [49.8, 50.4, 51.0]
        )
        watch_time = np.arange(
            0.0,
            60.0,
            0.02,
        )

        result = estimate_constant_offset(
            session_id="test-session",
            start_audio_marker=
                start_audio,
            start_watch_marker=
                start_watch,
            end_audio_marker=end_audio,
            end_watch_marker=end_watch,
            watch_time_seconds=
                watch_time,
        )

        self.assertAlmostEqual(
            result.start_offset_seconds,
            0.2,
        )
        self.assertAlmostEqual(
            result.end_offset_seconds,
            0.2,
        )
        self.assertAlmostEqual(
            result.selected_offset_seconds,
            0.2,
        )
        self.assertTrue(
            result.offset_consistent
        )
        self.assertEqual(
            result.selected_alignment_method,
            "CONSTANT_OFFSET",
        )
        self.assertEqual(
            result.warnings,
            (),
        )

    def test_offset_change_is_reported(
        self,
    ) -> None:
        start_audio = self._marker(
            [5.0, 5.6, 6.2]
        )
        start_watch = self._marker(
            [4.8, 5.4, 6.0]
        )
        end_audio = self._marker(
            [50.0, 50.6, 51.2]
        )
        end_watch = self._marker(
            [49.7, 50.3, 50.9]
        )

        result = estimate_constant_offset(
            session_id="test-session",
            start_audio_marker=
                start_audio,
            start_watch_marker=
                start_watch,
            end_audio_marker=end_audio,
            end_watch_marker=end_watch,
            watch_time_seconds=np.arange(
                0.0,
                60.0,
                0.02,
            ),
        )

        self.assertAlmostEqual(
            result.start_offset_seconds,
            0.2,
        )
        self.assertAlmostEqual(
            result.end_offset_seconds,
            0.3,
        )
        self.assertFalse(
            result.offset_consistent
        )
        self.assertEqual(
            result.selected_alignment_method,
            "CONSTANT_OFFSET_WITH_DRIFT_WARNING",
        )
        self.assertGreater(
            len(result.warnings),
            0,
        )

    def test_acceleration_marker_score_removes_static_baseline(
        self,
    ) -> None:
        time_seconds = np.arange(
            0.0,
            5.0,
            0.02,
        )
        magnitude = np.full(
            time_seconds.shape,
            9.81,
        )
        frame = pd.DataFrame({
            "time_seconds":
                time_seconds,
            "magnitude":
                magnitude,
        })

        score = (
            calculate_acceleration_marker_score(
                frame
            )
        )

        np.testing.assert_allclose(
            score,
            np.zeros_like(score),
        )

    def test_watch_offset_is_applied(
        self,
    ) -> None:
        aligned = apply_watch_offset(
            np.array(
                [0.0, 1.0, 2.0]
            ),
            0.2,
        )

        np.testing.assert_allclose(
            aligned,
            np.array(
                [0.2, 1.2, 2.2]
            ),
        )

    def test_default_marker_windows_do_not_overlap(
        self,
    ) -> None:
        start, end = (
            resolve_marker_windows(
                common_duration_seconds=60.0,
                start_window_seconds=None,
                end_window_seconds=None,
            )
        )

        self.assertLess(
            start[1],
            end[0],
        )
        self.assertEqual(
            start,
            (2.0, 15.0),
        )
        self.assertEqual(
            end,
            (45.0, 58.0),
        )

    def test_marker_exclusion_interval_has_margin(
        self,
    ) -> None:
        marker = self._marker(
            [5.0, 5.6, 6.2]
        )

        interval = marker_exclusion_interval(
            marker,
            margin_seconds=0.5,
        )

        self.assertEqual(
            interval,
            (4.5, 6.7),
        )

    @staticmethod
    def _marker(
        peak_times: list[float],
    ) -> MarkerGroup:
        values = np.array(
            [1.0] * len(peak_times),
            dtype=np.float64,
        )
        return MarkerGroup(
            peak_times_seconds=np.array(
                peak_times,
                dtype=np.float64,
            ),
            peak_values=values,
            window_start_seconds=0.0,
            window_end_seconds=60.0,
            detection_threshold=0.5,
            interval_coefficient_of_variation=0.0,
            detection_score=3.0,
        )


if __name__ == "__main__":
    unittest.main()
