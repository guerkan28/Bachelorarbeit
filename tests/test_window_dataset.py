from __future__ import annotations

import copy
import unittest

from trinkerkennung_analysis.video_alignment import (
    AUDIO_TIME_REFERENCE,
)
from trinkerkennung_analysis.window_dataset import (
    DRINK_LABEL,
    NON_DRINK_LABEL,
    WindowDatasetError,
    generate_windows_from_validated_annotation,
)


class WindowDatasetTest(unittest.TestCase):
    SESSION_ID = "11111111-2222-4333-8444-555555555555"

    def setUp(self) -> None:
        self.annotation = {
            "session_id": self.SESSION_ID,
            "participant_id": "P001",
            "session_mode": "CONTROLLED_MIXED_ACTIVITY",
            "time_reference": AUDIO_TIME_REFERENCE,
            "drink_events": [
                {
                    "event_id": "D001",
                    "scenario": "DRINK_FROM_GLASS",
                    "container_type": "GLASS",
                    "event_start_seconds": 10.0,
                    "event_end_seconds": 14.0,
                    "mouth_contact_intervals": [
                        {
                            "start_seconds": 11.0,
                            "end_seconds": 12.0,
                        }
                    ],
                }
            ],
            "negative_intervals": [
                {
                    "interval_id": "N001",
                    "scenario": (
                        "LIFT_CONTAINER_WITHOUT_DRINKING"
                    ),
                    "container_type": "BOTTLE",
                    "start_seconds": 20.0,
                    "end_seconds": 23.0,
                }
            ],
            "excluded_intervals": [
                {
                    "interval_id": "X001",
                    "reason": "SYNCHRONIZATION_MARKER",
                    "start_seconds": 30.0,
                    "end_seconds": 34.0,
                }
            ],
            "uncertain_intervals": [
                {
                    "interval_id": "U001",
                    "reason": "EVENT_BOUNDARY_NOT_VISIBLE",
                    "start_seconds": 40.0,
                    "end_seconds": 44.0,
                }
            ],
        }

    def test_full_drink_event_is_used_not_only_mouth_contact(
        self,
    ) -> None:
        self.annotation["negative_intervals"] = []

        windows = self._generate(
            window_length=2.0,
            stride=2.0,
        )

        self.assertEqual(len(windows), 2)
        self.assertEqual(
            [window.window_start_seconds for window in windows],
            [10.0, 12.0],
        )
        self.assertTrue(
            all(window.label == DRINK_LABEL for window in windows)
        )

    def test_negative_interval_generates_non_drink_window(
        self,
    ) -> None:
        self.annotation["drink_events"] = []

        windows = self._generate(
            window_length=2.0,
            stride=2.0,
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].label, NON_DRINK_LABEL)
        self.assertEqual(windows[0].window_start_seconds, 20.0)
        self.assertEqual(windows[0].window_end_seconds, 22.0)

    def test_partial_tail_is_not_windowed(self) -> None:
        self.annotation["drink_events"] = []

        windows = self._generate(
            window_length=2.0,
            stride=2.0,
        )

        self.assertEqual(len(windows), 1)
        self.assertNotIn(
            22.0,
            [window.window_start_seconds for window in windows],
        )

    def test_unlabeled_gap_is_not_generated_as_non_drink(
        self,
    ) -> None:
        windows = self._generate(
            window_length=2.0,
            stride=2.0,
        )

        for window in windows:
            self.assertFalse(
                14.0 <= window.window_start_seconds < 20.0
            )

    def test_excluded_and_uncertain_intervals_are_ignored(
        self,
    ) -> None:
        windows = self._generate(
            window_length=2.0,
            stride=2.0,
        )

        starts = {
            window.window_start_seconds
            for window in windows
        }

        self.assertFalse(
            any(30.0 <= start < 34.0 for start in starts)
        )
        self.assertFalse(
            any(40.0 <= start < 44.0 for start in starts)
        )

    def test_smaller_stride_can_create_overlapping_windows(
        self,
    ) -> None:
        self.annotation["negative_intervals"] = []

        windows = self._generate(
            window_length=2.0,
            stride=1.0,
        )

        self.assertEqual(
            [window.window_start_seconds for window in windows],
            [10.0, 11.0, 12.0],
        )

    def test_window_ids_are_stable_and_unique(self) -> None:
        windows = self._generate(
            window_length=2.0,
            stride=2.0,
        )

        ids = [window.window_id for window in windows]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            ids[0],
            f"{self.SESSION_ID}__D001__W001",
        )

    def test_negative_container_type_is_preserved(self) -> None:
        self.annotation["drink_events"] = []

        windows = self._generate(
            window_length=2.0,
            stride=2.0,
        )

        self.assertEqual(windows[0].container_type, "BOTTLE")

    def test_non_positive_window_length_is_rejected(self) -> None:
        with self.assertRaises(WindowDatasetError):
            self._generate(
                window_length=0.0,
                stride=1.0,
            )

    def test_non_positive_stride_is_rejected(self) -> None:
        with self.assertRaises(WindowDatasetError):
            self._generate(
                window_length=2.0,
                stride=0.0,
            )

    def test_wrong_time_reference_is_rejected(self) -> None:
        annotation = copy.deepcopy(self.annotation)
        annotation["time_reference"] = "VIDEO_SECONDS"

        with self.assertRaises(WindowDatasetError):
            generate_windows_from_validated_annotation(
                annotation,
                window_length_seconds=2.0,
                stride_seconds=2.0,
            )

    def _generate(
        self,
        *,
        window_length: float,
        stride: float,
    ):
        return generate_windows_from_validated_annotation(
            self.annotation,
            window_length_seconds=window_length,
            stride_seconds=stride,
        )


if __name__ == "__main__":
    unittest.main()