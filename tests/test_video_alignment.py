from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trinkerkennung_analysis.video_alignment import (
    VideoAlignmentError,
    add_video_alignment_to_report,
    apply_linear_mapping,
    build_watch_to_audio_mapping,
    estimate_video_to_audio_mapping,
)


class VideoAlignmentTest(unittest.TestCase):

    def test_known_linear_mapping_is_estimated(self) -> None:
        mapping = estimate_video_to_audio_mapping(
            audio_start_marker_times_seconds=[5.0, 5.6, 6.2],
            video_start_marker_times_seconds=[4.0, 4.5, 5.0],
            audio_end_marker_times_seconds=[50.0, 50.6, 51.2],
            video_end_marker_times_seconds=[41.5, 42.0, 42.5],
        )
        self.assertAlmostEqual(mapping["scale"], 1.2)
        self.assertAlmostEqual(mapping["offset_seconds"], 0.2)
        self.assertAlmostEqual(
            apply_linear_mapping(42.0, mapping),
            50.6,
        )

    def test_existing_report_is_augmented_without_losing_fields(self) -> None:
        session_id = "11111111-2222-4333-8444-555555555555"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"synchronization_report_{session_id}.json"
            original = self._legacy_report(session_id)
            path.write_text(json.dumps(original), encoding="utf-8")

            result = add_video_alignment_to_report(
                synchronization_report_path=path,
                reference_video_file=(
                    f"reference_video_{session_id}_test.mp4"
                ),
                video_start_marker_times_seconds=[4.0, 4.5, 5.0],
                video_end_marker_times_seconds=[41.5, 42.0, 42.5],
            )

            self.assertEqual(result["selected_offset_seconds"], 0.2)
            self.assertIn("watch_to_audio", result["time_mappings"])
            self.assertIn("video_to_audio", result["time_mappings"])
            self.assertEqual(result["schema_version"], 2)

    def test_unequal_marker_counts_are_rejected(self) -> None:
        with self.assertRaises(VideoAlignmentError):
            estimate_video_to_audio_mapping(
                audio_start_marker_times_seconds=[5.0, 5.6, 6.2],
                video_start_marker_times_seconds=[4.0, 4.5],
                audio_end_marker_times_seconds=[50.0, 50.6, 51.2],
                video_end_marker_times_seconds=[41.5, 42.0, 42.5],
            )

    def test_nonpositive_video_span_is_rejected(self) -> None:
        with self.assertRaises(VideoAlignmentError):
            estimate_video_to_audio_mapping(
                audio_start_marker_times_seconds=[5.0, 5.6, 6.2],
                video_start_marker_times_seconds=[50.0, 50.5, 51.0],
                audio_end_marker_times_seconds=[50.0, 50.6, 51.2],
                video_end_marker_times_seconds=[4.0, 4.5, 5.0],
            )

    def test_large_marker_residual_creates_warning(self) -> None:
        mapping = estimate_video_to_audio_mapping(
            audio_start_marker_times_seconds=[5.0, 5.6, 6.2],
            video_start_marker_times_seconds=[4.0, 4.5, 5.0],
            audio_end_marker_times_seconds=[50.0, 50.6, 52.0],
            video_end_marker_times_seconds=[41.5, 42.0, 42.5],
            maximum_marker_residual_seconds=0.05,
        )
        self.assertTrue(mapping["warnings"])

    def test_watch_mapping_is_built_from_legacy_fields(self) -> None:
        mapping = build_watch_to_audio_mapping(
            self._legacy_report("test-session")
        )
        self.assertEqual(mapping["scale"], 1.0)
        self.assertEqual(mapping["offset_seconds"], 0.2)
        self.assertEqual(mapping["method"], "CONSTANT_OFFSET")

    @staticmethod
    def _legacy_report(session_id: str) -> dict:
        return {
            "session_id": session_id,
            "start_audio_marker": {
                "peak_times_seconds": [5.0, 5.6, 6.2]
            },
            "end_audio_marker": {
                "peak_times_seconds": [50.0, 50.6, 51.2]
            },
            "start_offset_seconds": 0.2,
            "end_offset_seconds": 0.2,
            "selected_offset_seconds": 0.2,
            "selected_alignment_method": "CONSTANT_OFFSET",
            "offset_consistent": True,
        }


if __name__ == "__main__":
    unittest.main()
