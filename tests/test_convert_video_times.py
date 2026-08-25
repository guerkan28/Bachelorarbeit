from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trinkerkennung_analysis.convert_video_times import (
    VideoTimeConversionError,
    convert_video_times,
    create_conversion_report,
)


class VideoTimeConversionTest(unittest.TestCase):

    def test_known_video_times_are_converted(self) -> None:
        results = convert_video_times(
            [12.902, 57.391],
            scale=0.9957517588617413,
            offset_seconds=-6.487199609500851,
            audio_duration_seconds=57.16,
            labels=["start_anchor", "end_anchor"],
        )

        self.assertEqual(results[0]["label"], "start_anchor")
        self.assertAlmostEqual(
            results[0]["audio_seconds"],
            6.359989583333333,
            places=9,
        )
        self.assertAlmostEqual(
            results[1]["audio_seconds"],
            50.659989583333335,
            places=9,
        )
        self.assertTrue(results[0]["within_audio_recording"])
        self.assertTrue(results[1]["within_audio_recording"])

    def test_missing_video_alignment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "sync.json"
            report_path.write_text(
                json.dumps({
                    "session_id": "test",
                    "time_mappings": {
                        "watch_to_audio": {},
                        "video_to_audio": None,
                    },
                }),
                encoding="utf-8",
            )

            with self.assertRaises(VideoTimeConversionError):
                create_conversion_report(
                    report_path,
                    [10.0],
                )

    def test_label_count_must_match_video_times(self) -> None:
        with self.assertRaises(VideoTimeConversionError):
            convert_video_times(
                [10.0, 11.0],
                scale=1.0,
                offset_seconds=0.0,
                labels=["only_one_label"],
            )

    def test_times_outside_audio_are_flagged(self) -> None:
        results = convert_video_times(
            [1.0, 20.0],
            scale=1.0,
            offset_seconds=-5.0,
            audio_duration_seconds=10.0,
        )

        self.assertFalse(results[0]["within_audio_recording"])
        self.assertFalse(results[1]["within_audio_recording"])


if __name__ == "__main__":
    unittest.main()
