import tempfile
import unittest
from pathlib import Path

from trinkerkennung_analysis.video_alignment_comparison import (
    _parse_timestamp_token,
    fit_all_six_ols,
    fit_two_point,
    parse_manual_marker_file,
    residuals_seconds,
)


class VideoAlignmentComparisonTests(unittest.TestCase):
    def test_parse_timestamp_token_supports_seconds_and_minutes(self):
        self.assertAlmostEqual(_parse_timestamp_token("11.571"), 11.571)
        self.assertAlmostEqual(_parse_timestamp_token("11,571"), 11.571)
        self.assertAlmostEqual(_parse_timestamp_token("1:00.782"), 60.782)
        self.assertAlmostEqual(_parse_timestamp_token("1:37,026"), 97.026)

    def test_parse_manual_marker_file(self):
        session_id = "4fe52314-1b84-4a27-a9ab-37b697c6f721"
        content = """
Irgendein Kopf

ANFANG

Klopfer 1: 11.571
Text dazwischen
Klopfer 2: 11.970

Klopfer 3: 12.369

Weitere Notizen

ENDE

Klopfer 1: 1:36.028
Klopfer 2: 1:36.494
Klopfer 3: 1:37.026
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"smartphone_{session_id}_20260902_141545_589.txt"
            path.write_text(content, encoding="utf-8")
            parsed = parse_manual_marker_file(path)
        self.assertEqual(parsed["session_id"], session_id)
        self.assertEqual(parsed["video_start_markers_seconds"], [11.571, 11.970, 12.369])
        self.assertEqual(len(parsed["video_end_markers_seconds"]), 3)
        for actual, expected in zip(parsed["video_end_markers_seconds"], [96.028, 96.494, 97.026]):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_two_point_passes_exactly_through_middle_anchors(self):
        video_start = [10.0, 11.0, 12.0]
        video_end = [50.0, 51.0, 52.0]
        audio_start = [5.0, 6.0, 7.0]
        audio_end = [45.0, 46.0, 47.0]
        mapping = fit_two_point(video_start, video_end, audio_start, audio_end)
        self.assertAlmostEqual(mapping.map_seconds(11.0), 6.0)
        self.assertAlmostEqual(mapping.map_seconds(51.0), 46.0)

    def test_all_six_ols_recovers_exact_linear_mapping(self):
        video = [1, 2, 3, 10, 11, 12]
        audio = [2.5 * x - 4.0 for x in video]
        mapping = fit_all_six_ols(video, audio)
        self.assertAlmostEqual(mapping.scale, 2.5, places=12)
        self.assertAlmostEqual(mapping.offset_seconds, -4.0, places=12)
        errors = residuals_seconds(mapping, video, audio)
        self.assertTrue(all(abs(error) < 1e-10 for error in errors))


if __name__ == "__main__":
    unittest.main()
