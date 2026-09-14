from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path

from trinkerkennung_analysis.annotation import (
    AnnotationValidationError,
    validate_annotation_file,
)


class AnnotationValidationTest(unittest.TestCase):
    SESSION_ID = "11111111-2222-4333-8444-555555555555"

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.annotation = self._create_valid_files()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_valid_minimal_annotation(self) -> None:
        result = self._validate()
        self.assertEqual(result.session_id, self.SESSION_ID)
        self.assertEqual(result.drink_event_count, 1)
        self.assertTrue(result.implicit_unlabeled_time_is_not_negative)

    def test_valid_multiple_drink_events(self) -> None:
        self.annotation["drink_events"].append({
            "event_id": "D002",
            "scenario": "DRINK_FROM_BOTTLE",
            "container_type": "BOTTLE",
            "event_start_seconds": 25.0,
            "event_end_seconds": 29.0,
            "mouth_contact_intervals": [
                {"start_seconds": 26.0, "end_seconds": 27.0}
            ],
            "notes": None,
        })
        self.annotation["negative_intervals"] = []
        result = self._validate()
        self.assertEqual(result.drink_event_count, 2)

    def test_valid_multiple_mouth_contact_intervals(self) -> None:
        self.annotation["drink_events"][0]["mouth_contact_intervals"] = [
            {"start_seconds": 11.0, "end_seconds": 11.5},
            {"start_seconds": 11.8, "end_seconds": 12.3},
        ]
        result = self._validate()
        self.assertEqual(result.mouth_contact_interval_count, 2)

    def test_missing_mouth_contact_is_rejected(self) -> None:
        self.annotation["drink_events"][0]["mouth_contact_intervals"] = []
        self._assert_invalid("mindestens ein Intervall")

    def test_mouth_contact_outside_event_is_rejected(self) -> None:
        self.annotation["drink_events"][0]["mouth_contact_intervals"] = [
            {"start_seconds": 9.5, "end_seconds": 11.0}
        ]
        self._assert_invalid("nicht vollständig innerhalb")

    def test_overlapping_mouth_contacts_are_rejected(self) -> None:
        self.annotation["drink_events"][0]["mouth_contact_intervals"] = [
            {"start_seconds": 11.0, "end_seconds": 12.0},
            {"start_seconds": 11.9, "end_seconds": 12.5},
        ]
        self._assert_invalid("mouth_contact_intervals überlappen")

    def test_positive_and_negative_overlap_is_rejected(self) -> None:
        self.annotation["negative_intervals"][0]["start_seconds"] = 13.0
        self._assert_invalid("Explizite fachliche Intervalle")

    def test_overlap_with_excluded_is_rejected(self) -> None:
        self.annotation["excluded_intervals"].append({
            "interval_id": "X003",
            "reason": "INVALID_SIGNAL",
            "start_seconds": 13.0,
            "end_seconds": 15.0,
        })
        self._assert_invalid("Explizite fachliche Intervalle")

    def test_overlap_with_uncertain_is_rejected(self) -> None:
        self.annotation["uncertain_intervals"] = [{
            "interval_id": "U001",
            "reason": "EVENT_BOUNDARY_NOT_VISIBLE",
            "start_seconds": 13.0,
            "end_seconds": 15.0,
        }]
        self._assert_invalid("Explizite fachliche Intervalle")

    def test_negative_time_is_rejected(self) -> None:
        self.annotation["negative_intervals"][0]["start_seconds"] = -1.0
        self._assert_invalid("negative Zeit")

    def test_start_equal_end_is_rejected(self) -> None:
        self.annotation["negative_intervals"][0]["end_seconds"] = 20.0
        self._assert_invalid("start_seconds < end_seconds")

    def test_interval_outside_wav_is_rejected(self) -> None:
        self.annotation["negative_intervals"][0]["end_seconds"] = 61.0
        self._assert_invalid("außerhalb der WAV-Dauer")

    def test_duplicate_ids_are_rejected(self) -> None:
        self.annotation["negative_intervals"][0]["interval_id"] = "D001"
        self._assert_invalid("Doppelte Ereignis- oder Intervall-ID")

    def test_unknown_scenario_is_rejected(self) -> None:
        self.annotation["drink_events"][0]["scenario"] = "UNKNOWN"
        self._assert_invalid("scenario ist unbekannt")

    def test_unknown_exclusion_reason_is_rejected(self) -> None:
        self.annotation["excluded_intervals"][0]["reason"] = "UNKNOWN"
        self._assert_invalid("reason ist unbekannt")

    def test_wrong_participant_id_is_rejected(self) -> None:
        self.annotation["participant_id"] = "Max Mustermann"
        self._assert_invalid("participant_id")

    def test_wrong_session_id_is_rejected(self) -> None:
        self.annotation["session_id"] = (
            "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        )
        self._assert_invalid("Annotationsdateiname")

    def test_partial_coverage_with_unlabeled_remainder_is_valid(self) -> None:
        self.annotation["ground_truth"]["annotation_coverage"] = "PARTIAL"
        self.annotation["reviewed_intervals"] = [
            {"start_seconds": 8.0, "end_seconds": 24.0}
        ]
        result = self._validate()
        self.assertEqual(result.annotation_coverage, "PARTIAL")
        self.assertTrue(result.implicit_unlabeled_time_is_not_negative)

    def test_synchronized_video_without_reference_video_is_rejected(self) -> None:
        self.annotation["ground_truth"]["reference_video_file"] = ""
        self._assert_invalid("reference_video_file")

    def test_synchronized_video_without_video_alignment_is_rejected(self) -> None:
        report_path = self.root / self.annotation["synchronization_report_file"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["time_mappings"]["video_to_audio"] = None
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self._assert_invalid("Video-zu-Audio-Abbildung")

    def test_reference_video_absolute_mapping_path_is_valid(self) -> None:
        video_name = self.annotation["ground_truth"]["reference_video_file"]

        camera_directory = self.root / "camera"
        camera_directory.mkdir()

        source_video = self.root / video_name
        target_video = camera_directory / video_name
        source_video.replace(target_video)

        report_path = (
            self.root / self.annotation["synchronization_report_file"]
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))

        report["time_mappings"]["video_to_audio"][
            "reference_video_file"
        ] = str(target_video.resolve())

        report_path.write_text(
            json.dumps(report),
            encoding="utf-8",
        )

        result = self._validate()

        self.assertEqual(
            Path(result.reference_video_file).name,
            video_name,
        )

    def test_reference_video_mapping_filename_mismatch_is_rejected(
        self,
    ) -> None:
        report_path = (
            self.root / self.annotation["synchronization_report_file"]
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))

        wrong_video_name = (
            f"reference_video_{self.SESSION_ID}_different.mp4"
        )

        report["time_mappings"]["video_to_audio"][
            "reference_video_file"
        ] = str((self.root / "camera" / wrong_video_name).resolve())

        report_path.write_text(
            json.dumps(report),
            encoding="utf-8",
        )

        self._assert_invalid(
            "Referenzvideo in Annotation und Videoabbildung stimmt nicht überein."
        )

    def test_marker_match_uses_tolerance(self) -> None:
        report_path = self.root / self.annotation["synchronization_report_file"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.pop("audio_marker_exclusion_intervals_seconds")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self.annotation["excluded_intervals"] = [
            {
                "interval_id": "X001",
                "reason": "SYNCHRONIZATION_MARKER",
                "start_seconds": 5.04,
                "end_seconds": 6.21,
            },
            {
                "interval_id": "X002",
                "reason": "SYNCHRONIZATION_MARKER",
                "start_seconds": 50.54,
                "end_seconds": 51.71,
            },
        ]
        result = self._validate(marker_tolerance_seconds=0.05)
        self.assertEqual(result.excluded_interval_count, 2)

    def test_marker_outside_tolerance_is_rejected(self) -> None:
        report_path = self.root / self.annotation["synchronization_report_file"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.pop("audio_marker_exclusion_intervals_seconds")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self.annotation["excluded_intervals"][0]["start_seconds"] = 5.20
        self._assert_invalid("Audiomarker liegt außerhalb")

    def test_forbidden_window_field_is_rejected(self) -> None:
        self.annotation["window_length_seconds"] = 2.0
        self._assert_invalid("Verbotenes Fenster")

    def test_aborted_activity_can_be_uncertain(self) -> None:
        self.annotation["uncertain_intervals"] = [{
            "interval_id": "U001",
            "reason": "ABORTED_ACTIVITY",
            "start_seconds": 30.0,
            "end_seconds": 32.0,
            "notes": "Gefäßbewegung wurde vor Mundkontakt beendet.",
        }]
        result = self._validate()
        self.assertEqual(result.uncertain_interval_count, 1)

    def test_mixed_session_with_multiple_scenarios_is_valid(self) -> None:
        self.annotation["session_mode"] = "CONTINUOUS_MIXED_ACTIVITY"
        self.annotation["negative_intervals"].append({
            "interval_id": "N002",
            "scenario": "SMARTPHONE_USE",
            "start_seconds": 30.0,
            "end_seconds": 34.0,
            "notes": None,
        })
        self.annotation["drink_events"].append({
            "event_id": "D002",
            "scenario": "DRINK_FROM_BOTTLE",
            "container_type": "BOTTLE",
            "event_start_seconds": 38.0,
            "event_end_seconds": 42.0,
            "mouth_contact_intervals": [
                {"start_seconds": 39.0, "end_seconds": 40.0}
            ],
            "notes": None,
        })
        result = self._validate()
        self.assertEqual(result.drink_event_count, 2)
        self.assertEqual(result.negative_interval_count, 2)

    def test_lift_negative_with_glass_container_type_is_valid(self) -> None:
        self.annotation["negative_intervals"][0]["container_type"] = "GLASS"

        result = self._validate()

        self.assertEqual(result.negative_interval_count, 1)

    def test_move_negative_with_bottle_container_type_is_valid(self) -> None:
        self.annotation["negative_intervals"][0][
            "scenario"
        ] = "MOVE_CONTAINER_WITHOUT_DRINKING"
        self.annotation["negative_intervals"][0]["container_type"] = "BOTTLE"

        result = self._validate()

        self.assertEqual(result.negative_interval_count, 1)

    def test_unknown_negative_container_type_is_rejected(self) -> None:
        self.annotation["negative_intervals"][0]["container_type"] = "STRAW"

        self._assert_invalid("container_type ist unbekannt")

    def test_unknown_container_type_is_rejected(self) -> None:
        self.annotation["drink_events"][0]["container_type"] = "STRAW"
        self._assert_invalid("container_type ist unbekannt")

    def test_unknown_protocol_version_is_rejected(self) -> None:
        self.annotation["recording_protocol_version"] = "9.9"
        self._assert_invalid("recording_protocol_version")

    def test_reference_video_session_mismatch_is_rejected(self) -> None:
        self.annotation["ground_truth"]["reference_video_file"] = (
            "reference_video_aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee_test.mp4"
        )
        self._assert_invalid("reference_video_file enthält nicht")

    def test_partial_interval_outside_reviewed_area_is_rejected(self) -> None:
        self.annotation["ground_truth"]["annotation_coverage"] = "PARTIAL"
        self.annotation["reviewed_intervals"] = [
            {"start_seconds": 8.0, "end_seconds": 15.0}
        ]
        self._assert_invalid("außerhalb der dokumentierten reviewed_intervals")

    def test_overlapping_excluded_intervals_are_rejected(self) -> None:
        self.annotation["excluded_intervals"].append({
            "interval_id": "X003",
            "reason": "OTHER_TECHNICAL_ARTIFACT",
            "start_seconds": 6.0,
            "end_seconds": 7.0,
        })
        self._assert_invalid("Explizite fachliche Intervalle")

    def _validate(self, marker_tolerance_seconds: float = 0.05):
        annotation_path = self._write_annotation()
        return validate_annotation_file(
            data_root=self.root,
            annotation_path=annotation_path,
            marker_tolerance_seconds=marker_tolerance_seconds,
        )

    def _assert_invalid(self, expected_text: str) -> None:
        with self.assertRaises(AnnotationValidationError) as context:
            self._validate()
        self.assertIn(expected_text, str(context.exception))

    def _write_annotation(self) -> Path:
        path = self.root / f"annotation_{self.SESSION_ID}.json"
        path.write_text(
            json.dumps(self.annotation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def _create_valid_files(self) -> dict:
        session_id = self.SESSION_ID
        wav_name = f"smartphone_{session_id}_test.wav"
        csv_name = f"watch_{session_id}_test.csv"
        session_name = f"session_{session_id}.json"
        report_name = f"synchronization_report_{session_id}.json"
        video_name = f"reference_video_{session_id}_test.mp4"

        with wave.open(str(self.root / wav_name), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(100)
            wav_file.writeframes(b"\x00\x00" * 6_000)

        with (self.root / csv_name).open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["session_id", "sensor", "relative_time_ns"],
            )
            writer.writeheader()
            writer.writerows([
                {
                    "session_id": session_id,
                    "sensor": "ACCELEROMETER",
                    "relative_time_ns": 0,
                },
                {
                    "session_id": session_id,
                    "sensor": "GYROSCOPE",
                    "relative_time_ns": 20_000_000,
                },
            ])

        (self.root / session_name).write_text(
            json.dumps({
                "schema_version": 1,
                "session_id": session_id,
                "status": "COMPLETED",
                "phone_audio_file_name": wav_name,
                "watch_sensor_file_name": csv_name,
                "phone_audio_start_epoch_ms": 1000,
                "phone_audio_stop_epoch_ms": 61_000,
                "watch_start_epoch_ms": 1000,
                "watch_stop_epoch_ms": 1020,
                "phone_audio_success": True,
                "watch_recording_success": True,
            }),
            encoding="utf-8",
        )

        (self.root / video_name).write_bytes(b"synthetic-video-placeholder")

        synchronization_report = {
            "schema_version": 2,
            "session_id": session_id,
            "start_audio_marker": {
                "peak_times_seconds": [5.0, 5.6, 6.2]
            },
            "end_audio_marker": {
                "peak_times_seconds": [50.5, 51.1, 51.7]
            },
            "audio_marker_exclusion_intervals_seconds": [
                [4.5, 6.7],
                [50.0, 52.2],
            ],
            "time_mappings": {
                "watch_to_audio": {
                    "source_time_reference": (
                        "WATCH_SECONDS_FROM_SENSOR_SESSION_START"
                    ),
                    "target_time_reference": (
                        "AUDIO_SECONDS_FROM_WAV_START"
                    ),
                    "method": "CONSTANT_OFFSET",
                    "scale": 1.0,
                    "offset_seconds": 0.2,
                },
                "video_to_audio": {
                    "source_time_reference": (
                        "VIDEO_SECONDS_FROM_REFERENCE_VIDEO_START"
                    ),
                    "target_time_reference": (
                        "AUDIO_SECONDS_FROM_WAV_START"
                    ),
                    "method": "LINEAR_TWO_POINT",
                    "scale": 1.0001,
                    "offset_seconds": 0.3,
                    "reference_video_file": video_name,
                },
            },
        }
        (self.root / report_name).write_text(
            json.dumps(synchronization_report),
            encoding="utf-8",
        )

        return {
            "schema_version": 1,
            "session_id": session_id,
            "participant_id": "P001",
            "recording_protocol_version": "1.0",
            "session_mode": "CONTROLLED_SINGLE_ACTIVITY",
            "time_reference": "AUDIO_SECONDS_FROM_WAV_START",
            "synchronization_report_file": report_name,
            "ground_truth": {
                "source": "SYNCHRONIZED_VIDEO",
                "reference_video_file": video_name,
                "annotator_id": "A001",
                "annotation_coverage": "EXHAUSTIVE",
            },
            "drink_events": [{
                "event_id": "D001",
                "scenario": "DRINK_FROM_GLASS",
                "container_type": "GLASS",
                "event_start_seconds": 10.0,
                "event_end_seconds": 14.0,
                "mouth_contact_intervals": [
                    {"start_seconds": 11.0, "end_seconds": 12.0}
                ],
                "notes": None,
            }],
            "negative_intervals": [{
                "interval_id": "N001",
                "scenario": "LIFT_CONTAINER_WITHOUT_DRINKING",
                "start_seconds": 20.0,
                "end_seconds": 23.0,
                "notes": None,
            }],
            "excluded_intervals": [
                {
                    "interval_id": "X001",
                    "reason": "SYNCHRONIZATION_MARKER",
                    "start_seconds": 4.5,
                    "end_seconds": 6.7,
                },
                {
                    "interval_id": "X002",
                    "reason": "SYNCHRONIZATION_MARKER",
                    "start_seconds": 50.0,
                    "end_seconds": 52.2,
                },
            ],
            "uncertain_intervals": [],
        }


if __name__ == "__main__":
    unittest.main()
