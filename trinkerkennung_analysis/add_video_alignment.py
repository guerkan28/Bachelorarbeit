from __future__ import annotations

import argparse
import json
import sys

from .video_alignment import (
    VideoAlignmentError,
    add_video_alignment_to_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ergänzt einen Synchronisationsbericht um eine manuell "
            "bestimmte lineare Video-zu-Audio-Abbildung."
        )
    )
    parser.add_argument(
        "--synchronization-report",
        required=True,
    )
    parser.add_argument(
        "--reference-video-file",
        required=True,
    )
    parser.add_argument(
        "--video-start-markers",
        nargs="+",
        type=float,
        required=True,
        metavar="SEKUNDE",
    )
    parser.add_argument(
        "--video-end-markers",
        nargs="+",
        type=float,
        required=True,
        metavar="SEKUNDE",
    )
    parser.add_argument(
        "--output",
    )
    parser.add_argument(
        "--maximum-marker-residual-seconds",
        type=float,
        default=0.100,
    )
    arguments = parser.parse_args()

    try:
        report = add_video_alignment_to_report(
            synchronization_report_path=(
                arguments.synchronization_report
            ),
            reference_video_file=(
                arguments.reference_video_file
            ),
            video_start_marker_times_seconds=(
                arguments.video_start_markers
            ),
            video_end_marker_times_seconds=(
                arguments.video_end_markers
            ),
            output_path=arguments.output,
            maximum_marker_residual_seconds=(
                arguments.maximum_marker_residual_seconds
            ),
        )
    except VideoAlignmentError as exception:
        print(
            f"Videoausrichtungsfehler: {exception}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            report["time_mappings"]["video_to_audio"],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
