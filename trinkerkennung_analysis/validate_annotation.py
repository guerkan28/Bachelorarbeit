from __future__ import annotations

import argparse
import json
import os
import sys

from .annotation import (
    AnnotationValidationError,
    validate_annotation_file,
    validate_annotation_for_session,
)


DATA_ENV_NAME = "TRINKERKENNUNG_DATA_ROOT"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validiert eine fachliche Sitzungsannotation gegen WAV, "
            "Session-Metadaten, Synchronisationsbericht und Referenzvideo."
        )
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--annotation")
    selector.add_argument("--session-id")
    parser.add_argument("--data-root")
    parser.add_argument(
        "--marker-tolerance-seconds",
        type=float,
        default=0.050,
    )
    parser.add_argument("--report")
    arguments = parser.parse_args()

    data_root = arguments.data_root or os.environ.get(DATA_ENV_NAME)
    if not data_root:
        print(
            "Fehler: --data-root fehlt und "
            f"{DATA_ENV_NAME} ist nicht gesetzt.",
            file=sys.stderr,
        )
        return 2

    try:
        if arguments.annotation:
            result = validate_annotation_file(
                data_root=data_root,
                annotation_path=arguments.annotation,
                marker_tolerance_seconds=(
                    arguments.marker_tolerance_seconds
                ),
            )
        else:
            result = validate_annotation_for_session(
                data_root=data_root,
                session_id=arguments.session_id,
                marker_tolerance_seconds=(
                    arguments.marker_tolerance_seconds
                ),
            )
    except AnnotationValidationError as exception:
        print(str(exception), file=sys.stderr)
        return 1

    formatted = json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        indent=2,
    )
    print(formatted)

    if arguments.report:
        from pathlib import Path

        report_path = Path(arguments.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            formatted + "\n",
            encoding="utf-8",
        )
        print(f"\nPrüfbericht gespeichert: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
