from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .session_loader import (
    SessionValidationError,
    load_and_validate_session,
)

ENV_NAME = "TRINKERKENNUNG_DATA_ROOT"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validiert WAV, CSV und JSON "
            "einer Aufnahmesitzung."
        )
    )
    parser.add_argument(
        "--session-id",
        required=True,
    )
    parser.add_argument(
        "--data-root",
    )
    parser.add_argument(
        "--report",
    )
    arguments = parser.parse_args()

    data_root = (
        arguments.data_root or
        os.environ.get(ENV_NAME)
    )

    if not data_root:
        print(
            "Fehler: --data-root fehlt und "
            f"{ENV_NAME} ist nicht gesetzt.",
            file=sys.stderr,
        )
        return 2

    try:
        report = load_and_validate_session(
            data_root,
            arguments.session_id,
        )
    except SessionValidationError as exception:
        print(
            f"Validierungsfehler: {exception}",
            file=sys.stderr,
        )
        return 1

    formatted = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )
    print(formatted)

    if arguments.report:
        report_path = Path(
            arguments.report
        ).expanduser().resolve()
        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        report_path.write_text(
            formatted + "\n",
            encoding="utf-8",
        )
        print(
            f"\nPrüfbericht gespeichert: {report_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
