from __future__ import annotations

import argparse
import json
from pathlib import Path

from .video_alignment_comparison import run_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Vergleicht LINEAR_TWO_POINT und LINEAR_ALL_SIX_OLS auf bestehenden "
            "technischen Videoalignment-Sessions, ohne bestehende Synchronisationsberichte "
            "oder Annotationen zu verändern."
        )
    )
    parser.add_argument("--manual-root", type=Path, required=True)
    parser.add_argument("--sync-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--session-id", action="append", dest="session_ids")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_comparison(
        manual_root=args.manual_root,
        sync_root=args.sync_root,
        annotation_root=args.annotation_root,
        output_root=args.output_root,
        session_ids=args.session_ids,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
