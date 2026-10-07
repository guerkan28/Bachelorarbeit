"""Audio-, Watch- und Fusionsfeatures für die beiden Fensterkonfigurationen erzeugen."""

from __future__ import annotations

import argparse
from pathlib import Path


CONFIGURATIONS = (
    ("windows_1s_stride1s.csv", "1s", "primary_P101_P108", "1s_stride1s"),
    ("windows_2s_stride2s.csv", "2s", "sensitivity_P101_P108", "2s_stride2s"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--window-root", type=Path, default=Path("ML/01_Window_Datasets"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("ML/02_Feature_Datasets"),
    )
    parser.add_argument(
        "--participant-root", action="append", nargs=2,
        metavar=("PARTICIPANT_ID", "DATA_ROOT"),
        help="Abweichender Datenroot je Person; mehrfach möglich.",
    )
    parser.add_argument(
        "--naming", choices=("general", "main-study"), default="general",
        help="general: allgemeiner LOSO-Runner; main-study: Dateinamen für den Hauptstudienrunner.",
    )
    args = parser.parse_args()
    participant_roots: dict[str, Path] = {}
    for participant, root in args.participant_root or []:
        if participant in participant_roots:
            parser.error(f"Datenroot für {participant} mehrfach angegeben.")
        participant_roots[participant] = Path(root)

    # Der Hilfetext ist auch vor Installation der Feature-Abhängigkeiten verfügbar.
    from trinkerkennung_analysis.feature_dataset import (
        build_feature_datasets_from_window_csv,
        write_feature_datasets,
    )

    for filename, general_configuration, main_prefix, main_configuration in CONFIGURATIONS:
        tables = build_feature_datasets_from_window_csv(
            data_root=args.data_root,
            window_csv=args.window_root / filename,
            participant_data_roots=participant_roots,
        )
        prefix = "pilot" if args.naming == "general" else main_prefix
        configuration = general_configuration if args.naming == "general" else main_configuration
        paths = write_feature_datasets(
            tables=tables,
            output_root=args.output_root,
            file_prefix=prefix,
            configuration_name=configuration,
        )
        for modality, path in paths.items():
            print(f"{modality}: {path}")


if __name__ == "__main__":
    main()
