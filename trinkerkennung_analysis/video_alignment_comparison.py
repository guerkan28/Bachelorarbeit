from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


UUID_RE = re.compile(
    r"(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
CONTROL_INDICES = (0, 2, 3, 5)
GRID_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class LinearMapping:
    method: str
    scale: float
    offset_seconds: float

    def map_seconds(self, video_seconds: float) -> float:
        return self.scale * float(video_seconds) + self.offset_seconds

    def inverse_seconds(self, audio_seconds: float) -> float:
        if self.scale == 0.0:
            raise ValueError("Scale darf nicht 0 sein.")
        return (float(audio_seconds) - self.offset_seconds) / self.scale


def _parse_timestamp_token(token: str) -> float:
    token = token.strip().replace(",", ".")
    if ":" not in token:
        return float(token)
    parts = token.split(":")
    if len(parts) != 2:
        raise ValueError(f"Nicht unterstütztes Zeitformat: {token!r}")
    minutes = int(parts[0])
    seconds = float(parts[1])
    if not (0.0 <= seconds < 60.0):
        raise ValueError(f"Ungültige Sekunden im Zeitformat: {token!r}")
    return minutes * 60.0 + seconds


def _extract_three_markers(section_text: str, section_name: str) -> list[float]:
    values: list[float] = []
    time_pattern = r"([0-9]+(?::[0-9]{1,2})?(?:[.,][0-9]+)?)"
    for index in (1, 2, 3):
        match = re.search(
            rf"Klopfer\s*{index}\s*:\s*{time_pattern}",
            section_text,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                f"{section_name}: 'Klopfer {index}:' konnte nicht gefunden werden."
            )
        values.append(_parse_timestamp_token(match.group(1)))
    return values


def parse_manual_marker_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    session_match = UUID_RE.search(path.name) or UUID_RE.search(text)
    if not session_match:
        raise ValueError(f"Keine Session-UUID in {path.name} gefunden.")

    start_match = re.search(
        r"\bANFANG\b(.*?)(?=\bENDE\b)", text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    end_match = re.search(
        r"\bENDE\b(.*)$", text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not start_match or not end_match:
        raise ValueError(
            f"{path.name}: Abschnitte ANFANG und ENDE konnten nicht gelesen werden."
        )

    return {
        "session_id": session_match.group("uuid").lower(),
        "manual_file": str(path),
        "video_start_markers_seconds": _extract_three_markers(start_match.group(1), "ANFANG"),
        "video_end_markers_seconds": _extract_three_markers(end_match.group(1), "ENDE"),
    }


def discover_manual_marker_files(manual_root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(manual_root.rglob("*.txt")):
        match = UUID_RE.search(path.name)
        if not match:
            continue
        session_id = match.group("uuid").lower()
        if session_id in discovered:
            duplicates.setdefault(session_id, [discovered[session_id]]).append(path)
        else:
            discovered[session_id] = path
    if duplicates:
        details = "\n".join(
            f"{sid}: " + ", ".join(str(p) for p in paths)
            for sid, paths in duplicates.items()
        )
        raise ValueError(
            "Mehrere manuelle Auswertungsdateien für dieselbe Session gefunden:\n" + details
        )
    return discovered


def load_sync_report(sync_root: Path, session_id: str) -> tuple[Path, dict]:
    path = sync_root / session_id / f"synchronization_report_{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Synchronisationsbericht fehlt: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if str(data.get("session_id", "")).lower() != session_id.lower():
        raise ValueError(f"Session-ID im Synchronisationsbericht passt nicht: {path}")
    return path, data


def _extract_audio_markers(sync_report: dict) -> tuple[list[float], list[float]]:
    start = [float(x) for x in sync_report["start_audio_marker"]["peak_times_seconds"]]
    end = [float(x) for x in sync_report["end_audio_marker"]["peak_times_seconds"]]
    if len(start) != 3 or len(end) != 3:
        raise ValueError("Es werden genau drei Audio-Start- und drei Audio-Endmarker benötigt.")
    return start, end


def fit_two_point(
    video_start: Sequence[float],
    video_end: Sequence[float],
    audio_start: Sequence[float],
    audio_end: Sequence[float],
    start_index: int = 1,
    end_index: int = 1,
) -> LinearMapping:
    v1, a1 = float(video_start[start_index]), float(audio_start[start_index])
    v2, a2 = float(video_end[end_index]), float(audio_end[end_index])
    if v2 == v1:
        raise ValueError("Die beiden Two-Point-Videoanker dürfen nicht identisch sein.")
    scale = (a2 - a1) / (v2 - v1)
    offset = a1 - scale * v1
    return LinearMapping("LINEAR_TWO_POINT", float(scale), float(offset))


def fit_all_six_ols(video_markers: Sequence[float], audio_markers: Sequence[float]) -> LinearMapping:
    if len(video_markers) != 6 or len(audio_markers) != 6:
        raise ValueError("ALL_SIX_OLS benötigt genau sechs Markerpaare.")
    x = np.asarray(video_markers, dtype=float)
    y = np.asarray(audio_markers, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    return LinearMapping("LINEAR_ALL_SIX_OLS", float(scale), float(offset))


def residuals_seconds(mapping: LinearMapping, video_markers: Sequence[float], audio_markers: Sequence[float]) -> list[float]:
    return [
        float(audio) - mapping.map_seconds(float(video))
        for video, audio in zip(video_markers, audio_markers)
    ]


def error_metrics_seconds(errors: Sequence[float]) -> dict:
    if not errors:
        return {"mae_seconds": None, "rmse_seconds": None, "max_abs_seconds": None}
    arr = np.asarray(errors, dtype=float)
    return {
        "mae_seconds": float(np.mean(np.abs(arr))),
        "rmse_seconds": float(np.sqrt(np.mean(arr ** 2))),
        "max_abs_seconds": float(np.max(np.abs(arr))),
    }


def _grid_times(video_markers: Sequence[float]) -> list[float]:
    start, end = min(map(float, video_markers)), max(map(float, video_markers))
    span = end - start
    return [start + fraction * span for fraction in GRID_FRACTIONS]


def _mapping_delta_ms(left: LinearMapping, right: LinearMapping, grid: Sequence[float]) -> list[float]:
    return [1000.0 * (right.map_seconds(t) - left.map_seconds(t)) for t in grid]


def two_point_anchor_sensitivity(
    video_start: Sequence[float], video_end: Sequence[float],
    audio_start: Sequence[float], audio_end: Sequence[float],
    baseline: LinearMapping, grid: Sequence[float],
) -> dict:
    absolute_deltas: list[float] = []
    variants: list[dict] = []
    for start_index in range(3):
        for end_index in range(3):
            mapping = fit_two_point(
                video_start, video_end, audio_start, audio_end,
                start_index=start_index, end_index=end_index,
            )
            deltas = _mapping_delta_ms(baseline, mapping, grid)
            absolute_deltas.extend(abs(x) for x in deltas)
            variants.append({
                "start_marker_index": start_index + 1,
                "end_marker_index": end_index + 1,
                "scale": mapping.scale,
                "offset_seconds": mapping.offset_seconds,
                "mapping_delta_vs_middle_middle_ms": deltas,
            })
    return {
        "max_abs_mapping_delta_ms": max(absolute_deltas) if absolute_deltas else 0.0,
        "median_abs_mapping_delta_ms": statistics.median(absolute_deltas) if absolute_deltas else 0.0,
        "variants": variants,
    }


def _fit_ols_any_count(video_markers: Sequence[float], audio_markers: Sequence[float], method: str) -> LinearMapping:
    x = np.asarray(video_markers, dtype=float)
    y = np.asarray(audio_markers, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    return LinearMapping(method, float(scale), float(offset))


def all_six_leave_one_out_sensitivity(
    video_markers: Sequence[float], audio_markers: Sequence[float],
    baseline: LinearMapping, grid: Sequence[float],
) -> dict:
    absolute_deltas: list[float] = []
    variants: list[dict] = []
    for held_out in range(6):
        x = [v for i, v in enumerate(video_markers) if i != held_out]
        y = [a for i, a in enumerate(audio_markers) if i != held_out]
        mapping = _fit_ols_any_count(x, y, "LINEAR_ALL_SIX_OLS_LOO")
        deltas = _mapping_delta_ms(baseline, mapping, grid)
        absolute_deltas.extend(abs(x) for x in deltas)
        variants.append({
            "held_out_marker_index": held_out + 1,
            "scale": mapping.scale,
            "offset_seconds": mapping.offset_seconds,
            "mapping_delta_vs_full_all_six_ms": deltas,
        })
    return {
        "max_abs_mapping_delta_ms": max(absolute_deltas) if absolute_deltas else 0.0,
        "median_abs_mapping_delta_ms": statistics.median(absolute_deltas) if absolute_deltas else 0.0,
        "variants": variants,
    }


def all_six_control_holdout_errors(video_markers: Sequence[float], audio_markers: Sequence[float]) -> list[float]:
    errors: list[float] = []
    for held_out in CONTROL_INDICES:
        x = [v for i, v in enumerate(video_markers) if i != held_out]
        y = [a for i, a in enumerate(audio_markers) if i != held_out]
        mapping = _fit_ols_any_count(x, y, "LINEAR_ALL_SIX_OLS_CONTROL_HOLDOUT")
        prediction = mapping.map_seconds(video_markers[held_out])
        errors.append(float(audio_markers[held_out]) - prediction)
    return errors


def _annotation_boundaries(annotation: dict) -> list[tuple[str, float]]:
    boundaries: list[tuple[str, float]] = []
    for event in annotation.get("drink_events", []):
        event_id = str(event.get("event_id", "DRINK"))
        boundaries.append((f"{event_id}.event_start_seconds", float(event["event_start_seconds"])))
        boundaries.append((f"{event_id}.event_end_seconds", float(event["event_end_seconds"])))
        for index, contact in enumerate(event.get("mouth_contact_intervals", []), start=1):
            boundaries.append((f"{event_id}.mouth_contact_{index}.start_seconds", float(contact["start_seconds"])))
            boundaries.append((f"{event_id}.mouth_contact_{index}.end_seconds", float(contact["end_seconds"])))
    for category in ("negative_intervals", "uncertain_intervals"):
        for interval in annotation.get(category, []):
            interval_id = str(interval.get("interval_id", category))
            boundaries.append((f"{category}.{interval_id}.start_seconds", float(interval["start_seconds"])))
            boundaries.append((f"{category}.{interval_id}.end_seconds", float(interval["end_seconds"])))
    return boundaries


def annotation_sensitivity(annotation_path: Path | None, two_point: LinearMapping, all_six: LinearMapping) -> dict | None:
    if annotation_path is None or not annotation_path.exists():
        return None
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    items = []
    absolute_shifts_ms: list[float] = []
    for label, existing_audio_seconds in _annotation_boundaries(annotation):
        video_seconds = two_point.inverse_seconds(existing_audio_seconds)
        all_six_audio_seconds = all_six.map_seconds(video_seconds)
        shift_ms = 1000.0 * (all_six_audio_seconds - existing_audio_seconds)
        absolute_shifts_ms.append(abs(shift_ms))
        items.append({
            "label": label,
            "existing_two_point_audio_seconds": existing_audio_seconds,
            "reconstructed_video_seconds": video_seconds,
            "all_six_audio_seconds": all_six_audio_seconds,
            "all_six_minus_two_point_ms": shift_ms,
        })
    if not absolute_shifts_ms:
        return {"boundary_count": 0, "mean_abs_shift_ms": None, "median_abs_shift_ms": None, "max_abs_shift_ms": None, "boundaries": []}
    return {
        "boundary_count": len(absolute_shifts_ms),
        "mean_abs_shift_ms": float(statistics.mean(absolute_shifts_ms)),
        "median_abs_shift_ms": float(statistics.median(absolute_shifts_ms)),
        "max_abs_shift_ms": float(max(absolute_shifts_ms)),
        "boundaries": items,
    }


def compare_session(*, session_id: str, manual_file: Path, sync_root: Path, annotation_root: Path | None) -> dict:
    manual = parse_manual_marker_file(manual_file)
    if manual["session_id"] != session_id:
        raise ValueError(f"UUID aus manueller Datei passt nicht zu Session {session_id}: {manual_file}")
    sync_path, sync_report = load_sync_report(sync_root, session_id)
    audio_start, audio_end = _extract_audio_markers(sync_report)
    video_start = manual["video_start_markers_seconds"]
    video_end = manual["video_end_markers_seconds"]
    video_markers = video_start + video_end
    audio_markers = audio_start + audio_end

    two_point = fit_two_point(video_start, video_end, audio_start, audio_end)

    stored_video_map = sync_report.get("time_mappings", {}).get("video_to_audio")
    source_validation = {
        "stored_video_mapping_present": stored_video_map is not None,
        "manual_markers_match_stored_video_mapping": None,
        "recomputed_two_point_matches_stored_mapping": None,
    }
    if stored_video_map is None:
        raise ValueError(
            f"Finales video_to_audio-Mapping fehlt im Synchronisationsbericht: {sync_path}"
        )

    stored_start = [float(x) for x in stored_video_map["start_anchor"]["video_marker_times_seconds"]]
    stored_end = [float(x) for x in stored_video_map["end_anchor"]["video_marker_times_seconds"]]
    marker_differences = [
        abs(a - b)
        for a, b in zip(video_start + video_end, stored_start + stored_end)
    ]
    markers_match = max(marker_differences, default=0.0) <= 1e-6
    source_validation["manual_markers_match_stored_video_mapping"] = markers_match
    source_validation["manual_vs_stored_marker_max_abs_difference_seconds"] = max(
        marker_differences, default=0.0
    )
    if not markers_match:
        raise ValueError(
            f"Manuelle Videomarker stimmen nicht mit dem finalen Synchronisationsbericht überein: {manual_file}"
        )

    stored_scale = float(stored_video_map["scale"])
    stored_offset = float(stored_video_map["offset_seconds"])
    mapping_matches = (
        abs(two_point.scale - stored_scale) <= 1e-9
        and abs(two_point.offset_seconds - stored_offset) <= 1e-9
    )
    source_validation["recomputed_two_point_matches_stored_mapping"] = mapping_matches
    source_validation["two_point_scale_difference"] = two_point.scale - stored_scale
    source_validation["two_point_offset_difference_seconds"] = two_point.offset_seconds - stored_offset
    if not mapping_matches:
        raise ValueError(
            f"Rekonstruiertes LINEAR_TWO_POINT entspricht nicht dem final gespeicherten Mapping: {sync_path}"
        )

    all_six = fit_all_six_ols(video_markers, audio_markers)
    two_residuals = residuals_seconds(two_point, video_markers, audio_markers)
    all_residuals = residuals_seconds(all_six, video_markers, audio_markers)
    two_control_errors = [two_residuals[i] for i in CONTROL_INDICES]
    all_control_holdout = all_six_control_holdout_errors(video_markers, audio_markers)
    grid = _grid_times(video_markers)
    mapping_delta_ms = _mapping_delta_ms(two_point, all_six, grid)

    annotation_path = None
    if annotation_root is not None:
        candidate = annotation_root / f"annotation_{session_id}.json"
        if candidate.exists():
            annotation_path = candidate

    return {
        "schema_version": 1,
        "session_id": session_id,
        "inputs": {
            "manual_file": str(manual_file),
            "synchronization_report_file": str(sync_path),
            "annotation_file": str(annotation_path) if annotation_path else None,
            "video_start_markers_seconds": video_start,
            "video_end_markers_seconds": video_end,
            "audio_start_markers_seconds": audio_start,
            "audio_end_markers_seconds": audio_end,
        },
        "source_validation": source_validation,
        "two_point": {
            "method": two_point.method,
            "scale": two_point.scale,
            "offset_seconds": two_point.offset_seconds,
            "residuals_seconds": two_residuals,
            "all_marker_metrics": error_metrics_seconds(two_residuals),
            "control_marker_indices_one_based": [1, 3, 4, 6],
            "control_marker_errors_seconds": two_control_errors,
            "control_marker_metrics": error_metrics_seconds(two_control_errors),
            "anchor_sensitivity": two_point_anchor_sensitivity(video_start, video_end, audio_start, audio_end, two_point, grid),
        },
        "all_six_ols": {
            "method": all_six.method,
            "scale": all_six.scale,
            "offset_seconds": all_six.offset_seconds,
            "residuals_seconds": all_residuals,
            "all_marker_metrics": error_metrics_seconds(all_residuals),
            "control_holdout_marker_indices_one_based": [1, 3, 4, 6],
            "control_holdout_errors_seconds": all_control_holdout,
            "control_holdout_metrics": error_metrics_seconds(all_control_holdout),
            "leave_one_out_sensitivity": all_six_leave_one_out_sensitivity(video_markers, audio_markers, all_six, grid),
        },
        "method_difference": {
            "grid_video_seconds": grid,
            "grid_fractions": list(GRID_FRACTIONS),
            "all_six_minus_two_point_ms": mapping_delta_ms,
            "max_abs_mapping_delta_ms": max(abs(x) for x in mapping_delta_ms),
        },
        "annotation_sensitivity": annotation_sensitivity(annotation_path, two_point, all_six),
    }


def _seconds_to_ms(value: float | None) -> float | None:
    return None if value is None else 1000.0 * float(value)


def summary_row(result: dict) -> dict:
    tp_all = result["two_point"]["all_marker_metrics"]
    tp_control = result["two_point"]["control_marker_metrics"]
    all_all = result["all_six_ols"]["all_marker_metrics"]
    all_holdout = result["all_six_ols"]["control_holdout_metrics"]
    ann = result.get("annotation_sensitivity")
    return {
        "session_id": result["session_id"],
        "two_point_all_mae_ms": _seconds_to_ms(tp_all["mae_seconds"]),
        "two_point_all_rmse_ms": _seconds_to_ms(tp_all["rmse_seconds"]),
        "two_point_all_max_abs_ms": _seconds_to_ms(tp_all["max_abs_seconds"]),
        "all_six_all_mae_ms": _seconds_to_ms(all_all["mae_seconds"]),
        "all_six_all_rmse_ms": _seconds_to_ms(all_all["rmse_seconds"]),
        "all_six_all_max_abs_ms": _seconds_to_ms(all_all["max_abs_seconds"]),
        "two_point_control_mae_ms": _seconds_to_ms(tp_control["mae_seconds"]),
        "two_point_control_rmse_ms": _seconds_to_ms(tp_control["rmse_seconds"]),
        "two_point_control_max_abs_ms": _seconds_to_ms(tp_control["max_abs_seconds"]),
        "all_six_control_holdout_mae_ms": _seconds_to_ms(all_holdout["mae_seconds"]),
        "all_six_control_holdout_rmse_ms": _seconds_to_ms(all_holdout["rmse_seconds"]),
        "all_six_control_holdout_max_abs_ms": _seconds_to_ms(all_holdout["max_abs_seconds"]),
        "two_point_anchor_sensitivity_max_delta_ms": result["two_point"]["anchor_sensitivity"]["max_abs_mapping_delta_ms"],
        "all_six_loo_sensitivity_max_delta_ms": result["all_six_ols"]["leave_one_out_sensitivity"]["max_abs_mapping_delta_ms"],
        "methods_max_mapping_delta_ms": result["method_difference"]["max_abs_mapping_delta_ms"],
        "annotation_boundary_count": ann["boundary_count"] if ann else 0,
        "annotation_mean_abs_shift_ms": ann["mean_abs_shift_ms"] if ann else None,
        "annotation_median_abs_shift_ms": ann["median_abs_shift_ms"] if ann else None,
        "annotation_max_abs_shift_ms": ann["max_abs_shift_ms"] if ann else None,
    }


def _numeric_values(rows: Sequence[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]


def aggregate_rows(rows: Sequence[dict]) -> dict:
    metric_keys = [
        "two_point_all_mae_ms", "two_point_all_rmse_ms", "two_point_all_max_abs_ms",
        "all_six_all_mae_ms", "all_six_all_rmse_ms", "all_six_all_max_abs_ms",
        "two_point_control_mae_ms", "two_point_control_rmse_ms", "two_point_control_max_abs_ms",
        "all_six_control_holdout_mae_ms", "all_six_control_holdout_rmse_ms", "all_six_control_holdout_max_abs_ms",
        "two_point_anchor_sensitivity_max_delta_ms", "all_six_loo_sensitivity_max_delta_ms",
        "methods_max_mapping_delta_ms", "annotation_mean_abs_shift_ms", "annotation_median_abs_shift_ms", "annotation_max_abs_shift_ms",
    ]
    aggregates = {}
    for key in metric_keys:
        values = _numeric_values(rows, key)
        aggregates[key] = None if not values else {
            "count": len(values),
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "max": float(max(values)),
            "min": float(min(values)),
        }
    better = equal = worse = 0
    for row in rows:
        tp = row.get("two_point_control_mae_ms")
        al = row.get("all_six_control_holdout_mae_ms")
        if tp is None or al is None:
            continue
        difference = float(al) - float(tp)
        if abs(difference) < 1e-9:
            equal += 1
        elif difference < 0:
            better += 1
        else:
            worse += 1
    return {
        "session_count": len(rows),
        "all_six_holdout_mae_better_session_count": better,
        "all_six_holdout_mae_equal_session_count": equal,
        "all_six_holdout_mae_worse_session_count": worse,
        "metrics": aggregates,
    }


def write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("Keine Ergebniszeilen für summary.csv vorhanden.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, rows: Sequence[dict], aggregate: dict) -> None:
    def fmt(value):
        return "–" if value is None else f"{float(value):.2f}"
    lines = [
        "# Videoalignment-Methodenvergleich", "",
        "Verglichen werden `LINEAR_TWO_POINT` und `LINEAR_ALL_SIX_OLS`.", "",
        "| Session | TP Control MAE ms | ALL6 Holdout MAE ms | TP Anchor-Sens. max ms | ALL6 LOO-Sens. max ms | Mapping Δ max ms | Annotation Δ max ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['session_id']} | {fmt(row['two_point_control_mae_ms'])} | {fmt(row['all_six_control_holdout_mae_ms'])} | "
            f"{fmt(row['two_point_anchor_sensitivity_max_delta_ms'])} | {fmt(row['all_six_loo_sensitivity_max_delta_ms'])} | "
            f"{fmt(row['methods_max_mapping_delta_ms'])} | {fmt(row['annotation_max_abs_shift_ms'])} |"
        )
    lines += [
        "", "## Aggregierter Kontroll-/Holdout-Vergleich", "",
        f"- Sessions insgesamt: {aggregate['session_count']}",
        f"- ALL6 mit kleinerem Holdout-MAE: {aggregate['all_six_holdout_mae_better_session_count']}",
        f"- Gleich: {aggregate['all_six_holdout_mae_equal_session_count']}",
        f"- ALL6 mit größerem Holdout-MAE: {aggregate['all_six_holdout_mae_worse_session_count']}",
        "",
        "Die Auswahl einer finalen Methode darf nicht allein auf In-Sample-Residuen beruhen. Zu berücksichtigen sind insbesondere Holdout-Fehler, Stabilität gegenüber Markerwahl sowie die praktische Verschiebung bestehender Ground-Truth-Grenzen.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_comparison(*, manual_root: Path, sync_root: Path, annotation_root: Path | None, output_root: Path, session_ids: Iterable[str] | None = None) -> dict:
    manual_files = discover_manual_marker_files(manual_root)
    selected = sorted(manual_files) if session_ids is None else [str(s).lower() for s in session_ids]
    if not selected:
        raise ValueError("Keine Sessions für den Vergleich gefunden.")

    rows = []
    for session_id in selected:
        if session_id not in manual_files:
            raise FileNotFoundError(f"Keine manuelle Auswertungsdatei für Session {session_id} gefunden.")
        result = compare_session(
            session_id=session_id,
            manual_file=manual_files[session_id],
            sync_root=sync_root,
            annotation_root=annotation_root,
        )
        rows.append(summary_row(result))
        session_output = output_root / session_id
        session_output.mkdir(parents=True, exist_ok=True)
        (session_output / "comparison.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_rows(rows)
    write_summary_csv(output_root / "summary.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps({"schema_version": 1, "aggregate": aggregate, "sessions": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_markdown(output_root / "summary.md", rows, aggregate)
    return {
        "output_root": str(output_root),
        "session_count": len(rows),
        "summary_csv": str(output_root / "summary.csv"),
        "summary_json": str(output_root / "summary.json"),
        "summary_markdown": str(output_root / "summary.md"),
        "aggregate": aggregate,
    }
