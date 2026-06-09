from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cardinality_features import make_query_keys, parse_predicates, table_combo_from_row


EQ_CALIBRATION_KEY_ORDER = (
    "table_eq_signature",
    "table_predicate_signature",
    "eq_signature",
    "single_table_eq_col",
)
EQ_CALIBRATION_MIN_COUNT = 8
EQ_CALIBRATION_SMOOTHING = 16.0
EQ_CALIBRATION_MAX_ABS_SHIFT = 1.2


@dataclass(frozen=True)
class EqualityResidualCalibrator:
    shifts: dict[str, dict[str, float]]
    counts: dict[str, dict[str, int]]


def equality_calibration_keys(row: dict[str, object]) -> dict[str, str]:
    predicates = parse_predicates(row["Predicates"])
    eq_cols = [col for col, op, _ in predicates if op == "="]
    if not eq_cols:
        return {}

    query_keys = make_query_keys(row)
    table_combo = table_combo_from_row(row)
    eq_signature = "|".join(f"{col}:=" for col in eq_cols)
    keys = {
        "table_eq_signature": f"tables={table_combo};eq={eq_signature}",
        "table_predicate_signature": query_keys["tables_predicates"],
        "eq_signature": f"eq={eq_signature}",
    }
    if len(eq_cols) == 1:
        keys["single_table_eq_col"] = f"tables={table_combo};eq_col={eq_cols[0]}"
    return keys


def fit_equality_residual_calibrator(
    train_df: pd.DataFrame,
    true_log_y: np.ndarray,
    oof_log_pred: np.ndarray,
    min_count: int = EQ_CALIBRATION_MIN_COUNT,
) -> EqualityResidualCalibrator:
    residuals_by_key: dict[str, dict[str, list[float]]] = {
        key_name: defaultdict(list) for key_name in EQ_CALIBRATION_KEY_ORDER
    }
    residuals = np.asarray(true_log_y, dtype=float) - np.asarray(oof_log_pred, dtype=float)

    for row, residual in zip(train_df.to_dict("records"), residuals):
        for key_name, key_value in equality_calibration_keys(row).items():
            residuals_by_key[key_name][key_value].append(float(residual))

    shifts: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for key_name, grouped_residuals in residuals_by_key.items():
        shifts[key_name] = {}
        counts[key_name] = {}
        for key_value, values in grouped_residuals.items():
            if len(values) < min_count:
                continue
            raw_shift = float(np.median(np.asarray(values, dtype=float)))
            shrink = len(values) / (len(values) + EQ_CALIBRATION_SMOOTHING)
            shift = float(np.clip(raw_shift * shrink, -EQ_CALIBRATION_MAX_ABS_SHIFT, EQ_CALIBRATION_MAX_ABS_SHIFT))
            shifts[key_name][key_value] = shift
            counts[key_name][key_value] = len(values)

    return EqualityResidualCalibrator(shifts=shifts, counts=counts)


def apply_equality_residual_calibrator(
    target_df: pd.DataFrame,
    base_log_pred: np.ndarray,
    calibrator: EqualityResidualCalibrator,
) -> tuple[np.ndarray, dict[str, float]]:
    base = np.asarray(base_log_pred, dtype=float)
    calibrated = base.copy()
    applied_by_key = {key_name: 0.0 for key_name in EQ_CALIBRATION_KEY_ORDER}

    for row_idx, row in enumerate(target_df.to_dict("records")):
        keys = equality_calibration_keys(row)
        for key_name in EQ_CALIBRATION_KEY_ORDER:
            key_value = keys.get(key_name)
            if key_value is None:
                continue
            shift = calibrator.shifts[key_name].get(key_value)
            if shift is None:
                continue
            calibrated[row_idx] = base[row_idx] + shift
            applied_by_key[key_name] += 1.0
            break

    row_count = max(float(len(target_df)), 1.0)
    metrics = {
        "eq_residual_calibration_coverage": float(np.mean(calibrated != base)) if len(base) else 0.0,
        "eq_residual_calibration_group_count": float(
            sum(len(group_shifts) for group_shifts in calibrator.shifts.values())
        ),
    }
    for key_name, count in applied_by_key.items():
        metrics[f"eq_residual_calibration_{key_name}_coverage"] = count / row_count
    return calibrated, metrics
