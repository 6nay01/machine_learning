import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cardinality_features import parse_predicates, parse_tables, predicate_signature_from_predicates


MAX_LOG_PREDICTION = 25.0


def q_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.maximum(y_true.astype(float), 1.0)
    y_pred = np.maximum(y_pred.astype(float), 1.0)
    return np.maximum(y_pred / y_true, y_true / y_pred)


def log_predictions_to_cardinality(log_pred: np.ndarray) -> np.ndarray:
    safe_log = np.nan_to_num(log_pred, nan=0.0, posinf=MAX_LOG_PREDICTION, neginf=0.0)
    return np.maximum(np.rint(np.expm1(np.clip(safe_log, 0.0, MAX_LOG_PREDICTION))), 1)


def summarize_q_errors(prefix: str, errors: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean_q_error": float(np.mean(errors)),
        f"{prefix}_median_q_error": float(np.median(errors)),
        f"{prefix}_p90_q_error": float(np.quantile(errors, 0.90)),
        f"{prefix}_p95_q_error": float(np.quantile(errors, 0.95)),
        f"{prefix}_max_q_error": float(np.max(errors)),
    }


def evaluate_log_predictions(prefix: str, y_true: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    return summarize_q_errors(prefix, q_error(y_true, log_predictions_to_cardinality(log_pred)))


def cardinality_bucket(value: float) -> str:
    if value <= 10:
        return "<=10"
    if value <= 100:
        return "11-100"
    if value <= 1000:
        return "101-1k"
    if value <= 10000:
        return "1k-10k"
    if value <= 100000:
        return "10k-100k"
    return ">100k"


def table_combo_from_tables(value: object) -> str:
    _, aliases = parse_tables(value)
    return "|".join(aliases)


def equality_columns_from_predicates(value: object) -> list[str]:
    return [col for col, op, _ in parse_predicates(value) if op == "="] or ["NONE"]


def grouped_error_frame(rows: list[dict[str, object]], group_col: str) -> pd.DataFrame:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_col])].append(float(row["QError"]))

    result_rows = []
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        result_rows.append(
            {
                group_col: key,
                "count": int(len(arr)),
                "mean_q_error": float(np.mean(arr)),
                "median_q_error": float(np.median(arr)),
                "p95_q_error": float(np.quantile(arr, 0.95)),
                "max_q_error": float(np.max(arr)),
            }
        )
    return pd.DataFrame(result_rows).sort_values(["mean_q_error", "count"], ascending=[False, False])


def public_truth_diagnostics(
    truth_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    output_prefix: Path,
    top_n: int = 20,
) -> dict[str, float | str]:
    merged = truth_df.merge(submission_df, on="Id", how="inner", suffixes=("", "_pred"))
    y_true = merged["Cardinality"].to_numpy(dtype=float)
    y_pred = merged["Cardinality_pred"].to_numpy(dtype=float)
    errors = q_error(y_true, y_pred)

    detail = merged[["Id", "Tables", "Join Conditions", "Predicates", "Cardinality", "Cardinality_pred"]].copy()
    detail = detail.rename(columns={"Cardinality_pred": "Prediction"})
    detail["QError"] = errors
    detail["CardinalityBucket"] = [cardinality_bucket(value) for value in y_true]
    detail["TableCombo"] = [table_combo_from_tables(value) for value in detail["Tables"]]
    detail["PredicateSignature"] = [
        predicate_signature_from_predicates(parse_predicates(value)) for value in detail["Predicates"]
    ]

    eq_rows = []
    for row in detail.to_dict("records"):
        for eq_col in equality_columns_from_predicates(row["Predicates"]):
            eq_row = dict(row)
            eq_row["EqualityColumn"] = eq_col
            eq_rows.append(eq_row)

    top_errors = detail.sort_values("QError", ascending=False).head(top_n)
    top_errors.to_csv(output_prefix.with_name(f"{output_prefix.name}_top_errors.csv"), index=False)
    grouped_error_frame(detail.to_dict("records"), "CardinalityBucket").to_csv(
        output_prefix.with_name(f"{output_prefix.name}_by_bucket.csv"), index=False
    )
    grouped_error_frame(detail.to_dict("records"), "TableCombo").to_csv(
        output_prefix.with_name(f"{output_prefix.name}_by_table_combo.csv"), index=False
    )
    grouped_error_frame(detail.to_dict("records"), "PredicateSignature").to_csv(
        output_prefix.with_name(f"{output_prefix.name}_by_predicate_signature.csv"), index=False
    )
    grouped_error_frame(eq_rows, "EqualityColumn").to_csv(
        output_prefix.with_name(f"{output_prefix.name}_by_equality_column.csv"), index=False
    )

    metrics = summarize_q_errors("public_truth", errors)
    metrics["public_truth_rows"] = float(len(detail))
    output_prefix.with_name(f"{output_prefix.name}_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def select_candidate_by_metrics(candidate_metrics: pd.DataFrame) -> str:
    ranked = candidate_metrics.sort_values(
        [
            "public_mean_q_error",
            "public_p95_q_error",
            "public_max_q_error",
            "public_median_q_error",
        ],
        ascending=True,
    )
    return str(ranked.iloc[0]["candidate"])
