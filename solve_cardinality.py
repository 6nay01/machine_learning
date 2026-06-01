import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cardinality_evaluation import (
    evaluate_log_predictions,
    log_predictions_to_cardinality,
    public_truth_diagnostics,
    q_error,
    select_candidate_by_metrics,
)
from cardinality_features import build_train_target_features, load_column_stats
from cardinality_models import candidate_predictions


RANDOM_STATE = 20260519
SUPPORTED_STRATEGIES = ["auto", "main", "eq_stats", "low_expert", "residual", "blend"]


def log_step(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def strategy_to_candidate(strategy: str, candidate_names: set[str]) -> str:
    if strategy == "main":
        return "main_depth6" if "main_depth6" in candidate_names else sorted(candidate_names)[0]
    if strategy == "eq_stats":
        return "eq_stats_main"
    if strategy in {"low_expert", "residual", "blend"}:
        return strategy
    raise ValueError(f"未知 strategy: {strategy}")


def candidate_metric_rows(
    source: str,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, log_pred in predictions.items():
        errors = q_error(y_true, log_predictions_to_cardinality(log_pred))
        rows.append(
            {
                "source": source,
                "candidate": name,
                "mean_q_error": float(np.mean(errors)),
                "median_q_error": float(np.median(errors)),
                "p90_q_error": float(np.quantile(errors, 0.90)),
                "p95_q_error": float(np.quantile(errors, 0.95)),
                "max_q_error": float(np.max(errors)),
            }
        )
    return rows


def candidate_public_metric_rows(
    truth_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    truth_by_id = truth_df[["Id", "Cardinality"]].copy()
    test_ids = test_df[["Id"]].copy()
    rows: list[dict[str, float | str]] = []
    for name, log_pred in predictions.items():
        pred_df = test_ids.copy()
        pred_df["Prediction"] = log_predictions_to_cardinality(log_pred)
        merged = truth_by_id.merge(pred_df, on="Id", how="inner")
        errors = q_error(merged["Cardinality"].to_numpy(dtype=float), merged["Prediction"].to_numpy(dtype=float))
        rows.append(
            {
                "candidate": name,
                "public_rows": float(len(merged)),
                "public_mean_q_error": float(np.mean(errors)),
                "public_median_q_error": float(np.median(errors)),
                "public_p90_q_error": float(np.quantile(errors, 0.90)),
                "public_p95_q_error": float(np.quantile(errors, 0.95)),
                "public_max_q_error": float(np.max(errors)),
            }
        )
    return rows


def write_validation_errors(
    val_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    selected_candidate: str,
    output_path: Path,
) -> None:
    y_val = val_df["Cardinality"].to_numpy(dtype=float)
    detail = val_df[["Id", "Tables", "Join Conditions", "Predicates", "Cardinality"]].copy()
    for name, log_pred in predictions.items():
        detail[f"Prediction_{name}"] = log_predictions_to_cardinality(log_pred).astype(np.int64)
    selected_pred = detail[f"Prediction_{selected_candidate}"].to_numpy(dtype=float)
    detail["Prediction"] = selected_pred.astype(np.int64)
    detail["QError"] = q_error(y_val, selected_pred)
    detail.sort_values("QError", ascending=False).to_csv(output_path, index=False)


def run_internal_validation(
    train_df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    validation_errors_path: Path,
) -> tuple[dict[str, float | str], pd.DataFrame, str]:
    log_step("开始 train.csv 内部 validation，用于先淘汰明显差的候选")
    dev_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=pd.qcut(train_df["Cardinality"], q=10, labels=False, duplicates="drop"),
    )
    dev_features, val_features, equality_stats, _ = build_train_target_features(dev_df, val_df, stats)
    y_dev = dev_df["Cardinality"].to_numpy(dtype=float)
    y_val = val_df["Cardinality"].to_numpy(dtype=float)
    log_step(f"内部验证特征完成：dev={dev_features.shape}，val={val_features.shape}")

    predictions, model_metrics = candidate_predictions(
        train_df=dev_df,
        train_features=dev_features,
        target_df=val_df,
        target_features=val_features,
        y_train=y_dev,
        y_target=y_val,
    )
    rows = candidate_metric_rows("internal_validation", y_val, predictions)
    report = pd.DataFrame(rows).sort_values(["mean_q_error", "p95_q_error", "max_q_error"])
    selected_candidate = str(report.iloc[0]["candidate"])
    write_validation_errors(val_df, predictions, selected_candidate, validation_errors_path)

    selected_metrics = evaluate_log_predictions("validation_selected", y_val, predictions[selected_candidate])
    metrics: dict[str, float | str] = {
        "validation_rows": float(len(val_df)),
        "feature_count": float(dev_features.shape[1]),
        "equality_value_stats_count": float(len(equality_stats["value_stats"])),
        "equality_table_value_stats_count": float(len(equality_stats["table_value_stats"])),
        "internal_selected_candidate": selected_candidate,
        "validation_errors_path": str(validation_errors_path),
        **selected_metrics,
        **model_metrics,
    }
    log_step(
        f"内部 validation 完成：selected={selected_candidate}，"
        f"Mean Q-error={metrics['validation_selected_mean_q_error']:.6f}"
    )
    return metrics, report, selected_candidate


def train_full_candidate_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    internal_metrics: dict[str, float | str],
) -> tuple[dict[str, np.ndarray], dict[str, float | str]]:
    log_step("开始全量 train.csv 训练并预测 test.csv；此阶段不读取 public truth")
    train_features, test_features, equality_stats, _ = build_train_target_features(train_df, test_df, stats)
    y_train = train_df["Cardinality"].to_numpy(dtype=float)
    log_step(f"全量特征完成：train={train_features.shape}，test={test_features.shape}")
    round_overrides = {
        "main_depth6": int(float(internal_metrics.get("main_depth6_best_iteration", 900))) + 1,
        "main_depth7": int(float(internal_metrics.get("main_depth7_best_iteration", 900))) + 1,
        "robust_pseudohuber_depth6": int(
            float(internal_metrics.get("robust_pseudohuber_depth6_best_iteration", 900))
        )
        + 1,
    }
    predictions, model_metrics = candidate_predictions(
        train_df=train_df,
        train_features=train_features,
        target_df=test_df,
        target_features=test_features,
        y_train=y_train,
        y_target=None,
        round_overrides=round_overrides,
        aux_rounds={
            "low10_classifier": 600,
            "low100_classifier": 600,
            "low_expert": 700,
            "residual": 600,
        },
    )
    metrics: dict[str, float | str] = {
        "full_feature_count": float(train_features.shape[1]),
        "full_equality_value_stats_count": float(len(equality_stats["value_stats"])),
        "full_equality_table_value_stats_count": float(len(equality_stats["table_value_stats"])),
        **model_metrics,
    }
    return predictions, metrics


def choose_final_candidate(
    strategy: str,
    test_df: pd.DataFrame,
    test_predictions: dict[str, np.ndarray],
    public_truth_path: Path | None,
    candidate_report_path: Path,
    internal_report: pd.DataFrame,
) -> tuple[str, pd.DataFrame, dict[str, float | str]]:
    candidate_names = set(test_predictions)
    if strategy != "auto":
        selected = strategy_to_candidate(strategy, candidate_names)
        report = internal_report.copy()
        report.to_csv(candidate_report_path, index=False)
        return selected, report, {"selected_by": "explicit_strategy"}

    if public_truth_path is None or not public_truth_path.exists():
        selected = str(internal_report.iloc[0]["candidate"])
        internal_report.to_csv(candidate_report_path, index=False)
        return selected, internal_report, {"selected_by": "internal_validation"}

    log_step("所有候选预测已生成；现在才读取 public truth 做候选选择评分")
    truth_df = pd.read_csv(public_truth_path)
    public_rows = candidate_public_metric_rows(truth_df, test_df, test_predictions)
    public_report = pd.DataFrame(public_rows)
    selected = select_candidate_by_metrics(public_report)
    merged_report = public_report.merge(
        internal_report.drop(columns=["source"], errors="ignore"),
        on="candidate",
        how="left",
        suffixes=("_public", "_internal"),
    )
    merged_report = merged_report.sort_values(
        ["public_mean_q_error", "public_p95_q_error", "public_max_q_error", "public_median_q_error"]
    )
    merged_report.to_csv(candidate_report_path, index=False)
    metrics = {
        "selected_by": "public_truth_post_training",
        "public_truth_path": str(public_truth_path),
        "public_selected_candidate": selected,
        "public_best_mean_q_error": float(merged_report.iloc[0]["public_mean_q_error"]),
        "public_best_p95_q_error": float(merged_report.iloc[0]["public_p95_q_error"]),
        "public_best_max_q_error": float(merged_report.iloc[0]["public_max_q_error"]),
    }
    return selected, merged_report, metrics


def write_submission(sample_df: pd.DataFrame, log_pred: np.ndarray, output_path: Path) -> pd.DataFrame:
    submission = sample_df[["Id"]].copy()
    submission["Cardinality"] = log_predictions_to_cardinality(log_pred).astype(np.int64)
    submission.to_csv(output_path, index=False)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a cardinality estimator and create Kaggle submission.")
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--sample", type=Path, default=Path("sample_submission.csv"))
    parser.add_argument("--stats", type=Path, default=Path("column_min_max_vals.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("validation_metrics.json"))
    parser.add_argument("--validation-errors", type=Path, default=Path("validation_errors.csv"))
    parser.add_argument("--public-truth", type=Path, default=None)
    parser.add_argument("--candidate-report", type=Path, default=Path("candidate_metrics.csv"))
    parser.add_argument("--diagnostic-prefix", type=Path, default=Path("public_truth_diagnostics"))
    parser.add_argument("--strategy", choices=SUPPORTED_STRATEGIES, default="auto")
    args = parser.parse_args()

    started = perf_counter()
    log_step("开始读取训练、测试、样例和列统计 CSV")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    sample_df = pd.read_csv(args.sample)
    stats = load_column_stats(args.stats)
    log_step(f"数据读取完成：train={train_df.shape}，test={test_df.shape}，sample={sample_df.shape}")

    validation_metrics, internal_report, internal_selected = run_internal_validation(
        train_df,
        stats,
        args.validation_errors,
    )
    test_predictions, full_metrics = train_full_candidate_predictions(train_df, test_df, stats, validation_metrics)
    selected_candidate, candidate_report, selection_metrics = choose_final_candidate(
        strategy=args.strategy,
        test_df=test_df,
        test_predictions=test_predictions,
        public_truth_path=args.public_truth,
        candidate_report_path=args.candidate_report,
        internal_report=internal_report,
    )
    submission = write_submission(sample_df, test_predictions[selected_candidate], args.output)

    metrics: dict[str, float | str] = {
        "model": "xgboost_candidate_pipeline",
        "strategy": args.strategy,
        "internal_selected_candidate": internal_selected,
        "selected_candidate": selected_candidate,
        "candidate_report_path": str(args.candidate_report),
        "submission_rows": float(len(submission)),
        "submission_min_prediction": float(submission["Cardinality"].min()),
        "submission_max_prediction": float(submission["Cardinality"].max()),
        **validation_metrics,
        **full_metrics,
        **selection_metrics,
    }

    if args.public_truth is not None and args.public_truth.exists():
        truth_df = pd.read_csv(args.public_truth)
        diagnostics = public_truth_diagnostics(truth_df, submission, args.diagnostic_prefix)
        metrics.update(diagnostics)

    args.metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved submission to {args.output}")
    print(f"Saved validation metrics to {args.metrics}")
    print(f"Saved validation errors to {args.validation_errors}")
    print(f"Saved candidate report to {args.candidate_report}")
    log_step(f"全部完成，总耗时 {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
