"""训练主流程与提交生成脚本。

此模块提供完整的训练流程：
- 在 train.csv 上做内部切分进行候选模型的验证
- 使用全部训练数据训练并生成对 test.csv 的多个候选预测
- 选择固定的提交候选并输出 submission.csv 与评估指标

脚本可以从命令行接收输入/输出路径，用于 Kaggke 风格的实验流程。
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cardinality_evaluation import (
    log_predictions_to_cardinality,
    q_error,
)
from cardinality_features import build_train_target_features, load_column_stats
from cardinality_models import UNSEEN_EQ_SWITCH_CANDIDATE, candidate_predictions


RANDOM_STATE = 20260519
BLEND_PREFIX = "blend_low_"
FINAL_SUBMISSION_CANDIDATE = UNSEEN_EQ_SWITCH_CANDIDATE


def log_step(message: str) -> None:
    """打印带进度前缀的日志。"""
    print(f"[progress] {message}", flush=True)


def is_blend_candidate(candidate: str) -> bool:
    """判断候选名是否属于 blend 比例模型。"""
    return candidate.startswith(BLEND_PREFIX)


def blend_weight_metrics(candidate: str) -> dict[str, float]:
    """从 blend 候选名中提取 low_expert 和另一分支的权重。"""
    if not is_blend_candidate(candidate):
        return {}
    low_expert_weight = int(candidate.removeprefix(BLEND_PREFIX)) / 100.0
    return {
        "selected_blend_low_expert_weight": low_expert_weight,
        "selected_blend_other_weight": 1.0 - low_expert_weight,
    }

def candidate_metric_rows(
    source: str,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    """为每个候选模型生成一行内部评估指标。"""
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

def write_validation_errors(
    val_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """写出内部验证集上按 Q-error 排序的逐条误差明细。"""
    y_val = val_df["Cardinality"].to_numpy(dtype=float)
    detail = val_df[["Id", "Tables", "Join Conditions", "Predicates", "Cardinality"]].copy()
    for name, log_pred in predictions.items():
        detail[f"Prediction_{name}"] = log_predictions_to_cardinality(log_pred).astype(np.int64)
    detail["QError"] = q_error(y_val, detail[f"Prediction_{FINAL_SUBMISSION_CANDIDATE}"].to_numpy(dtype=float))
    detail.sort_values("QError", ascending=False).to_csv(output_path, index=False)


def run_internal_validation(
    train_df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    validation_errors_path: Path,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """在 train.csv 内部切分验证集，评估候选并返回报告。"""
    log_step("开始 train.csv 内部 validation")
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
    # selected_candidate = str(report.iloc[0]["candidate"])
    write_validation_errors(val_df, predictions, validation_errors_path)

    # selected_metrics = evaluate_log_predictions("validation_selected", y_val, predictions[selected_candidate])
    metrics: dict[str, float | str] = {
        "validation_rows": float(len(val_df)),
        "feature_count": float(dev_features.shape[1]),
        "equality_value_stats_count": float(len(equality_stats["value_stats"])),
        "equality_table_value_stats_count": float(len(equality_stats["table_value_stats"])),
        # "internal_selected_candidate": selected_candidate,
        # "validation_errors_path": str(validation_errors_path),
        # **selected_metrics,
        **model_metrics,
    }
    log_step(
        f"内部 validation 完成"
    )
    return metrics, report


def train_full_candidate_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    internal_metrics: dict[str, float | str],
) -> tuple[dict[str, np.ndarray], dict[str, float | str]]:
    """使用全量训练数据训练所有候选并生成 test 预测。"""
    log_step("开始全量 train.csv 训练并预测 test.csv")
    train_features, test_features, equality_stats, _ = build_train_target_features(train_df, test_df, stats)
    y_train = train_df["Cardinality"].to_numpy(dtype=float)
    log_step(f"全量特征完成：train={train_features.shape}，test={test_features.shape}")
    round_overrides = {
        "main_depth": int(float(internal_metrics.get("main_depth6_best_iteration", 900))) + 1,
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


def require_final_submission_candidate(test_predictions: dict[str, np.ndarray]) -> str:
    """校验最终提交所需的固定 blend 候选存在。"""
    if FINAL_SUBMISSION_CANDIDATE not in test_predictions:
        raise ValueError(f"缺少固定提交候选: {FINAL_SUBMISSION_CANDIDATE}")
    return FINAL_SUBMISSION_CANDIDATE


def select_submission_candidate(
    test_predictions: dict[str, np.ndarray],
    candidate_report_path: Path,
    internal_report: pd.DataFrame,
) -> tuple[str, dict[str, float | str]]:
    """写出内部验证报告，并固定返回最终提交候选。"""
    internal_report.to_csv(candidate_report_path, index=False)
    fixed_candidate = require_final_submission_candidate(test_predictions)
    metrics = {
        "selected_by": "fixed_submission_candidate",
        "internal_best_candidate": str(internal_report.iloc[0]["candidate"]),
        **blend_weight_metrics(fixed_candidate),
    }
    return fixed_candidate, metrics


def write_submission(sample_df: pd.DataFrame, log_pred: np.ndarray, output_path: Path) -> pd.DataFrame:
    """将对数预测转换为整数基数并写出 submission.csv。"""
    submission = sample_df[["Id"]].copy()
    submission["Cardinality"] = log_predictions_to_cardinality(log_pred).astype(np.int64)
    submission.to_csv(output_path, index=False)
    return submission


def main() -> None:
    """执行训练、候选评估和最终提交流程。"""
    parser = argparse.ArgumentParser(description="Train a cardinality estimator and create Kaggle submission.")
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--sample", type=Path, default=Path("sample_submission.csv"))
    parser.add_argument("--stats", type=Path, default=Path("column_min_max_vals.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("validation_metrics.json"))
    parser.add_argument("--validation-errors", type=Path, default=Path("validation_errors.csv"))
    parser.add_argument("--candidate-report", type=Path, default=Path("candidate_metrics.csv"))
    args = parser.parse_args()

    started = perf_counter()
    log_step("开始读取训练、测试、样例和列统计 CSV")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    sample_df = pd.read_csv(args.sample)
    stats = load_column_stats(args.stats)
    log_step(f"数据读取完成：train={train_df.shape}，test={test_df.shape}，sample={sample_df.shape}")

    validation_metrics, internal_report = run_internal_validation(
        train_df,
        stats,
        args.validation_errors,
    )
    test_predictions, full_metrics = train_full_candidate_predictions(train_df, test_df, stats, validation_metrics)
    selected_candidate, selection_metrics = select_submission_candidate(
        test_predictions=test_predictions,
        candidate_report_path=args.candidate_report,
        internal_report=internal_report,
    )
    submission = write_submission(sample_df, test_predictions[selected_candidate], args.output)

    metrics: dict[str, float | str] = {
        "model": "xgboost_candidate_pipeline",
        "selected_candidate": selected_candidate,
        "candidate_report_path": str(args.candidate_report),
        "submission_rows": float(len(submission)),
        "submission_min_prediction": float(submission["Cardinality"].min()),
        "submission_max_prediction": float(submission["Cardinality"].max()),
        **validation_metrics,
        **full_metrics,
        **selection_metrics,
    }

    args.metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved submission to {args.output}")
    print(f"Saved validation metrics to {args.metrics}")
    print(f"Saved validation errors to {args.validation_errors}")
    print(f"Saved candidate report to {args.candidate_report}")
    log_step(f"全部完成，总耗时 {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
