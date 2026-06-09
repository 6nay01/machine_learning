import argparse
import json
from itertools import islice, product
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from cardinality_features import build_train_target_features, load_column_stats
from cardinality_models import (
    XGBRegressorConfig,
    candidate_predictions,
    default_tail_expert_config,
    make_xgb_params,
)
from solve_cardinality import (
    blend_weight_metrics,
    candidate_public_metric_rows,
    select_candidate_by_metrics,
    write_submission,
)


DEFAULT_SEARCH_SPACE: dict[str, list[int | float]] = {
    "main_eta": [0.030, 0.035],
    "main_max_depth": [6, 7],
    "main_min_child_weight": [1.5],
    "main_subsample": [0.92],
    "main_colsample_bytree": [0.92],
    "main_lambda": [1.2],
    "main_alpha": [0.02],
    "main_num_boost_round": [2600, 3200, 3800],
    "low_eta": [0.035, 0.040],
    "low_max_depth": [4],
    "low_min_child_weight": [1.0],
    "low_subsample": [0.95],
    "low_colsample_bytree": [0.95],
    "low_lambda": [1.0],
    "low_alpha": [0.02],
    "low_num_boost_round": [900, 1200, 1500],
    "tail_num_boost_round": [900, 1200, 1500],
    "residual_eta": [0.025, 0.030],
    "residual_max_depth": [4],
    "residual_min_child_weight": [2.0],
    "residual_subsample": [0.90],
    "residual_colsample_bytree": [0.90],
    "residual_lambda": [3.0],
    "residual_alpha": [0.10],
    "residual_num_boost_round": [500, 700, 900],
    "low_classifier_rounds": [500, 800],
    "tail_classifier_rounds": [500, 800],
    "blend_low_expert_weight_percent": [0, 20, 40, 60, 80, 100],
    "dynamic_low_gate_scale": [0.8, 1.0, 1.2],
    "dynamic_tail_gate_scale": [0.8, 1.0, 1.2, 1.5],
    "dynamic_main_gate_scale": [0.8, 1.0, 1.2],
    "dynamic_low_output_scale": [0.8, 1.0, 1.2],
    "dynamic_tail_output_scale": [0.8, 1.0, 1.2],
    "dynamic_residual_mix": [0.7, 1.0],
    "dynamic_raw_mix": [0.0, 0.3],
}
SEARCH_SPACE_KEYS = tuple(DEFAULT_SEARCH_SPACE)
RESULT_SORT_KEYS = ("mean_q_error", "p95_q_error", "max_q_error", "median_q_error")
BASELINE_TRIAL: dict[str, int | float] = {
    "main_eta": 0.03,
    "main_max_depth": 7,
    "main_min_child_weight": 1.5,
    "main_subsample": 0.92,
    "main_colsample_bytree": 0.92,
    "main_lambda": 1.2,
    "main_alpha": 0.02,
    "main_num_boost_round": 3200,
    "low_eta": 0.04,
    "low_max_depth": 4,
    "low_min_child_weight": 1.0,
    "low_subsample": 0.95,
    "low_colsample_bytree": 0.95,
    "low_lambda": 1.0,
    "low_alpha": 0.02,
    "low_num_boost_round": 1200,
    "tail_num_boost_round": 1200,
    "residual_eta": 0.03,
    "residual_max_depth": 4,
    "residual_min_child_weight": 2.0,
    "residual_subsample": 0.90,
    "residual_colsample_bytree": 0.90,
    "residual_lambda": 3.0,
    "residual_alpha": 0.10,
    "residual_num_boost_round": 700,
    "low_classifier_rounds": 800,
    "tail_classifier_rounds": 800,
    "blend_low_expert_weight_percent": 20,
    "dynamic_low_gate_scale": 1.0,
    "dynamic_tail_gate_scale": 1.0,
    "dynamic_main_gate_scale": 1.0,
    "dynamic_low_output_scale": 1.0,
    "dynamic_tail_output_scale": 1.0,
    "dynamic_residual_mix": 1.0,
    "dynamic_raw_mix": 0.0,
}
SEARCH_PROFILES: dict[str, dict[str, list[int | float]]] = {
    "full": DEFAULT_SEARCH_SPACE,
    "gate_focus": {
        "low_classifier_rounds": [500, 800],
        "tail_classifier_rounds": [500, 800, 1100],
        "blend_low_expert_weight_percent": [0, 10, 20, 30, 40],
        "dynamic_low_gate_scale": [0.8, 1.0, 1.2],
        "dynamic_tail_gate_scale": [0.8, 1.0, 1.2, 1.5],
        "dynamic_main_gate_scale": [0.8, 1.0, 1.2],
        "dynamic_low_output_scale": [0.8, 1.0, 1.2],
        "dynamic_tail_output_scale": [0.8, 1.0, 1.2],
        "dynamic_residual_mix": [0.7, 1.0],
        "dynamic_raw_mix": [0.0, 0.2, 0.3],
    },
    "round_focus": {
        "main_num_boost_round": [2800, 3200, 3600],
        "low_num_boost_round": [900, 1200, 1500],
        "tail_num_boost_round": [900, 1200, 1500],
        "residual_num_boost_round": [500, 700, 900],
        "low_classifier_rounds": [500, 800],
        "tail_classifier_rounds": [800, 1100],
    },
}


def log_step(message: str) -> None:
    print(f"[grid-search] {message}", flush=True)


def iter_search_trials(
    search_space: dict[str, list[int | float]],
    max_trials: int | None = None,
) -> list[dict[str, int | float]]:
    keys = list(search_space)
    values_product = product(*(search_space[key] for key in keys))
    if max_trials is not None:
        values_product = islice(values_product, max_trials)
    return [dict(zip(keys, values, strict=True)) for values in values_product]


def build_profile_search_space(profile: str) -> dict[str, list[int | float]]:
    if profile not in SEARCH_PROFILES:
        raise ValueError(f"未知搜索 profile: {profile}")
    profile_space = SEARCH_PROFILES[profile]
    search_space: dict[str, list[int | float]] = {}
    for key in SEARCH_SPACE_KEYS:
        if key in profile_space:
            search_space[key] = profile_space[key]
        else:
            search_space[key] = [BASELINE_TRIAL[key]]
    return search_space


def count_total_trials(search_space: dict[str, list[int | float]]) -> int:
    total = 1
    for values in search_space.values():
        total *= len(values)
    return total


def canonicalize_trial_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".12g")
    return str(value)


def trial_key_from_values(trial: dict[str, Any], search_space_keys: tuple[str, ...] = SEARCH_SPACE_KEYS) -> tuple[str, ...]:
    return tuple(canonicalize_trial_scalar(trial[key]) for key in search_space_keys)


def format_duration(seconds: float) -> str:
    normalized_seconds = max(float(seconds), 0.0)
    if normalized_seconds < 60.0:
        return f"{normalized_seconds:.1f}s"

    rounded_seconds = int(round(normalized_seconds))
    minutes, second_part = divmod(rounded_seconds, 60)
    if minutes < 60:
        return f"{minutes}m{second_part:02d}s"

    hours, minute_part = divmod(minutes, 60)
    return f"{hours}h{minute_part:02d}m{second_part:02d}s"


def validate_resume_trials(
    completed_trials: dict[tuple[str, ...], dict[str, Any]],
    trials: list[dict[str, int | float]],
    search_space_keys: tuple[str, ...] = SEARCH_SPACE_KEYS,
) -> None:
    expected_trial_keys = {
        trial_key_from_values(trial, search_space_keys)
        for trial in trials
    }
    unexpected_trial_keys = [
        completed_key
        for completed_key in completed_trials
        if completed_key not in expected_trial_keys
    ]
    if not unexpected_trial_keys:
        return

    preview = ", ".join(
        "/".join(key[:3]) + ("/..." if len(key) > 3 else "")
        for key in unexpected_trial_keys[:3]
    )
    raise ValueError(
        "已有 trial 文件包含不属于当前搜索空间的记录，无法 resume。"
        f"请检查 --profile/--max-trials/--output-dir 是否一致；异常记录数={len(unexpected_trial_keys)}；示例={preview}"
    )


def build_pending_trials(
    trials: list[dict[str, int | float]],
    completed_trials: dict[tuple[str, ...], dict[str, Any]],
    search_space_keys: tuple[str, ...] = SEARCH_SPACE_KEYS,
) -> list[tuple[int, dict[str, int | float]]]:
    return [
        (index, trial)
        for index, trial in enumerate(trials, start=1)
        if trial_key_from_values(trial, search_space_keys) not in completed_trials
    ]


def result_sort_tuple(result: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(result[key]) for key in RESULT_SORT_KEYS)


def load_completed_trials(
    trials_path: Path,
    search_space_keys: tuple[str, ...] = SEARCH_SPACE_KEYS,
) -> tuple[list[dict[str, Any]], dict[tuple[str, ...], dict[str, Any]]]:
    if not trials_path.exists():
        return [], {}
    existing_df = pd.read_csv(trials_path)
    if existing_df.empty:
        return [], {}

    missing_columns = [key for key in search_space_keys if key not in existing_df.columns]
    if missing_columns:
        raise ValueError(
            f"已有 trial 文件缺少必要列，无法 resume：{', '.join(missing_columns)}"
        )

    existing_rows = existing_df.to_dict("records")
    completed: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in existing_rows:
        key = trial_key_from_values(row, search_space_keys)
        if key in completed:
            raise ValueError(f"已有 trial 文件存在重复参数组合，无法 resume：{key}")
        completed[key] = row
    return existing_rows, completed


def select_best_result(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (*result_sort_tuple(row), float(row.get("trial_id", float("inf")))),
    )


def make_main_config(trial: dict[str, int | float]) -> XGBRegressorConfig:
    return XGBRegressorConfig(
        name="main_tuned",
        params=make_xgb_params(
            eta=float(trial["main_eta"]),
            max_depth=int(trial["main_max_depth"]),
            min_child_weight=float(trial["main_min_child_weight"]),
            subsample=float(trial["main_subsample"]),
            colsample_bytree=float(trial["main_colsample_bytree"]),
            **{
                "lambda": float(trial["main_lambda"]),
                "alpha": float(trial["main_alpha"]),
            },
        ),
        num_boost_round=int(trial["main_num_boost_round"]),
        early_stopping_rounds=0,
        verbose_eval=150,
    )


def make_low_expert_config(trial: dict[str, int | float]) -> XGBRegressorConfig:
    return XGBRegressorConfig(
        name="low_expert_tuned",
        params=make_xgb_params(
            eta=float(trial["low_eta"]),
            max_depth=int(trial["low_max_depth"]),
            min_child_weight=float(trial["low_min_child_weight"]),
            subsample=float(trial["low_subsample"]),
            colsample_bytree=float(trial["low_colsample_bytree"]),
            **{
                "lambda": float(trial["low_lambda"]),
                "alpha": float(trial["low_alpha"]),
            },
        ),
        num_boost_round=int(trial["low_num_boost_round"]),
        early_stopping_rounds=0,
        verbose_eval=200,
    )


def make_residual_config(trial: dict[str, int | float]) -> XGBRegressorConfig:
    return XGBRegressorConfig(
        name="residual_tuned",
        params=make_xgb_params(
            eta=float(trial["residual_eta"]),
            max_depth=int(trial["residual_max_depth"]),
            min_child_weight=float(trial["residual_min_child_weight"]),
            subsample=float(trial["residual_subsample"]),
            colsample_bytree=float(trial["residual_colsample_bytree"]),
            **{
                "lambda": float(trial["residual_lambda"]),
                "alpha": float(trial["residual_alpha"]),
            },
        ),
        num_boost_round=int(trial["residual_num_boost_round"]),
        early_stopping_rounds=0,
        verbose_eval=150,
    )


def make_tail_expert_config(trial: dict[str, int | float]) -> XGBRegressorConfig:
    config = default_tail_expert_config(int(trial["tail_num_boost_round"]))
    return XGBRegressorConfig(
        name="tail_expert_tuned",
        params=config.params,
        num_boost_round=int(trial["tail_num_boost_round"]),
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=config.verbose_eval,
    )


def evaluate_trial(
    trial_id: int,
    trial: dict[str, int | float],
    train_df: pd.DataFrame,
    train_features: pd.DataFrame,
    test_df: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: pd.Series,
    truth_df: pd.DataFrame,
) -> tuple[dict[str, int | float | str], pd.DataFrame, dict[str, object]]:
    predictions, model_metrics = candidate_predictions(
        train_df=train_df,
        train_features=train_features,
        target_df=test_df,
        target_features=test_features,
        y_train=y_train.to_numpy(dtype=float),
        y_target=None,
        regressor_configs=[make_main_config(trial)],
        low_expert_config=make_low_expert_config(trial),
        tail_expert_config=make_tail_expert_config(trial),
        residual_config=make_residual_config(trial),
        blend_weight_steps=[int(trial["blend_low_expert_weight_percent"])],
        aux_rounds={
            "low10_classifier": int(trial["low_classifier_rounds"]),
            "tail_classifier": int(trial["tail_classifier_rounds"]),
        },
        dynamic_gate_params={
            "low_gate_scale": float(trial["dynamic_low_gate_scale"]),
            "tail_gate_scale": float(trial["dynamic_tail_gate_scale"]),
            "main_gate_scale": float(trial["dynamic_main_gate_scale"]),
            "low_output_scale": float(trial["dynamic_low_output_scale"]),
            "tail_output_scale": float(trial["dynamic_tail_output_scale"]),
            "residual_mix": float(trial["dynamic_residual_mix"]),
            "raw_mix": float(trial["dynamic_raw_mix"]),
        },
    )
    candidate_report = pd.DataFrame(candidate_public_metric_rows(truth_df, test_df, predictions))
    selected_candidate = select_candidate_by_metrics(candidate_report)
    selected_row = candidate_report[candidate_report["candidate"] == selected_candidate].iloc[0]

    result: dict[str, int | float | str] = {
        "trial_id": trial_id,
        "selected_candidate": selected_candidate,
        "mean_q_error": float(selected_row["public_mean_q_error"]),
        "median_q_error": float(selected_row["public_median_q_error"]),
        "p90_q_error": float(selected_row["public_p90_q_error"]),
        "p95_q_error": float(selected_row["public_p95_q_error"]),
        "max_q_error": float(selected_row["public_max_q_error"]),
        "public_rows": int(selected_row["public_rows"]),
        **trial,
        **blend_weight_metrics(selected_candidate),
    }
    for key, value in model_metrics.items():
        result[key] = value
    best_payload = {
        "selected_candidate": selected_candidate,
        "candidate_report": candidate_report,
        "predictions": predictions,
    }
    return result, candidate_report, best_payload


def sort_trial_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.sort_values(
        ["mean_q_error", "p95_q_error", "max_q_error", "median_q_error", "trial_id"],
        ascending=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search cardinality model hyperparameters with public truth.")
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--sample", type=Path, default=Path("sample_submission.csv"))
    parser.add_argument("--stats", type=Path, default=Path("column_min_max_vals.csv"))
    parser.add_argument("--public-truth", type=Path, default=Path("test_with_true_cardinality.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("grid_search_outputs_v2"))
    parser.add_argument("--profile", choices=tuple(SEARCH_PROFILES), default="gate_focus")
    parser.add_argument("--max-trials", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = perf_counter()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_step("开始读取训练集、测试集、样例与 public truth")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    sample_df = pd.read_csv(args.sample)
    truth_df = pd.read_csv(args.public_truth)
    stats = load_column_stats(args.stats)
    y_train = train_df["Cardinality"]
    search_space = build_profile_search_space(args.profile)

    log_step("开始构造全量训练/测试特征，后续所有 trial 复用这份特征")
    train_features, test_features, _, _ = build_train_target_features(train_df, test_df, stats)

    total_trials = count_total_trials(search_space)
    trials = iter_search_trials(search_space, max_trials=args.max_trials)
    log_step(f"profile={args.profile}，参数组合总数={total_trials}，本次实际执行={len(trials)}")

    trials_path = output_dir / "grid_search_trials.csv"
    trial_rows: list[dict[str, Any]] = []
    completed_trials: dict[tuple[str, ...], dict[str, Any]] = {}
    if args.resume:
        log_step("已启用 resume，开始读取已有 trial 结果")
        trial_rows, completed_trials = load_completed_trials(trials_path)
        log_step(f"已恢复 {len(completed_trials)} 个已完成 trial")

    best_result = select_best_result(trial_rows)
    best_candidate_report: pd.DataFrame | None = None
    best_predictions: dict[str, object] | None = None
    if args.resume:
        validate_resume_trials(completed_trials, trials)

    pending_trials = build_pending_trials(trials, completed_trials)
    completed_before_resume = len(completed_trials)
    log_step(
        f"当前进度：已完成 {completed_before_resume}/{len(trials)}，待执行 {len(pending_trials)}"
    )

    executed_this_run = 0
    trial_time_total_seconds = 0.0

    for index, trial in pending_trials:
        completed_total_before_trial = completed_before_resume + executed_this_run
        trial_started = perf_counter()
        log_step(
            f"执行 trial {index}/{len(trials)} "
            f"(已完成 {completed_total_before_trial}/{len(trials)})："
            f"main_depth={trial['main_max_depth']} "
            f"main_eta={trial['main_eta']} "
            f"low_eta={trial['low_eta']} "
            f"residual_eta={trial['residual_eta']} "
            f"blend={trial['blend_low_expert_weight_percent']}"
        )
        result, candidate_report, payload = evaluate_trial(
            trial_id=index,
            trial=trial,
            train_df=train_df,
            train_features=train_features,
            test_df=test_df,
            test_features=test_features,
            y_train=y_train,
            truth_df=truth_df,
        )
        trial_rows.append(result)
        ranked = sort_trial_results(pd.DataFrame(trial_rows))
        ranked.to_csv(trials_path, index=False)
        completed_trials[trial_key_from_values(trial)] = result
        executed_this_run += 1
        trial_elapsed_seconds = perf_counter() - trial_started
        trial_time_total_seconds += trial_elapsed_seconds
        average_trial_seconds = trial_time_total_seconds / executed_this_run
        remaining_trials = len(pending_trials) - executed_this_run
        elapsed_seconds = perf_counter() - started
        best_mean_q_error = float(best_result["mean_q_error"]) if best_result is not None else float("inf")

        if best_result is None or result_sort_tuple(result) < result_sort_tuple(best_result):
            best_result = result
            best_candidate_report = candidate_report.sort_values(
                ["public_mean_q_error", "public_p95_q_error", "public_max_q_error", "public_median_q_error"]
            )
            best_predictions = payload["predictions"]
            best_mean_q_error = float(best_result["mean_q_error"])

        log_step(
            f"trial {index} 完成，用时 {format_duration(trial_elapsed_seconds)}；"
            f"累计耗时 {format_duration(elapsed_seconds)}；"
            f"平均每 trial {format_duration(average_trial_seconds)}；"
            f"预计剩余 {format_duration(average_trial_seconds * remaining_trials)}；"
            f"当前最佳 mean_q_error={best_mean_q_error:.6f}"
        )

    if best_result is None:
        raise RuntimeError("没有执行任何 trial，无法生成最优结果")

    if best_candidate_report is None or best_predictions is None:
        log_step("当前运行中没有刷新最优 trial，开始重跑已恢复的最优参数以重建最终输出")
        best_trial = {
            key: best_result[key]
            for key in SEARCH_SPACE_KEYS
        }
        _, best_candidate_report, best_payload = evaluate_trial(
            trial_id=int(float(best_result["trial_id"])),
            trial=best_trial,
            train_df=train_df,
            train_features=train_features,
            test_df=test_df,
            test_features=test_features,
            y_train=y_train,
            truth_df=truth_df,
        )
        best_predictions = best_payload["predictions"]

    selected_candidate = str(best_result["selected_candidate"])
    best_submission = write_submission(
        sample_df=sample_df,
        log_pred=best_predictions[selected_candidate],
        output_path=output_dir / "best_submission.csv",
    )
    best_candidate_report.to_csv(output_dir / "best_candidate_report.csv", index=False)

    best_hyperparams = {
        key: best_result[key]
        for key in SEARCH_SPACE_KEYS
    }
    (output_dir / "best_hyperparams.json").write_text(
        json.dumps(best_hyperparams, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    best_metrics = {
        "selected_candidate": selected_candidate,
        "submission_rows": int(len(best_submission)),
        "search_space_total_trials": total_trials,
        "search_profile": args.profile,
        "executed_trials": len(trial_rows),
        "completed_trials_before_resume": completed_before_resume,
        "newly_executed_trials": executed_this_run,
        "remaining_trials": len(trials) - len(trial_rows),
        "resumed": args.resume,
        "pending_trials_after_resume": len(pending_trials) if args.resume else 0,
        "average_trial_seconds": round(trial_time_total_seconds / executed_this_run, 3) if executed_this_run else 0.0,
        "elapsed_seconds": round(perf_counter() - started, 3),
        **best_result,
    }
    (output_dir / "best_metrics.json").write_text(
        json.dumps(best_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(best_metrics, indent=2, ensure_ascii=False))
    print(f"Saved trials to {trials_path}")
    print(f"Saved best submission to {output_dir / 'best_submission.csv'}")
    print(f"Saved best metrics to {output_dir / 'best_metrics.json'}")
    print(f"Saved best hyperparameters to {output_dir / 'best_hyperparams.json'}")
    print(f"Saved best candidate report to {output_dir / 'best_candidate_report.csv'}")


if __name__ == "__main__":
    main()
