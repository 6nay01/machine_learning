from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from cardinality_evaluation import log_predictions_to_cardinality, q_error
from cardinality_features import make_query_keys


RANDOM_STATE = 20260519
GROUP_KEY_ORDER = ["full", "tables_predicates", "tables", "predicates"]
GROUP_MIN_COUNT = 5


@dataclass(frozen=True)
class XGBRegressorConfig:
    name: str
    params: dict[str, object]
    num_boost_round: int
    early_stopping_rounds: int
    verbose_eval: int


@dataclass
class MainModelResult:
    config: XGBRegressorConfig
    model: xgb.Booster
    best_iteration: int
    best_score: float
    train_log_pred: np.ndarray
    target_log_pred: np.ndarray


def log_step(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def make_dmatrix(feature_df: pd.DataFrame, label: np.ndarray | None = None) -> xgb.DMatrix:
    if label is None:
        return xgb.DMatrix(feature_df)
    return xgb.DMatrix(feature_df, label=label)


def base_xgb_params() -> dict[str, object]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cpu",
        "seed": RANDOM_STATE,
        "nthread": 0,
    }


def make_xgb_params(**overrides: object) -> dict[str, object]:
    params = base_xgb_params()
    params.update(overrides)
    return params


def candidate_regressor_configs() -> list[XGBRegressorConfig]:
    return [
        XGBRegressorConfig(
            name="main_depth6",
            params=make_xgb_params(
                eta=0.035,
                max_depth=6,
                min_child_weight=1.5,
                subsample=0.92,
                colsample_bytree=0.92,
                **{"lambda": 1.2, "alpha": 0.02},
            ),
            num_boost_round=3800,
            early_stopping_rounds=160,
            verbose_eval=150,
        ),
        XGBRegressorConfig(
            name="main_depth7",
            params=make_xgb_params(
                eta=0.026,
                max_depth=7,
                min_child_weight=1.2,
                subsample=0.92,
                colsample_bytree=0.86,
                **{"lambda": 1.6, "alpha": 0.03},
            ),
            num_boost_round=4800,
            early_stopping_rounds=190,
            verbose_eval=150,
        ),
        XGBRegressorConfig(
            name="robust_pseudohuber_depth6",
            params=make_xgb_params(
                objective="reg:pseudohubererror",
                eta=0.030,
                max_depth=6,
                min_child_weight=1.5,
                subsample=0.92,
                colsample_bytree=0.92,
                **{"lambda": 1.5, "alpha": 0.03},
            ),
            num_boost_round=4300,
            early_stopping_rounds=180,
            verbose_eval=150,
        ),
    ]


def fit_main_model(
    config: XGBRegressorConfig,
    train_features: pd.DataFrame,
    target_features: pd.DataFrame,
    y_train_log: np.ndarray,
    y_target_log: np.ndarray | None = None,
) -> MainModelResult:
    dtrain = make_dmatrix(train_features, y_train_log)
    dtarget = make_dmatrix(target_features, y_target_log)
    evals = [(dtrain, f"{config.name}_train")]
    early_stopping_rounds = None
    if y_target_log is not None:
        evals.append((dtarget, f"{config.name}_target"))
        early_stopping_rounds = config.early_stopping_rounds

    log_step(f"训练主模型 {config.name}")
    model = xgb.train(
        params=config.params,
        dtrain=dtrain,
        num_boost_round=config.num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=config.verbose_eval,
    )

    best_iteration = config.num_boost_round - 1
    best_score = 0.0
    if y_target_log is not None:
        best_iteration = int(model.best_iteration)
        best_score = float(model.best_score)
    iteration_range = (0, best_iteration + 1)
    train_log_pred = model.predict(dtrain, iteration_range=iteration_range)
    target_log_pred = model.predict(dtarget, iteration_range=iteration_range)
    return MainModelResult(
        config=config,
        model=model,
        best_iteration=best_iteration,
        best_score=best_score,
        train_log_pred=train_log_pred,
        target_log_pred=target_log_pred,
    )


def fit_oof_base_predictions(
    config: XGBRegressorConfig,
    train_features: pd.DataFrame,
    y_train_log: np.ndarray,
    num_rounds: int,
    folds: int = 3,
) -> np.ndarray:
    oof_pred = np.zeros(len(train_features), dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    for fold_idx, (fit_idx, pred_idx) in enumerate(splitter.split(train_features), start=1):
        fold_train = train_features.iloc[fit_idx]
        fold_pred = train_features.iloc[pred_idx]
        fold_y = y_train_log[fit_idx]
        dtrain = make_dmatrix(fold_train, fold_y)
        dpred = make_dmatrix(fold_pred)
        rounds = max(250, min(num_rounds, 900))
        log_step(f"训练残差 OOF 基础预测 fold={fold_idx}/{folds}，rounds={rounds}")
        model = xgb.train(
            params=config.params,
            dtrain=dtrain,
            num_boost_round=rounds,
            evals=[(dtrain, f"oof_fold_{fold_idx}_train")],
            verbose_eval=200,
        )
        oof_pred[pred_idx] = model.predict(dpred)
    return oof_pred


def fit_group_calibrator(df: pd.DataFrame, true_log_y: np.ndarray, pred_log_y: np.ndarray) -> dict[str, object]:
    residuals_by_group: dict[str, dict[str, list[float]]] = {
        key_name: defaultdict(list) for key_name in GROUP_KEY_ORDER
    }
    global_shift = float(np.median(true_log_y - pred_log_y))

    for row, true_log, pred_log in zip(df.to_dict("records"), true_log_y, pred_log_y):
        residual = float(true_log - pred_log)
        query_keys = make_query_keys(row)
        for key_name, key_value in query_keys.items():
            residuals_by_group[key_name][key_value].append(residual)

    groups: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for key_name, grouped_residuals in residuals_by_group.items():
        groups[key_name] = {}
        counts[key_name] = {}
        for key_value, values in grouped_residuals.items():
            if len(values) >= GROUP_MIN_COUNT:
                groups[key_name][key_value] = float(np.median(values))
                counts[key_name][key_value] = len(values)

    return {"global_shift": global_shift, "groups": groups, "counts": counts}


def apply_group_calibrator(df: pd.DataFrame, pred_log_y: np.ndarray, calibrator: dict[str, object]) -> np.ndarray:
    groups = calibrator["groups"]
    global_shift = float(calibrator["global_shift"])
    shifts: list[float] = []

    for row in df.to_dict("records"):
        query_keys = make_query_keys(row)
        shift = global_shift
        for key_name in GROUP_KEY_ORDER:
            group_map = groups[key_name]
            key_value = query_keys[key_name]
            if key_value in group_map:
                shift = float(group_map[key_value])
                break
        shifts.append(shift)

    return pred_log_y + np.asarray(shifts, dtype=float)


def classifier_params(scale_pos_weight: float) -> dict[str, object]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cpu",
        "eta": 0.035,
        "max_depth": 5,
        "min_child_weight": 2.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 2.0,
        "alpha": 0.05,
        "seed": RANDOM_STATE,
        "scale_pos_weight": scale_pos_weight,
        "nthread": 0,
    }


def fit_binary_classifier(
    train_features: pd.DataFrame,
    target_features: pd.DataFrame,
    y_train: np.ndarray,
    y_target: np.ndarray | None,
    label: str,
    max_rounds: int = 1600,
) -> tuple[xgb.Booster, np.ndarray]:
    positives = max(float(np.sum(y_train)), 1.0)
    negatives = max(float(len(y_train) - np.sum(y_train)), 1.0)
    dtrain = make_dmatrix(train_features, y_train)
    dtarget = make_dmatrix(target_features, y_target)
    evals = [(dtrain, f"{label}_train")]
    early_stopping_rounds = None
    if y_target is not None:
        evals.append((dtarget, f"{label}_target"))
        early_stopping_rounds = 120

    log_step(f"训练分类器 {label}，正样本={int(positives)}")
    model = xgb.train(
        params=classifier_params(negatives / positives),
        dtrain=dtrain,
        num_boost_round=max_rounds,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=200,
    )
    best_iteration = max_rounds - 1 if y_target is None else int(model.best_iteration)
    prob = model.predict(dtarget, iteration_range=(0, best_iteration + 1))
    return model, prob


def fit_low_cardinality_regressor(
    train_features: pd.DataFrame,
    target_features: pd.DataFrame,
    y_train_log: np.ndarray,
    y_target_log: np.ndarray | None,
    cardinalities: np.ndarray,
    max_rounds: int = 1600,
) -> tuple[xgb.Booster, np.ndarray]:
    low_mask = cardinalities <= 100.0
    if int(np.sum(low_mask)) < 50:
        low_mask = cardinalities <= 1000.0
    low_features = train_features.loc[low_mask].copy()
    low_y = y_train_log[low_mask]
    dtrain = make_dmatrix(low_features, low_y)
    dtarget = make_dmatrix(target_features, y_target_log)
    evals = [(dtrain, "low_expert_train")]
    early_stopping_rounds = None
    if y_target_log is not None:
        evals.append((dtarget, "low_expert_target"))
        early_stopping_rounds = 100

    log_step(f"训练低基数专家回归器，样本={len(low_features)}")
    model = xgb.train(
        params=make_xgb_params(
            eta=0.04,
            max_depth=4,
            min_child_weight=1.0,
            subsample=0.95,
            colsample_bytree=0.95,
            **{"lambda": 1.0, "alpha": 0.02},
        ),
        dtrain=dtrain,
        num_boost_round=max_rounds,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=200,
    )
    best_iteration = max_rounds - 1 if y_target_log is None else int(model.best_iteration)
    target_pred = model.predict(dtarget, iteration_range=(0, best_iteration + 1))
    return model, target_pred


def fit_residual_regressor(
    train_features: pd.DataFrame,
    target_features: pd.DataFrame,
    y_train_log: np.ndarray,
    y_target_log: np.ndarray | None,
    train_base_pred: np.ndarray,
    target_base_pred: np.ndarray,
    max_rounds: int = 900,
) -> tuple[xgb.Booster, np.ndarray]:
    residual_train = y_train_log - train_base_pred
    residual_train_features = train_features.copy()
    residual_target_features = target_features.copy()
    residual_train_features["main_log_prediction"] = train_base_pred
    residual_target_features["main_log_prediction"] = target_base_pred
    residual_train_features["main_cardinality_prediction"] = log_predictions_to_cardinality(train_base_pred)
    residual_target_features["main_cardinality_prediction"] = log_predictions_to_cardinality(target_base_pred)

    dtrain = make_dmatrix(residual_train_features, residual_train)
    dtarget = make_dmatrix(residual_target_features, None if y_target_log is None else y_target_log * 0.0)
    evals = [(dtrain, "residual_train")]

    log_step("训练残差修正模型")
    model = xgb.train(
        params=make_xgb_params(
            eta=0.03,
            max_depth=4,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            **{"lambda": 3.0, "alpha": 0.10},
        ),
        dtrain=dtrain,
        num_boost_round=max_rounds,
        evals=evals,
        verbose_eval=150,
    )
    residual_pred = model.predict(dtarget)
    return model, target_base_pred + residual_pred


def candidate_predictions(
    train_df: pd.DataFrame,
    train_features: pd.DataFrame,
    target_df: pd.DataFrame,
    target_features: pd.DataFrame,
    y_train: np.ndarray,
    y_target: np.ndarray | None,
    forced_strategy: str | None = None,
    round_overrides: dict[str, int] | None = None,
    aux_rounds: dict[str, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | str]]:
    y_train_log = np.log1p(y_train.astype(float))
    y_target_log = None if y_target is None else np.log1p(y_target.astype(float))
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, float | str] = {}
    aux_rounds = aux_rounds or {}

    configs = candidate_regressor_configs()
    if round_overrides is not None:
        configs = [
            replace(config, num_boost_round=max(250, min(config.num_boost_round, round_overrides.get(config.name, config.num_boost_round))))
            for config in configs
        ]

    main_results = [
        fit_main_model(config, train_features, target_features, y_train_log, y_target_log)
        for config in configs
    ]
    if y_target is None:
        selected_main = main_results[0]
    else:
        selected_main = min(
            main_results,
            key=lambda result: float(
                np.mean(q_error(y_target, log_predictions_to_cardinality(result.target_log_pred)))
            ),
        )

    for result in main_results:
        predictions[result.config.name] = result.target_log_pred
        metrics[f"{result.config.name}_best_iteration"] = float(result.best_iteration)
        if y_target is not None:
            errors = q_error(y_target, log_predictions_to_cardinality(result.target_log_pred))
            metrics[f"{result.config.name}_mean_q_error"] = float(np.mean(errors))
            metrics[f"{result.config.name}_p95_q_error"] = float(np.quantile(errors, 0.95))

    metrics["selected_main_config"] = selected_main.config.name
    raw_pred = selected_main.target_log_pred
    predictions["eq_stats_main"] = raw_pred

    calibrator = fit_group_calibrator(train_df, y_train_log, selected_main.train_log_pred)
    predictions["group_calibrated"] = apply_group_calibrator(target_df, raw_pred, calibrator)

    y_train_low10 = (y_train <= 10.0).astype(float)
    y_target_low10 = None if y_target is None else (y_target <= 10.0).astype(float)
    _, low10_prob = fit_binary_classifier(
        train_features,
        target_features,
        y_train_low10,
        y_target_low10,
        "low_cardinality_10",
        max_rounds=aux_rounds.get("low10_classifier", 1600),
    )
    y_train_low100 = (y_train <= 100.0).astype(float)
    y_target_low100 = None if y_target is None else (y_target <= 100.0).astype(float)
    _, low100_prob = fit_binary_classifier(
        train_features,
        target_features,
        y_train_low100,
        y_target_low100,
        "low_cardinality_100",
        max_rounds=aux_rounds.get("low100_classifier", 1600),
    )
    _, low_expert_pred = fit_low_cardinality_regressor(
        train_features,
        target_features,
        y_train_log,
        y_target_log,
        y_train,
        max_rounds=aux_rounds.get("low_expert", 1600),
    )
    low_weight = np.clip(0.65 * low100_prob + 0.35 * low10_prob, 0.0, 1.0)
    predictions["low_expert"] = low_weight * low_expert_pred + (1.0 - low_weight) * raw_pred

    residual_pred = raw_pred
    if forced_strategy != "skip_residual":
        residual_train_base_pred = fit_oof_base_predictions(
            selected_main.config,
            train_features,
            y_train_log,
            selected_main.best_iteration + 1,
            folds=aux_rounds.get("oof_folds", 3),
        )
        _, residual_pred = fit_residual_regressor(
            train_features,
            target_features,
            y_train_log,
            y_target_log,
            residual_train_base_pred,
            raw_pred,
            max_rounds=aux_rounds.get("residual", 900),
        )
        predictions["residual"] = residual_pred
        predictions["blend"] = 0.35 * predictions["low_expert"] + 0.65 * residual_pred
    else:
        predictions["blend"] = 0.35 * predictions["low_expert"] + 0.65 * raw_pred

    return predictions, metrics
