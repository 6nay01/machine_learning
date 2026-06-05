from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from cardinality_evaluation import log_predictions_to_cardinality, q_error
from cardinality_features import make_query_keys


RANDOM_STATE = 20260519
GROUP_KEY_ORDER = ["full", "tables_predicates", "tables", "predicates"]
GROUP_MIN_COUNT = 5
BLEND_WEIGHT_STEPS = tuple(range(0, 101))
DYNAMIC_GATE_CANDIDATE = "dynamic_gate"
LOW_GATE_QUANTILE = 0.10
LOW_EXPERT_QUANTILE = 0.25
TAIL_FALLBACK_QUANTILE = 0.90
TAIL_GATE_QUANTILE = 0.95
EXPERT_MIN_SAMPLE_COUNT = 24
TAIL_WEIGHT_P90 = 2.5
TAIL_WEIGHT_P95 = 5.0


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


def blend_candidate_name(low_expert_weight_percent: int) -> str:
    return f"blend_low_{low_expert_weight_percent:03d}"


def normalize_blend_weight_steps(blend_weight_steps: Iterable[int] | None) -> tuple[int, ...]:
    if blend_weight_steps is None:
        return BLEND_WEIGHT_STEPS
    normalized = tuple(dict.fromkeys(int(step) for step in blend_weight_steps))
    if not normalized:
        raise ValueError("blend 权重列表不能为空")
    for step in normalized:
        if step < 0 or step > 100:
            raise ValueError(f"blend 权重必须在 0 到 100 之间，收到 {step}")
    return normalized


def add_blend_predictions_from_arrays(
    predictions: dict[str, np.ndarray],
    low_expert_pred: np.ndarray,
    other_pred: np.ndarray,
    blend_weight_steps: Iterable[int] | None = None,
) -> None:
    for low_expert_weight_percent in normalize_blend_weight_steps(blend_weight_steps):
        low_expert_weight = low_expert_weight_percent / 100.0
        other_weight = 1.0 - low_expert_weight
        predictions[blend_candidate_name(low_expert_weight_percent)] = (
            low_expert_weight * low_expert_pred + other_weight * other_pred
        )


def add_blend_predictions(
    predictions: dict[str, np.ndarray],
    low_expert_name: str,
    other_name: str,
    blend_weight_steps: Iterable[int] | None = None,
) -> None:
    add_blend_predictions_from_arrays(
        predictions,
        predictions[low_expert_name],
        predictions[other_name],
        blend_weight_steps=blend_weight_steps,
    )

def make_dmatrix(
    feature_df: pd.DataFrame,
    label: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> xgb.DMatrix:
    if label is None:
        return xgb.DMatrix(feature_df, weight=weight)
    return xgb.DMatrix(feature_df, label=label, weight=weight)


def cardinality_quantiles(cardinalities: np.ndarray) -> dict[str, float]:
    values = np.asarray(cardinalities, dtype=float)
    return {
        "low_gate": float(np.quantile(values, LOW_GATE_QUANTILE)),
        "low_expert": float(np.quantile(values, LOW_EXPERT_QUANTILE)),
        "tail_fallback": float(np.quantile(values, TAIL_FALLBACK_QUANTILE)),
        "tail_gate": float(np.quantile(values, TAIL_GATE_QUANTILE)),
    }


def build_tail_sample_weights(
    cardinalities: np.ndarray,
    quantiles: dict[str, float],
) -> np.ndarray:
    values = np.asarray(cardinalities, dtype=float)
    weights = np.ones(len(values), dtype=float)
    weights[values >= quantiles["tail_fallback"]] = TAIL_WEIGHT_P90
    weights[values >= quantiles["tail_gate"]] = TAIL_WEIGHT_P95
    return weights


def normalized_gate_weights(
    low_prob: np.ndarray,
    tail_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_weight = np.clip(np.asarray(low_prob, dtype=float), 0.0, 1.0)
    tail_weight = np.clip(np.asarray(tail_prob, dtype=float), 0.0, 1.0)
    main_weight = np.clip(1.0 - low_weight - tail_weight, 0.0, 1.0)
    total_weight = np.maximum(low_weight + tail_weight + main_weight, 1.0)
    return (
        low_weight / total_weight,
        tail_weight / total_weight,
        main_weight / total_weight,
    )


def combine_expert_predictions(
    low_prob: np.ndarray,
    tail_prob: np.ndarray,
    low_expert_pred: np.ndarray,
    tail_expert_pred: np.ndarray,
    backbone_pred: np.ndarray,
) -> np.ndarray:
    low_weight, tail_weight, main_weight = normalized_gate_weights(low_prob, tail_prob)
    return (
        low_weight * np.asarray(low_expert_pred, dtype=float)
        + tail_weight * np.asarray(tail_expert_pred, dtype=float)
        + main_weight * np.asarray(backbone_pred, dtype=float)
    )


def low_expert_mask_from_quantiles(
    cardinalities: np.ndarray,
    quantiles: dict[str, float],
) -> np.ndarray:
    values = np.asarray(cardinalities, dtype=float)
    return values <= quantiles["low_expert"]


def tail_expert_mask_from_quantiles(
    cardinalities: np.ndarray,
    quantiles: dict[str, float],
) -> np.ndarray:
    values = np.asarray(cardinalities, dtype=float)
    tail_mask = values >= quantiles["tail_gate"]
    if int(np.sum(tail_mask)) >= EXPERT_MIN_SAMPLE_COUNT:
        return tail_mask
    return values >= quantiles["tail_fallback"]


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
    ]


def default_low_expert_config(max_rounds: int = 1600) -> XGBRegressorConfig:
    return XGBRegressorConfig(
        name="low_expert_model",
        params=make_xgb_params(
            eta=0.04,
            max_depth=4,
            min_child_weight=1.0,
            subsample=0.95,
            colsample_bytree=0.95,
            **{"lambda": 1.0, "alpha": 0.02},
        ),
        num_boost_round=max_rounds,
        early_stopping_rounds=100,
        verbose_eval=200,
    )


def default_residual_config(max_rounds: int = 900) -> XGBRegressorConfig:
    return XGBRegressorConfig(
        name="residual_model",
        params=make_xgb_params(
            eta=0.03,
            max_depth=4,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            **{"lambda": 3.0, "alpha": 0.10},
        ),
        num_boost_round=max_rounds,
        early_stopping_rounds=0,
        verbose_eval=150,
    )


def fit_main_model(
    config: XGBRegressorConfig,
    train_features: pd.DataFrame,
    target_features: pd.DataFrame,
    y_train_log: np.ndarray,
    y_target_log: np.ndarray | None = None,
    train_weight: np.ndarray | None = None,
    target_weight: np.ndarray | None = None,
) -> MainModelResult:
    dtrain = make_dmatrix(train_features, y_train_log, train_weight)
    dtarget = make_dmatrix(target_features, y_target_log, target_weight)
    evals = [(dtrain, f"{config.name}_train")]
    early_stopping_rounds = None
    if y_target_log is not None and config.early_stopping_rounds > 0:
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
    train_weight: np.ndarray | None = None,
    folds: int = 3,
) -> np.ndarray:
    oof_pred = np.zeros(len(train_features), dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    for fold_idx, (fit_idx, pred_idx) in enumerate(splitter.split(train_features), start=1):
        fold_train = train_features.iloc[fit_idx]
        fold_pred = train_features.iloc[pred_idx]
        fold_y = y_train_log[fit_idx]
        fold_weight = None if train_weight is None else np.asarray(train_weight, dtype=float)[fit_idx]
        dtrain = make_dmatrix(fold_train, fold_y, fold_weight)
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


def fit_oof_binary_predictions(
    train_features: pd.DataFrame,
    y_train: np.ndarray,
    label: str,
    max_rounds: int = 800,
    folds: int = 3,
) -> np.ndarray:
    oof_pred = np.zeros(len(train_features), dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    for fold_idx, (fit_idx, pred_idx) in enumerate(splitter.split(train_features), start=1):
        fold_train = train_features.iloc[fit_idx]
        fold_pred = train_features.iloc[pred_idx]
        fold_y = y_train[fit_idx]
        positives = max(float(np.sum(fold_y)), 1.0)
        negatives = max(float(len(fold_y) - np.sum(fold_y)), 1.0)
        dtrain = make_dmatrix(fold_train, fold_y)
        dpred = make_dmatrix(fold_pred)
        rounds = max(200, min(max_rounds, 900))
        log_step(f"训练 {label} OOF 分类器 fold={fold_idx}/{folds}，rounds={rounds}")
        model = xgb.train(
            params=classifier_params(negatives / positives),
            dtrain=dtrain,
            num_boost_round=rounds,
            evals=[(dtrain, f"{label}_oof_fold_{fold_idx}_train")],
            verbose_eval=200,
        )
        oof_pred[pred_idx] = model.predict(dpred)
    return oof_pred


def fit_oof_low_cardinality_predictions(
    train_features: pd.DataFrame,
    y_train_log: np.ndarray,
    cardinalities: np.ndarray,
    config: XGBRegressorConfig,
    folds: int = 3,
) -> np.ndarray:
    oof_pred = np.zeros(len(train_features), dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    for fold_idx, (fit_idx, pred_idx) in enumerate(splitter.split(train_features), start=1):
        fold_train_features = train_features.iloc[fit_idx]
        fold_pred_features = train_features.iloc[pred_idx]
        fold_y_log = y_train_log[fit_idx]
        fold_cards = cardinalities[fit_idx]
        low_mask = fold_cards <= 100.0
        if int(np.sum(low_mask)) < 50:
            low_mask = fold_cards <= 1000.0
        low_features = fold_train_features.loc[low_mask].copy()
        low_y = fold_y_log[low_mask]
        dtrain = make_dmatrix(low_features, low_y)
        dpred = make_dmatrix(fold_pred_features)
        rounds = max(200, min(config.num_boost_round, 900))
        log_step(f"训练低基数专家 OOF fold={fold_idx}/{folds}，样本={len(low_features)}")
        model = xgb.train(
            params=config.params,
            dtrain=dtrain,
            num_boost_round=rounds,
            evals=[(dtrain, f"low_expert_oof_fold_{fold_idx}_train")],
            verbose_eval=config.verbose_eval,
        )
        oof_pred[pred_idx] = model.predict(dpred)
    return oof_pred


def add_main_prediction_features(feature_df: pd.DataFrame, base_pred: np.ndarray) -> pd.DataFrame:
    features = feature_df.copy()
    features["main_log_prediction"] = np.asarray(base_pred, dtype=float)
    features["main_cardinality_prediction"] = log_predictions_to_cardinality(base_pred)
    return features


def fit_oof_residual_predictions(
    main_config: XGBRegressorConfig,
    residual_config: XGBRegressorConfig,
    train_features: pd.DataFrame,
    y_train_log: np.ndarray,
    num_rounds: int,
    folds: int = 3,
) -> np.ndarray:
    oof_pred = np.zeros(len(train_features), dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    for fold_idx, (fit_idx, pred_idx) in enumerate(splitter.split(train_features), start=1):
        fold_train_features = train_features.iloc[fit_idx]
        fold_pred_features = train_features.iloc[pred_idx]
        fold_y_log = y_train_log[fit_idx]
        dtrain_main = make_dmatrix(fold_train_features, fold_y_log)
        dpred_main = make_dmatrix(fold_pred_features)
        rounds = max(250, min(num_rounds, 900))
        log_step(f"训练动态融合残差 OOF 主模型 fold={fold_idx}/{folds}，rounds={rounds}")
        main_model = xgb.train(
            params=main_config.params,
            dtrain=dtrain_main,
            num_boost_round=rounds,
            evals=[(dtrain_main, f"dynamic_residual_oof_main_fold_{fold_idx}_train")],
            verbose_eval=200,
        )
        fold_train_base_pred = main_model.predict(dtrain_main)
        fold_pred_base_pred = main_model.predict(dpred_main)
        residual_train = fold_y_log - fold_train_base_pred
        residual_train_features = add_main_prediction_features(fold_train_features, fold_train_base_pred)
        residual_pred_features = add_main_prediction_features(fold_pred_features, fold_pred_base_pred)
        dtrain_residual = make_dmatrix(residual_train_features, residual_train)
        dpred_residual = make_dmatrix(residual_pred_features)
        residual_rounds = max(200, min(residual_config.num_boost_round, 900))
        log_step(f"训练动态融合残差 OOF 修正 fold={fold_idx}/{folds}，rounds={residual_rounds}")
        residual_model = xgb.train(
            params=residual_config.params,
            dtrain=dtrain_residual,
            num_boost_round=residual_rounds,
            evals=[(dtrain_residual, f"dynamic_residual_oof_fold_{fold_idx}_train")],
            verbose_eval=residual_config.verbose_eval,
        )
        residual_delta = residual_model.predict(dpred_residual)
        oof_pred[pred_idx] = fold_pred_base_pred + residual_delta
    return oof_pred


def compute_low_expert_preference_labels(
    y_true: np.ndarray,
    low_expert_pred: np.ndarray,
    other_pred: np.ndarray,
) -> np.ndarray:
    low_error = q_error(y_true, log_predictions_to_cardinality(low_expert_pred))
    other_error = q_error(y_true, log_predictions_to_cardinality(other_pred))
    return (low_error <= other_error).astype(float)


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
    quantiles: dict[str, float],
    config: XGBRegressorConfig | None = None,
) -> tuple[xgb.Booster, np.ndarray]:
    config = config or default_low_expert_config()
    low_mask = low_expert_mask_from_quantiles(cardinalities, quantiles)
    low_features = train_features.loc[low_mask].copy()
    low_y = y_train_log[low_mask]
    dtrain = make_dmatrix(low_features, low_y)
    dtarget = make_dmatrix(target_features, y_target_log)
    evals = [(dtrain, "low_expert_train")]
    early_stopping_rounds = None
    if y_target_log is not None and config.early_stopping_rounds > 0:
        evals.append((dtarget, "low_expert_target"))
        early_stopping_rounds = config.early_stopping_rounds

    log_step(f"训练低基数专家回归器，样本={len(low_features)}")
    model = xgb.train(
        params=config.params,
        dtrain=dtrain,
        num_boost_round=config.num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=config.verbose_eval,
    )
    best_iteration = config.num_boost_round - 1 if y_target_log is None else int(model.best_iteration)
    target_pred = model.predict(dtarget, iteration_range=(0, best_iteration + 1))
    return model, target_pred


def fit_tail_regressor(
    train_features: pd.DataFrame,
    target_features: pd.DataFrame,
    y_train_log: np.ndarray,
    y_target_log: np.ndarray | None,
    cardinalities: np.ndarray,
    quantiles: dict[str, float],
    config: XGBRegressorConfig | None = None,
) -> tuple[xgb.Booster, np.ndarray]:
    config = config or default_low_expert_config()
    tail_mask = tail_expert_mask_from_quantiles(cardinalities, quantiles)
    tail_features = train_features.loc[tail_mask].copy()
    tail_y = y_train_log[tail_mask]
    dtrain = make_dmatrix(tail_features, tail_y)
    dtarget = make_dmatrix(target_features, y_target_log)
    evals = [(dtrain, "tail_expert_train")]
    early_stopping_rounds = None
    if y_target_log is not None and config.early_stopping_rounds > 0:
        evals.append((dtarget, "tail_expert_target"))
        early_stopping_rounds = config.early_stopping_rounds

    log_step(f"训练高尾专家回归器，样本={len(tail_features)}")
    model = xgb.train(
        params=config.params,
        dtrain=dtrain,
        num_boost_round=config.num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=config.verbose_eval,
    )
    best_iteration = config.num_boost_round - 1 if y_target_log is None else int(model.best_iteration)
    target_pred = model.predict(dtarget, iteration_range=(0, best_iteration + 1))
    return model, target_pred


def fit_residual_regressor(
    train_features: pd.DataFrame,
    target_features: pd.DataFrame,
    y_train_log: np.ndarray,
    y_target_log: np.ndarray | None,
    train_base_pred: np.ndarray,
    target_base_pred: np.ndarray,
    train_weight: np.ndarray | None = None,
    config: XGBRegressorConfig | None = None,
) -> tuple[xgb.Booster, np.ndarray]:
    config = config or default_residual_config()
    residual_train = y_train_log - train_base_pred
    residual_train_features = train_features.copy()
    residual_target_features = target_features.copy()
    residual_train_features["main_log_prediction"] = train_base_pred
    residual_target_features["main_log_prediction"] = target_base_pred
    residual_train_features["main_cardinality_prediction"] = log_predictions_to_cardinality(train_base_pred)
    residual_target_features["main_cardinality_prediction"] = log_predictions_to_cardinality(target_base_pred)

    dtrain = make_dmatrix(residual_train_features, residual_train, train_weight)
    dtarget = make_dmatrix(residual_target_features, None if y_target_log is None else y_target_log * 0.0)
    evals = [(dtrain, "residual_train")]

    log_step("训练残差修正模型")
    model = xgb.train(
        params=config.params,
        dtrain=dtrain,
        num_boost_round=config.num_boost_round,
        evals=evals,
        verbose_eval=config.verbose_eval,
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
    regressor_configs: list[XGBRegressorConfig] | None = None,
    low_expert_config: XGBRegressorConfig | None = None,
    residual_config: XGBRegressorConfig | None = None,
    blend_weight_steps: Iterable[int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | str]]:
    y_train_log = np.log1p(y_train.astype(float))
    y_target_log = None if y_target is None else np.log1p(y_target.astype(float))
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, float | str] = {}
    aux_rounds = aux_rounds or {}
    quantiles = cardinality_quantiles(y_train)
    train_weight = build_tail_sample_weights(y_train, quantiles)
    target_weight = None if y_target is None else build_tail_sample_weights(y_target, quantiles)

    configs = regressor_configs or candidate_regressor_configs()
    if round_overrides is not None:
        configs = [
            replace(config, num_boost_round=max(250, min(config.num_boost_round, round_overrides.get(config.name, config.num_boost_round))))
            for config in configs
        ]
    low_expert_config = low_expert_config or default_low_expert_config(aux_rounds.get("low_expert", 1600))
    residual_config = residual_config or default_residual_config(aux_rounds.get("residual", 900))

    main_results = [
        fit_main_model(
            config,
            train_features,
            target_features,
            y_train_log,
            y_target_log,
            train_weight=train_weight,
            target_weight=target_weight,
        )
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
    metrics["low_gate_threshold"] = quantiles["low_gate"]
    metrics["low_expert_threshold"] = quantiles["low_expert"]
    metrics["tail_fallback_threshold"] = quantiles["tail_fallback"]
    metrics["tail_gate_threshold"] = quantiles["tail_gate"]
    raw_pred = selected_main.target_log_pred
    predictions["eq_stats_main"] = raw_pred

    calibrator = fit_group_calibrator(train_df, y_train_log, selected_main.train_log_pred)
    predictions["group_calibrated"] = apply_group_calibrator(target_df, raw_pred, calibrator)

    y_train_low = (y_train <= quantiles["low_gate"]).astype(float)
    y_target_low = None if y_target is None else (y_target <= quantiles["low_gate"]).astype(float)
    _, low_prob = fit_binary_classifier(
        train_features,
        target_features,
        y_train_low,
        y_target_low,
        "low_cardinality_quantile",
        max_rounds=aux_rounds.get("low10_classifier", 1600),
    )
    y_train_tail = (y_train >= quantiles["tail_gate"]).astype(float)
    y_target_tail = None if y_target is None else (y_target >= quantiles["tail_gate"]).astype(float)
    _, tail_prob = fit_binary_classifier(
        train_features,
        target_features,
        y_train_tail,
        y_target_tail,
        "tail_quantile",
        max_rounds=aux_rounds.get("tail_classifier", 1000),
    )
    _, low_expert_pred = fit_low_cardinality_regressor(
        train_features,
        target_features,
        y_train_log,
        y_target_log,
        y_train,
        quantiles,
        config=low_expert_config,
    )
    _, tail_expert_pred = fit_tail_regressor(
        train_features,
        target_features,
        y_train_log,
        y_target_log,
        y_train,
        quantiles,
        config=low_expert_config,
    )
    low_weight = np.clip(low_prob, 0.0, 1.0)
    tail_weight = np.clip(tail_prob, 0.0, 1.0)
    predictions["low_expert"] = low_weight * low_expert_pred + (1.0 - low_weight) * raw_pred
    predictions["tail_expert"] = tail_weight * tail_expert_pred + (1.0 - tail_weight) * raw_pred

    residual_pred = raw_pred
    residual_train_base_pred = fit_oof_base_predictions(
        selected_main.config,
        train_features,
        y_train_log,
        selected_main.best_iteration + 1,
        train_weight=train_weight,
        folds=aux_rounds.get("oof_folds", 3),
    )
    if forced_strategy != "skip_residual":
        _, residual_pred = fit_residual_regressor(
            train_features,
            target_features,
            y_train_log,
            y_target_log,
            residual_train_base_pred,
            raw_pred,
            train_weight=train_weight,
            config=residual_config,
        )
        predictions["residual"] = residual_pred
        predictions[DYNAMIC_GATE_CANDIDATE] = combine_expert_predictions(
            low_prob=low_prob,
            tail_prob=tail_prob,
            low_expert_pred=low_expert_pred,
            tail_expert_pred=tail_expert_pred,
            backbone_pred=residual_pred,
        )
        add_blend_predictions(
            predictions,
            "low_expert",
            "residual",
            blend_weight_steps=blend_weight_steps,
        )
    else:
        predictions[DYNAMIC_GATE_CANDIDATE] = combine_expert_predictions(
            low_prob=low_prob,
            tail_prob=tail_prob,
            low_expert_pred=low_expert_pred,
            tail_expert_pred=tail_expert_pred,
            backbone_pred=raw_pred,
        )
        add_blend_predictions_from_arrays(
            predictions,
            predictions["low_expert"],
            raw_pred,
            blend_weight_steps=blend_weight_steps,
        )
        
    return predictions, metrics
