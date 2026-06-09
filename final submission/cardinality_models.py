"""模型训练与候选预测管道。

该模块定义用于训练多个 XGBoost 回归/分类器候选的配置与工具，
包括主模型、低基数专家、稀有等值专家、残差修正模型以及混合策略。
提供训练单个模型、生成 OOF 基础预测、组合候选预测并返回评估指标。
"""

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from cardinality_evaluation import log_predictions_to_cardinality, q_error
from cardinality_features import equality_key, parse_predicates

RANDOM_STATE = 20260519
FINAL_BLEND_LOW_EXPERT_WEIGHT_PERCENT = 20
RARE_EQUALITY_MAX_COUNT = 3
UNSEEN_EQ_SWITCH_CANDIDATE = "switch_unseen_eq"

@dataclass(frozen=True)
class XGBRegressorConfig:
    name: str
    params: dict[str, object]
    num_boost_round: int
    early_stopping_rounds: int
    verbose_eval: int
    """XGBoost 回归器配置的不可变数据类。

    字段说明：
    - `name`: 配置名称（用于标识候选模型）。
    - `params`: 传递给 xgboost.train 的参数字典。
    - `num_boost_round`: 最大迭代轮数。
    - `early_stopping_rounds`: 早停轮数（在有验证集时使用）。
    - `verbose_eval`: 日志输出频率。
    """


@dataclass
class MainModelResult:
    config: XGBRegressorConfig
    model: xgb.Booster
    best_iteration: int
    best_score: float
    train_log_pred: np.ndarray
    target_log_pred: np.ndarray
    """主模型训练结果容器。

    包含训练时使用的配置、训练完成后的 `xgb.Booster` 对象、
    最佳迭代与得分，以及在训练集与目标集上的对数预测数组。
    """

def log_step(message: str) -> None:
    """打印带进度前缀的训练日志。"""
    print(f"[progress] {message}", flush=True)

def blend_candidate_name(low_expert_weight_percent: int) -> str:
    """按约定格式生成 blend 候选名。"""
    return f"blend_low_{low_expert_weight_percent:03d}"

def add_blend_predictions_from_arrays(
    predictions: dict[str, np.ndarray],
    low_expert_pred: np.ndarray,
    other_pred: np.ndarray,
) -> None:
    """基于两个预测数组生成固定的 20% low_expert blend 候选。"""
    low_expert_weight = FINAL_BLEND_LOW_EXPERT_WEIGHT_PERCENT / 100.0
    other_weight = 1.0 - low_expert_weight
    predictions[blend_candidate_name(FINAL_BLEND_LOW_EXPERT_WEIGHT_PERCENT)] = (
        low_expert_weight * low_expert_pred + other_weight * other_pred
    )

def add_blend_predictions(
    predictions: dict[str, np.ndarray],
    low_expert_name: str,
    other_name: str,
) -> None:
    """使用已有候选预测批量生成 blend 候选。"""
    add_blend_predictions_from_arrays(predictions, predictions[low_expert_name], predictions[other_name])


def equality_value_counts(df: pd.DataFrame) -> dict[str, int]:
    """统计训练数据中每个等值谓词取值出现的次数。"""
    counts: dict[str, int] = defaultdict(int)
    for row in df.to_dict("records"):
        for col, op, val in parse_predicates(row["Predicates"]):
            if op == "=":
                counts[equality_key(col, val)] += 1
    return counts


def build_rare_and_unseen_equality_masks(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    rare_max_count: int = RARE_EQUALITY_MAX_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """构造训练集稀有等值掩码和目标集未见等值掩码。"""
    value_counts = equality_value_counts(train_df)
    train_mask = np.zeros(len(train_df), dtype=bool)
    target_mask = np.zeros(len(target_df), dtype=bool)

    for idx, row in enumerate(train_df.to_dict("records")):
        eq_counts = [
            value_counts[equality_key(col, val)]
            for col, op, val in parse_predicates(row["Predicates"])
            if op == "="
        ]
        train_mask[idx] = bool(eq_counts) and min(eq_counts) <= rare_max_count

    for idx, row in enumerate(target_df.to_dict("records")):
        eq_counts = [
            value_counts.get(equality_key(col, val), 0)
            for col, op, val in parse_predicates(row["Predicates"])
            if op == "="
        ]
        target_mask[idx] = any(count == 0 for count in eq_counts)

    return train_mask, target_mask


def make_dmatrix(feature_df: pd.DataFrame, label: np.ndarray | None = None) -> xgb.DMatrix:
    """将特征和可选标签封装为 XGBoost DMatrix。"""
    if label is None:
        return xgb.DMatrix(feature_df)
    return xgb.DMatrix(feature_df, label=label)


def base_xgb_params() -> dict[str, object]:
    """返回所有 XGBoost 模型共享的基础参数。"""
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cpu",
        "seed": RANDOM_STATE,
        "nthread": 0,
    }


def make_xgb_params(**overrides: object) -> dict[str, object]:
    """在基础参数上叠加模型专属参数。"""
    params = base_xgb_params()
    params.update(overrides)
    return params


def candidate_regressor_configs() -> list[XGBRegressorConfig]:
    """定义主回归候选模型的配置列表。"""
    return [
        XGBRegressorConfig(
            name="main_model",
            params=make_xgb_params(
                eta=0.03,
                max_depth=7,
                min_child_weight=1.5,
                subsample=0.92,
                colsample_bytree=0.92,
                **{"lambda": 1.2, "alpha": 0.02},
            ),
            num_boost_round=3200,
            early_stopping_rounds=160,
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
    """训练一个主回归模型并返回训练集与目标集预测。"""
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


def classifier_params(scale_pos_weight: float) -> dict[str, object]:
    """返回低基数分类器的 XGBoost 参数。"""
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
    """训练一个二分类器并返回目标集正类概率。"""
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


def fit_oof_base_predictions(
    config: XGBRegressorConfig,
    train_features: pd.DataFrame,
    y_train_log: np.ndarray,
    num_rounds: int,
    folds: int = 3,
) -> np.ndarray:
    """为残差模型生成主模型的 OOF 基础预测。"""
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


def fit_low_cardinality_regressor(
    train_features: pd.DataFrame,
    target_features: pd.DataFrame,
    y_train_log: np.ndarray,
    y_target_log: np.ndarray | None,
    cardinalities: np.ndarray,
    max_rounds: int = 1200,
) -> tuple[xgb.Booster, np.ndarray]:
    """训练专注低基数样本的专家回归器。"""
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


def fit_rare_equality_regressor(
    train_features: pd.DataFrame,
    target_features: pd.DataFrame,
    y_train_log: np.ndarray,
    y_target_log: np.ndarray | None,
    rare_mask: np.ndarray,
    max_rounds: int = 1200,
) -> tuple[xgb.Booster, np.ndarray]:
    """训练专注稀有等值查询的专家回归器。"""
    effective_mask = rare_mask.copy()
    if int(np.sum(effective_mask)) < 50:
        effective_mask = np.ones(len(train_features), dtype=bool)
    rare_features = train_features.loc[effective_mask].copy()
    rare_y = y_train_log[effective_mask]
    dtrain = make_dmatrix(rare_features, rare_y)
    dtarget = make_dmatrix(target_features, y_target_log)
    evals = [(dtrain, "rare_eq_train")]
    early_stopping_rounds = None
    if y_target_log is not None:
        evals.append((dtarget, "rare_eq_target"))
        early_stopping_rounds = 80

    log_step(f"训练稀有等值专家回归器，样本={len(rare_features)}")
    model = xgb.train(
        params=make_xgb_params(
            eta=0.035,
            max_depth=5,
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
    max_rounds: int = 700,
) -> tuple[xgb.Booster, np.ndarray]:
    """训练残差修正模型并叠加到基础预测上。"""
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
    round_overrides: dict[str, int] | None = None,
    aux_rounds: dict[str, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | str]]:
    """训练整套候选模型并返回每个候选的预测与指标。"""
    y_train_log = np.log1p(y_train.astype(float))
    y_target_log = None if y_target is None else np.log1p(y_target.astype(float))
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, float | str] = {}
    aux_rounds = aux_rounds or {}
    rare_train_mask, unseen_target_mask = build_rare_and_unseen_equality_masks(train_df, target_df)
    metrics["rare_eq_train_samples"] = float(np.sum(rare_train_mask))
    metrics["unseen_eq_target_samples"] = float(np.sum(unseen_target_mask))

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
    add_blend_predictions(predictions, "low_expert", "residual")
    _, rare_eq_pred = fit_rare_equality_regressor(
        train_features,
        target_features,
        y_train_log,
        y_target_log,
        rare_train_mask,
        max_rounds=aux_rounds.get("rare_eq_expert", 700),
    )
    predictions["rare_eq_expert"] = rare_eq_pred
    switched_pred = predictions[blend_candidate_name(FINAL_BLEND_LOW_EXPERT_WEIGHT_PERCENT)].copy()
    switched_pred[unseen_target_mask] = rare_eq_pred[unseen_target_mask]
    predictions[UNSEEN_EQ_SWITCH_CANDIDATE] = switched_pred
    
    return predictions, metrics
