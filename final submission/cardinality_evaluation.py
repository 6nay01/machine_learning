"""卡方/误差评估工具。

此模块提供用于计算 Q-error（基数估计误差）、将对数预测
转换回整数基数以及汇总 Q-error 统计量的辅助函数。
所有函数均以 numpy 数组为输入/输出。
"""

import numpy as np

MAX_LOG_PREDICTION = 25.0

def q_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """计算逐条样本的 Q-error。"""
    y_true = np.maximum(y_true.astype(float), 1.0)
    y_pred = np.maximum(y_pred.astype(float), 1.0)
    return np.maximum(y_pred / y_true, y_true / y_pred)


def log_predictions_to_cardinality(log_pred: np.ndarray) -> np.ndarray:
    """将对数空间预测还原为正整数基数。"""
    safe_log = np.nan_to_num(log_pred, nan=0.0, posinf=MAX_LOG_PREDICTION, neginf=0.0)
    return np.maximum(np.rint(np.expm1(np.clip(safe_log, 0.0, MAX_LOG_PREDICTION))), 1)


def summarize_q_errors(prefix: str, errors: np.ndarray) -> dict[str, float]:
    """汇总一组 Q-error 的统计量。"""
    return {
        f"{prefix}_mean_q_error": float(np.mean(errors)),
        f"{prefix}_median_q_error": float(np.median(errors)),
        f"{prefix}_p90_q_error": float(np.quantile(errors, 0.90)),
        f"{prefix}_p95_q_error": float(np.quantile(errors, 0.95)),
        f"{prefix}_max_q_error": float(np.max(errors)),
    }


def evaluate_log_predictions(prefix: str, y_true: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    """评估对数预测并返回带前缀的指标字典。"""
    return summarize_q_errors(prefix, q_error(y_true, log_predictions_to_cardinality(log_pred)))
