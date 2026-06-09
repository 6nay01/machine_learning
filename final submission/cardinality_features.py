"""特征工程工具集。

负责从原始查询数据构建用于模型训练的特征表，包含：
- 解析 CSV 字段（表、谓词、连接条件）
- 基于列统计估计选择性
- 构建等值谓词的目标统计
- 将类别特征编码为稳定整数映射

模块中主要函数返回 Pandas DataFrame 或用于训练/预测的映射对象。
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TABLE_ALIASES = ["t", "mc", "ci", "mi", "mi_idx", "mk"]
PREDICATE_COLUMNS = [
    "t.kind_id",
    "t.production_year",
    "mc.company_id",
    "mc.company_type_id",
    "ci.person_id",
    "ci.role_id",
    "mi.info_type_id",
    "mi_idx.info_type_id",
    "mk.keyword_id",
]
OPERATORS = ["<", "=", ">"]
OP_NAME = {"<": "lt", "=": "eq", ">": "gt"}
EQUALITY_STATS_MIN_COUNT = 2
CATEGORICAL_COLS = [
    "table_combo",
    "join_combo",
    "predicate_signature",
    "full_signature",
    "tables_predicates_signature",
]


@dataclass(frozen=True)
class FeatureBundle:
    features: pd.DataFrame
    category_maps: dict[str, dict[str, int]]


def load_column_stats(path: Path) -> dict[str, dict[str, float]]:
    """读取列统计 CSV 并转成按列名索引的字典。"""
    stats_df = pd.read_csv(path)
    stats: dict[str, dict[str, float]] = {}
    for row in stats_df.to_dict("records"):
        stats[row["name"]] = {
            "min": float(row["min"]),
            "max": float(row["max"]),
            "cardinality": float(row["cardinality"]),
            "num_unique_values": float(row["num_unique_values"]),
        }
    return stats


def split_csv_field(value: object) -> list[str]:
    """将 CSV 中的逗号分隔字段拆成去空白的字符串列表。"""
    if pd.isna(value) or value == "":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_tables(value: object) -> tuple[list[str], list[str]]:
    """解析 Tables 字段，返回表名列表和别名列表。"""
    tables = split_csv_field(value)
    names: list[str] = []
    aliases: list[str] = []
    for item in tables:
        pieces = item.split()
        names.append(pieces[0])
        aliases.append(pieces[-1])
    return names, aliases


def parse_predicates(value: object) -> list[tuple[str, str, float]]:
    """将 Predicates 字段解析为 `(列, 操作符, 值)` 三元组列表。"""
    parts = split_csv_field(value)
    if len(parts) % 3 != 0:
        raise ValueError(f"Predicates 字段无法按三元组解析: {value!r}")
    predicates: list[tuple[str, str, float]] = []
    for idx in range(0, len(parts), 3):
        predicates.append((parts[idx], parts[idx + 1], float(parts[idx + 2])))
    return predicates


def table_combo_from_row(row: dict[str, object]) -> str:
    """从单条查询记录中提取按别名拼接的表组合键。"""
    _, aliases = parse_tables(row["Tables"])
    return "|".join(aliases)


def predicate_signature_from_predicates(predicates: list[tuple[str, str, float]]) -> str:
    """将谓词列表编码成仅保留列和操作符的签名。"""
    return "|".join(f"{col}:{op}" for col, op, _ in predicates) or "NONE"


def make_query_keys(row: dict[str, object]) -> dict[str, str]:
    """为一条查询构造多粒度分组键。"""
    _, aliases = parse_tables(row["Tables"])
    joins = split_csv_field(row["Join Conditions"])
    predicates = parse_predicates(row["Predicates"])

    table_key = "|".join(aliases)
    join_key = "|".join(sorted(joins))
    predicate_key = predicate_signature_from_predicates(predicates)
    return {
        "full": f"tables={table_key};joins={join_key};predicates={predicate_key}",
        "tables_predicates": f"tables={table_key};predicates={predicate_key}",
        "tables": f"tables={table_key}",
        "predicates": f"predicates={predicate_key}",
    }


def estimate_selectivity(col: str, op: str, val: float, stats: dict[str, dict[str, float]]) -> float:
    """根据列统计粗略估计单个谓词的选择率。"""
    col_stats = stats[col]
    col_min = col_stats["min"]
    col_max = col_stats["max"]
    cardinality = col_stats["cardinality"]
    unique = col_stats["num_unique_values"]
    floor = 1.0 / max(cardinality, 1.0)

    if op == "=":
        return max(floor, 1.0 / max(unique, 1.0))

    span = max(col_max - col_min, 1.0)
    if op == "<":
        ratio = (val - col_min) / span
    else:
        ratio = (col_max - val) / span
    return min(1.0, max(floor, ratio))


def equality_key(col: str, val: float) -> str:
    """构造不区分表组合的等值谓词键。"""
    return f"{col}={int(val)}"


def table_equality_key(table_combo: str, col: str, val: float) -> str:
    """构造带表组合上下文的等值谓词键。"""
    return f"{table_combo};{equality_key(col, val)}"


def fit_equality_target_stats(train_df: pd.DataFrame) -> dict[str, object]:
    """统计训练集中等值谓词对应的目标分布特征。"""
    value_logs: dict[str, list[float]] = defaultdict(list)
    table_value_logs: dict[str, list[float]] = defaultdict(list)
    counts_by_col: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in train_df.to_dict("records"):
        log_cardinality = math.log1p(float(row["Cardinality"]))
        table_combo = table_combo_from_row(row)
        for col, op, val in parse_predicates(row["Predicates"]):
            if op != "=":
                continue
            key = equality_key(col, val)
            value_logs[key].append(log_cardinality)
            table_value_logs[table_equality_key(table_combo, col, val)].append(log_cardinality)
            counts_by_col[col][key] += 1

    value_stats: dict[str, dict[str, float]] = {}
    for key, logs in value_logs.items():
        if len(logs) >= EQUALITY_STATS_MIN_COUNT:
            arr = np.asarray(logs, dtype=float)
            value_stats[key] = {
                "count": float(len(logs)),
                "mean_log": float(np.mean(arr)),
                "median_log": float(np.median(arr)),
                "p10_log": float(np.quantile(arr, 0.10)),
                "p90_log": float(np.quantile(arr, 0.90)),
            }

    table_value_stats: dict[str, dict[str, float]] = {}
    for key, logs in table_value_logs.items():
        if len(logs) >= EQUALITY_STATS_MIN_COUNT:
            arr = np.asarray(logs, dtype=float)
            table_value_stats[key] = {
                "count": float(len(logs)),
                "median_log": float(np.median(arr)),
            }

    frequency_rank: dict[str, float] = {}
    for col_counts in counts_by_col.values():
        sorted_counts = sorted(col_counts.items(), key=lambda item: item[1])
        denom = max(len(sorted_counts) - 1, 1)
        for rank, (key, _) in enumerate(sorted_counts):
            frequency_rank[key] = float(rank / denom)

    return {
        "value_stats": value_stats,
        "table_value_stats": table_value_stats,
        "frequency_rank": frequency_rank,
    }


def init_feature_row() -> dict[str, float | str]:
    """初始化一行特征字典及默认值。"""
    feat: dict[str, float | str] = {}
    for name in [
        "eq_query_count_max",
        "eq_query_count_sum",
        "eq_seen_count",
        "eq_unseen_count",
        "eq_frequency_rank_max",
        "eq_frequency_rank_mean",
        "eq_target_mean_log_mean",
        "eq_target_median_log_min",
        "eq_target_median_log_max",
        "eq_target_median_log_mean",
        "eq_target_p10_log_min",
        "eq_target_p90_log_max",
        "eq_table_target_seen_count",
        "eq_table_target_median_log_mean",
    ]:
        feat[name] = 0.0

    for alias in TABLE_ALIASES:
        feat[f"has_table_{alias}"] = 0.0
    for col in PREDICATE_COLUMNS:
        safe_col = col.replace(".", "_")
        feat[f"{safe_col}_has"] = 0.0
        feat[f"{safe_col}_count"] = 0.0
        feat[f"{safe_col}_min_norm"] = 1.0
        feat[f"{safe_col}_max_norm"] = 0.0
        feat[f"{safe_col}_mean_norm"] = 0.0
        feat[f"{safe_col}_min_selectivity"] = 1.0
        feat[f"{safe_col}_sum_log_selectivity"] = 0.0
        for op in OPERATORS:
            feat[f"{safe_col}_op_{OP_NAME[op]}"] = 0.0
    return feat


def build_feature_frame(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    equality_stats: dict[str, object] | None = None,
) -> pd.DataFrame:
    """将原始查询数据转换为模型使用的特征表。"""
    rows: list[dict[str, float | str]] = []
    value_stats = {}
    table_value_stats = {}
    frequency_rank = {}
    if equality_stats is not None:
        value_stats = equality_stats["value_stats"]
        table_value_stats = equality_stats["table_value_stats"]
        frequency_rank = equality_stats["frequency_rank"]

    for row in df.to_dict("records"):
        table_names, aliases = parse_tables(row["Tables"])
        joins = split_csv_field(row["Join Conditions"])
        predicates = parse_predicates(row["Predicates"])
        query_keys = make_query_keys(row)
        table_combo = "|".join(aliases)

        feat = init_feature_row()
        feat["table_combo"] = table_combo
        feat["join_combo"] = "|".join(sorted(joins)) or "NONE"
        feat["predicate_signature"] = query_keys["predicates"]
        feat["full_signature"] = query_keys["full"]
        feat["tables_predicates_signature"] = query_keys["tables_predicates"]
        feat["n_tables"] = float(len(table_names))
        feat["n_joins"] = float(len(joins))
        feat["n_predicates"] = float(len(predicates))

        table_cards: list[float] = []
        for alias in aliases:
            feat[f"has_table_{alias}"] = 1.0
            table_cards.append(stats[f"{alias}.id"]["cardinality"])
        log_product_card = sum(math.log1p(card) for card in table_cards)
        feat["log_product_table_card"] = log_product_card
        feat["mean_log_table_card"] = log_product_card / max(len(table_cards), 1)

        selectivities: list[float] = []
        eq_counts: list[float] = []
        eq_frequency_ranks: list[float] = []
        eq_mean_logs: list[float] = []
        eq_median_logs: list[float] = []
        eq_p10_logs: list[float] = []
        eq_p90_logs: list[float] = []
        eq_table_medians: list[float] = []
        norm_values_by_col: dict[str, list[float]] = defaultdict(list)
        selectivities_by_col: dict[str, list[float]] = defaultdict(list)

        for col, op, val in predicates:
            col_stats = stats[col]
            col_min = col_stats["min"]
            col_max = col_stats["max"]
            norm_val = (val - col_min) / max(col_max - col_min, 1.0)
            sel = estimate_selectivity(col, op, val, stats)
            safe_col = col.replace(".", "_")

            selectivities.append(sel)
            norm_values_by_col[col].append(norm_val)
            selectivities_by_col[col].append(sel)
            feat[f"{safe_col}_has"] = 1.0
            feat[f"{safe_col}_count"] = float(feat[f"{safe_col}_count"]) + 1.0
            op_name = OP_NAME[op]
            feat[f"{safe_col}_op_{op_name}"] = float(feat[f"{safe_col}_op_{op_name}"]) + 1.0
            feat[f"{safe_col}_{op_name}_norm"] = norm_val
            feat[f"{safe_col}_{op_name}_value"] = val
            feat[f"{safe_col}_{op_name}_log_value"] = math.log1p(max(val, 0.0))
            feat[f"{safe_col}_{op_name}_selectivity"] = sel
            feat[f"{safe_col}_{op_name}_log_selectivity"] = math.log(max(sel, 1e-12))

            if equality_stats is not None and op == "=":
                key = equality_key(col, val)
                stat = value_stats.get(key)
                if stat is None:
                    feat["eq_unseen_count"] = float(feat["eq_unseen_count"]) + 1.0
                else:
                    eq_counts.append(stat["count"])
                    eq_mean_logs.append(stat["mean_log"])
                    eq_median_logs.append(stat["median_log"])
                    eq_p10_logs.append(stat["p10_log"])
                    eq_p90_logs.append(stat["p90_log"])
                eq_frequency_ranks.append(float(frequency_rank.get(key, 0.0)))
                table_stat = table_value_stats.get(table_equality_key(table_combo, col, val))
                if table_stat is not None:
                    eq_table_medians.append(table_stat["median_log"])

        for col, values in norm_values_by_col.items():
            safe_col = col.replace(".", "_")
            feat[f"{safe_col}_min_norm"] = min(values)
            feat[f"{safe_col}_max_norm"] = max(values)
            feat[f"{safe_col}_mean_norm"] = sum(values) / len(values)
        for col, values in selectivities_by_col.items():
            safe_col = col.replace(".", "_")
            feat[f"{safe_col}_min_selectivity"] = min(values)
            feat[f"{safe_col}_sum_log_selectivity"] = sum(math.log(max(v, 1e-12)) for v in values)

        if selectivities:
            sum_log_sel = sum(math.log(max(sel, 1e-12)) for sel in selectivities)
            feat["sum_log_selectivity"] = sum_log_sel
            feat["product_selectivity"] = math.exp(sum_log_sel)
            feat["min_selectivity"] = min(selectivities)
            feat["max_selectivity"] = max(selectivities)
            feat["mean_selectivity"] = sum(selectivities) / len(selectivities)
            feat["independent_log_card_estimate"] = log_product_card + sum_log_sel
        else:
            feat["sum_log_selectivity"] = 0.0
            feat["product_selectivity"] = 1.0
            feat["min_selectivity"] = 1.0
            feat["max_selectivity"] = 1.0
            feat["mean_selectivity"] = 1.0
            feat["independent_log_card_estimate"] = log_product_card

        if eq_counts:
            feat["eq_query_count_max"] = max(eq_counts)
            feat["eq_query_count_sum"] = sum(eq_counts)
            feat["eq_seen_count"] = float(len(eq_counts))
            feat["eq_target_mean_log_mean"] = sum(eq_mean_logs) / len(eq_mean_logs)
            feat["eq_target_median_log_min"] = min(eq_median_logs)
            feat["eq_target_median_log_max"] = max(eq_median_logs)
            feat["eq_target_median_log_mean"] = sum(eq_median_logs) / len(eq_median_logs)
            feat["eq_target_p10_log_min"] = min(eq_p10_logs)
            feat["eq_target_p90_log_max"] = max(eq_p90_logs)
        if eq_frequency_ranks:
            feat["eq_frequency_rank_max"] = max(eq_frequency_ranks)
            feat["eq_frequency_rank_mean"] = sum(eq_frequency_ranks) / len(eq_frequency_ranks)
        if eq_table_medians:
            feat["eq_table_target_seen_count"] = float(len(eq_table_medians))
            feat["eq_table_target_median_log_mean"] = sum(eq_table_medians) / len(eq_table_medians)

        rows.append(feat)

    return pd.DataFrame(rows)


def fit_category_maps(feature_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """为类别特征拟合稳定的整数编码映射。"""
    maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_COLS:
        values = feature_df[col].fillna("NONE").astype(str).unique()
        maps[col] = {value: idx for idx, value in enumerate(sorted(values), start=1)}
    return maps


def apply_category_maps(feature_df: pd.DataFrame, maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    """按给定映射对类别特征做整数编码。"""
    encoded = feature_df.copy()
    for col, mapping in maps.items():
        encoded[col] = encoded[col].fillna("NONE").astype(str).map(mapping).fillna(0).astype(np.int32)
    return encoded.fillna(0.0)


def align_feature_columns(reference_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """按参考特征列对齐目标特征表。"""
    return target_df.reindex(columns=reference_df.columns, fill_value=0.0)


def build_train_target_features(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, dict[str, int]]]:
    """构建训练集和目标集的对齐特征及其辅助统计。"""
    equality_stats = fit_equality_target_stats(source_df)
    source_features = build_feature_frame(source_df, stats, equality_stats)
    target_features = build_feature_frame(target_df, stats, equality_stats)
    category_maps = fit_category_maps(source_features)
    source_features = apply_category_maps(source_features, category_maps)
    target_features = apply_category_maps(target_features, category_maps)
    target_features = align_feature_columns(source_features, target_features)
    return source_features, target_features, equality_stats, category_maps
