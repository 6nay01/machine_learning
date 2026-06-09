"""计算提交文件与带真实基数的 CSV 之间平均 Q-error 的小工具。

该脚本通过命令行参数接收真实值 CSV 与提交 CSV，合并后
计算并打印均值 Q-error 与匹配行数，常用于本地验证提交质量。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cardinality_evaluation import q_error


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回包含输入/输出路径的命名空间。

    返回值包含 `truth_csv`、`submission_csv` 两个 `Path` 类型的属性，
    对应真实基数文件和提交文件的路径。
    """
    parser = argparse.ArgumentParser(
        description="Compute the mean q-error between test_with_true_cardinality.csv and submission.csv."
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path("test_with_true_cardinality.csv"),
        help="Path to the CSV containing true cardinalities.",
    )
    parser.add_argument(
        "--submission-csv",
        type=Path,
        default=Path("submission.csv"),
        help="Path to the submission CSV with predicted cardinalities.",
    )
    return parser.parse_args()


def compute_mean_q_error(truth_csv: Path, submission_csv: Path) -> tuple[float, int]:
    """读取两个 CSV，按 Id 合并并计算平均 Q-error。

    返回 (mean_q_error, matched_row_count)。当两个文件没有重叠 Id 时抛出 ValueError。
    """
    truth_df = pd.read_csv(truth_csv)
    submission_df = pd.read_csv(submission_csv)

    required_columns = {"Id", "Cardinality"}
    missing_truth = required_columns - set(truth_df.columns)
    missing_submission = required_columns - set(submission_df.columns)
    if missing_truth:
        raise ValueError(f"truth CSV missing columns: {sorted(missing_truth)}")
    if missing_submission:
        raise ValueError(f"submission CSV missing columns: {sorted(missing_submission)}")

    merged = truth_df[["Id", "Cardinality"]].merge(
        submission_df[["Id", "Cardinality"]],
        on="Id",
        how="inner",
        suffixes=("_true", "_pred"),
    )
    if merged.empty:
        raise ValueError("No overlapping Id values found between the two CSV files.")

    y_true = merged["Cardinality_true"].to_numpy(dtype=float)
    y_pred = merged["Cardinality_pred"].to_numpy(dtype=float)
    errors = q_error(y_true, y_pred)
    return float(errors.mean()), int(len(merged))


def main() -> None:
    """命令行入口：解析参数并打印均值 Q-error 与匹配行数。

    直接读取文件并将结果输出到标准输出，适合在脚本/CI 中调用。
    """
    args = parse_args()
    mean_q_error, row_count = compute_mean_q_error(args.truth_csv, args.submission_csv)
    print(f"mean_q_error={mean_q_error:.6f}")
    print(f"matched_rows={row_count}")


if __name__ == "__main__":
    main()