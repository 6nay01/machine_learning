# 查询规模估算作业说明

本目录已提供一个可复现 baseline，用 SQL 查询结构和列统计信息预测查询结果规模。

## 环境

已创建项目内 conda 环境：

```bash
conda activate /home/shiyanleo/machine_learning/.conda-cardinality
```

主要依赖：

- pandas
- numpy
- scikit-learn
- xgboost

## 运行

```bash
/home/shiyanleo/machine_learning/.conda-cardinality/bin/python solve_cardinality.py
```

脚本会读取：

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `column_min_max_vals.csv`

并生成：

- `submission.csv`：Kaggle 提交文件
- `validation_metrics.json`：本地验证指标
- `validation_errors.csv`：验证集预测明细，按 Q-error 从高到低排序

## 方法摘要

特征构造包含表组合、连接条件、谓词列、谓词操作符、谓词值归一化、列统计选择率、表基数估计、等值谓词目标统计等信息。模型对 `log1p(Cardinality)` 训练，使用 `XGBoost` 梯度提升树，并在训练集内部学习分组残差校准。脚本还会训练低基数和高基数风险分类器，用于降低极端 Q-error。评价指标为 Mean Q-error。脚本运行时会输出 `[progress]` 进度提醒，XGBoost 本身每 100 或 200 轮输出一次训练进度。

验证阶段只用 dev 子集训练模型和校准器，再评估 validation 子集。最终提交阶段只用完整 `train.csv` 训练模型和校准器，`test.csv` 只用于特征转换与预测，不参与训练、调参或任何目标统计。

## 建议测试

- 检查 `submission.csv` 是否只有 `Id,Cardinality` 两列。
- 检查提交行数是否等于 `sample_submission.csv`。
- 检查预测值是否全部为正整数。
- 检查 `validation_metrics.json` 中的 Mean Q-error，作为本地效果参考。
- 检查 `validation_errors.csv` 前几行，定位极端 Q-error 的查询结构。
