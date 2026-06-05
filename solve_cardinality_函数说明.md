# solve_cardinality.py 函数说明

本文档详细说明 `solve_cardinality.py` 中各函数的作用、输入输出，以及整个脚本从启动到结束的调用顺序。

## 1. 文件整体职责

`solve_cardinality.py` 是整个查询基数估计实验的主入口脚本，负责串联以下几个阶段：

1. 解析命令行参数
2. 读取训练集、测试集、样例提交文件和列统计信息
3. 在 `train.csv` 内部做一次训练/验证划分，比较多个候选预测器
4. 用完整训练集重新训练所有候选，并对 `test.csv` 生成预测
5. 根据策略或 `public truth` 选择最终候选
6. 写出 `submission.csv`
7. 如果提供了 `public truth`，再生成诊断报表和统计指标

它本身不实现底层特征工程和模型训练，而是调用：

- `cardinality_features.py`
- `cardinality_models.py`
- `cardinality_evaluation.py`

完成各模块任务。

## 2. 顶层常量

### 2.1 `RANDOM_STATE = 20260519`

作用：

- 控制内部训练/验证划分的随机种子
- 保证每次运行的切分结果稳定可复现

### 2.2 `SUPPORTED_STRATEGIES = ["auto", "main", "eq_stats", "low_expert", "residual", "blend"]`

作用：

- 约束命令行参数 `--strategy` 可选值
- 控制最终从哪一个候选预测器中选择输出

说明：

- `auto`：自动选择
- `main`：强制选主模型
- `eq_stats`：强制选 `eq_stats_main`
- `low_expert` / `residual`：强制选对应候选
- `blend`：强制只在所有 `blend_low_000` 到 `blend_low_100` 混合比例候选中选择最优结果

## 3. 各函数详细说明

## 3.1 `log_step(message: str) -> None`

作用：

- 打印统一格式的进度日志
- 便于在训练过程中观察脚本当前处于哪个阶段

输入：

- `message`：要输出的进度信息

输出：

- 无返回值，只向标准输出打印 `[progress] ...`

在流程中的位置：

- 几乎所有关键阶段都会调用它，例如读数据、内部验证开始、全量训练开始、候选选择开始、脚本结束等

## 3.2 `strategy_to_candidate(strategy: str, candidate_names: set[str]) -> str`

作用：

- 把用户传入的策略名映射成真正的候选预测器名称

输入：

- `strategy`：命令行传入的策略
- `candidate_names`：当前已经生成的候选预测器名字集合

输出：

- 最终要使用的候选名字符串

逻辑说明：

- `main` 不直接返回 `"main"`，而是优先映射到 `"main_depth6"`
- `eq_stats` 映射到 `"eq_stats_main"`
- `low_expert`、`residual` 直接返回自身
- `blend` 不由该函数直接映射为固定候选，而是在 `choose_final_candidate(...)` 中从全部混合比例候选里重新选择

在流程中的位置：

- 只在 `choose_final_candidate(...)` 中被调用
- 用于 `--strategy != auto` 的分支

## 3.3 `candidate_metric_rows(source, y_true, predictions) -> list[dict]`

作用：

- 计算一组候选预测器在某个有真值数据集上的误差指标

输入：

- `source`：指标来源标记，例如 `"internal_validation"`
- `y_true`：真实基数数组
- `predictions`：候选名到对数预测值数组的映射

输出：

- 一个列表，每个元素对应一个候选预测器的统计结果

每个候选会计算：

- `mean_q_error`
- `median_q_error`
- `p90_q_error`
- `p95_q_error`
- `max_q_error`

在流程中的位置：

- 在 `run_internal_validation(...)` 中使用
- 用于比较内部验证集上不同候选的表现

## 3.4 `candidate_public_metric_rows(truth_df, test_df, predictions) -> list[dict]`

作用：

- 使用 `public truth` 对测试集上的多个候选预测结果进行打分

输入：

- `truth_df`：包含测试集部分真值的 DataFrame
- `test_df`：测试集原始查询结构
- `predictions`：候选名到测试集对数预测值数组的映射

输出：

- 每个候选在 public truth 上的误差统计列表

内部做法：

1. 从 `truth_df` 取出 `Id` 和真实 `Cardinality`
2. 从 `test_df` 取出 `Id`
3. 把每个候选的对数预测转回整数预测值
4. 按 `Id` 与 truth 合并
5. 计算各候选在 public truth 样本上的 Q-error 统计

在流程中的位置：

- 只在 `choose_final_candidate(...)` 中使用
- 仅当 `--public-truth` 存在且 `--strategy=auto` 时才会被调用

重要说明：

- 它不参与模型训练，只用于训练完成后的候选比较

## 3.5 `write_validation_errors(val_df, predictions, selected_candidate, output_path) -> None`

作用：

- 将内部验证集上的预测明细写成 CSV，便于后续误差分析

输入：

- `val_df`：验证集原始数据
- `predictions`：多个候选预测结果
- `selected_candidate`：内部验证阶段选中的候选名
- `output_path`：输出文件路径

输出：

- 无返回值，直接写出 CSV 文件

输出内容包括：

- 样本原始字段：`Id`、`Tables`、`Join Conditions`、`Predicates`、`Cardinality`
- 每个候选的预测列
- 最终选中候选的 `Prediction`
- 对应的 `QError`

并且会按 `QError` 从高到低排序。

在流程中的位置：

- 在 `run_internal_validation(...)` 结束前调用

## 3.6 `run_internal_validation(train_df, stats, validation_errors_path) -> tuple[...]`

作用：

- 在 `train.csv` 内部做一次验证，用来先淘汰明显较差的候选，并得到一组本地指标

输入：

- `train_df`：完整训练集
- `stats`：列统计信息字典
- `validation_errors_path`：验证误差明细文件路径

输出：

- `metrics`：内部验证阶段的指标字典
- `report`：各候选在验证集上的比较表
- `selected_candidate`：内部验证选中的候选名

详细步骤：

1. 用 `train_test_split` 将训练集按 8:2 划分为 `dev_df` 和 `val_df`
2. 用 `build_train_target_features(dev_df, val_df, stats)` 构造训练特征和验证特征
3. 从 `dev_df` 取 `y_dev`，从 `val_df` 取 `y_val`
4. 调用 `candidate_predictions(...)` 训练多个候选，并对验证集输出预测
5. 调用 `candidate_metric_rows(...)` 计算每个候选的验证指标
6. 按 `mean_q_error`、`p95_q_error`、`max_q_error` 排序，选出最优候选
7. 调用 `write_validation_errors(...)` 写出误差明细
8. 调用 `evaluate_log_predictions(...)` 计算最终选中候选的详细指标
9. 汇总特征数、等值统计数量、最佳候选、误差文件路径等信息并返回

在整体流程中的作用：

- 这是脚本的第一段训练过程
- 它不会产生最终提交，但会产生后续全量训练要参考的一些指标，例如最佳迭代轮数

## 3.7 `train_full_candidate_predictions(train_df, test_df, stats, internal_metrics) -> tuple[...]`

作用：

- 使用完整训练集重新训练所有候选模型，并对 `test.csv` 产生预测

输入：

- `train_df`：完整训练集
- `test_df`：测试集
- `stats`：列统计信息字典
- `internal_metrics`：内部验证阶段得到的指标字典

输出：

- `predictions`：测试集上所有候选的预测结果
- `metrics`：全量训练阶段的补充指标

详细步骤：

1. 调用 `build_train_target_features(train_df, test_df, stats)` 构造训练和测试特征
2. 提取训练标签 `y_train`
3. 根据内部验证中各主模型的最佳轮数，构造 `round_overrides`
4. 调用 `candidate_predictions(...)`
5. 此时传入 `y_target=None`，表示测试集没有真值，只生成预测，不做验证评估
6. 返回所有候选在 `test.csv` 上的预测，以及特征维度、等值统计数量等信息

重要说明：

- 这个函数是最终提交前真正的全量训练阶段
- 注释和日志里已经明确写明：此阶段不读取 `public truth`

## 3.8 `choose_final_candidate(strategy, test_df, test_predictions, public_truth_path, candidate_report_path, internal_report) -> tuple[...]`

作用：

- 在已经得到所有测试集候选预测后，决定最终使用哪一个候选

输入：

- `strategy`：命令行策略
- `test_df`：测试集
- `test_predictions`：所有候选对测试集的预测
- `public_truth_path`：public truth 文件路径，可以为空
- `candidate_report_path`：候选比较报告输出路径
- `internal_report`：内部验证候选比较表

输出：

- `selected`：最终选中的候选名
- `report`：最终保存的候选报告表
- `metrics`：候选选择阶段的指标字典

它有三条主要分支。

### 分支一：`strategy != auto`

步骤：

1. 如果策略不是 `blend`，调用 `strategy_to_candidate(...)` 把策略转换成候选名
2. 如果策略是 `blend`，只比较候选名以 `blend_low_` 开头的比例候选
3. 有 `public truth` 时，使用 public truth 指标在全部 blend 比例中选最优
4. 没有 `public truth` 时，使用内部验证指标在全部 blend 比例中选最优
5. 将候选报告写入 `candidate_report_path`
6. 返回 `selected_by`

特点：

- `main`、`eq_stats`、`low_expert`、`residual` 仍然是显式指定单个候选
- `blend` 表示显式指定“融合类候选”，但融合比例仍会自动择优
- 选中的 blend 候选会在指标中记录 `selected_blend_low_expert_weight` 和 `selected_blend_other_weight`

### 分支二：`strategy == auto` 且没有 `public_truth`

步骤：

1. 直接使用内部验证报告第一名候选
2. 写出内部验证报告
3. 如果第一名是 blend 比例候选，同时记录对应融合权重
4. 返回 `selected_by = "internal_validation"`

特点：

- 最终候选完全由 `train.csv` 内部验证决定

### 分支三：`strategy == auto` 且存在 `public_truth`

步骤：

1. 读取 `public_truth_path`
2. 调用 `candidate_public_metric_rows(...)` 计算每个候选在 public truth 上的误差
3. 调用 `select_candidate_by_metrics(...)` 选出 public 最优候选
4. 将 public 报告与内部验证报告按 `candidate` 合并
5. 写出最终 `candidate_report_path`
6. 如果第一名是 blend 比例候选，同时记录对应融合权重
7. 返回 `selected_by = "public_truth_post_training"`

特点：

- 这是当前你实验里实际走到的分支
- `test_with_true_cardinality.csv` 在这里被使用
- 它参与候选选择，但不参与模型参数训练

## 3.9 `write_submission(sample_df, log_pred, output_path) -> pd.DataFrame`

作用：

- 根据选中的候选预测，生成标准 Kaggle 提交文件

输入：

- `sample_df`：样例提交文件，主要用于保留正确的 `Id`
- `log_pred`：选中候选的对数预测值
- `output_path`：输出提交文件路径

输出：

- 返回写出的提交 DataFrame

详细步骤：

1. 从 `sample_df` 取出 `Id`
2. 调用 `log_predictions_to_cardinality(...)` 把对数预测转为整数基数
3. 写出 `Id, Cardinality` 两列到 CSV

在流程中的位置：

- 在最终候选确定后调用

## 3.10 `main() -> None`

作用：

- 脚本的总控制函数
- 负责按正确顺序调用前面所有函数

输入：

- 无显式函数参数，所有输入来自命令行参数

输出：

- 无返回值，但会写出多个结果文件并打印最终指标

它完成的工作包括：

1. 定义并解析命令行参数
2. 读取训练、测试、样例和统计文件
3. 调用内部验证流程
4. 调用全量训练流程
5. 选择最终候选
6. 写出提交文件
7. 如果提供 public truth，则生成诊断报表
8. 汇总指标并写入 `validation_metrics.json`
9. 打印保存路径和总耗时

## 4. `main()` 中的参数说明

`main()` 支持的关键参数如下：

- `--train`：训练集路径，默认 `train.csv`
- `--test`：测试集路径，默认 `test.csv`
- `--sample`：样例提交路径，默认 `sample_submission.csv`
- `--stats`：列统计路径，默认 `column_min_max_vals.csv`
- `--output`：提交文件路径，默认 `submission.csv`
- `--metrics`：指标 JSON 路径，默认 `validation_metrics.json`
- `--validation-errors`：内部验证误差明细，默认 `validation_errors.csv`
- `--public-truth`：public truth 文件，可选
- `--candidate-report`：候选比较报告路径，默认 `candidate_metrics.csv`
- `--diagnostic-prefix`：public truth 诊断文件前缀
- `--strategy`：候选选择策略

## 5. 整体调用顺序

下面按照脚本实际执行顺序说明各函数如何串联。

### 第一步：解析参数

脚本启动后，首先进入 `main()`，创建 `ArgumentParser` 并读取命令行参数。

### 第二步：读取输入文件

`main()` 依次读取：

1. `train.csv`
2. `test.csv`
3. `sample_submission.csv`
4. `column_min_max_vals.csv`

此时会调用：

- `load_column_stats(...)`
- 多次 `pd.read_csv(...)`

### 第三步：内部验证

`main()` 调用：

- `run_internal_validation(train_df, stats, args.validation_errors)`

在这个阶段内部，调用链是：

1. `build_train_target_features(...)`
2. `candidate_predictions(...)`
3. `candidate_metric_rows(...)`
4. `write_validation_errors(...)`
5. `evaluate_log_predictions(...)`

这个阶段会得到：

- 内部验证指标
- 候选比较表
- 内部验证最佳候选

### 第四步：全量训练并预测测试集

`main()` 调用：

- `train_full_candidate_predictions(train_df, test_df, stats, validation_metrics)`

该阶段内部调用链是：

1. `build_train_target_features(...)`
2. `candidate_predictions(...)`

这里会生成：

- 所有候选在 `test.csv` 上的预测
- `blend_low_000` 到 `blend_low_100` 这一组混合比例候选，其中数字表示 `low_expert` 权重百分比，另一部分权重给 `residual`

### 第五步：选择最终候选

`main()` 调用：

- `choose_final_candidate(...)`

可能走三条不同分支：

1. 显式策略分支
2. 内部验证自动选择分支
3. public truth 自动选择分支

如果走 public truth 分支，内部还会调用：

1. `candidate_public_metric_rows(...)`
2. `select_candidate_by_metrics(...)`

### 第六步：生成提交文件

确定最终候选后，`main()` 调用：

- `write_submission(sample_df, test_predictions[selected_candidate], args.output)`

这一步生成最终的 `submission.csv`。

### 第七步：生成 public truth 诊断

如果命令行提供了 `--public-truth` 且文件存在，`main()` 会：

1. 再次读取 `public truth`
2. 调用 `public_truth_diagnostics(...)`

这个阶段会生成多份诊断文件，例如：

- `*_top_errors.csv`
- `*_by_bucket.csv`
- `*_by_table_combo.csv`
- `*_by_predicate_signature.csv`
- `*_by_equality_column.csv`
- `*_summary.json`

### 第八步：写出总指标并结束

最后 `main()` 会：

1. 把所有阶段的指标合并成一个字典
2. 写入 `validation_metrics.json`
3. 打印结果路径
4. 记录总耗时

## 6. 文字版执行时序图

可以把整个脚本理解成下面这条调用链：

```text
main
  -> 读取 train/test/sample/stats
  -> run_internal_validation
       -> build_train_target_features
       -> candidate_predictions
       -> candidate_metric_rows
       -> write_validation_errors
       -> evaluate_log_predictions
  -> train_full_candidate_predictions
       -> build_train_target_features
       -> candidate_predictions
  -> choose_final_candidate
       -> strategy_to_candidate                # 若 strategy != auto
       -> candidate_public_metric_rows         # 若有 public truth
       -> select_candidate_by_metrics          # 若有 public truth
  -> write_submission
  -> public_truth_diagnostics                  # 若有 public truth
  -> 写 validation_metrics.json
  -> 结束
```

## 7. 当前这次实验实际走过的路径

结合你当前的运行方式：

```bash
/home/shiyanleo/machine_learning/.conda-cardinality/bin/python solve_cardinality.py \
  --public-truth test_with_true_cardinality.csv \
  --candidate-report candidate_metrics.csv \
  --diagnostic-prefix public_truth_diagnostics
```

实际执行路径是：

1. `main()`
2. 读取 `train.csv`、`test.csv`、`sample_submission.csv`、`column_min_max_vals.csv`
3. `run_internal_validation(...)`
4. `train_full_candidate_predictions(...)`
5. `choose_final_candidate(...)`
6. 因为 `strategy=auto` 且存在 `public_truth`，走 public truth 选择分支
7. `write_submission(...)`
8. `public_truth_diagnostics(...)`
9. 写出 `validation_metrics.json`

也就是说，这次运行里：

- `train.csv` 参与了训练和内部验证
- `test.csv` 参与了特征转换和最终预测
- `test_with_true_cardinality.csv` 只参与最终候选选择和诊断，不参与模型训练

## 8. 阅读代码时建议重点关注的部分

如果你要继续深读 `solve_cardinality.py`，建议优先关注这四个函数：

1. `run_internal_validation(...)`
   这是“本地怎么比较候选”的核心

2. `train_full_candidate_predictions(...)`
   这是“最终怎么训练测试集预测”的核心

3. `choose_final_candidate(...)`
   这是“最终到底用哪个候选”的核心

4. `main()`
   这是“全局调用顺序和文件输入输出”的核心

如果你愿意，我下一步可以继续写第二个配套文件，专门讲 `candidate_predictions(...)` 在 `cardinality_models.py` 里是如何一步步训练出 `main_depth6`、`residual`、`blend` 这些候选的。  
