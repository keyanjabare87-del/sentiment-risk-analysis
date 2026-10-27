# 阶段 2 报告：固定实验基线

## 0. 结论先行：现有四模型对比表不成立

项目里存在**两套互不相同的划分**：

- `train.csv / val.csv / test.csv` —— `train_bert.py` 使用（下称 BERT 家族）
- `train_split.csv / validation_split.csv / test_split.csv` —— `train_baselines.py` 在内部重新 `train_test_split` 生成（下称基线家族）

两族来自同一语料（62705 条唯一文本），但**划分结果不同**：

| 基线族划分 | 与 BERT-test 共同文本 | 其中 BERT 标签为 0 | 为 1 | 性质 |
|---|---|---|---|---|
| train_split | 1547 | 0 | 1547 | BERT的测试样本落在基线族的训练/验证集 → 交叉污染 |
| validation_split | 303 | 0 | 303 | BERT的测试样本落在基线族的训练/验证集 → 交叉污染 |
| test_split | 7556 | 4653 | 2903 | 同一批样本（可对齐） |

**关键后果**：BERT 测试集的 9406 条中，有 **1850 条**（19.7%）出现在基线家族的 train/val 里，且全部是正类。这意味着 TF-IDF 基线在这部分样本上是「见过答案」的，其 0.9304 与 BERT 的 0.9466 **不能直接比较**。

另外，两族的 `sample_id` 存在 ±1 漂移（共同文本中约一半 ID 不同），**跨族对齐只能按文本，不能按 sample_id**。

## 1. 基线结果（各自原划分，仅作存档）

| 实验 | 模型 | Accuracy | Macro-F1 | 负面 Recall | 负面 Precision | 可比性 |
|---|---|---|---|---|---|---|
| B0 | TF-IDF + LogisticRegression | 0.9243 | 0.9243 | 0.9329 | 0.9156 | 不可比：不同 test 划分 |
| B0 | TF-IDF + NaiveBayes | 0.9254 | 0.9254 | 0.9235 | 0.9255 | 不可比：不同 test 划分 |
| B0 | TF-IDF + LinearSVM | 0.9304 | 0.9304 | 0.9329 | 0.9268 | 不可比：不同 test 划分 |
| A | BERT-base-chinese | 0.9466 | 0.9466 | 0.9443 | 0.9476 | 可比（参考口径） |

## 2. 干净子集上的可比对照

取 BERT-test 中未被基线族 train/val 覆盖的 **7556 条**（负 4653 / 正 2903）重算，四个模型面对同一批样本：

| 模型 | Accuracy | Macro-F1 | 负面 Recall | 负面 Precision |
|---|---|---|---|---|
| TF-IDF + LogisticRegression | 0.9255 | 0.9216 | 0.9329 | 0.9453 |
| TF-IDF + NaiveBayes | 0.9254 | 0.9218 | 0.9235 | 0.9538 |
| TF-IDF + LinearSVM | 0.9305 | 0.9270 | 0.9329 | 0.9532 |
| BERT-base-chinese | 0.9447 | 0.9419 | 0.9443 | 0.9651 |

注意：该子集负类占 61.6%，明显高于全集 49.5%，因此**绝对值不能与全集指标直接对比**，只用于判断四个模型的相对排序。

## 3. 单族内部完整性

| 家族 | split | 行数 | 唯一文本 | 内部重复 | 负样本占比 |
|---|---|---|---|---|---|
| BERT家族 | train | 43894 | 43894 | 0 | 49.46% |
| BERT家族 | val | 9406 | 9406 | 0 | 49.47% |
| BERT家族 | test | 9406 | 9406 | 0 | 49.47% |
| 基线家族(历史) | train_split | 43893 | 43893 | 0 | 49.46% |
| 基线家族(历史) | validation_split | 9406 | 9406 | 0 | 49.47% |
| 基线家族(历史) | test_split | 9406 | 9406 | 0 | 49.47% |

- 两族内部均无重复文本；`sample_id` 在各自族内唯一。
- BERT 家族内 train 与 test 有 1 条文本重叠，`train_bert.py` 已从训练集剔除。
- 基线家族由 `train_test_split` 生成，本身无跨集合重叠。

## 4. 模型选择口径

- BERT：只按 `val macro-F1` 选 epoch，最佳 epoch=2（val Macro-F1=0.9480），测试集仅最终评估一次。
- TF-IDF 基线：按 `validation_f1_macro` 选型，最佳 `linear_svm`，同样未用测试集选型。
- 两族的选型流程都合规，问题只出在「划分本身不同」。

## 5. 统一划分的执行结果

已以 BERT 家族为准完成统一（原因：人工审计、错误分析、修正清单全部基于 `test.csv` 的 `sample_id`，换划分会让全部结论失效）：

| 动作 | 内容 |
|---|---|
| 唯一入口 | 新增 `src/data_split.py`，所有脚本通过它读取划分 |
| 修复根因 | `train_baselines.py` 删除内部 `train_test_split`，改读 `train.csv/val.csv/test.csv` |
| 隔离旧文件 | `*_split.csv`、旧基线预测、旧 joblib 移入 `outputs/predictions/_deprecated_alt_split/` |
| 诊断脚本 | `audit_data_leakage.py`、`check_duplicate_detail.py` 已改为读统一划分 |
| 校验 | `python src/verify_split_unified.py` |

**待你执行**：`python src/train_baselines.py` 重训三个 TF-IDF 基线（本机无 sklearn，无法代跑）。完成后 `outputs/metrics/test_results.csv` 才是可比的 B0 行。

推理耗时仍未测（本机无 torch/sklearn 环境），保持「待测」，不得预填。

## 6. 未测项

- 推理耗时 / 吞吐量：待测。
- B / C / D 三组实验：全部留空，由真实运行填写。
