# 数据说明

## 1. 语料

`raw/online_shopping_10_cats.csv`：中文电商评论情感数据集，10 个类目
（书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店）。

- 总行数：62,706
- 唯一文本：62,705（全语料仅 1 条重复文本）
- 标签定义：**`0` = 负面，`1` = 正面**（全项目统一，无例外）

`processed/reviews_clean.csv` 是清洗后的版本，仅供追溯；
**训练与评估不再从它划分数据**（见第 3 节）。

## 2. 唯一合法划分

划分文件位于 `outputs/predictions/`，是项目内**唯一**的划分来源：

| split | 文件 | 行数 | 负样本 | 正样本 | 负样本占比 |
|---|---|---|---|---|---|
| train | `train.csv` | 43,894 | 21,711 | 22,183 | 49.46% |
| val | `val.csv` | 9,406 | 4,653 | 4,753 | 49.47% |
| test | `test.csv` | 9,406 | 4,653 | 4,753 | 49.47% |

字段：`sample_id`（语料级唯一 ID）、`cat`（类目）、`label`（0/1）、`review`（原文）。

划分的完整性由 `src/data_split.py` 保证，任何脚本都必须通过它读取：

```python
from data_split import load, load_df, load_all, assert_valid

rows = load("test")        # list[dict]，纯标准库
df = load_df("test")       # pandas DataFrame
assert_valid()             # 训练前自检，失败即抛异常
```

## 3. 为什么不能自行划分

早期 `train_baselines.py` 在内部用 `sklearn.train_test_split` 重新划分了一遍，
产生了第二套划分，导致：

- 传统基线（LR/NB/SVM）与 BERT 的测试集不是同一批样本，指标不可比；
- BERT 测试集 9,406 条中有 **1,850 条**落在基线的训练/验证集里，且全部为正类；
- 两套 `sample_id` 存在 ±1 漂移，跨套只能按文本对齐。

**该问题已修复**：`train_baselines.py` 改为读取统一划分，旧的
`train_split.csv / validation_split.csv / test_split.csv` 已归档到
`outputs/predictions/_deprecated_alt_split/`，禁止再被读取。

验收命令：

```bash
python src/verify_split_unified.py     # 通过后写 outputs/experiments/split_manifest.json
```

## 4. 已知数据问题

1. **train 与 test 有 1 条文本重复**。`train_bert.py` 与 `train_baselines.py`
   都会在训练前从训练集剔除，test 保持不变。清洗后训练集为 43,893 条。
2. **标签噪声**。对 test 集 502 条误判做人工审计，确认 246 条标签极性写反
   （163 条差评标正、83 条好评标负、7 条评价对象错位）。
   修正清单见 `outputs/bert/category_analysis/label_fixes_test.csv`，
   **只用于评测口径分析，不得进入训练数据**。
3. `outputs/metrics/label_conflicts.csv` 记录了 36 条「同文本、不同标签」的冲突，
   属同一噪声来源。
4. 少量脏样本：`计算机` 类有一句重复 21 次；`水果` 类有纯商品标题；
   若干乱码与拼音残句。

## 5. 标签修正的使用边界

`label_fixes_test.csv` 中的 246 条全部属于 **test** 集。
用它们重算指标属于「人工复核后的测试集口径」，必须与原口径并列报告：

- 原标签口径：Accuracy 0.9466 / Macro-F1 0.9466
- 修正口径：Accuracy 0.9728 / Macro-F1 0.9728

**这 2.6 个百分点来自标签修正，不是模型改进**，不能用于宣称性能提升。
