# 标签清洗交接说明

## 一、字段：修正标签存在哪一列

`manual_audit_all_502_error_cases.csv`（502 行）在原字段基础上新增 4 列：

| 列名 | 含义 | 取值 |
|---|---|---|
| `sample_id` | 与原数据集对齐的唯一键（来自 `outputs/predictions/*.csv`） | 字符串 |
| `split` | 该样本所属划分 | 全部为 `test` |
| `corrected_label` | **人工修正后的标签** | `0`/`1`；非标签错的行留空 |
| `label_action` | 该行应如何处理 | `flip`(246) / `keep`(211) / `undecided`(45) |

判定依据列 `audit_label`、`audit_verdict`、`audit_reason` 保留，便于复核。

**权威清单**是 `label_fixes_test.csv`（246 行，仅含 `flip`），训练脚本只读这个文件，
不要直接拿 502 行的审计表去改标签——里面还有 211 条是"模型错"，改了就等于篡改真值。

翻转方向：`1→0` 163 条，`0→1` 83 条。

## 二、能否与原始数据集准确匹配

可以，已逐条验证（`src/validate_label_fixes.py`、`src/dryrun_label_fix.py`）：

- 502 条通过 `(review_norm, label, prediction)` 映射到 `sample_id`，**502/502 唯一命中**，
  与 `test_predictions.csv` 中 `correct=False` 的 502 条完全一致（集合相等）。
- test.csv 内部文本无重复（9406 行 / 9406 个唯一文本），不存在同文多行歧义。
- 修正清单 246 条全部存在于 test.csv，**0 条出现在 train/val**（无跨集合污染）。
- 未命中样本零改动，已断言验证。
- 补充：train 与 test 有 1 条文本重叠，`train_bert.py` 原本就会从训练集剔除，
  且该条不在审计集中，不影响结论。

## 三、必须先明确的方法论前提

这 502 条**全部来自测试集**。因此：

1. 修正它们的标签 = **修改评测真值**，不是清洗训练数据。指标变好只代表"原来的分数被
   标签噪声压低了"，**不能**说模型变强。
2. 这些样本**绝不能**挪进训练集（会造成测试集泄漏）。
3. `train_bert.py` 的 `--allow-split` 默认只放行 `train`，改测试集必须显式传
   `--allow-split test`，就是为了防止这一步被误操作。

## 四、最小改造与实验矩阵

`train_bert.py` 改动（原文件已备份为 `train_bert.py.bak`），不传参时行为与原脚本完全一致：

- 新增 `--label-fixes` / `--run-tag` / `--allow-split` / `--seed`
- 新增 `load_label_fixes()` 与 `apply_label_fixes()`（按 `sample_id` 精确替换，未命中即跳过）
- `load_data()` 保留 `label_orig` 列
- 最终评估同时输出**原标签口径**与**修正后口径**，并写 `metric_compare.csv`

三次运行即可完成对照：

```bash
# R0 基线：复现历史结果
python src/train_bert.py --run-tag baseline

# R1 口径修正：训练完全不变，只换评测真值（一次运行同时给出两个口径）
python src/train_bert.py --run-tag eval_fixed \
  --label-fixes outputs/bert/category_analysis/label_fixes_test.csv \
  --allow-split test

# R2 训练侧清洗（需先产出 label_fixes_train.csv，见第五节）
python src/train_bert.py --run-tag clean_train \
  --label-fixes outputs/bert/category_analysis/label_fixes_all.csv \
  --allow-split train test

# 汇总对照
python src/compare_clean_experiments.py
```

| 运行 | 训练集 | 评测口径 | 回答什么问题 |
|---|---|---|---|
| R0 | 原标签 | 原标签 | 历史 94.66% 能否复现 |
| R1 | 原标签 | 修正后 | 标签噪声压低了几个点（**非**模型提升） |
| R2 | 修正后 | 修正后 | 清洗是否真的提升模型 |

只有 R0 与 R2 在**同一列指标**上的差异，才是"清洗带来的模型提升"。

## 五、训练侧清洗的缺口

目前只审了测试集。train 有 43893 行（去重后），按 5% 左右的误判率估计约 2000 条噪声，
逐条人工审不现实。建议：

1. **先估噪声率**：每类分层随机抽 100 条人工审，判断是否值得投入清洗。
2. **规则筛查 + 人工确认**：把已识别的噪声模板（"好评打底 + 赠品/物流吐槽"、
   "差评 + 局部褒义转折"）写成规则，人工只确认命中样本。
3. **避免自我确认**：不要用"训练在该数据上的模型预测"去找训练集标签错误，
   应使用 out-of-fold 预测或纯人工判断。

## 六、已知残留问题

- val 集未审计，仍用于早停与选模型；各次运行一致，横向可比，但选择信号本身含噪声。
- 45 条 `undecided`（中性文本、纯标题、乱码、跑题）建议直接从评测集中剔除并单独报告，
  不要强行二分类。
- `计算机` 类存在同一句重复 21 次的样本，`水果` 类存在纯商品标题样本，建议单独清洗。
