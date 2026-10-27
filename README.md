# 电商评论情感分类（中文 BERT）

面向电商负面评论识别的二分类项目。核心工作不是刷分数，而是**先确认这份数据的标签有多脏，再判断改进手段值不值得做**。

**一句话结论**：标签噪声是「评测指标被低估」和「错误分析被扭曲」的主因（测试集 2.6% 的标签极性写反，修正后同一模型准确率 0.9466 → 0.9728），但**不是**「模型性能」的主因——清洗与增强带来的 +0.27 个百分点在统计上不可检测。

详细分析与结论见 [`reports/final_report.md`](reports/final_report.md)。

---

## 1. 业务问题

平台需要对商品评论做负面识别。漏判一条真实差评（FN）的代价高于误报（FP），
因此评估不能只看 Accuracy，必须同时报告**负面类 Recall / Precision** 与混淆矩阵。

## 2. 数据

| 项目 | 说明 |
|---|---|
| 数据集 | online_shopping_10_cats，62,706 条，10 个类目（书籍/平板/手机/水果/洗发水/热水器/蒙牛/衣服/计算机/酒店） |
| 标签定义 | **0 = 负面，1 = 正面** |
| 划分 | train 43,893（去重去泄漏后）/ val 9,406 / test 9,406 |
| 划分来源 | `outputs/predictions/{train,val,test}.csv`，唯一合法来源 |

所有脚本必须通过 `src/data_split.py` 读取划分，禁止自行 `train_test_split`：

```python
from data_split import load, load_df, assert_valid
assert_valid()          # 训练前自检，划分异常直接抛错
rows = load("test")
```

数据细节与已知问题见 [`data/README.md`](data/README.md)。

> 历史坑：早期 `train_baselines.py` 在内部重新划分了一遍，导致 BERT 测试集有 1,850 条落在传统基线的训练/验证集里，四模型对比不成立。已修复，旧划分归档到 `outputs/predictions/_deprecated_alt_split/`。

## 3. 基线对比

| 模型 | Accuracy | Macro-F1 | 负面 Recall | 负面 Precision |
|---|---|---|---|---|
| TF-IDF + LogisticRegression | 0.9239 | 0.9239 | — | — |
| TF-IDF + NaiveBayes | 0.9253 | 0.9253 | — | — |
| TF-IDF + LinearSVM | 0.9299 | 0.9299 | 0.9329 | 0.9268 |
| **BERT-base-chinese** | **0.9466** | **0.9466** | 0.9443 | 0.9476 |

BERT 超参：`max_length=192, batch_size=8, lr=2e-5, epochs=3, patience=2, seed=42`，
按验证集 Macro-F1 选模型（最佳 epoch=2），测试集只评估一次。

## 4. 标签质量审计

对测试集全部 **502 条误判**逐条读原文审计（不看模型概率）：

| 审计结论 | 条数 | 占比 |
|---|---|---|
| 标签错（模型对、标签反） | 246 | 49.0% |
| 模型错（真错） | 211 | 42.0% |
| 双向存疑（中性/跑题/乱码） | 45 | 9.0% |

关键发现：

- **误差方向不对称**：FN 侧 243 条里 163 条（67%）是标签写反；FP 侧 259 条里 156 条是真错 → 模型存在**系统性正面偏置**。
- **修正后指标**：Accuracy 0.9466 → 0.9728，负面 Recall 0.9443 → 0.9628。**这 2.6 个点全部来自标签，模型零改动**，不能算模型提升。
- 噪声集中在两类模板：「好评打底 + 赠品/物流吐槽」「差评 + 局部褒义」。

修正清单：`outputs/bert/category_analysis/label_fixes_test.csv`（246 条，**只用于评测口径，不得进训练**）。

## 5. 改进实验

| 实验 | 改动 | Accuracy | Macro-F1 | 负面 Recall | p (vs A) |
|---|---|---|---|---|---|
| A 原始 BERT | 对照 | 0.9466 | 0.9466 | 0.9443 | — |
| B 训练集清洗 | 137 条训练集错标翻转（0.31%） | 0.9474 | 0.9474 | 0.9506 | 0.681 |
| C 转折句过采样 | 933 条真转折样本 ×2 | 0.9486 | 0.9486 | 0.9482 | 0.232 |
| BC 组合 | B + C | 0.9493 | 0.9493 | 0.9508 | 0.125 |
| D 方面级规则基线 | 8 个方面 + 权重聚合，不训练 | 0.6969 | 0.6817 | 0.4833 | — |

**三组提升均不显著**（McNemar，95% CI 全部跨 0），且单 seed 方差（±0.2~0.5pt）大于效应量（+0.27pt）。

唯一稳健的类目级信号是**平板**（n=1467，三次运行 p=0.003/0.001/<0.0001，Macro-F1 +2pt），
但**无法用标签噪声率解释**（平板噪声率仅 1.70%，相关系数 r=+0.28），机制未明。

D 失败：规则基线比 BERT 低 26 个百分点，且测试集误判里只有 3 条属于"评价对象不明确"，理论提升空间本就极小。

## 6. 推理 API

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
# 指定模型/设备
MODEL_DIR=outputs/bert/best_model DEVICE=cpu uvicorn src.api:app --port 8000
```

| 接口 | 说明 |
|---|---|
| `POST /predict` | 单条评论 |
| `POST /predict/batch` | 批量（1~256 条） |
| `GET /health` | 健康检查 |

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review":"物流很快，但是客服态度不好"}'
```

```json
{
  "review": "物流很快，但是客服态度不好",
  "label": 0,
  "sentiment": "negative",
  "probability": 0.993612,
  "prob_negative": 0.993612,
  "prob_positive": 0.006388,
  "latency_ms": 21.2
}
```

> 上面的概率是实际推理结果。`latency_ms` 取基准测试的 p50（首次请求含 CUDA 预热会到 ~500ms）。

工程要点：模型在 lifespan 中只加载一次；同步端点跑线程池 + 锁串行化 GPU 前向；空文本/纯空白/超长/批量超限返回 422。

**实测性能**（RTX 3060，max_length=192，128 条样本）：

| 设备 | batch=1 | batch=8 | batch=32 |
|---|---|---|---|
| GPU | 15.21 ms/条（66 条/秒） | **5.02 ms/条（199 条/秒）** | 8.08 ms/条（124 条/秒） |
| CPU | 110.26 ms/条（9 条/秒） | 132.22 ms/条 | 180.50 ms/条 |

HTTP 端到端单条延迟：p50 21.2ms / p95 26.4ms（含框架与序列化开销）。
GPU 在 batch=32 反而慢于 batch=8，原因是变长文本 padding 变多、显存压力上升。

## 7. 环境安装

```bash
conda create -n ml311 python=3.11 -y
conda activate ml311
pip install -r requirements.txt

# 离线环境（本机代理会拦 HuggingFace，必须加）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

首次运行需要 `bert-base-chinese` 的本地缓存；无网络时请提前下载。

## 8. 训练与评估

```bash
# 划分校验
python src/verify_split_unified.py

# 传统基线（TF-IDF）
python src/train_baselines.py

# BERT 基线
python src/train_bert.py

# 改进实验（见第 5 节）
python src/train_bert.py --run-tag B_clean_train \
  --label-fixes outputs/experiments/stage3/label_fixes_train.csv --allow-split train
python src/train_bert.py --run-tag C_contrastive \
  --augment-file outputs/experiments/stage3/contrastive_oversample.csv
python src/train_bert.py --run-tag BC_clean_contrastive \
  --label-fixes outputs/experiments/stage3/label_fixes_train.csv --allow-split train \
  --augment-file outputs/experiments/stage3/contrastive_oversample.csv

# 汇总、显著性检验、图表
python src/stage3_report.py
python src/make_figures.py

# API 测试与性能基准
python -m pytest tests/test_api.py -q
python src/benchmark_inference.py
```

`train_bert.py` 的可选参数：

| 参数 | 说明 |
|---|---|
| `--label-fixes` | 标签修正清单，按 `sample_id` 精确替换；一条未命中会直接中止 |
| `--allow-split` | 允许被修正的 split，默认只有 `train`（防止误改测试集） |
| `--augment-file` | 过采样清单，只允许作用于 train |
| `--run-tag` | 实验标签，输出到 `outputs/bert/runs/<tag>/` |
| `--seed` | 覆盖随机种子 |

## 9. 目录结构

```
sentiment-risk-analysis/
├── data/
│   ├── README.md                  # 语料、划分、已知问题
│   ├── raw/online_shopping_10_cats.csv
│   └── processed/reviews_clean.csv
├── src/
│   ├── data_split.py              # 唯一划分入口（含校验）
│   ├── train_bert.py              # BERT 训练（支持标签修正/过采样）
│   ├── train_baselines.py         # TF-IDF 基线
│   ├── api.py                     # FastAPI 推理服务
│   ├── benchmark_inference.py     # CPU/GPU 性能基准
│   ├── mine_train_label_errors.py # 训练集标签错误筛查
│   ├── mine_contrastive_train.py  # 转折句挖掘
│   ├── aspect_rule_baseline.py    # 方面级规则基线（实验 D）
│   ├── stage1_align_and_verify.py # 审计结果对齐与复核
│   ├── stage2_baseline_record.py  # 基线与划分冲突核验
│   ├── stage3_report.py           # 实验汇总 + 显著性检验
│   ├── make_figures.py            # 报告图表
│   ├── verify_split_unified.py    # 划分验收
│   ├── validate_label_fixes.py    # 修正清单校验
│   └── analyze_*.py               # 早期错误分析脚本
├── tests/test_api.py              # API 契约与一致性测试（14 项）
├── outputs/
│   ├── predictions/               # 唯一划分文件
│   ├── bert/                      # 基线模型与预测
│   ├── bert/runs/                 # B / C / BC 实验产物
│   ├── experiments/               # 实验记录、显著性、清单
│   └── metrics/                   # TF-IDF 基线指标
├── reports/
│   ├── final_report.md            # 最终报告
│   ├── experiment_comparison.csv
│   ├── significance_test.csv
│   ├── category_significance.csv
│   ├── category_macro_f1.csv
│   ├── inference_benchmark.csv
│   ├── http_latency.csv
│   └── figures/                   # 4 张图
├── models/                        # TF-IDF joblib
├── requirements.txt
├── .gitignore
└── README.md
```

## 10. 关键图表

| 文件 | 内容 |
|---|---|
| `reports/figures/fig1_experiments.png` | A/B/C/BC/D 的 Macro-F1 与负面 Recall |
| `reports/figures/fig2_noise_vs_gain.png` | 类别标签错误率 vs 清洗增益（r=+0.28，假设不成立） |
| `reports/figures/fig3_confusion.png` | A 与 BC 混淆矩阵对比 |
| `reports/figures/fig4_latency.png` | CPU/GPU 推理耗时随 batch size 变化 |

## 11. 局限

- 只审计了 502 条误判，正确分类的 8,904 条未审计，真实噪声率可能高于 2.6%。
- 训练集只做规则筛查（365 候选），未做随机抽样审计，噪声率无无偏估计。
- 每组实验单 seed，未做多次重复；因此所有 <0.5pt 的差异都不可靠。
- 分类别检验属探索性分析，30 次比较存在多重比较问题。
- 平板类目的改善机制未明，不能推广为"清洗在高噪声类目更有效"。
