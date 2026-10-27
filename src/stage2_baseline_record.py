# -*- coding: utf-8 -*-
"""
阶段 2：固定实验基线（记录 + 一致性核验）

⚠ 核验要点：项目里存在两套划分，必须按【文本】而不是 sample_id 匹配。
   - train.csv / val.csv / test.csv            ← train_bert.py 使用（BERT 家族）
   - train_split.csv / validation_split.csv / test_split.csv  ← train_baselines.py 自己重新划分
   两族来自同一语料但划分不同，且 sample_id 存在 ±1 漂移，不能跨族对齐。

产出（outputs/experiments/）：
  baseline_results.csv              实验记录表
  data_integrity_check.csv          单族内部的重复/泄漏核验
  split_family_conflict.csv         两族划分冲突矩阵（按文本）
  split_consistency_check.csv       各模型是否使用同一 test 样本
  baseline_results_clean_subset.csv 泄漏干净子集上的可比对照
  baseline_config.json              配置快照
  stage2_report.md                  基线结论与阻塞项
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BERT = ROOT / "outputs" / "bert"
PRED = ROOT / "outputs" / "predictions"
ALT = PRED / "_deprecated_alt_split"   # 历史第二套划分（已归档，仅作证据）
MET = ROOT / "outputs" / "metrics"
OUT = ROOT / "outputs" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dump(path, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def cm_of(rows):
    c = Counter()
    for r in rows:
        y, p = int(r["label"]), int(r["prediction"])
        c[("TN", "FP", "FN", "TP")[{(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}[(y, p)]]] += 1
    return c


def metrics_of(c, n):
    neg, pn = c["TN"] + c["FP"], c["TN"] + c["FN"]
    acc = (c["TN"] + c["TP"]) / n
    pre = c["TN"] / pn if pn else 0.0
    rec = c["TN"] / neg if neg else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0.0
    pp = c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) else 0.0
    pr = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) else 0.0
    f1p = 2 * pp * pr / (pp + pr) if (pp + pr) else 0.0
    return acc, pre, rec, f1, (f1 + f1p) / 2


# ============================================================
# 1. 单族内部完整性
# ============================================================
fam = {
    "BERT家族": {n: load(PRED / f"{n}.csv") for n in ("train", "val", "test")},
    "基线家族(历史)": {n: load(ALT / f"{n}.csv")
                       for n in ("train_split", "validation_split", "test_split")},
}

integrity = []
for fname, parts in fam.items():
    tsets = {n: {norm(r["review"]) for r in rows} for n, rows in parts.items()}
    for n, rows in parts.items():
        ns = [norm(r["review"]) for r in rows]
        lab = Counter(int(r["label"]) for r in rows)
        row = {"划分家族": fname, "split": n, "行数": len(rows),
               "唯一文本数": len(set(ns)), "内部重复条数": len(ns) - len(set(ns)),
               "负样本(0)": lab[0], "正样本(1)": lab[1],
               "负样本占比": f"{lab[0]/len(rows)*100:.2f}%"}
        for m in parts:
            if m != n:
                row[f"与{m}文本重叠"] = len(tsets[n] & tsets[m])
        integrity.append(row)
dump(OUT / "data_integrity_check.csv", integrity, cols=[
    "划分家族", "split", "行数", "唯一文本数", "内部重复条数", "负样本(0)", "正样本(1)",
    "负样本占比", "与train文本重叠", "与val文本重叠", "与test文本重叠",
    "与train_split文本重叠", "与validation_split文本重叠", "与test_split文本重叠"])

# ============================================================
# 2. 两族划分冲突（按文本匹配）
# ============================================================
bert_test = fam["BERT家族"]["test"]
base_parts = {n: {norm(r["review"]) for r in rows} for n, rows in fam["基线家族(历史)"].items()}
bert_test_texts = {norm(r["review"]) for r in bert_test}

conflict = []
for n, ts in base_parts.items():
    inter = bert_test_texts & ts
    lab = Counter(int(r["label"]) for r in bert_test if norm(r["review"]) in ts)
    conflict.append({
        "基线族划分": n,
        "与BERT-test共同文本数": len(inter),
        "占BERT-test比例": f"{len(inter)/len(bert_test)*100:.2f}%",
        "其中BERT-test标签0": lab[0],
        "其中BERT-test标签1": lab[1],
        "性质": "同一批样本（可对齐）" if n == "test_split"
                else "BERT的测试样本落在基线族的训练/验证集 → 交叉污染",
    })
dump(OUT / "split_family_conflict.csv", conflict)

contaminated = set()
for n in ("train_split", "validation_split"):
    contaminated |= bert_test_texts & base_parts[n]

# ============================================================
# 3. 各模型是否使用同一 test 样本（按文本）
# ============================================================
model_files = {
    "TF-IDF + LogisticRegression": ALT / "logistic_regression_test_predictions.csv",
    "TF-IDF + NaiveBayes": ALT / "naive_bayes_test_predictions.csv",
    "TF-IDF + LinearSVM": ALT / "linear_svm_test_predictions.csv",
    "BERT-base-chinese": BERT / "test_predictions.csv",
}
model_files = {k: v for k, v in model_files.items() if v.exists()}
new_baseline = ROOT / "outputs" / "predictions" / "linear_svm_test_predictions.csv"
if new_baseline.exists():
    for nm, f in (("TF-IDF + LogisticRegression", "logistic_regression_test_predictions.csv"),
                  ("TF-IDF + NaiveBayes", "naive_bayes_test_predictions.csv"),
                  ("TF-IDF + LinearSVM", "linear_svm_test_predictions.csv")):
        p2 = ROOT / "outputs" / "predictions" / f
        if p2.exists():
            model_files[nm] = p2   # 重跑后的统一划分版本优先
by_text = {}
for name, p in model_files.items():
    rows = load(p)
    by_text[name] = {norm(r["review"]): r for r in rows}

consistency = []
for name, d in by_text.items():
    common = set(d) & bert_test_texts
    lab_diff = sum(1 for t in common
                   if int(d[t]["label"]) != int(by_text["BERT-base-chinese"][t]["label"]))
    consistency.append({
        "model": name,
        "test样本数": len(d),
        "与BERT-test共同文本": len(common),
        "缺失(BERT-test有而该模型无)": len(bert_test_texts - set(d)),
        "标签不一致条数": lab_diff,
        "结论": "同一划分" if set(d) == bert_test_texts and lab_diff == 0
                else "不同划分，不可直接比较",
    })
dump(OUT / "split_consistency_check.csv", consistency)

# ============================================================
# 4. 结果表
# ============================================================
def row_of(eid, name, role, rows, n, comparable, note, src):
    c = cm_of(rows)
    acc, pre, rec, f1, mf1 = metrics_of(c, n)
    return {
        "experiment_id": eid, "experiment": name, "role": role, "model": name,
        "seed": 42, "train_size": "", "val_size": "", "test_size": n,
        "error_count": c["FP"] + c["FN"],
        "accuracy": f"{acc:.4f}", "macro_f1": f"{mf1:.4f}",
        "negative_precision": f"{pre:.4f}", "negative_recall": f"{rec:.4f}",
        "negative_f1": f"{f1:.4f}",
        "TN": c["TN"], "FP": c["FP"], "FN": c["FN"], "TP": c["TP"],
        "inference_ms_per_sample": "待测",
        "comparability": comparable, "status": "已完成", "notes": note,
        "source_file": src,
    }


results = []
for name, p in model_files.items():
    rows = load(p)
    is_bert = name.startswith("BERT")
    results.append(row_of(
        "A" if is_bert else "B0", name,
        "基线（对照组）" if is_bert else "基线（传统模型，待重跑）",
        rows, len(rows),
        "可比（参考口径）" if is_bert else "不可比：不同 test 划分",
        "test.csv 原标签" if is_bert else "test_split.csv 原标签（与 BERT 不同划分）",
        str(p.relative_to(ROOT)).replace("\\", "/")))

for eid, nm in [("B", "训练集标签清洗"), ("C", "转折句定向增强"), ("D", "方面级情感分析")]:
    results.append({
        "experiment_id": eid, "experiment": nm, "role": "待验证",
        "model": "bert-base-chinese", "seed": 42, "train_size": "", "val_size": "",
        "test_size": "", "error_count": "", "accuracy": "", "macro_f1": "",
        "negative_precision": "", "negative_recall": "", "negative_f1": "",
        "TN": "", "FP": "", "FN": "", "TP": "",
        "inference_ms_per_sample": "待测", "comparability": "",
        "status": "未开始", "notes": "", "source_file": "",
    })
dump(OUT / "baseline_results.csv", results)

# ---- 干净子集：BERT-test 中未出现在基线族 train/val 的行 ----
clean_texts = bert_test_texts - contaminated
clean = [r for r in bert_test if norm(r["review"]) in clean_texts]
clean_rows = []
for name, d in by_text.items():
    sub = [d[t] for t in clean_texts if t in d]
    if len(sub) != len(clean):
        continue
    clean_rows.append(row_of(
        "A" if name.startswith("BERT") else "B0", name, "干净子集对照",
        sub, len(sub), "可比（同一批样本）",
        "仅保留基线族未参与训练/验证的样本",
        str(model_files[name].relative_to(ROOT)).replace("\\", "/")))
dump(OUT / "baseline_results_clean_subset.csv", clean_rows)

# ============================================================
# 5. 配置快照
# ============================================================
cfg = {
    "dataset": {
        "name": "online_shopping_10_cats",
        "label_definition": {"0": "负面", "1": "正面"},
        "bert_family_split": {"train": 43894, "val": 9406, "test": 9406},
        "baseline_family_split": {"train": 43893, "val": 9406, "test": 9406},
        "warning": "两族划分不同，比较前必须统一",
    },
    "bert": json.load(open(BERT / "test_metrics.json", encoding="utf-8")),
    "tfidf_baselines": json.load(open(MET / "experiment_config.json", encoding="utf-8")),
    "model_selection": {"criterion": "validation macro-F1", "best_epoch": 2,
                        "test_used_for_selection": False, "patience": 2},
    "blockers": [
        "train_bert.py 与 train_baselines.py 使用两套不同划分，4 模型对比表当前不成立",
        "BERT-test 中有 1850 条落在基线族 train/val，直接重打分会被污染",
    ],
}
json.dump(cfg, open(OUT / "baseline_config.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# ============================================================
# 6. 报告
# ============================================================
clean_neg = sum(1 for r in clean if r["label"] == "0")
L = []
L.append("# 阶段 2 报告：固定实验基线\n")
L.append("## 0. 结论先行：现有四模型对比表不成立\n")
L.append("项目里存在**两套互不相同的划分**：\n")
L.append("- `train.csv / val.csv / test.csv` —— `train_bert.py` 使用（下称 BERT 家族）")
L.append("- `train_split.csv / validation_split.csv / test_split.csv` —— "
         "`train_baselines.py` 在内部重新 `train_test_split` 生成（下称基线家族）\n")
L.append("两族来自同一语料（62705 条唯一文本），但**划分结果不同**：\n")
L.append("| 基线族划分 | 与 BERT-test 共同文本 | 其中 BERT 标签为 0 | 为 1 | 性质 |")
L.append("|---|---|---|---|---|")
for x in conflict:
    L.append(f"| {x['基线族划分']} | {x['与BERT-test共同文本数']} | {x['其中BERT-test标签0']} "
             f"| {x['其中BERT-test标签1']} | {x['性质']} |")
L.append("")
L.append(f"**关键后果**：BERT 测试集的 9406 条中，有 **{len(contaminated)} 条**"
         f"（{len(contaminated)/len(bert_test)*100:.1f}%）出现在基线家族的 train/val 里，"
         "且全部是正类。这意味着 TF-IDF 基线在这部分样本上是「见过答案」的，"
         "其 0.9304 与 BERT 的 0.9466 **不能直接比较**。\n")
L.append("另外，两族的 `sample_id` 存在 ±1 漂移（共同文本中约一半 ID 不同），"
         "**跨族对齐只能按文本，不能按 sample_id**。\n")

L.append("## 1. 基线结果（各自原划分，仅作存档）\n")
L.append("| 实验 | 模型 | Accuracy | Macro-F1 | 负面 Recall | 负面 Precision | 可比性 |")
L.append("|---|---|---|---|---|---|---|")
for r in results:
    if r["status"] != "已完成":
        continue
    L.append(f"| {r['experiment_id']} | {r['model']} | {r['accuracy']} | {r['macro_f1']} "
             f"| {r['negative_recall']} | {r['negative_precision']} | {r['comparability']} |")
L.append("")

L.append("## 2. 干净子集上的可比对照\n")
L.append(f"取 BERT-test 中未被基线族 train/val 覆盖的 **{len(clean)} 条**"
         f"（负 {clean_neg} / 正 {len(clean)-clean_neg}）重算，四个模型面对同一批样本：\n")
L.append("| 模型 | Accuracy | Macro-F1 | 负面 Recall | 负面 Precision |")
L.append("|---|---|---|---|---|")
for r in clean_rows:
    L.append(f"| {r['model']} | {r['accuracy']} | {r['macro_f1']} "
             f"| {r['negative_recall']} | {r['negative_precision']} |")
L.append("")
L.append(f"注意：该子集负类占 {clean_neg/len(clean)*100:.1f}%，明显高于全集 49.5%，"
         "因此**绝对值不能与全集指标直接对比**，只用于判断四个模型的相对排序。\n")

L.append("## 3. 单族内部完整性\n")
L.append("| 家族 | split | 行数 | 唯一文本 | 内部重复 | 负样本占比 |")
L.append("|---|---|---|---|---|---|")
for x in integrity:
    L.append(f"| {x['划分家族']} | {x['split']} | {x['行数']} | {x['唯一文本数']} "
             f"| {x['内部重复条数']} | {x['负样本占比']} |")
L.append("")
L.append("- 两族内部均无重复文本；`sample_id` 在各自族内唯一。")
L.append("- BERT 家族内 train 与 test 有 1 条文本重叠，`train_bert.py` 已从训练集剔除。")
L.append("- 基线家族由 `train_test_split` 生成，本身无跨集合重叠。\n")

L.append("## 4. 模型选择口径\n")
L.append("- BERT：只按 `val macro-F1` 选 epoch，最佳 epoch=2（val Macro-F1=0.9480），"
         "测试集仅最终评估一次。")
L.append("- TF-IDF 基线：按 `validation_f1_macro` 选型，最佳 `linear_svm`，同样未用测试集选型。")
L.append("- 两族的选型流程都合规，问题只出在「划分本身不同」。\n")

L.append("## 5. 统一划分的执行结果\n")
L.append("已以 BERT 家族为准完成统一（原因：人工审计、错误分析、修正清单全部基于 "
         "`test.csv` 的 `sample_id`，换划分会让全部结论失效）：\n")
L.append("| 动作 | 内容 |")
L.append("|---|---|")
L.append("| 唯一入口 | 新增 `src/data_split.py`，所有脚本通过它读取划分 |")
L.append("| 修复根因 | `train_baselines.py` 删除内部 `train_test_split`，改读 "
         "`train.csv/val.csv/test.csv` |")
L.append("| 隔离旧文件 | `*_split.csv`、旧基线预测、旧 joblib 移入 "
         "`outputs/predictions/_deprecated_alt_split/` |")
L.append("| 诊断脚本 | `audit_data_leakage.py`、`check_duplicate_detail.py` 已改为读统一划分 |")
L.append("| 校验 | `python src/verify_split_unified.py` |")
L.append("")
L.append("**待你执行**：`python src/train_baselines.py` 重训三个 TF-IDF 基线"
         "（本机无 sklearn，无法代跑）。完成后 `outputs/metrics/test_results.csv` "
         "才是可比的 B0 行。")
L.append("")
L.append("推理耗时仍未测（本机无 torch/sklearn 环境），保持「待测」，不得预填。\n")

L.append("## 6. 未测项\n")
L.append("- 推理耗时 / 吞吐量：待测。")
L.append("- B / C / D 三组实验：全部留空，由真实运行填写。\n")
(OUT / "stage2_report.md").write_text("\n".join(L), encoding="utf-8")

print("=== 两族冲突（按文本）===")
for x in conflict:
    print(x)
print(f"\n被污染的 BERT-test 行数：{len(contaminated)}")
print(f"干净子集：{len(clean)} 条（负 {clean_neg} / 正 {len(clean)-clean_neg}）")
print("\n=== 干净子集对照 ===")
for r in clean_rows:
    print(f"  {r['model'][:30]:<30} acc={r['accuracy']} mf1={r['macro_f1']} "
          f"nrec={r['negative_recall']} npre={r['negative_precision']}")
print("\n=== 一致性 ===")
for x in consistency:
    print(f"  {x['model'][:30]:<30} 共同{x['与BERT-test共同文本']} "
          f"缺失{x['缺失(BERT-test有而该模型无)']} {x['结论']}")
print(f"\n产出：{OUT}")
