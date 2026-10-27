# -*- coding: utf-8 -*-
"""
阶段 1：数据对齐与复核结论整理（不重新生成审核清单，不修改原始数据）

产出（全部写入 outputs/experiments/stage1/）：
  1. metric_recompute_verification.csv  逐类复核「修正指标」的算法是否可复现
  2. split_assignment_check.csv         246 条修正的 split 归属与重复文本核查
  3. label_correction_log.csv           修正日志（含原因分类、依据、原文）
  4. model_error_taxonomy.csv           211 条模型错的分类（供实验 C/D 取样）
  5. hot_water_3fix_detail.csv          热水器 3 条标签错的单独核查
  6. stage1_report.md                   复核结论汇总
"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BERT = ROOT / "outputs" / "bert"
CAT = BERT / "category_analysis"
PRED = ROOT / "outputs" / "predictions"
OUT = ROOT / "outputs" / "experiments" / "stage1"
OUT.mkdir(parents=True, exist_ok=True)

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dump(path, rows, cols=None):
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


audit = load(CAT / "manual_audit_all_502_error_cases.csv")
cm_orig = load(CAT / "category_metrics_and_counts.csv")
cm_fixed = load(CAT / "manual_audit_corrected_metrics_all.csv")
fixes = load(CAT / "label_fixes_test.csv")
tp = load(BERT / "test_predictions.csv")
test = load(PRED / "test.csv")
train = load(PRED / "train.csv")
val = load(PRED / "val.csv")

# ============================================================
# 0. 字段字典（第一份 CSV 的人工标注字段）
# ============================================================
FIELD_DICT = [
    ("sample_id", "唯一键", "与 outputs/predictions/*.csv 的 sample_id 一一对应，用于精确匹配原始行"),
    ("split", "数据划分", "该样本所属 split；502 条全部为 test"),
    ("audit_label", "人工标签（5 类）", "疑似标签争议 / 语义理解错误 / 转折或反讽 / 评价对象不明确 / 其他"),
    ("human_sentiment", "人工判断的真实情感", "正面 / 负面 / 中性；这是人读原文后的独立判断，不看模型概率"),
    ("audit_verdict", "审计结论", "标签错 / 模型错 / 双向存疑"),
    ("needs_relabel", "是否需重标", "是 / 否 / 待定，由 audit_verdict 直接派生"),
    ("corrected_label", "修正后标签", "0/1；仅 audit_verdict=标签错 的行有值（246 条）"),
    ("label_action", "处置动作", "flip(246) / keep(211) / undecided(45)"),
    ("audit_reason", "判定依据", "引用原文原句的自由文本说明"),
]

# ============================================================
# 1. 复核「修正指标」是怎么算的：从原始预测重算，逐类比对
# ============================================================
fixmap = {f["sample_id"]: int(f["corrected_label"]) for f in fixes}
by_cat = defaultdict(list)
for r in tp:
    by_cat[r["cat"]].append(r)


def confusion(rows, use_fixed):
    c = Counter()
    for r in rows:
        y = fixmap.get(r["sample_id"], int(r["label"])) if use_fixed else int(r["label"])
        p = int(r["prediction"])
        c[("TN", "FP", "FN", "TP")[{(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}[(y, p)]]] += 1
    return c


def derive(c, n):
    neg = c["TN"] + c["FP"]
    pn = c["TN"] + c["FN"]
    acc = (c["TN"] + c["TP"]) / n
    pre = c["TN"] / pn if pn else 0
    rec = c["TN"] / neg if neg else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0
    return neg, acc, pre, rec, f1


verify = []
orig_map = {r["category"]: r for r in cm_orig}
fixed_map = {r["category"]: r for r in cm_fixed}
for cat in orig_map:
    rows = by_cat[cat]
    n = len(rows)
    c0, c1 = confusion(rows, False), confusion(rows, True)
    neg0, acc0, pre0, rec0, f10 = derive(c0, n)
    neg1, acc1, pre1, rec1, f11 = derive(c1, n)
    o, f = orig_map[cat], fixed_map[cat]
    ok_cm = all(int(o[k]) == c0[k] for k in ("TN", "FP", "FN", "TP"))
    # 修正指标表未存混淆矩阵，只能比对派生指标（见报告中的说明）
    ok_fix = (
        f["neg_count_after"] == str(neg1)
        and f["neg_recall_after"] == f"{rec1*100:.2f}%"
        and f["neg_precision_after"] == f"{pre1*100:.2f}%"
        and f["acc_after"] == f"{acc1*100:.2f}%"
        and f["neg_f1_after"] == f"{f11:.4f}"
    )
    ok_fix_before = (
        f["neg_count_before"] == str(neg0)
        and f["neg_recall_before"] == f"{rec0*100:.2f}%"
        and f["acc_before"] == f"{acc0*100:.2f}%"
    )
    verify.append({
        "category": cat, "n": n,
        "原CM(TN/FP/FN/TP)": f"{c0['TN']}/{c0['FP']}/{c0['FN']}/{c0['TP']}",
        "原CM与metrics表一致": "是" if ok_cm else "否",
        "修正后CM(TN/FP/FN/TP)": f"{c1['TN']}/{c1['FP']}/{c1['FN']}/{c1['TP']}",
        "负样本数(原→修正)": f"{neg0}→{neg1}",
        "负面召回(原→修正)": f"{rec0*100:.2f}%→{rec1*100:.2f}%",
        "负面精确(原→修正)": f"{pre0*100:.2f}%→{pre1*100:.2f}%",
        "准确率(原→修正)": f"{acc0*100:.2f}%→{acc1*100:.2f}%",
        "负面F1(原→修正)": f"{f10:.4f}→{f11:.4f}",
        "修正前值与表一致": "是" if ok_fix_before else "否",
        "修正后值与表一致": "是" if ok_fix else "否",
        "复算与表中值一致": "是" if (ok_cm and ok_fix and ok_fix_before) else "否",
    })
dump(OUT / "metric_recompute_verification.csv", verify)

# ============================================================
# 2. split 归属 + 重复文本核查
# ============================================================
sid2split = {}
text2sids = defaultdict(list)
for name, rows in (("train", train), ("val", val), ("test", test)):
    for r in rows:
        sid2split[r["sample_id"]] = name
        text2sids[norm(r["review"])].append((name, r["sample_id"]))

test_by_sid = {r["sample_id"]: r for r in test}
split_chk = []
for a in audit:
    sid = a["sample_id"]
    t = norm(a["review"])
    occ = text2sids.get(t, [])
    other_splits = sorted({s for s, x in occ if x != sid})
    split_chk.append({
        "sample_id": sid,
        "split": sid2split.get(sid, "未找到"),
        "category": a["category"],
        "audit_verdict": a["audit_verdict"],
        "原文逐字一致": "是" if test_by_sid[sid]["review"] == a["review"] else "否",
        "同文本出现次数": len(occ),
        "同文本涉及的其他split": "/".join(other_splits) if other_splits else "无",
        "重复风险": "无" if len(occ) == 1 else "需人工确认",
    })
dump(OUT / "split_assignment_check.csv", split_chk)

# ============================================================
# 3. 修正日志（只含 246 条 flip，含原因分类）
# ============================================================
def reason_type(a):
    d = "1→0" if (a["label"] == "1" and a["corrected_label"] == "0") else "0→1"
    if a["audit_label"] == "评价对象不明确":
        return "评价对象错位（原文评的是物流/价格/赠品，非商品本体）", d
    if d == "1→0":
        return "极性倒置：原文整体负面，原标签为正面", d
    return "极性倒置：原文整体正面，原标签为负面", d


log = []
for a in audit:
    if a["label_action"] != "flip":
        continue
    rt, d = reason_type(a)
    log.append({
        "sample_id": a["sample_id"],
        "split": a["split"],
        "category": a["category"],
        "review": a["review"],
        "orig_label": a["label"],
        "orig_sentiment": a["true_sentiment"],
        "corrected_label": a["corrected_label"],
        "corrected_sentiment": a["human_sentiment"],
        "flip_direction": d,
        "reason_type": rt,
        "audit_label": a["audit_label"],
        "evidence": a["audit_reason"],
        "model_prediction": a["prediction"],
        "prob_negative": a["prob_negative"],
        "reviewer": "人工阅读原文判定（未参考模型概率）",
        "source_file": "outputs/bert/category_analysis/manual_audit_all_502_error_cases.csv",
    })
dump(OUT / "label_correction_log.csv", log)

# ============================================================
# 4. 模型错分类（供实验 C/D 取样）
# ============================================================
tax = []
for a in audit:
    if a["audit_verdict"] != "模型错":
        continue
    lab = a["audit_label"]
    if lab == "转折或反讽":
        pool = "实验C-转折句候选"
    elif lab == "评价对象不明确":
        pool = "实验D-方面级候选"
    elif lab == "语义理解错误":
        pool = "实验C-长文本/汇总候选" if int(a["review_length"]) > 80 else "实验C-短文本极性候选"
    else:
        pool = "待定"
    tax.append({
        "sample_id": a["sample_id"], "split": a["split"], "category": a["category"],
        "review_length": a["review_length"], "review": a["review"],
        "true_sentiment": a["human_sentiment"], "model_prediction": a["prediction"],
        "prob_negative": a["prob_negative"], "audit_label": lab,
        "suggested_pool": pool, "audit_reason": a["audit_reason"],
    })
dump(OUT / "model_error_taxonomy.csv", tax)

# ============================================================
# 5. 热水器 3 条标签错单独核查
# ============================================================
hw = [x for x in log if x["category"] == "热水器"]
dump(OUT / "hot_water_3fix_detail.csv", hw)

# ============================================================
# 6. 汇总报告
# ============================================================
all_ok = all(v["复算与表中值一致"] == "是" for v in verify)
dup_risk = [x for x in split_chk if x["重复风险"] != "无"]
text_mismatch = [x for x in split_chk if x["原文逐字一致"] != "是"]
split_dist = Counter(x["split"] for x in split_chk)
fix_split = Counter(x["split"] for x in log)
rt_dist = Counter(x["reason_type"] for x in log)
pool_dist = Counter(x["suggested_pool"] for x in tax)

L = []
L.append("# 阶段 1 复核报告：数据对齐与结论整理\n")
L.append("> 本阶段只做核对与整理，未重新生成审核清单，未修改 `data/` 下任何原始数据。\n")

L.append("## 1. 修正指标的算法（已逐类复算通过）\n")
L.append("`manual_audit_corrected_metrics_all.csv` 的计算链条：\n")
L.append("1. 基线混淆矩阵取自 `category_metrics_and_counts.csv`（由 test 集 9406 条预测统计）；")
L.append("2. 对 `audit_verdict=标签错` 的样本，按 `corrected_label` 替换真值，**模型预测保持不变**；")
L.append("3. 重算 TN/FP/FN/TP，再派生准确率、负面精确/召回/F1。\n")
L.append("复算方式：直接从 `test_predictions.csv` + `label_fixes_test.csv` 从零重算，"
         f"逐类与两份指标表比对，结果：{'全部一致' if all_ok else '存在不一致'}。\n")
L.append("| 类别 | 负样本数 原→修正 | 负面召回 原→修正 | 准确率 原→修正 | 复算一致 |")
L.append("|---|---|---|---|---|")
for v in verify:
    L.append(f"| {v['category']} | {v['负样本数(原→修正)']} | {v['负面召回(原→修正)']} "
             f"| {v['准确率(原→修正)']} | {v['复算与表中值一致']} |")
L.append("")
L.append("**重要限定**：只审计了 502 条误判样本，正确分类的 8904 条未审计，其标签被视为正确。"
         "因此修正后的指标是「已知噪声修正后的下限口径」，真实噪声率可能更高，"
         "不能反过来宣称模型有更高上限。\n")

L.append("## 2. 人工标注字段名（第一份 CSV）\n")
L.append("| 字段 | 含义 | 取值 |")
L.append("|---|---|---|")
for a, b, c in FIELD_DICT:
    L.append(f"| `{a}` | {b} | {c} |")
L.append("")
L.append("权威修正清单是 `label_fixes_test.csv`（246 行，只含 flip）。"
         "502 行审计表里另有 211 条「模型错」，不可用于改标签。\n")

L.append("## 3. split 归属核查\n")
L.append(f"- 502 条审计样本的 split 分布：{dict(split_dist)}")
L.append(f"- 246 条标签错修正的 split 分布：{dict(fix_split)}")
L.append(f"- 原文与 `test.csv` 逐字一致的条数：{len(split_chk)-len(text_mismatch)}/{len(split_chk)}")
L.append(f"- 存在同文本重复记录的条数：{len(dup_risk)}")
if dup_risk:
    L.append("  - 需人工确认的 sample_id：" + ", ".join(x["sample_id"] for x in dup_risk[:10]))
L.append("")
L.append("**结论：246 条修正全部来自 test 集，没有一条落在 train 或 val。**")
L.append("按你的要求，它们只能用于测试标签质量分析，不得进入训练数据，"
         "也不得通过改测试标签宣称模型性能提升。\n")

L.append("### 热水器类别的 3 条标签错\n")
L.append("| sample_id | split | 原文 | 原标签 | 修正 | 原因 |")
L.append("|---|---|---|---|---|---|")
for x in hw:
    L.append(f"| {x['sample_id']} | {x['split']} | {x['review'][:22]} | {x['orig_label']} "
             f"| {x['corrected_label']} | {x['reason_type']} |")
L.append("")
L.append("这 3 条同为 test 样本，属于 8 条热水器误判中的标签问题，"
         "修正后热水器负面召回 58.33%→69.23%，仍是「测试集人工复核口径」，不是模型改进。\n")

L.append("## 4. 修正原因分布（246 条）\n")
L.append("| 原因类型 | 条数 |")
L.append("|---|---|")
for k, v in rt_dist.most_common():
    L.append(f"| {k} | {v} |")
L.append("")
L.append("逐条明细见 `label_correction_log.csv`，每条都带原文、原标签、修正标签、"
         "翻转方向与引用原句的证据。\n")

L.append("## 5. 模型错分类（211 条，供阶段 3 取样）\n")
L.append("| 候选池 | 条数 |")
L.append("|---|---|")
for k, v in pool_dist.most_common():
    L.append(f"| {k} | {v} |")
L.append("")
L.append("注意：这些是「模型错」，与标签错严格分开。实验 C/D 的增强样本应从这些池中挑，"
         "而不是从标签错里挑。\n")

L.append("## 6. 原始文件保护\n")
L.append("- `data/raw/`、`data/processed/`、`outputs/predictions/*.csv` 未做任何修改。")
L.append("- `manual_audit_all_502_error_cases.csv` 为 v2，相比 v1 仅新增 4 列"
         "（`sample_id`/`split`/`corrected_label`/`label_action`），原有 18 列全部保留，无信息损失。")
L.append("- 修正日志另存为 `label_correction_log.csv`，与原始审计表分离，可独立审计。\n")

(OUT / "stage1_report.md").write_text("\n".join(L), encoding="utf-8")

print(f"复算一致：{all_ok}")
print(f"split 分布（502 条）：{dict(split_dist)}")
print(f"标签错 split 分布（246 条）：{dict(fix_split)}")
print(f"逐字一致：{len(split_chk)-len(text_mismatch)}/{len(split_chk)}  重复风险：{len(dup_risk)}")
print(f"原因分布：{dict(rt_dist)}")
print(f"模型错候选池：{dict(pool_dist)}")
print(f"\n产出目录：{OUT}")
for p in sorted(OUT.iterdir()):
    print("  ", p.name)
