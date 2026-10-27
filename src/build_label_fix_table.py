# -*- coding: utf-8 -*-
"""
把人工审计结果升级为「可精确匹配 + 可直接驱动训练脚本」的形式。

产出：
1. manual_audit_all_502_error_cases.csv  追加 sample_id / split / corrected_label / label_action
2. label_fixes_test.csv                  唯一权威修正清单（仅 246 条确认翻转）
3. label_fix_impact_test.csv             修正对测试集分布与指标的影响
"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BERT = ROOT / "outputs" / "bert"
CAT = BERT / "category_analysis"
PRED = ROOT / "outputs" / "predictions"

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


audit = load(CAT / "manual_audit_all_502_error_cases.csv")
tp = load(BERT / "test_predictions.csv")
test = load(PRED / "test.csv")
train = load(PRED / "train.csv")
val = load(PRED / "val.csv")

# ---------- 1. 用 (review_norm, label, prediction) 映射 sample_id ----------
# test 文本唯一，故该键在 test 内唯一
key2sid = {}
for r in tp:
    k = (norm(r["review"]), str(r["label"]), str(r["prediction"]))
    if k in key2sid:
        raise SystemExit(f"test 内出现重复键，无法唯一定位：{k[0][:30]}")
    key2sid[k] = r["sample_id"]

sid2split = {}
for name, rows in (("train", train), ("val", val), ("test", test)):
    for r in rows:
        sid2split[r["sample_id"]] = name

missing = []
for a in audit:
    k = (norm(a["review"]), str(a["label"]), str(a["prediction"]))
    a["sample_id"] = key2sid.get(k, "")
    if not a["sample_id"]:
        missing.append(a["case_no"])
if missing:
    raise SystemExit(f"有 {len(missing)} 条无法匹配 sample_id：{missing[:10]}")

# 反向校验：审计表必须完整覆盖 test 的全部误判
mis_sids = {r["sample_id"] for r in tp if str(r["correct"]).lower() == "false"}
aud_sids = {a["sample_id"] for a in audit}
assert mis_sids == aud_sids, f"覆盖不一致：缺{len(mis_sids-aud_sids)} 多{len(aud_sids-mis_sids)}"

# ---------- 2. 补 split / corrected_label / label_action ----------
for a in audit:
    a["split"] = sid2split.get(a["sample_id"], "unknown")
    ver, hum, orig = a["audit_verdict"], a["human_sentiment"], str(a["label"])
    if ver == "标签错":
        new = "0" if hum == "负面" else "1"
        assert new != orig, f"case {a['case_no']} 判为标签错但与原标签相同"
        a["corrected_label"] = new
        a["label_action"] = "flip"
    elif ver == "模型错":
        a["corrected_label"] = ""
        a["label_action"] = "keep"
    else:
        a["corrected_label"] = ""
        a["label_action"] = "undecided"

COLS = ["sample_id", "split", "category", "case_no", "review", "review_length",
        "label", "true_sentiment", "prediction", "predicted_sentiment", "error_type",
        "prob_negative", "prob_positive", "confidence", "probability_margin",
        "audit_label", "human_sentiment", "audit_verdict", "needs_relabel",
        "corrected_label", "label_action", "audit_reason"]

with open(CAT / "manual_audit_all_502_error_cases.csv", "w",
          encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(audit)

# ---------- 3. 唯一权威修正清单 ----------
FIX_COLS = ["sample_id", "split", "category", "orig_label", "corrected_label",
            "audit_label", "audit_reason"]
fixes = [{
    "sample_id": a["sample_id"], "split": a["split"], "category": a["category"],
    "orig_label": a["label"], "corrected_label": a["corrected_label"],
    "audit_label": a["audit_label"], "audit_reason": a["audit_reason"],
} for a in audit if a["label_action"] == "flip"]

with open(CAT / "label_fixes_test.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIX_COLS)
    w.writeheader()
    w.writerows(fixes)

# ---------- 4. 影响评估 ----------
def cm(rows, label_key):
    c = Counter()
    for r in rows:
        y, p = int(r[label_key]), int(r["prediction"])
        c[("TN", "FP", "FN", "TP")[{ (0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3 }[(y, p)]]] += 1
    return c


fixmap = {f["sample_id"]: int(f["corrected_label"]) for f in fixes}
rows = []
for r in tp:
    rows.append({**r, "label_new": fixmap.get(r["sample_id"], int(r["label"]))})

before, after = cm(rows, "label"), cm(rows, "label_new")
N = len(rows)
out = []
for name, c in (("修正前(原标签)", before), ("修正后(人工标签)", after)):
    neg = c["TN"] + c["FP"]
    pn = c["TN"] + c["FN"]
    out.append({
        "口径": name, "样本数": N, "真实负样本": neg, "真实正样本": c["TP"] + c["FN"],
        "TN": c["TN"], "FP": c["FP"], "FN": c["FN"], "TP": c["TP"],
        "准确率": f"{(c['TN']+c['TP'])/N*100:.2f}%",
        "负面精确率": f"{c['TN']/pn*100:.2f}%" if pn else "-",
        "负面召回率": f"{c['TN']/neg*100:.2f}%" if neg else "-",
        "负面F1": f"{2*c['TN']/(2*c['TN']+c['FN']+c['FP']):.4f}",
    })

with open(CAT / "label_fix_impact_test.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

print("审计表已补 sample_id/split/corrected_label/label_action")
print("split 分布：", Counter(a["split"] for a in audit))
print("修正清单条数：", len(fixes), " 原标签分布：", Counter(f["orig_label"] for f in fixes))
print("翻转方向：1->0", sum(1 for f in fixes if f["orig_label"] == "1" and f["corrected_label"] == "0"),
      " 0->1", sum(1 for f in fixes if f["orig_label"] == "0" and f["corrected_label"] == "1"))
print()
for o in out:
    print(o)
print()
# 与 train 重叠的那 1 条是否落在审计集
tr_texts = {norm(r["review"]) for r in train}
ov = [a for a in audit if norm(a["review"]) in tr_texts]
print("审计集中与 train 文本重叠的条数：", len(ov))
for a in ov:
    print("  ", a["category"], a["sample_id"], a["review"][:40])
