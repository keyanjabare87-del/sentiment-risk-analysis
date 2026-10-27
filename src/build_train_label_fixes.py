# -*- coding: utf-8 -*-
"""
阶段 3B 第二步：把人工判定结果落成修正清单与审计日志。

判定口径
--------
候选共 365 条（R2 101 + R3 264），**逐条读原文判定**，只翻转「整体情感与标签相反」的样本；
混合评价（褒贬参半）、中性文本、跑题文本一律保持原标签不动。

- R2 = 含 ≥2 个非否定负面词、却标为正面(1)
- R3 = 含 ≥2 个非否定正面词、却标为负面(0)

产出
----
outputs/experiments/stage3/train_label_audit_log.csv   365 条逐条判定记录
outputs/experiments/stage3/label_fixes_train.csv       137 条确认翻转（供 train_bert.py 使用）
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs" / "predictions"
OUT = ROOT / "outputs" / "experiments" / "stage3"
OUT.mkdir(parents=True, exist_ok=True)

# 人工判定为「标签错」的序号（序号 = train_label_candidates.csv 中同规则内的位置）
FLIP_R2 = [4, 5, 8, 14, 15, 16, 19, 30, 32, 34, 41, 43, 50, 54, 55, 56, 57,
           60, 61, 70, 71, 76, 81, 83, 84, 86, 88, 89, 95, 97]
FLIP_R3 = [0, 1, 3, 5, 13, 16, 21, 24, 25, 28, 30, 33, 34, 35, 36, 37, 38,
           42, 45, 47, 49, 53, 54, 56, 57, 62, 65, 66, 67, 68, 72, 78, 81,
           82, 86, 89, 90, 91, 94, 96, 98, 102, 104, 106, 108, 112, 113, 114,
           115, 118, 119, 120, 122, 124, 125, 128,
           132, 135, 136, 145, 148, 153, 155, 161, 163, 165, 171, 173, 175,
           176, 178, 179, 181, 184, 187, 190, 199, 200, 201, 204, 206, 209,
           211, 212, 216, 219, 220, 221, 224, 227, 229, 230, 232, 237, 240,
           242, 244, 245, 248, 249, 252, 253, 255, 256, 257, 258, 259]


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


cands = load(OUT / "train_label_candidates.csv")
r2 = [x for x in cands if x["rules"] == "R2"]
r3 = [x for x in cands if x["rules"] == "R3"]

flip_ids = set()
for i in FLIP_R2:
    flip_ids.add(r2[i]["sample_id"])
for i in FLIP_R3:
    flip_ids.add(r3[i]["sample_id"])

log, fixes = [], []
for x in cands:
    sid = x["sample_id"]
    is_flip = sid in flip_ids
    new = ("0" if x["label"] == "1" else "1") if is_flip else x["label"]
    rule_desc = ("负面词≥2却标正面" if x["rules"] == "R2"
                 else "正面词≥2却标负面")
    log.append({
        "sample_id": sid,
        "split": "train",
        "cat": x["cat"],
        "review": x["review"],
        "review_length": x["review_length"],
        "orig_label": x["label"],
        "rule": x["rules"],
        "rule_hit": x["evidence"],
        "audit_verdict": "标签错" if is_flip else "标签正确",
        "corrected_label": new if is_flip else "",
        "reason": (f"{rule_desc}，人工读原文确认整体情感与标签相反"
                   if is_flip else "混合评价/中性/跑题，整体情感与原标签一致"),
        "reviewer": "人工阅读原文判定（未参考模型预测）",
    })
    if is_flip:
        fixes.append({
            "sample_id": sid,
            "split": "train",
            "cat": x["cat"],
            "orig_label": x["label"],
            "corrected_label": new,
            "rule": x["rules"],
            "evidence": x["evidence"],
            "review": x["review"],
        })

with open(OUT / "train_label_audit_log.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
    w.writeheader()
    w.writerows(log)

FIX_COLS = ["sample_id", "split", "cat", "orig_label", "corrected_label",
            "rule", "evidence", "review"]
with open(OUT / "label_fixes_train.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIX_COLS)
    w.writeheader()
    w.writerows(fixes)

print(f"候选 {len(cands)} 条 → 确认标签错 {len(fixes)} 条")
print(f"  R2: {len(FLIP_R2)}/{len(r2)} = {len(FLIP_R2)/len(r2)*100:.1f}%")
print(f"  R3: {len(FLIP_R3)}/{len(r3)} = {len(FLIP_R3)/len(r3)*100:.1f}%")
print("翻转方向：", dict(Counter(f"{f['orig_label']}→{f['corrected_label']}" for f in fixes)))
print("类目分布：", dict(Counter(f["cat"] for f in fixes)))
print(f"占训练集比例：{len(fixes)/43894*100:.2f}%")
print(f"输出：{OUT / 'label_fixes_train.csv'}")
