# -*- coding: utf-8 -*-
"""
阶段 3B 第一步（v2）：在训练集里筛查「疑似标签错误」候选。

v2 相比 v1 的修正
-----------------
1. **加入否定处理**：`不甜` / `不好吃` / `没满意` 这类不再算作正面词，
   而是转成负面证据。v1 因缺这一步把大量正确样本误判为候选。
2. 去掉裸的「极短文本」规则（`很好` 这类 2 字短评本来就该是正面）。
3. 要求至少 2 个非否定极性词且反向词为 0，追求高精确率。

仍然**不使用模型预测**，避免把模型偏好当成标注错误。

输出：outputs/experiments/stage3/train_label_candidates.csv
"""
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs" / "predictions"
OUT = ROOT / "outputs" / "experiments" / "stage3"
OUT.mkdir(parents=True, exist_ok=True)

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()

POS = ["满意", "很好", "不错", "推荐", "喜欢", "好评", "棒", "赞", "好吃", "划算",
       "实惠", "正品", "舒服", "漂亮", "精美", "值得", "给力", "超值", "新鲜", "甜",
       "快", "方便", "清晰", "耐用"]
NEG = ["失望", "垃圾", "差评", "退货", "太差", "很差", "难吃", "骗", "假货", "不满",
       "后悔", "糟糕", "破损", "烂了", "态度差", "不推荐", "无语", "坑", "慢", "贵",
       "坏", "瑕疵", "异味", "假"]
NEGATORS = "不没无未非别难否"


def polarity_hits(text):
    """返回 (非否定正面词, 非否定负面词, 被否定的正面词)"""
    pos, neg, negated_pos = [], [], []
    for w in POS:
        start, found, negated = 0, False, False
        while True:
            i = text.find(w, start)
            if i < 0:
                break
            pre = text[max(0, i - 2):i]
            if any(ch in NEGATORS for ch in pre):
                negated = True
            else:
                found = True
            start = i + 1
        if found:
            pos.append(w)
        elif negated:
            negated_pos.append(w)

    for w in NEG:
        start, found = 0, False
        while True:
            i = text.find(w, start)
            if i < 0:
                break
            pre = text[max(0, i - 2):i]
            # 「不贵」「不慢」「没坏」这类否定负词不算负面
            if not any(ch in NEGATORS for ch in pre):
                found = True
            start = i + 1
        if found:
            neg.append(w)
    return pos, neg, negated_pos


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


rows = load(PRED / "train.csv")
cands = []
for i, r in enumerate(rows):
    t = norm(r["review"])
    if len(t) < 8:
        continue
    lab = r["label"]
    pos, neg, negated_pos = polarity_hits(t)
    neg_eff = neg + negated_pos

    rule, evidence = None, ""
    if lab == "0" and len(pos) >= 2 and not neg_eff:
        rule = "R3"
        evidence = "正面词:" + "/".join(pos)
    elif lab == "1" and len(neg_eff) >= 2 and not pos:
        rule = "R2"
        evidence = "负面词:" + "/".join(neg_eff)

    if rule:
        cands.append({
            "train_row": i,
            "sample_id": r["sample_id"],
            "cat": r["cat"],
            "review": r["review"],
            "review_length": len(t),
            "label": lab,
            "rules": rule,
            "evidence": evidence,
            "pos_hits": "/".join(pos),
            "neg_hits": "/".join(neg_eff),
        })

with open(OUT / "train_label_candidates.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cands[0].keys()))
    w.writeheader()
    w.writerows(cands)

print(f"训练集 {len(rows)} 行 → 候选 {len(cands)} 条")
print("按规则：", dict(Counter(c["rules"] for c in cands)))
print("按类目：", dict(Counter(c["cat"] for c in cands)))
print("按标签：", dict(Counter(c["label"] for c in cands)))
print(f"输出：{OUT / 'train_label_candidates.csv'}")
