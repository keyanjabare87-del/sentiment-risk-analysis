# -*- coding: utf-8 -*-
"""
阶段 3C 第一步：挖掘训练集里的「真·转折句」样本。

为什么要专门挖
--------------
test 集 502 条误判中，92 条属于「转折或反讽」——模型抓住前半段褒义就判正面，
忽略「就是/不过/但是」之后的抱怨。这是模型最集中的短板。

做法
----
1. 按转折连词把评论切成前后两段
2. 分别算两段极性，只保留「前后极性相反」的（真转折，而不是单纯并列）
3. 输出候选，由人工确认标签是否正确
4. 确认正确的样本在训练时做**过采样**（重复 k 次），让模型多看这类结构

注意：过采样只作用于 train，绝不引入 val/test 样本。

输出：outputs/experiments/stage3/contrastive_train_candidates.csv
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_split import load  # noqa: E402

OUT = ROOT / "outputs" / "experiments" / "stage3"
OUT.mkdir(parents=True, exist_ok=True)

MARKERS = ["但是", "但", "不过", "只是", "就是", "可惜", "然而", "却", "虽然", "尽管"]

POS = ["满意", "很好", "不错", "推荐", "喜欢", "好评", "棒", "赞", "好吃", "划算",
       "实惠", "正品", "舒服", "漂亮", "精美", "值得", "给力", "超值", "新鲜", "甜",
       "快", "方便", "清晰", "耐用", "香", "合适"]
NEG = ["失望", "垃圾", "差评", "退货", "太差", "很差", "难吃", "骗", "假货", "不满",
       "后悔", "糟糕", "破损", "烂", "态度差", "不推荐", "无语", "坑", "慢", "贵",
       "坏", "瑕疵", "异味", "假", "脏", "旧", "卡", "酸"]
NEGATORS = "不没无未非别难否"


def polarity(s):
    v = 0
    for w in POS:
        for m in re.finditer(re.escape(w), s):
            pre = s[max(0, m.start() - 2):m.start()]
            v += -1 if any(c in NEGATORS for c in pre) else 1
    for w in NEG:
        for m in re.finditer(re.escape(w), s):
            pre = s[max(0, m.start() - 2):m.start()]
            if not any(c in NEGATORS for c in pre):
                v -= 1
    return (v > 0) - (v < 0)


rows = load("train")
cands = []
for i, r in enumerate(rows):
    t = str(r["review"]).replace("\n", " ")
    pos_m = [(t.find(m), m) for m in MARKERS if m in t]
    if not pos_m:
        continue
    idx, mk = min(pos_m)
    if idx < 4 or len(t) - idx < 6:
        continue
    left, right = t[:idx], t[idx + len(mk):]
    pl, pr = polarity(left), polarity(right)
    if pl == 0 or pr == 0 or pl == pr:
        continue
    cands.append({
        "train_row": i,
        "sample_id": r["sample_id"],
        "cat": r["cat"],
        "label": r["label"],
        "marker": mk,
        "left_polarity": pl,
        "right_polarity": pr,
        "review": t,
        "review_length": len(t),
    })

with open(OUT / "contrastive_train_candidates.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cands[0].keys()))
    w.writeheader()
    w.writerows(cands)

print(f"训练集 {len(rows)} 行 → 真转折候选 {len(cands)} 条")
print("按标签：", dict(Counter(c["label"] for c in cands)))
print("按连词：", dict(Counter(c["marker"] for c in cands).most_common(6)))
print("按「前正后负/前负后正」：",
      dict(Counter(f"{c['left_polarity']}→{c['right_polarity']}" for c in cands)))
print(f"输出：{OUT / 'contrastive_train_candidates.csv'}")
