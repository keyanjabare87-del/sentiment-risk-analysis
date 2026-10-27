# -*- coding: utf-8 -*-
"""
阶段 3D 第一步：方面级情感分析的**规则基线**（不训练模型）。

目的不是拿高分，而是用最低成本回答一个问题：
「把评论拆成 质量/价格/物流/客服/售后/包装/外观 等维度分别判极性，再聚合，
是否值得再投入一个方面级模型？」

方法
----
1. 方面识别：关键词表命中（如 物流/快递/配送 → aspect=物流）
2. 方面极性：在该方面词所在的**分句内**（按 ，。；！？ 切分）找极性词，
   带否定处理；分句内没有极性词则视为未表态
3. 聚合：按方面权重加权求和（商品本体权重最高，服务类次之）
4. 判定：加权得分 < 0 → 负面(0)，否则正面(1)

局限（必须写进结论）
------------------
- 没有方面级标注，只能在**文档级**评估，无法衡量方面识别本身准不准
- 规则对反讽、隐含否定无能为力
- 只能作为「是否值得上模型」的下界参考

输出：outputs/experiments/stage3/aspect_rule_baseline.csv
      outputs/experiments/stage3/aspect_rule_metrics.csv
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_split import load  # noqa: E402

OUT = ROOT / "outputs" / "experiments" / "stage3"
OUT.mkdir(parents=True, exist_ok=True)

# 方面 → 关键词
ASPECT_KW = {
    "质量": ["质量", "做工", "材质", "面料", "料子", "手感", "耐用", "正品", "假货", "次品"],
    "口感": ["好吃", "难吃", "口感", "味道", "甜", "酸", "新鲜", "烂", "坏", "脆", "面"],
    "价格": ["价格", "价钱", "贵", "便宜", "实惠", "划算", "性价比", "降价", "促销"],
    "物流": ["物流", "快递", "配送", "送货", "到货", "发货", "顺丰", "京东小哥"],
    "客服售后": ["客服", "售后", "服务", "态度", "退货", "换货", "维修", "安装"],
    "包装赠品": ["包装", "赠品", "礼盒", "泡沫", "破损", "漏", "发票"],
    "外观": ["外观", "款式", "颜色", "漂亮", "好看", "设计", "版型"],
    "尺寸": ["尺寸", "尺码", "大小", "合身", "偏大", "偏小", "太长", "太短"],
}
# 方面权重：商品本体 > 服务 > 附加
ASPECT_W = {"质量": 1.0, "口感": 1.0, "价格": 0.9, "尺寸": 0.8, "外观": 0.7,
            "客服售后": 0.7, "物流": 0.5, "包装赠品": 0.5}

POS = ["满意", "很好", "不错", "推荐", "喜欢", "好评", "棒", "赞", "好吃", "划算",
       "实惠", "正品", "舒服", "漂亮", "精美", "值得", "给力", "超值", "新鲜", "甜",
       "快", "方便", "清晰", "耐用", "香", "合适", "准"]
NEG = ["失望", "垃圾", "差评", "退货", "太差", "很差", "难吃", "骗", "假货", "不满",
       "后悔", "糟糕", "破损", "烂", "态度差", "不推荐", "无语", "坑", "慢", "贵",
       "坏", "瑕疵", "异味", "假", "脏", "旧", "卡", "酸"]
NEGATORS = "不没无未非别难否"

CLAUSE = re.compile(r"[，,。；;！!？?\n]")


def polarity_of(clause):
    """分句极性：+1 正面 / -1 负面 / 0 未表态"""
    s = 0
    for w in POS:
        for m in re.finditer(re.escape(w), clause):
            pre = clause[max(0, m.start() - 2):m.start()]
            s += -1 if any(c in NEGATORS for c in pre) else 1
    for w in NEG:
        for m in re.finditer(re.escape(w), clause):
            pre = clause[max(0, m.start() - 2):m.start()]
            if not any(c in NEGATORS for c in pre):
                s -= 1
    return (s > 0) - (s < 0)


def analyze(text):
    """返回 (加权得分, 命中的方面及其极性)

    若全篇没有识别到任何方面（如「很好，很喜欢」），退化为整句极性求和，
    避免无方面样本一律被判为正面。
    """
    total, hits = 0.0, {}
    global_score = 0.0
    for clause in CLAUSE.split(str(text)):
        if not clause.strip():
            continue
        pol = polarity_of(clause)
        if pol == 0:
            continue
        global_score += pol
        for asp, kws in ASPECT_KW.items():
            if any(k in clause for k in kws):
                total += pol * ASPECT_W[asp]
                hits.setdefault(asp, []).append(pol)
    if not hits:
        return global_score, hits
    return total, hits


rows = load("test")
preds, aspect_cnt = [], Counter()
per_aspect_pol = defaultdict(Counter)
for r in rows:
    score, hits = analyze(r["review"])
    pred = 0 if score < 0 else 1
    preds.append({**r, "prediction": pred, "score": round(score, 2),
                  "aspects": "|".join(hits)})
    for a, pols in hits.items():
        aspect_cnt[a] += 1
        for p in pols:
            per_aspect_pol[a][p] += 1

with open(OUT / "aspect_rule_baseline.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
    w.writeheader()
    w.writerows(preds)

c = Counter()
for p in preds:
    y, pr = int(p["label"]), p["prediction"]
    c[("TN", "FP", "FN", "TP")[{(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}[(y, pr)]]] += 1
n = len(preds)
neg = c["TN"] + c["FP"]
pn = c["TN"] + c["FN"]
acc = (c["TN"] + c["TP"]) / n
pre = c["TN"] / pn
rec = c["TN"] / neg
f1n = 2 * pre * rec / (pre + rec)
pp = c["TP"] / (c["TP"] + c["FP"])
rp = c["TP"] / (c["TP"] + c["FN"])
f1p = 2 * pp * rp / (pp + rp)
macro = (f1n + f1p) / 2

no_aspect = sum(1 for p in preds if not p["aspects"])
metrics = [
    ("样本数", n), ("准确率", f"{acc:.4f}"), ("Macro-F1", f"{macro:.4f}"),
    ("负面精确率", f"{pre:.4f}"), ("负面召回率", f"{rec:.4f}"), ("负面F1", f"{f1n:.4f}"),
    ("TN", c["TN"]), ("FP", c["FP"]), ("FN", c["FN"]), ("TP", c["TP"]),
    ("未识别到任何方面的样本", no_aspect),
]
with open(OUT / "aspect_rule_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["指标", "值"])
    w.writerows(metrics)

print("=" * 56)
print("方面级规则基线（文档级评估）")
print("=" * 56)
for k, v in metrics:
    print(f"  {k:<22} {v}")
print("\n方面命中次数：")
for a, v in aspect_cnt.most_common():
    print(f"  {a:<8} {v:>6}  正面表态 {per_aspect_pol[a][1]:>5} / 负面表态 {per_aspect_pol[a][-1]:>5}")
print(f"\n输出：{OUT}")
