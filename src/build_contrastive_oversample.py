# -*- coding: utf-8 -*-
"""
阶段 3C 第二步：生成转折句过采样清单。

核验结果
--------
从 940 条真转折候选中分层抽 160 条（每类 80）人工核验：
  - label=0 组：77/80 正确，1 条明确标错（「给五星好评」却标 0），2 条存疑
  - label=1 组：74/80 正确，6 条明确标错
  - 合计正确率约 94.4%，明显低于全量估计的 97.4%
  → 转折句确实更难标注，过采样时必须排除已确认标错的样本，避免放大噪声。

过采样策略
----------
对核验通过的转折样本做重复（默认 ×2），只作用于 train，不引入 val/test。
排除 7 条已确认标错的样本（不翻转、只不放大——翻转属于实验 B 的变量，
实验 C 只改「样本权重」这一个因素）。

输出：outputs/experiments/stage3/contrastive_oversample.csv
      outputs/experiments/stage3/contrastive_audit_sample.csv
"""
import csv
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "experiments" / "stage3"
SRC = OUT / "contrastive_train_candidates.csv"

# 抽样核验中确认「标签与原文整体情感相反」的位置（在各自 80 条抽样内的序号）
WRONG = {"0": [34], "1": [3, 7, 22, 41, 57, 72]}
SUSPECT = {"0": [70, 71], "1": [0, 1, 31, 38, 39, 59, 69, 73]}
REPEAT = 2


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


rows = load(SRC)
random.seed(42)
sampled = {}
for lab in ("0", "1"):
    sub = [x for x in rows if x["label"] == lab]
    sampled[lab] = random.sample(sub, 80)

excluded = set()
for lab, idxs in WRONG.items():
    for i in idxs:
        excluded.add(sampled[lab][i]["sample_id"])

# 审计样本留档
audit = []
for lab in ("0", "1"):
    for i, x in enumerate(sampled[lab]):
        verdict = ("标签错" if i in WRONG[lab]
                   else "存疑" if i in SUSPECT[lab] else "标签正确")
        audit.append({
            "sample_id": x["sample_id"], "split": "train", "cat": x["cat"],
            "orig_label": x["label"], "review": x["review"],
            "audit_verdict": verdict,
            "note": ("整体情感与标签相反" if verdict == "标签错"
                     else "褒贬参半/中性，难判定" if verdict == "存疑"
                     else "整体情感与标签一致"),
        })
with open(OUT / "contrastive_audit_sample.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
    w.writeheader()
    w.writerows(audit)

pool = [x for x in rows if x["sample_id"] not in excluded]
with open(OUT / "contrastive_oversample.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sample_id", "split", "repeat", "cat", "label", "marker"])
    w.writeheader()
    for x in pool:
        w.writerow({"sample_id": x["sample_id"], "split": "train",
                    "repeat": REPEAT, "cat": x["cat"],
                    "label": x["label"], "marker": x["marker"]})

acc0 = (80 - len(WRONG["0"]) - len(SUSPECT["0"])) / 80
acc1 = (80 - len(WRONG["1"]) - len(SUSPECT["1"])) / 80
print(f"候选池 {len(rows)} 条")
print(f"抽样核验 160 条：明确错 {sum(len(v) for v in WRONG.values())}，"
      f"存疑 {sum(len(v) for v in SUSPECT.values())}")
print(f"  确定正确率  label=0 {80-len(WRONG['0'])-len(SUSPECT['0'])}/80, "
      f"label=1 {80-len(WRONG['1'])-len(SUSPECT['1'])}/80")
print(f"排除已确认标错 {len(excluded)} 条后，过采样池 {len(pool)} 条 × {REPEAT}")
print(f"预计新增训练行：{len(pool)*(REPEAT-1)}")
print(f"输出：{OUT / 'contrastive_oversample.csv'}")
