# -*- coding: utf-8 -*-
"""
训练前预检：不依赖 torch / pandas，验证「修正清单 → 三个 split」这条链路是否正确。

它做的事与 train_bert.py 中的 apply_label_fixes 完全一致（按 sample_id 精确替换），
并复算测试集在两种口径下的指标，用于确认补丁逻辑无误后再启动训练。

用法：
    python src/dryrun_label_fix.py --fixes outputs/bert/category_analysis/label_fixes_test.csv --allow-split test
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs" / "predictions"
BERT = ROOT / "outputs" / "bert"

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def apply_fixes(rows, fixes, split):
    """与 train_bert.py 中 apply_label_fixes 等价的逻辑"""
    m = fixes.get(split)
    if not m:
        return rows, 0, 0
    hit = changed = 0
    for r in rows:
        if r["sample_id"] in m:
            hit += 1
            new = m[r["sample_id"]]
            if int(r["label"]) != new:
                r["label"] = str(new)
                changed += 1
    return rows, hit, changed


def report(rows, tag):
    c = Counter()
    for r in rows:
        y, p = int(r["label"]), int(r["prediction"])
        c["TP" if (y, p) == (1, 1) else "TN" if (y, p) == (0, 0)
          else "FP" if (y, p) == (0, 1) else "FN"] += 1
    n = len(rows)
    neg = c["TN"] + c["FP"]
    pn = c["TN"] + c["FN"]
    acc = (c["TN"] + c["TP"]) / n
    rec = c["TN"] / neg if neg else float("nan")
    pre = c["TN"] / pn if pn else float("nan")
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else float("nan")
    print(f"[{tag}] N={n} 负样本={neg} TN={c['TN']} FP={c['FP']} "
          f"FN={c['FN']} TP={c['TP']}")
    print(f"      准确率={acc*100:.2f}%  负面精确={pre*100:.2f}%  "
          f"负面召回={rec*100:.2f}%  负面F1={f1:.4f}")
    return {"acc": acc, "prec": pre, "rec": rec, "f1": f1, "cm": dict(c)}


ap = argparse.ArgumentParser()
ap.add_argument("--fixes", required=True)
ap.add_argument("--allow-split", nargs="*", default=["train"])
a = ap.parse_args()

fix_rows = load(Path(a.fixes) if Path(a.fixes).is_absolute() else ROOT / a.fixes)
fixes = {}
for r in fix_rows:
    sp = str(r["split"]).strip()
    if sp not in a.allow_split:
        raise SystemExit(f"split={sp} 未在 --allow-split={a.allow_split} 中放行")
    fixes.setdefault(sp, {})[str(r["sample_id"]).strip()] = int(float(r["corrected_label"]))

print(f"修正清单 {len(fix_rows)} 条，按 split 分组："
      f"{ {k: len(v) for k, v in fixes.items()} }\n")

data = {}
for name in ("train", "val", "test"):
    p = PRED / f"{name}.csv"
    if p.exists():
        data[name] = load(p)

# 把模型预测挂到测试集行上（供复算指标）
tp = {r["sample_id"]: r for r in load(BERT / "test_predictions.csv")}
for r in data.get("test", []):
    r["prediction"] = tp[r["sample_id"]]["prediction"]

orig_test = [dict(r) for r in data.get("test", [])]  # 修正前快照

for name in ("train", "val", "test"):
    if name not in data:
        continue
    rows, hit, changed = apply_fixes(data[name], fixes, name)
    print(f"{name}: 命中 {hit} 条，翻转 {changed} 条")

print()
# 训练集去重/去泄漏后的规模（与 train_bert.py 一致）
if "train" in data:
    prot = {norm(r["review"]) for n in ("val", "test") if n in data
            for r in data[n]}
    before_n = len(data["train"])
    kept = [r for r in data["train"] if norm(r["review"]) not in prot]
    leak = before_n - len(kept)
    seen, ded = set(), []
    for r in kept:
        k = norm(r["review"])
        if k not in seen:
            seen.add(k)
            ded.append(r)
    print(f"训练集：原始 {before_n} → 剔除跨集合重复 {leak} → 内部去重 "
          f"{len(kept)-len(ded)} → 最终 {len(ded)}")
    fixed_in_train = sum(1 for r in ded if r["sample_id"] in fixes.get("train", {}))
    print(f"最终训练集中被修正的样本数：{fixed_in_train}")

print()
before = report(orig_test, "测试集-原标签")
after = report(data["test"], "测试集-修正后")

# 与先前生成的影响表对齐校验
imp = load(BERT / "category_analysis" / "label_fix_impact_test.csv")
exp = {r["口径"]: r for r in imp}
ok = (exp["修正前(原标签)"]["准确率"] == f"{before['acc']*100:.2f}%"
      and exp["修正后(人工标签)"]["准确率"] == f"{after['acc']*100:.2f}%")
print(f"\n与 label_fix_impact_test.csv 一致性校验：{'通过' if ok else '不一致'}")

# 校验未命中样本完全没被动过
untouched = all(
    a1["label"] == a2["label"]
    for a1, a2 in zip(orig_test, data["test"])
    if a1["sample_id"] not in fixes.get("test", {})
)
print(f"未在清单内的测试样本标签是否零改动：{'是' if untouched else '否'}")
sys.exit(0 if (ok and untouched) else 1)
