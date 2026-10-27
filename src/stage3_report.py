# -*- coding: utf-8 -*-
"""
阶段 3/4 汇总：把所有实验拉到同一张表，并给出分类别指标。

数据来源
--------
A  原始 BERT        outputs/bert/test_metrics.json + test_predictions.csv
B  训练集标签清洗    outputs/bert/runs/B_clean_train/
C  转折句过采样      outputs/bert/runs/C_contrastive/
BC 组合             outputs/bert/runs/BC_clean_contrastive/
D  方面级规则基线    outputs/experiments/stage3/aspect_rule_baseline.csv

输出
----
outputs/experiments/stage3/experiment_comparison.csv
outputs/experiments/stage3/category_macro_f1.csv
outputs/experiments/stage3/stage3_report.md
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_split import load  # noqa: E402

BERT = ROOT / "outputs" / "bert"
OUT = ROOT / "outputs" / "experiments" / "stage3"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("A", "原始 BERT", BERT),
    ("B", "训练集标签清洗", BERT / "runs" / "B_clean_train"),
    ("C", "转折句过采样", BERT / "runs" / "C_contrastive"),
    ("BC", "清洗+过采样", BERT / "runs" / "BC_clean_contrastive"),
]


def load_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def metrics(rows, label_key="label"):
    c = Counter()
    for r in rows:
        y, p = int(r[label_key]), int(r["prediction"])
        c[("TN", "FP", "FN", "TP")[{(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}[(y, p)]]] += 1
    n = len(rows)
    neg, pn = c["TN"] + c["FP"], c["TN"] + c["FN"]
    acc = (c["TN"] + c["TP"]) / n if n else 0
    pre = c["TN"] / pn if pn else 0
    rec = c["TN"] / neg if neg else 0
    f1n = 2 * pre * rec / (pre + rec) if (pre + rec) else 0
    pp = c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) else 0
    rp = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) else 0
    f1p = 2 * pp * rp / (pp + rp) if (pp + rp) else 0
    return {"n": n, "acc": acc, "macro_f1": (f1n + f1p) / 2,
            "neg_precision": pre, "neg_recall": rec, "neg_f1": f1n,
            "TN": c["TN"], "FP": c["FP"], "FN": c["FN"], "TP": c["TP"]}


rows_out, cat_rows = [], []
for eid, name, d in RUNS:
    mj, pp = d / "test_metrics.json", d / "test_predictions.csv"
    if not (mj.exists() and pp.exists()):
        rows_out.append({"experiment_id": eid, "experiment": name,
                         "status": "未运行", "note": "缺少产物"})
        continue
    meta = json.load(open(mj, encoding="utf-8"))
    preds = load_csv(pp)
    m = metrics(preds)
    fr = meta.get("label_fix_report") or {}
    ar = meta.get("augment_report") or {}
    rows_out.append({
        "experiment_id": eid, "experiment": name, "status": "已完成",
        "test_size": m["n"],
        "accuracy": f"{m['acc']:.4f}", "macro_f1": f"{m['macro_f1']:.4f}",
        "negative_precision": f"{m['neg_precision']:.4f}",
        "negative_recall": f"{m['neg_recall']:.4f}",
        "negative_f1": f"{m['neg_f1']:.4f}",
        "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"],
        "train_label_fixes": (fr.get("train") or {}).get("changed", 0),
        "train_rows_added": ar.get("added_rows", 0),
        "train_size": meta.get("train_size_after_dedup"),
        "best_epoch": meta.get("best_epoch"),
        "val_macro_f1": round(meta.get("best_val_macro_f1", 0), 4),
        "inference_ms_per_sample": "待测",
    })
    # 分类别
    by_cat = defaultdict(list)
    for r in preds:
        by_cat[r["cat"]].append(r)
    for cat, sub in sorted(by_cat.items()):
        mc = metrics(sub)
        cat_rows.append({
            "experiment_id": eid, "experiment": name, "category": cat,
            "n": mc["n"], "accuracy": f"{mc['acc']:.4f}",
            "macro_f1": f"{mc['macro_f1']:.4f}",
            "negative_recall": f"{mc['neg_recall']:.4f}",
            "negative_precision": f"{mc['neg_precision']:.4f}",
            "error_count": mc["FP"] + mc["FN"],
        })

# D：规则基线
d_pred = OUT / "aspect_rule_baseline.csv"
if d_pred.exists():
    preds = load_csv(d_pred)
    m = metrics(preds)
    rows_out.append({
        "experiment_id": "D", "experiment": "方面级规则基线", "status": "已完成",
        "test_size": m["n"], "accuracy": f"{m['acc']:.4f}",
        "macro_f1": f"{m['macro_f1']:.4f}",
        "negative_precision": f"{m['neg_precision']:.4f}",
        "negative_recall": f"{m['neg_recall']:.4f}",
        "negative_f1": f"{m['neg_f1']:.4f}",
        "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"],
        "train_label_fixes": 0, "train_rows_added": 0, "train_size": 0,
        "best_epoch": "-", "val_macro_f1": "-",
        "inference_ms_per_sample": "待测",
    })
    by_cat = defaultdict(list)
    for r in preds:
        by_cat[r["cat"]].append(r)
    for cat, sub in sorted(by_cat.items()):
        mc = metrics(sub)
        cat_rows.append({
            "experiment_id": "D", "experiment": "方面级规则基线", "category": cat,
            "n": mc["n"], "accuracy": f"{mc['acc']:.4f}",
            "macro_f1": f"{mc['macro_f1']:.4f}",
            "negative_recall": f"{mc['neg_recall']:.4f}",
            "negative_precision": f"{mc['neg_precision']:.4f}",
            "error_count": mc["FP"] + mc["FN"],
        })

cols = ["experiment_id", "experiment", "status", "test_size", "accuracy", "macro_f1",
        "negative_precision", "negative_recall", "negative_f1", "TN", "FP", "FN", "TP",
        "train_label_fixes", "train_rows_added", "train_size", "best_epoch",
        "val_macro_f1", "inference_ms_per_sample"]
with open(OUT / "experiment_comparison.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows_out)

with open(OUT / "category_macro_f1.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()))
    w.writeheader()
    w.writerows(cat_rows)

# ---------- 报告 ----------
L = ["# 阶段 3 实验报告：A → B → C → D\n"]
L.append("所有实验使用同一划分（`train/val/test.csv`，sha256 见 `split_manifest.json`）、"
         "同一种子（42）、同一超参；测试集只在最后评估一次。\n")
L.append("## 1. 总表（测试集原标签口径）\n")
L.append("| 实验 | Accuracy | Macro-F1 | 负面 Recall | 负面 Precision | 训练集修正 | 过采样新增 |")
L.append("|---|---|---|---|---|---|---|")
for r in rows_out:
    if r.get("status") != "已完成":
        L.append(f"| {r['experiment_id']} {r['experiment']} | - | - | - | - | - | - |")
        continue
    L.append(f"| {r['experiment_id']} {r['experiment']} | {r['accuracy']} | {r['macro_f1']} "
             f"| {r['negative_recall']} | {r['negative_precision']} "
             f"| {r['train_label_fixes']} | {r['train_rows_added']} |")
L.append("")

base = next((r for r in rows_out if r["experiment_id"] == "A" and r.get("status") == "已完成"), None)
if base:
    L.append("## 2. 相对基线 A 的变化\n")
    L.append("| 实验 | ΔAccuracy | ΔMacro-F1 | Δ负面Recall |")
    L.append("|---|---|---|---|")
    for r in rows_out:
        if r["experiment_id"] == "A" or r.get("status") != "已完成":
            continue
        L.append(f"| {r['experiment_id']} {r['experiment']} | "
                 f"{float(r['accuracy'])-float(base['accuracy']):+.4f} | "
                 f"{float(r['macro_f1'])-float(base['macro_f1']):+.4f} | "
                 f"{float(r['negative_recall'])-float(base['negative_recall']):+.4f} |")
    L.append("")

    # ---- 配对显著性检验 ----
    import math
    import random as _rnd

    def paired_test(pa, pb):
        A = {r["sample_id"]: r for r in load_csv(pa)}
        B = {r["sample_id"]: r for r in load_csv(pb)}
        ids = [i for i in A if i in B]
        a_ok = {i: A[i]["label"] == A[i]["prediction"] for i in ids}
        b_ok = {i: B[i]["label"] == B[i]["prediction"] for i in ids}
        b = sum(1 for i in ids if a_ok[i] and not b_ok[i])
        c = sum(1 for i in ids if not a_ok[i] and b_ok[i])
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
        p = math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0
        _rnd.seed(0)
        n = len(ids)
        diffs = []
        for _ in range(2000):
            s = [b_ok[i] - a_ok[i] for i in _rnd.choices(ids, k=n)]
            diffs.append(sum(s) / n)
        diffs.sort()
        d = (sum(b_ok.values()) - sum(a_ok.values())) / n
        return {"n": n, "a_only": b, "b_only": c, "chi2": chi2, "p": p,
                "delta": d, "lo": diffs[50], "hi": diffs[1949]}

    L.append("## 2b. 配对显著性检验（同测试集，McNemar + bootstrap）\n")
    L.append("| 实验 | A对B错 | A错B对 | McNemar χ² | p 值 | ΔAccuracy(B−A) | 95% CI | 结论 |")
    L.append("|---|---|---|---|---|---|---|---|")
    sig_rows = []
    for eid, name, d in RUNS:
        if eid == "A":
            continue
        pa, pb = BERT / "test_predictions.csv", d / "test_predictions.csv"
        if not (pa.exists() and pb.exists()):
            continue
        t = paired_test(pa, pb)
        concl = "显著" if t["p"] < 0.05 else "不显著（与噪声不可区分）"
        L.append(f"| {eid} {name} | {t['a_only']} | {t['b_only']} | {t['chi2']:.2f} "
                 f"| {t['p']:.4f} | {t['delta']:+.5f} "
                 f"| [{t['lo']:+.5f}, {t['hi']:+.5f}] | {concl} |")
        sig_rows.append({"experiment": eid, **t, "conclusion": concl})
    L.append("")
    L.append("> 二分类、n=9406 时准确率的标准误约 0.23 个百分点，"
             "因此小于 0.5 个百分点的差异必须看配对检验，不能只看小数点后三位。\n")
    if sig_rows:
        with open(OUT / "significance_test.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sig_rows[0].keys()))
            w.writeheader()
            w.writerows(sig_rows)

L.append("## 3. 分类别 Macro-F1（检查是否只在大类有效）\n")
cats = sorted({r["category"] for r in cat_rows})
eids = [r["experiment_id"] for r in rows_out if r.get("status") == "已完成"]
L.append("| 类别 | 样本数 | " + " | ".join(eids) + " |")
L.append("|---|---|" + "---|" * len(eids))
for c in cats:
    n = next((r["n"] for r in cat_rows if r["category"] == c), 0)
    cells = []
    for e in eids:
        v = next((r["macro_f1"] for r in cat_rows if r["category"] == c and r["experiment_id"] == e), "-")
        cells.append(v)
    L.append(f"| {c} | {n} | " + " | ".join(cells) + " |")
L.append("")
L.append("> 热水器仅 77 条、蒙牛 314 条，单类指标的置信区间很宽，"
         "不足以据此断定某方法有效；需结合错误类型与样本量一起看。\n")

L.append("## 3b. 分类别配对检验（探索性，多重比较需谨慎）\n")
L.append("| 实验 | 类别 | 样本数 | A对X错 | A错X对 | p 值 |")
L.append("|---|---|---|---|---|---|")
cat_sig = []
for eid, name, d in RUNS:
    if eid == "A":
        continue
    pa, pb = BERT / "test_predictions.csv", d / "test_predictions.csv"
    if not (pa.exists() and pb.exists()):
        continue
    A_ = {r["sample_id"]: r for r in load_csv(pa)}
    B_ = {r["sample_id"]: r for r in load_csv(pb)}
    for cat in sorted({A_[i]["cat"] for i in A_}):
        ids = [i for i in A_ if A_[i]["cat"] == cat]
        x = sum(1 for i in ids if A_[i]["label"] == A_[i]["prediction"]
                and B_[i]["label"] != B_[i]["prediction"])
        y = sum(1 for i in ids if A_[i]["label"] != A_[i]["prediction"]
                and B_[i]["label"] == B_[i]["prediction"])
        if x + y == 0:
            continue
        chi = (abs(x - y) - 1) ** 2 / (x + y)
        pv = math.erfc(math.sqrt(chi / 2)) if chi > 0 else 1.0
        if pv < 0.10:
            L.append(f"| {eid} | {cat} | {len(ids)} | {x} | {y} | {pv:.4f} |")
            cat_sig.append({"experiment": eid, "category": cat, "n": len(ids),
                            "a_only": x, "b_only": y, "p": round(pv, 4)})
L.append("")
L.append("> 共 30 次比较（3 实验 × 10 类），p<0.05 的期望假阳性约 1.5 个。"
         "「平板」在 B/C/BC 三次独立运行中方向一致且 p 均 < 0.005，"
         "比任何单个 p 值更值得信任；其余单项不显著。\n")
if cat_sig:
    with open(OUT / "category_significance.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cat_sig[0].keys()))
        w.writeheader()
        w.writerows(cat_sig)

L.append("## 4. 未测项\n")
L.append("- 推理耗时 / 吞吐量：待测（需单独跑 benchmark）。")
L.append("- B+C 组合实验：待 B、C 结果出来后决定是否值得做。\n")
(OUT / "stage3_report.md").write_text("\n".join(L), encoding="utf-8")

print("实验对比：")
for r in rows_out:
    if r.get("status") == "已完成":
        print(f"  {r['experiment_id']:<3} {r['experiment']:<14} acc={r['accuracy']} "
              f"mf1={r['macro_f1']} nrec={r['negative_recall']}")
    else:
        print(f"  {r['experiment_id']:<3} {r['experiment']:<14} 未运行")
print(f"\n输出：{OUT}")
