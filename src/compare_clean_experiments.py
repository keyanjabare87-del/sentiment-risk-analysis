# -*- coding: utf-8 -*-
"""
汇总多次运行的 test_metrics.json，生成清洗前后对照表。

用法：
    python src/compare_clean_experiments.py
    python src/compare_clean_experiments.py --dirs outputs/bert outputs/bert/runs/clean_train

输出：
    outputs/bert/experiment_compare.csv
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BERT = ROOT / "outputs" / "bert"

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="*", default=None,
                help="要比较的运行目录；默认自动扫描 outputs/bert 与 runs/*")
ap.add_argument("--out", default=str(BERT / "experiment_compare.csv"))
a = ap.parse_args()

dirs = []
if a.dirs:
    dirs = [Path(d) if Path(d).is_absolute() else ROOT / d for d in a.dirs]
else:
    if (BERT / "test_metrics.json").exists():
        dirs.append(BERT)
    runs = BERT / "runs"
    if runs.exists():
        dirs += sorted(p for p in runs.iterdir() if p.is_dir())

rows = []
for d in dirs:
    p = d / "test_metrics.json"
    if not p.exists():
        print(f"[跳过] 无 test_metrics.json：{d}")
        continue
    m = json.load(open(p, encoding="utf-8"))
    rep = m.get("label_fix_report") or {}
    train_changed = (rep.get("train") or {}).get("changed", 0)
    test_changed = (rep.get("test") or {}).get("changed", 0)
    row = {
        "run": m.get("run_tag") or d.name,
        "输出目录": str(d.relative_to(ROOT)).replace("\\", "/"),
        "训练集修正条数": train_changed,
        "验证集修正条数": (rep.get("val") or {}).get("changed", 0),
        "测试集修正条数": test_changed,
        "seed": m.get("seed"),
        "best_epoch": m.get("best_epoch"),
        "训练集规模": m.get("train_size_after_dedup"),
        # 统一口径：全部用修正后标签，才能横向比
        "accuracy": round(m.get("test_accuracy", float("nan")), 4),
        "macro_f1": round(m.get("test_macro_f1", float("nan")), 4),
        # 历史口径（原标签），仅基线运行有
        "accuracy_orig": round(m["test_orig_accuracy"], 4) if "test_orig_accuracy" in m else "",
        "macro_f1_orig": round(m["test_orig_macro_f1"], 4) if "test_orig_macro_f1" in m else "",
    }
    rows.append(row)

if not rows:
    raise SystemExit("没有可比较的运行结果。")

cols = list(rows[0].keys())
with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print(f"共 {len(rows)} 次运行\n")
head = ["run", "训练集修正条数", "测试集修正条数", "accuracy", "macro_f1", "accuracy_orig"]
print(" | ".join(f"{h:>14}" for h in head))
for r in rows:
    print(" | ".join(f"{str(r[h]):>14}" for h in head))
print(f"\n对照表已写入：{a.out}")
print("\n读数提醒：")
print("  1) accuracy / macro_f1 列是「统一用修正后测试标签」的口径，只有这一列可跨运行比较。")
print("  2) 基线运行（训练集修正=0）的 accuracy 提升只反映「评测口径修正」，不代表模型变强。")
print("  3) 要证明清洗有效，需让训练集修正条数 > 0，且在同一份修正后测试集上比较。")
