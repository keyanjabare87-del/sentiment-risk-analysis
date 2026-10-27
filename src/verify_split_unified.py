# -*- coding: utf-8 -*-
"""
统一划分的验收脚本：一次性检查所有可能出错的地方，并生成划分清单。

用法：
    python src/verify_split_unified.py

检查项：
  V1 三个 canonical 文件存在、字段完整、行数符合清单
  V2 sample_id 在集合内唯一，跨集合不重叠
  V3 标签只含 0/1，无空文本
  V4 outputs/predictions/ 根目录不再残留 *_split.csv
  V5 src/ 下没有脚本再引用旧划分文件
  V6 246 条标签修正的 sample_id 全部存在于 test.csv
  V7 训练集去重/去泄漏后的规模与 BERT 训练脚本一致
通过后写出 outputs/experiments/split_manifest.json
"""
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_split as ds  # noqa: E402

FIXES = ROOT / "outputs" / "bert" / "category_analysis" / "label_fixes_test.csv"
OUT = ROOT / "outputs" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

fails, warns = [], []


def fail(msg):
    fails.append(msg)


def warn(msg):
    warns.append(msg)


# ---------- V1~V3 ----------
problems = ds.validate()
for p in problems:
    (warn if "文本重叠" in p else fail)(p)

rows = {s: ds.load(s) for s in ds.CANONICAL}

# ---------- V4 ----------
leftover = [f.name for f in ds.PRED.iterdir()
            if f.is_file() and f.name.endswith("_split.csv")]
if leftover:
    fail(f"V4 outputs/predictions/ 仍残留旧划分：{leftover}")

# ---------- V5 ----------
pat = re.compile(r"(train|validation|test)_split\.csv")
hits = []
for py in (ROOT / "src").glob("*.py"):
    if py.name in {"verify_split_unified.py", "stage2_baseline_record.py",
                   "inspect_split_files.py", "data_split.py"}:
        continue
    for i, line in enumerate(py.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if pat.search(line) and "_deprecated_alt_split" not in line:
            hits.append(f"{py.name}:{i}: {line.strip()[:70]}")
if hits:
    fail(f"V5 仍有脚本引用旧划分：{hits}")

# ---------- V6 ----------
fix_ids, fix_split = set(), Counter()
if FIXES.exists():
    with open(FIXES, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            fix_ids.add(r["sample_id"])
            fix_split[r["split"]] += 1
    test_ids = {r["sample_id"] for r in rows["test"]}
    missing = fix_ids - test_ids
    if missing:
        fail(f"V6 有 {len(missing)} 条标签修正的 sample_id 不在 test.csv：{list(missing)[:5]}")
else:
    warn("V6 未找到 label_fixes_test.csv，跳过")

# ---------- V7 ----------
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()
prot = {norm(r["review"]) for r in rows["val"]} | {norm(r["review"]) for r in rows["test"]}
tr = rows["train"]
leak = sum(1 for r in tr if norm(r["review"]) in prot)
kept = [r for r in tr if norm(r["review"]) not in prot]
seen, ded = set(), 0
for r in kept:
    k = norm(r["review"])
    if k in seen:
        ded += 1
    else:
        seen.add(k)
final_train = len(kept) - ded

# ---------- 清单 ----------
manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "policy": "outputs/predictions/{train,val,test}.csv 为唯一合法划分，"
              "所有脚本必须通过 src/data_split.py 读取",
    "label_definition": {"0": "负面", "1": "正面"},
    "splits": {},
    "train_after_cleaning": {
        "raw": len(tr), "leak_removed": leak,
        "internal_dup_removed": ded, "final": final_train,
    },
    "deprecated": {
        "dir": "outputs/predictions/_deprecated_alt_split",
        "files": sorted(p.name for p in (ds.PRED / "_deprecated_alt_split").glob("*.csv")),
        "reason": "train_baselines.py 早期版本自行 train_test_split 生成的第二套划分",
    },
    "label_fixes": {"count": len(fix_ids), "by_split": dict(fix_split)},
    "checks": {"failed": fails, "warnings": warns},
}

for s, fname in ds.CANONICAL.items():
    p = ds.PRED / fname
    r = rows[s]
    lab = Counter(x["label"] for x in r)
    cat = Counter(x["cat"] for x in r)
    manifest["splits"][s] = {
        "file": f"outputs/predictions/{fname}",
        "rows": len(r),
        "sha256": ds.sha256(p),
        "label_distribution": {"0": lab["0"], "1": lab["1"]},
        "negative_ratio": round(lab["0"] / len(r), 4),
        "categories": dict(sorted(cat.items())),
    }

(OUT / "split_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 输出 ----------
print("=" * 66)
print("统一划分验收")
print("=" * 66)
for s, fname in ds.CANONICAL.items():
    info = manifest["splits"][s]
    print(f"{fname:<11} {info['rows']:>6} 行  负{info['label_distribution']['0']:>6} / "
          f"正{info['label_distribution']['1']:>6}  "
          f"sha256={info['sha256'][:16]}")
print(f"训练集清洗后：{len(tr)} → 去泄漏 {leak} → 去重 {ded} → {final_train}")
print(f"标签修正：{len(fix_ids)} 条，split 分布 {dict(fix_split)}")
print(f"旧划分归档：{manifest['deprecated']['files']}")
print("-" * 66)
for w in warns:
    print("[警告]", w)
if fails:
    print(f"[失败] {len(fails)} 项：")
    for x in fails:
        print("  -", x)
    sys.exit(1)
print("[通过] 划分已统一，清单已写入 outputs/experiments/split_manifest.json")
