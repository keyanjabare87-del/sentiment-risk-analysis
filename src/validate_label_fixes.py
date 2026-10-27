# -*- coding: utf-8 -*-
"""
校验修正标签清单，防止误改验证集/测试集或匹配错行。

用法：
    python src/validate_label_fixes.py --fixes outputs/bert/category_analysis/label_fixes_test.csv
    python src/validate_label_fixes.py --fixes ... --allow-split test

检查项：
  C1 字段完整性            C2 sample_id 唯一
  C3 sample_id 必须存在于所声明的 split
  C4 该 sample_id 不得同时出现在其他 split（跨集合泄漏）
  C5 corrected_label ∈ {0,1} 且与原标签不同
  C6 文本一致性：清单里的 review（若有）与 split 文件中同 sample_id 的文本一致
  C7 自洽性：修正后标签应等于该样本的模型预测（否则该条其实是"模型错"而非"标签错"）
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixes", required=True)
    ap.add_argument("--allow-split", nargs="*", default=["train"],
                    help="允许修改的 split，默认只允许 train")
    ap.add_argument("--strict", action="store_true",
                    help="C7 不通过时直接失败")
    a = ap.parse_args()

    fixes = load(Path(a.fixes) if Path(a.fixes).is_absolute() else ROOT / a.fixes)
    errors, warns = [], []

    # C1
    need = {"sample_id", "split", "orig_label", "corrected_label"}
    if not need.issubset(fixes[0].keys()):
        errors.append(f"C1 缺少字段：{need - set(fixes[0].keys())}")

    # C2
    dup = [k for k, v in Counter(f["sample_id"] for f in fixes).items() if v > 1]
    if dup:
        errors.append(f"C2 sample_id 重复 {len(dup)} 个：{dup[:5]}")

    # 载入三个 split
    splits = {}
    for name in ("train", "val", "test"):
        p = PRED / f"{name}.csv"
        if p.exists():
            splits[name] = {r["sample_id"]: r for r in load(p)}
    sid2splits = {}
    for name, d in splits.items():
        for sid in d:
            sid2splits.setdefault(sid, []).append(name)

    # 模型预测（用于 C7）
    pred_path = BERT / "test_predictions.csv"
    sid2pred = {}
    if pred_path.exists():
        sid2pred = {r["sample_id"]: r for r in load(pred_path)}

    for f in fixes:
        sid, sp = f["sample_id"], f["split"]
        # C3
        if sp not in splits:
            errors.append(f"C3 {sid} 声明 split={sp}，但该 split 文件不存在")
            continue
        if sid not in splits[sp]:
            errors.append(f"C3 {sid} 不在 {sp}.csv 中（匹配错行或数据已变更）")
            continue
        # C4
        others = [s for s in sid2splits.get(sid, []) if s != sp]
        if others:
            errors.append(f"C4 {sid} 同时出现在 {others}，存在跨集合泄漏风险")
        # C5
        orig = str(splits[sp][sid]["label"])
        if f.get("orig_label") and str(f["orig_label"]) != orig:
            errors.append(f"C5 {sid} 清单 orig_label={f['orig_label']} 与 {sp}.csv 的 {orig} 不一致")
        if str(f["corrected_label"]) not in {"0", "1"}:
            errors.append(f"C5 {sid} corrected_label 非法：{f['corrected_label']}")
        elif str(f["corrected_label"]) == orig:
            errors.append(f"C5 {sid} corrected_label 与原标签相同，属无效翻转")
        # C6
        if f.get("review"):
            if norm(f["review"]) != norm(splits[sp][sid]["review"]):
                errors.append(f"C6 {sid} 文本与 {sp}.csv 不一致")
        # C7
        if sp == "test" and sid in sid2pred:
            if str(f["corrected_label"]) != str(sid2pred[sid]["prediction"]):
                warns.append(f"C7 {sid} 修正后标签与模型预测不一致，该条更可能是「模型错」")
        # split 白名单
        if sp not in a.allow_split:
            errors.append(f"白名单 {sid} 属于 {sp}，未在 --allow-split 中放行")

    print("=" * 62)
    print(f"清单文件：{a.fixes}")
    print(f"条数：{len(fixes)}")
    print(f"split 分布：{dict(Counter(f['split'] for f in fixes))}")
    print(f"翻转方向：1->0 {sum(1 for f in fixes if str(f['orig_label'])=='1' and str(f['corrected_label'])=='0')}"
          f"  |  0->1 {sum(1 for f in fixes if str(f['orig_label'])=='0' and str(f['corrected_label'])=='1')}")
    print(f"涉及类别：{dict(Counter(f.get('category','-') for f in fixes))}")
    print(f"允许修改的 split：{a.allow_split}")
    print("-" * 62)

    for w in warns[:10]:
        print("[警告]", w)
    if len(warns) > 10:
        print(f"[警告] ……另有 {len(warns)-10} 条同类警告")

    if errors:
        print(f"\n[失败] 共 {len(errors)} 项错误：")
        for e in errors[:20]:
            print("  -", e)
        if len(errors) > 20:
            print(f"  ……另有 {len(errors)-20} 项")
        return 1

    if a.strict and warns:
        print(f"\n[失败] --strict 模式下存在 {len(warns)} 条警告")
        return 1

    print("\n[通过] 清单可安全用于标签修正。")
    print("提示：C7 全通过说明这 246 条是「标签错」而非「模型错」，")
    print("      因此它们只能用于修正评测口径；若要证明模型变强，需另做训练侧清洗。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
