# -*- coding: utf-8 -*-
"""
唯一数据划分入口（Single Source of Truth）。

所有训练 / 评估 / 分析脚本都必须通过本模块读取划分，禁止各自重新 split。

规范划分（canonical）：
    outputs/predictions/train.csv   43894 行
    outputs/predictions/val.csv      9406 行
    outputs/predictions/test.csv     9406 行

历史遗留的另一套划分已移至：
    outputs/predictions/_deprecated_alt_split/
（train_split.csv / validation_split.csv / test_split.csv，禁止再用于任何实验）

用法：
    from data_split import load, load_df, load_all, validate

    rows = load("test")                 # list[dict]，纯标准库
    df   = load_df("test")              # pandas.DataFrame，需要 pandas
    tr, va, te = load_all()
    problems = validate()               # 返回问题列表，空列表表示通过
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs" / "predictions"

CANONICAL = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
REQUIRED_COLS = {"sample_id", "cat", "label", "review"}
EXPECTED = {"train": 43894, "val": 9406, "test": 9406}

DEPRECATED_DIR = PRED / "_deprecated_alt_split"
DEPRECATED = ("train_split.csv", "validation_split.csv", "test_split.csv")

_norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()


class SplitError(RuntimeError):
    pass


def path_of(split: str) -> Path:
    if split not in CANONICAL:
        raise SplitError(f"未知 split：{split}，可选 {list(CANONICAL)}")
    return PRED / CANONICAL[split]


def load(split: str) -> list[dict]:
    """读取某个 split，返回 list[dict]（纯标准库，无 pandas 依赖）"""
    p = path_of(split)
    if not p.exists():
        raise SplitError(f"找不到划分文件：{p}")
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = REQUIRED_COLS - set(rows[0].keys())
    if missing:
        raise SplitError(f"{p.name} 缺少字段：{missing}")
    return rows


def load_df(split: str):
    """读取某个 split，返回 pandas.DataFrame（延迟导入 pandas）"""
    import pandas as pd
    return pd.read_csv(path_of(split), encoding="utf-8-sig")


def load_all() -> tuple[list[dict], list[dict], list[dict]]:
    return load("train"), load("val"), load("test")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(strict_counts: bool = True) -> list[str]:
    """全量校验，返回问题列表（空列表 = 通过）"""
    problems: list[str] = []
    data: dict[str, list[dict]] = {}

    for split, fname in CANONICAL.items():
        p = PRED / fname
        if not p.exists():
            problems.append(f"缺少文件 {fname}")
            continue
        rows = load(split)
        data[split] = rows

        if not rows:
            problems.append(f"{fname} 为空")
            continue

        ids = [r["sample_id"] for r in rows]
        if len(ids) != len(set(ids)):
            problems.append(f"{fname} 内 sample_id 重复 {len(ids)-len(set(ids))} 个")

        bad = [r["label"] for r in rows if str(r["label"]).strip() not in {"0", "1"}]
        if bad:
            problems.append(f"{fname} 存在非法标签：{set(bad)}")

        empty = sum(1 for r in rows if not str(r["review"]).strip())
        if empty:
            problems.append(f"{fname} 存在 {empty} 条空文本")

        if strict_counts and len(rows) != EXPECTED[split]:
            problems.append(
                f"{fname} 行数 {len(rows)} 与清单 {EXPECTED[split]} 不一致（数据可能被改动）")

    # 跨集合：sample_id 与文本都不得重叠
    names = list(data)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sid_a = {r["sample_id"] for r in data[a]}
            sid_b = {r["sample_id"] for r in data[b]}
            if sid_a & sid_b:
                problems.append(f"{a}/{b} 的 sample_id 重叠 {len(sid_a & sid_b)} 个")

            tx_a = {_norm(r["review"]) for r in data[a]}
            tx_b = {_norm(r["review"]) for r in data[b]}
            if tx_a & tx_b:
                problems.append(
                    f"{a}/{b} 文本重叠 {len(tx_a & tx_b)} 条（train_bert.py 会从训练集剔除）")

    # 遗留划分必须已隔离
    for fname in DEPRECATED:
        if (PRED / fname).exists():
            problems.append(
                f"{fname} 仍在 outputs/predictions/ 根目录，请移入 {DEPRECATED_DIR.name}/")
    return problems


def assert_valid(strict_counts: bool = True) -> None:
    problems = validate(strict_counts)
    # 文本重叠属于已知情况，单独提示而不阻断
    blocking = [p for p in problems if "文本重叠" not in p]
    if blocking:
        raise SplitError("划分校验失败：\n  - " + "\n  - ".join(blocking))
    for p in problems:
        print(f"[提示] {p}")


if __name__ == "__main__":
    ps = validate()
    print("=" * 60)
    for split, fname in CANONICAL.items():
        p = PRED / fname
        if p.exists():
            rows = load(split)
            neg = sum(1 for r in rows if r["label"] == "0")
            print(f"{fname:<12} {len(rows):>6} 行  负 {neg} / 正 {len(rows)-neg}  "
                  f"sha256={sha256(p)[:16]}")
    print("=" * 60)
    if ps:
        print(f"发现 {len(ps)} 项问题：")
        for x in ps:
            print("  -", x)
    else:
        print("校验通过：划分唯一、无重复、无跨集合泄漏。")
