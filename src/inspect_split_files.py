
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"

# 统一划分：前三个是唯一合法来源；后三个已归档，仅作历史对照
ALT_DIR = PRED_DIR / "_deprecated_alt_split"
FILES = [
    ("canonical", PRED_DIR / "train.csv"),
    ("canonical", PRED_DIR / "val.csv"),
    ("canonical", PRED_DIR / "test.csv"),
    ("deprecated", ALT_DIR / "train_split.csv"),
    ("deprecated", ALT_DIR / "validation_split.csv"),
    ("deprecated", ALT_DIR / "test_split.csv"),
]

def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

data = {}

print("=" * 70)
print("数据划分检查")
print("=" * 70)

for status, path in FILES:
    filename = path.name

    if not path.exists():
        print(f"\n[缺失] {filename}")
        continue

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"\n文件：{filename}  [{status}]")
    print(f"样本数：{len(df)}")
    print(f"字段：{list(df.columns)}")

    if {"review", "label"}.issubset(df.columns):
        df["review_norm"] = normalize_text(df["review"])
        print("标签分布：")
        print(df["label"].value_counts(dropna=False).sort_index().to_dict())
        print("标签缺失数：", df["label"].isna().sum())
        print("评论缺失数：", df["review_norm"].eq("").sum())
        print("规范化后重复评论数：",
              int(df["review_norm"].duplicated().sum()))
        data[filename] = df

# 检查常见三份数据的文本交集
groups = [
    ("train.csv", "val.csv"),
    ("train.csv", "test.csv"),
    ("val.csv", "test.csv"),
    ("train_split.csv", "validation_split.csv"),
    ("train_split.csv", "test_split.csv"),
    ("validation_split.csv", "test_split.csv"),
]

print("\n" + "=" * 70)
print("跨数据集重复评论检查")
print("=" * 70)

for name_a, name_b in groups:
    if name_a not in data or name_b not in data:
        continue

    a = set(data[name_a]["review_norm"])
    b = set(data[name_b]["review_norm"])

    # 排除空评论
    a.discard("")
    b.discard("")

    overlap = a & b
    print(f"{name_a} 与 {name_b}："
          f"{len(overlap)} 条不同评论文本存在交集")

print("\n检查完成。这里只检查文件结构与文本重合，"
      "尚未证明这些文件就是正确的数据划分。")
