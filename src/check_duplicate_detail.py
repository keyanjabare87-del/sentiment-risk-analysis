
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"

# 统一划分：唯一合法来源（旧的 *_split.csv 已归档到 _deprecated_alt_split/）
files = {
    "train": PRED_DIR / "train.csv",
    "validation": PRED_DIR / "val.csv",
    "test": PRED_DIR / "test.csv",
}

def load_data(path):
    df = pd.read_csv(path)
    df["normalized_review"] = (
        df["review"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df

data = {name: load_data(path) for name, path in files.items()}

pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

for name_a, name_b in pairs:
    a, b = data[name_a], data[name_b]

    common = set(a["normalized_review"]) & set(b["normalized_review"])
    common.discard("")

    print(f"\n{name_a} vs {name_b}: {len(common)} 条重复文本")

    if common:
        for review in sorted(common)[:5]:
            print("\n重复评论：", review[:300])

            cols = [
                c for c in ["cat", "label", "review"]
                if c in a.columns and c in b.columns
            ]

            print(f"{name_a} 中的记录：")
            print(a.loc[a["normalized_review"] == review, cols].to_string(index=False))

            print(f"{name_b} 中的记录：")
            print(b.loc[b["normalized_review"] == review, cols].to_string(index=False))
