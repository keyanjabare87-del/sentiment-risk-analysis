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

spilts = {
    name: pd.read_csv(path,encoding="utf-8-sig")
    for name, path in files.items()
}

def normalize(text):
    return " ".join(str(text).split()).strip()

sets = {
    name: set(df["review"].map(normalize))
    for name, df in spilts.items()
}

print("跨集合完全相同评论审计")
pairs = [
    ("train", "validation"),
    ("train","test"),
    ("test", "validation")
]

for a, b in pairs:
    overlap = sets[a] & sets[b]
    print(f"(a) vs (b): {len(overlap)} 条重复文本")

    if overlap:
        examples = list(overlap)[:5]
        print("示例：")
        for text in examples:
            print(" -", text[:120])

# 检查每个划分内部是否存在重复评论
for name, df in spilts.items():
    normalized = df["review"].map(normalize)
    duplicate_rows = int(normalized.duplicated().sum())
    print(f"{name} 集内部重复行数：{duplicate_rows}")

print("\n审计完成。")
print("若跨集合存在重复文本，需进一步判断是否构成实质性泄漏。")