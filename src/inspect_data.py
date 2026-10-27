
from pathlib import Path
import pandas as pd


# ============================================================
# 0. 路径配置
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "online_shopping_10_cats.csv"
OUT_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "outputs" / "metrics"

OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_PATH = OUT_DIR / "reviews_clean.csv"
CONFLICT_PATH = METRICS_DIR / "label_conflicts.csv"


# ============================================================
# 1. 读取原始数据
# ============================================================
if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"找不到数据文件：{RAW_PATH}\n"
        "请将 online_shopping_10_cats.csv 放入 data/raw/ 目录。"
    )

df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")

print("=" * 60)
print("第一步：读取原始数据")
print("=" * 60)

print("原始数据形状：", df.shape)
print("原始字段：", df.columns.tolist())

required = {"cat", "label", "review"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"数据缺少必要字段：{missing}")

# 只保留本项目需要的字段
df = df[["cat", "label", "review"]].copy()

original_count = len(df)


# ============================================================
# 2. 基础数据清理
# ============================================================
print("\n" + "=" * 60)
print("第二步：基础数据清理")
print("=" * 60)

# 清理商品类别
df["cat"] = (
    df["cat"]
    .fillna("未知")
    .astype(str)
    .str.strip()
)

# 清理评论文本：
# 1. 将缺失值转为空字符串
# 2. 将连续空白字符（空格、制表符、换行符等）合并成一个空格
# 3. 删除首尾空白
df["review"] = (
    df["review"]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 标签转换为数值，无法转换的内容设为 NaN
df["label"] = pd.to_numeric(df["label"], errors="coerce")

# 统计并删除空评论
empty_count = int(df["review"].eq("").sum())
df = df[df["review"].ne("")].copy()

# 统计并删除不合法标签
invalid_label_count = int((~df["label"].isin([0, 1])).sum())
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

print("原始样本数：", original_count)
print("删除空评论数：", empty_count)
print("删除缺失或非法标签数：", invalid_label_count)
print("基础清理后样本数：", len(df))


# ============================================================
# 3. 检查同一规范化文本是否对应冲突标签
# ============================================================
print("\n" + "=" * 60)
print("第三步：检查标签冲突")
print("=" * 60)

# 此时 review 已规范化，因此空格数量不同的相同评论
# 会被识别为同一条文本
label_counts = df.groupby("review")["label"].nunique()

conflict_texts = label_counts[label_counts > 1].index
conflict_count = len(conflict_texts)

print("存在冲突标签的文本数：", conflict_count)

if conflict_count > 0:
    # 保存冲突记录，方便后续人工检查
    conflict_df = df[df["review"].isin(conflict_texts)].copy()

    conflict_df.to_csv(
        CONFLICT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("冲突记录已保存：", CONFLICT_PATH)

    # 保守处理：同一评论对应不同标签时，暂不用于训练
    df = df[~df["review"].isin(conflict_texts)].copy()
else:
    # 没有冲突时，删除可能存在的旧冲突报告
    if CONFLICT_PATH.exists():
        CONFLICT_PATH.unlink()

print("冲突处理后样本数：", len(df))


# ============================================================
# 4. 按规范化评论文本去重
# ============================================================
print("\n" + "=" * 60)
print("第四步：评论文本去重")
print("=" * 60)

before_dedup = len(df)

# 同一规范化文本只保留一条。
# 即使商品类别不同，也不允许相同文本进入不同数据记录。
df = df.drop_duplicates(
    subset=["review"],
    keep="first"
).copy()

duplicate_removed = before_dedup - len(df)

print("删除重复评论数：", duplicate_removed)
print("去重后样本数：", len(df))


# ============================================================
# 5. 数据质量检查
# ============================================================
print("\n" + "=" * 60)
print("第五步：数据质量检查")
print("=" * 60)

# 检查是否仍有重复文本
remaining_duplicates = int(df["review"].duplicated().sum())

# 检查标签是否合法
invalid_labels_remaining = int((~df["label"].isin([0, 1])).sum())

# 检查是否仍有空文本
empty_reviews_remaining = int(df["review"].eq("").sum())

print("剩余重复文本数：", remaining_duplicates)
print("剩余非法标签数：", invalid_labels_remaining)
print("剩余空评论数：", empty_reviews_remaining)

if (
    remaining_duplicates > 0
    or invalid_labels_remaining > 0
    or empty_reviews_remaining > 0
):
    raise RuntimeError("数据质量检查未通过，请检查清洗逻辑。")


# ============================================================
# 6. 输出数据概况
# ============================================================
print("\n" + "=" * 60)
print("第六步：清洗后数据概况")
print("=" * 60)

print("清洗后总样本数：", len(df))

print("\n标签分布：")
print(df["label"].value_counts().sort_index())

print("\n标签比例：")
print(df["label"].value_counts(normalize=True).sort_index().round(4))

print("\n商品类别分布：")
print(df["cat"].value_counts())

print("\n评论长度统计：")
print(df["review"].str.len().describe())

print("\n各类别的标签分布：")
print(
    pd.crosstab(
        df["cat"],
        df["label"],
        margins=True
    )
)


# ============================================================
# 7. 保存清洗结果
# ============================================================
print("\n" + "=" * 60)
print("第七步：保存数据")
print("=" * 60)

df.to_csv(
    CLEAN_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("清洗后数据已保存：", CLEAN_PATH)
print("最终数据形状：", df.shape)

print("\n前 5 条数据：")
print(df.head(5).to_string(index=False))

print("\n数据清洗完成！")
