from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
classification_report,
confusion_matrix,
)

# 路径配置
ROOT_DIR = Path(__file__).resolve().parents[1]
PRED_PATH = (
    ROOT_DIR / "outputs" / "predictions" / "linear_svm_test_predictions.csv"
)
OUT_DIR = ROOT_DIR / "outputs" / "error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not PRED_PATH.exists():
    raise FileNotFoundError(
        f"找不到预测文件：{PRED_PATH}\n"
        "请先运行 src/train_baseline.py\n"
    )

df = pd.read_csv(PRED_PATH, encoding="utf-8-sig")

required = {"cat", "review", "label", "prediction", "decision_score"}
missing = required -  set(df.columns)

if missing:
    raise ValueError(
        f"预测文件缺少字段： {missing}\n"
    )

df = df.dropna(
    subset=["cat","review","label","prediction","decision_score"]
).copy()

df["label"] = df["label"].astype(int)
df["prediction"] = df["prediction"].astype(int)
df["decision_score"] = pd.to_numeric(df["decision_score"], errors="coerce")

df["correct"] = df["label"] == df["prediction"]
df["review_length"] = df["review"].astype(str).str.len()
df["abs_margin"] = df["decision_score"].abs()

# 对二分类 LinearSVC，decision_score > 0 通常对应类别 1，
# decision_score < 0 通常对应类别 0。
# 绝对值越小，样本越接近分类边界，但不等于概率。

print("=" * 65)
print("第二阶段：线性 SVM 错误分析")
print("=" * 65)

# ============================================================
# 1. 整体测试集指标
# ============================================================
y_pred = df["prediction"]
y_true = df["label"]

print("\n[1] 整体指标")
print(f"测试样本数：{len(df)}")
print(f"预测正确数：{int(df['correct'].sum())}")
print(f"预测错误数：{int((~df['correct']).sum())}")
print(f"错误率：{(~df['correct']).mean():.4%}")
print(f"Accuracy：{accuracy_score(y_true, y_pred):.4f}")
print(
    "Macro-F1：",
    f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
)
print(
    "Macro-Precision：",
    f"{precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
)
print(
    "Macro-Recall：",
    f"{recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
)

print("\n混淆矩阵（行=真实标签，列=预测标签；顺序为 0、1）：")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\n分类报告：")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["负面（0）", "正面（1）"],
        digits=4,
        zero_division=0,
    )
)

# ============================================================
# 2. 按商品类别分析
# ============================================================
print("\n[2] 按商品类别分析")

category_analysis = (
    df.groupby("cat")
    .agg(
        samples=("correct", "size"),
        errors=("correct", lambda x: int((~x).sum())),
        accuracy=("correct", "mean"),
        mean_abs_margin=("abs_margin", "mean"),
        mean_review_length=("review_length", "mean"),
    )
    .reset_index()
)

category_analysis["error_rate"] = (
    category_analysis["errors"] / category_analysis["samples"]
)

# 计算每个类别的错误率 Wilson 95% 置信区间
def wilson_interval(errors, n, z=1.96):
    if n == 0:
        return np.nan, np.nan

    p = errors / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = (
        z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        / denominator
    )
    return max(0, center - half), min(1, center + half)

intervals = category_analysis.apply(
    lambda row: wilson_interval(
        int(row["errors"]), int(row["samples"])
    ),
    axis=1,
)

category_analysis["error_rate_ci_lower"] = [
    x[0] for x in intervals
]
category_analysis["error_rate_ci_upper"] = [
    x[1] for x in intervals
]

category_analysis = category_analysis.sort_values(
    ["error_rate", "samples"],
    ascending=[False, False],
)

category_path = OUT_DIR / "category_error_analysis.csv"
category_analysis.to_csv(
    category_path, index=False, encoding="utf-8-sig"
)

print(
    category_analysis[
        [
            "cat", "samples", "errors", "error_rate",
            "error_rate_ci_lower", "error_rate_ci_upper",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

print(f"已保存：{category_path}")

# ============================================================
# 3. 按评论长度分析
# ============================================================
print("\n[3] 按评论长度分析")

length_bins = [0, 10, 30, 60, 120, 300, np.inf]
length_labels = [
    "1-10", "11-30", "31-60",
    "61-120", "121-300", "301+",
]

df["length_group"] = pd.cut(
    df["review_length"],
    bins=length_bins,
    labels=length_labels,
    include_lowest=True,
)

length_analysis = (
    df.groupby("length_group", observed=False)
    .agg(
        samples=("correct", "size"),
        errors=("correct", lambda x: int((~x).sum())),
        accuracy=("correct", "mean"),
        mean_abs_margin=("abs_margin", "mean"),
    )
    .reset_index()
)

length_analysis["error_rate"] = (
    length_analysis["errors"]
    / length_analysis["samples"].replace(0, np.nan)
)

length_path = OUT_DIR / "length_error_analysis.csv"
length_analysis.to_csv(
    length_path, index=False, encoding="utf-8-sig"
)

print(length_analysis.to_string(index=False))
print(f"已保存：{length_path}")

# ============================================================
# 4. 导出全部误判样本
# ============================================================
print("\n[4] 导出误判样本")

errors = df.loc[~df["correct"]].copy()

errors["error_type"] = np.select(
    [
        (errors["label"] == 0) & (errors["prediction"] == 1),
        (errors["label"] == 1) & (errors["prediction"] == 0),
    ],
    [
        "负面被误判为正面",
        "正面被误判为负面",
    ],
    default="其他",
)

errors = errors.sort_values("abs_margin", ascending=True)

error_path = OUT_DIR / "misclassified_reviews.csv"
errors.to_csv(error_path, index=False, encoding="utf-8-sig")

print(f"误判样本数：{len(errors)}")
print(f"已保存：{error_path}")

# ============================================================
# 5. 分析靠近分类边界和远离分类边界的错误
# ============================================================
print("\n[5] 误判样本的决策分数分析")

low_margin_errors = errors.nsmallest(
    min(30, len(errors)), "abs_margin"
)

high_margin_errors = errors.nlargest(
    min(30, len(errors)), "abs_margin"
)

show_cols = [
    "cat", "label", "prediction", "decision_score",
    "abs_margin", "review",
]

low_path = OUT_DIR / "low_margin_errors.csv"
high_path = OUT_DIR / "high_margin_errors.csv"

low_margin_errors[show_cols].to_csv(
    low_path, index=False, encoding="utf-8-sig"
)
high_margin_errors[show_cols].to_csv(
    high_path, index=False, encoding="utf-8-sig"
)

print("\n最接近分类边界的误判样本：")
if len(low_margin_errors):
    print(
        low_margin_errors[show_cols]
        .head(10)
        .to_string(index=False)
    )
else:
    print("没有误判样本。")

print("\n远离分类边界的误判样本：")
if len(high_margin_errors):
    print(
        high_margin_errors[show_cols]
        .head(10)
        .to_string(index=False)
    )
else:
    print("没有误判样本。")

print(f"\n已保存：{low_path}")
print(f"已保存：{high_path}")

# ============================================================
# 6. 汇总决策分数和错误关系
# ============================================================
print("\n[6] 正确与错误样本的决策分数对比")

margin_summary = (
    df.groupby("correct")
    .agg(
        samples=("abs_margin", "size"),
        mean_abs_margin=("abs_margin", "mean"),
        median_abs_margin=("abs_margin", "median"),
    )
    .reset_index()
)

margin_summary["result"] = margin_summary["correct"].map(
    {True: "预测正确", False: "预测错误"}
)

print(
    margin_summary[
        [
            "result", "samples",
            "mean_abs_margin", "median_abs_margin",
        ]
    ].to_string(index=False)
)

margin_path = OUT_DIR / "margin_summary.csv"
margin_summary.to_csv(
    margin_path, index=False, encoding="utf-8-sig"
)

print(f"\n已保存：{margin_path}")

print("\n" + "=" * 65)
print("错误分析完成！")
print(f"所有结果目录：{OUT_DIR}")
print("=" * 65)
