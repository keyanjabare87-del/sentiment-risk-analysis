
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# 1. 路径与配置
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "outputs" / "bert" / "test_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "bert" / "category_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 每个商品类别最多挑选多少条典型错误案例
CASES_PER_CATEGORY = 5

# 标签定义：0 = 负面，1 = 正面
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1


# ============================================================
# 2. 通用辅助函数
# ============================================================
def find_column(df, candidates, required=True):
    """从候选列名中寻找实际存在的列。"""
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"找不到需要的列。候选列名：{candidates}\n"
            f"文件实际列名：{list(df.columns)}"
        )
    return None


def normalize_text(series):
    """统一空格，便于文本查看和后续对比。"""
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def safe_divide(a, b):
    """分母为 0 时返回 NaN，避免产生误导性指标。"""
    return float(a / b) if b else np.nan


def calc_binary_metrics(group):
    """将标签 1 作为正类，计算二分类指标。"""
    y_true = group["label"].astype(int)
    y_pred = group["prediction"].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "sample_count": len(group),

        # 真实标签分布
        "true_negative_count": int((y_true == 0).sum()),
        "true_positive_count": int((y_true == 1).sum()),

        # 预测标签分布
        "predicted_negative_count": int((y_pred == 0).sum()),
        "predicted_positive_count": int((y_pred == 1).sum()),

        # 混淆矩阵四项
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),

        # 整体指标
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        # 负面类指标：负面为正类时的 Recall/Precision
        "negative_precision": precision_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "negative_recall": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),

        # 正面类指标
        "positive_precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "positive_recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),

        "error_count": int((y_true != y_pred).sum()),
        "error_rate": float((y_true != y_pred).mean()),
    }


# ============================================================
# 3. 读取测试集预测
# ============================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"找不到预测文件：{INPUT_FILE}\n"
        "请先确认 BERT 训练脚本已生成 test_predictions.csv。"
    )

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

review_col = find_column(df, ["review", "text", "content"])
label_col = find_column(df, ["label", "true_label", "y_true"])
pred_col = find_column(
    df,
    ["prediction", "predicted_label", "pred_label", "y_pred", "pred"],
)
category_col = find_column(
    df,
    ["cat", "category", "product_category"],
)

# 统一列名，后续所有分析使用同一套字段
rename_map = {
    review_col: "review",
    label_col: "label",
    pred_col: "prediction",
    category_col: "category",
}
df = df.rename(columns=rename_map).copy()

df["review"] = df["review"].fillna("").astype(str)
df["review_normalized"] = normalize_text(df["review"])
df["category"] = df["category"].fillna("未知类别").astype(str)

df["label"] = pd.to_numeric(
    df["label"], errors="raise"
).astype(int)

df["prediction"] = pd.to_numeric(
    df["prediction"], errors="raise"
).astype(int)

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("真实标签不是 0/1，请检查标签定义。")

if not set(df["prediction"].unique()).issubset({0, 1}):
    raise ValueError("预测标签不是 0/1，请检查预测文件。")

# 如果有概率列，尝试自动识别
prob_neg_col = find_column(
    df,
    ["prob_negative", "prob_0", "probability_0", "negative_probability"],
    required=False,
)
prob_pos_col = find_column(
    df,
    ["prob_positive", "prob_1", "probability_1", "positive_probability"],
    required=False,
)

# 兼容模型导出的概率字段
if prob_neg_col and prob_pos_col:
    df["prob_negative"] = pd.to_numeric(
        df[prob_neg_col], errors="coerce"
    )
    df["prob_positive"] = pd.to_numeric(
        df[prob_pos_col], errors="coerce"
    )

    # 预测类别对应的置信度
    df["confidence"] = np.where(
        df["prediction"] == 1,
        df["prob_positive"],
        df["prob_negative"],
    )

    df["probability_margin"] = (
        df["prob_positive"] - df["prob_negative"]
    ).abs()
else:
    df["prob_negative"] = np.nan
    df["prob_positive"] = np.nan
    df["confidence"] = np.nan
    df["probability_margin"] = np.nan

df["correct"] = df["label"] == df["prediction"]
df["review_length"] = df["review"].str.len()

print("=" * 70)
print("BERT 商品类别与错误案例分析")
print("=" * 70)
print(f"输入文件：{INPUT_FILE}")
print(f"测试集样本数：{len(df)}")
print(f"商品类别数：{df['category'].nunique()}")
print(f"错误样本数：{int((~df['correct']).sum())}")
print(f"整体准确率：{accuracy_score(df['label'], df['prediction']):.4f}")
print(
    "整体 Macro-F1："
    f"{f1_score(df['label'], df['prediction'], average='macro'):.4f}"
)

if not (prob_neg_col and prob_pos_col):
    print(
        "\n提示：未识别到正负面概率列。"
        "混淆矩阵和错误统计仍然有效，"
        "但典型案例无法按预测置信度排序。"
    )


# ============================================================
# 4. 每个商品类别的正负样本数量与分类指标
# ============================================================
category_rows = []

for category, group in df.groupby("category", dropna=False):
    row = {"category": category}
    row.update(calc_binary_metrics(group))
    category_rows.append(row)

category_report = pd.DataFrame(category_rows)

# 同时保存按 Macro-F1 和样本数排序的报告
category_report = category_report.sort_values(
    ["macro_f1", "sample_count"],
    ascending=[True, False],
)

category_report.to_csv(
    OUTPUT_DIR / "category_metrics_and_counts.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n===== 各类别样本数量与指标 =====")
print(
    category_report[
        [
            "category",
            "sample_count",
            "true_negative_count",
            "true_positive_count",
            "predicted_negative_count",
            "predicted_positive_count",
            "TN", "FP", "FN", "TP",
            "macro_f1",
            "negative_recall",
            "positive_recall",
            "error_rate",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 5. 各类别混淆矩阵：每个类别一行
# ============================================================
confusion_rows = []

for category, group in df.groupby("category", dropna=False):
    tn, fp, fn, tp = confusion_matrix(
        group["label"],
        group["prediction"],
        labels=[0, 1],
    ).ravel()

    confusion_rows.append({
        "category": category,
        "actual_negative_predicted_negative_TN": int(tn),
        "actual_negative_predicted_positive_FP": int(fp),
        "actual_positive_predicted_negative_FN": int(fn),
        "actual_positive_predicted_positive_TP": int(tp),
    })

confusion_df = pd.DataFrame(confusion_rows).sort_values("category")

confusion_df.to_csv(
    OUTPUT_DIR / "category_confusion_matrices.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 6. 错误类型统计
# ============================================================
def classify_error(row):
    if row["label"] == 0 and row["prediction"] == 1:
        return "负面误判为正面（FP）"
    if row["label"] == 1 and row["prediction"] == 0:
        return "正面误判为负面（FN）"
    if row["label"] == 0 and row["prediction"] == 0:
        return "正确识别负面（TN）"
    return "正确识别正面（TP）"


df["result_type"] = df.apply(classify_error, axis=1)

error_type_summary = (
    df.groupby(["category", "result_type"], dropna=False)
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# 确保即使某类没有某种错误，也保留该列
expected_types = [
    "负面误判为正面（FP）",
    "正面误判为负面（FN）",
    "正确识别负面（TN）",
    "正确识别正面（TP）",
]

for col in expected_types:
    if col not in error_type_summary.columns:
        error_type_summary[col] = 0

error_type_summary["total_errors"] = (
    error_type_summary["负面误判为正面（FP）"]
    + error_type_summary["正面误判为负面（FN）"]
)

error_type_summary.to_csv(
    OUTPUT_DIR / "category_error_type_counts.csv",
    index=False,
    encoding="utf-8-sig",
)

# 全局错误类型统计
overall_error_summary = (
    df["result_type"]
    .value_counts()
    .rename_axis("result_type")
    .reset_index(name="count")
)

overall_error_summary.to_csv(
    OUTPUT_DIR / "overall_error_type_counts.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n===== 全局错误类型统计 =====")
print(overall_error_summary.to_string(index=False))


# ============================================================
# 7. 生成完整误判案例表
# ============================================================
errors = df.loc[~df["correct"]].copy()

errors["error_type"] = errors.apply(
    lambda row: (
        "负面误判为正面（FP）"
        if row["label"] == 0 and row["prediction"] == 1
        else "正面误判为负面（FN）"
    ),
    axis=1,
)

errors["true_sentiment"] = errors["label"].map(
    {0: "负面", 1: "正面"}
)
errors["predicted_sentiment"] = errors["prediction"].map(
    {0: "负面", 1: "正面"}
)

case_columns = [
    "category",
    "review",
    "review_length",
    "label",
    "true_sentiment",
    "prediction",
    "predicted_sentiment",
    "error_type",
    "prob_negative",
    "prob_positive",
    "confidence",
    "probability_margin",
]

errors[case_columns].sort_values(
    ["category", "error_type"]
).to_csv(
    OUTPUT_DIR / "all_misclassified_cases.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 8. 典型案例：每类优先选置信度最高的误判
# ============================================================
typical_cases = []

for category, group in errors.groupby("category", dropna=False):
    if group["confidence"].notna().any():
        # 高置信度误判：模型对错误答案很有把握
        selected = group.sort_values(
            "confidence", ascending=False
        ).head(CASES_PER_CATEGORY).copy()
        selected["case_selection_reason"] = "高置信度误判"
    else:
        # 没有概率时，选取评论长度有差异的前几条作为人工阅读起点
        selected = group.sort_values(
            "review_length", ascending=False
        ).head(CASES_PER_CATEGORY).copy()
        selected["case_selection_reason"] = (
            "无概率字段，按较长评论选取供人工检查"
        )

    typical_cases.append(selected)

if typical_cases:
    typical_df = pd.concat(typical_cases, ignore_index=True)
else:
    typical_df = pd.DataFrame(
        columns=case_columns + ["case_selection_reason"]
    )

output_columns = case_columns + ["case_selection_reason"]

typical_df.reindex(columns=output_columns).to_csv(
    OUTPUT_DIR / "typical_error_cases_by_category.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 9. 每类输出可直接阅读的文本报告
# ============================================================
report_lines = [
    "# BERT 商品类别与错误案例分析报告",
    "",
    f"- 测试集样本数：{len(df)}",
    f"- 商品类别数：{df['category'].nunique()}",
    f"- 错误样本数：{len(errors)}",
    f"- 整体准确率：{accuracy_score(df['label'], df['prediction']):.4f}",
    (
        "- 整体 Macro-F1："
        f"{f1_score(df['label'], df['prediction'], average='macro'):.4f}"
    ),
    "",
    "## 各类别结果",
    "",
    "| 类别 | 样本数 | 真实负面 | 真实正面 | TN | FP | FN | TP | Macro-F1 | 负面召回率 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for _, row in category_report.iterrows():
    report_lines.append(
        f"| {row['category']} "
        f"| {int(row['sample_count'])} "
        f"| {int(row['true_negative_count'])} "
        f"| {int(row['true_positive_count'])} "
        f"| {int(row['TN'])} "
        f"| {int(row['FP'])} "
        f"| {int(row['FN'])} "
        f"| {int(row['TP'])} "
        f"| {row['macro_f1']:.4f} "
        f"| {row['negative_recall']:.4f} |"
    )

report_lines += [
    "",
    "## 典型误判案例",
    "",
    "请打开 typical_error_cases_by_category.csv，结合原文人工判断错误原因。",
    "不要仅凭模型预测就修改测试集标签。",
    "",
    "## 指标解释",
    "",
    "- TN：真实负面，预测负面。",
    "- FP：真实负面，预测正面。",
    "- FN：真实正面，预测负面。",
    "- TP：真实正面，预测正面。",
    "- 负面召回率 = TN / (TN + FP)。",
    "- 正面召回率 = TP / (TP + FN)。",
    "",
    "注意：热水器等样本量较少的类别，指标可能有较大波动。",
]

(OUTPUT_DIR / "analysis_report.md").write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("\n===== 典型错误案例抽样 =====")
print(f"全部误判案例：{OUTPUT_DIR / 'all_misclassified_cases.csv'}")
print(f"每类典型案例：{OUTPUT_DIR / 'typical_error_cases_by_category.csv'}")
print(f"混淆矩阵：{OUTPUT_DIR / 'category_confusion_matrices.csv'}")
print(f"类别报告：{OUTPUT_DIR / 'analysis_report.md'}")

print("\n分析完成，输出目录：")
print(OUTPUT_DIR)
