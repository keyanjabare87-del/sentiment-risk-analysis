
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =====================================================
# 1. 路径配置
# =====================================================
ROOT = Path(__file__).resolve().parents[1]
BERT_PRED_PATH = ROOT / "outputs" / "bert" / "test_predictions.csv"
OUT_DIR = ROOT / "outputs" / "bert" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 如果你有完整的 LinearSVC 测试集预测文件，
# 把下面路径改成它的真实路径。
# 文件至少需要包含 review、label、prediction 三列；
# 最好也包含 cat 列。
SVC_PRED_PATH = (
    ROOT / "outputs" / "predictions" / "linearsvc_test_predictions.csv"
)


def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def get_metrics(y_true, y_pred):
    return {
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
        "negative_recall": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "positive_recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
    }


# =====================================================
# 2. 读取 BERT 测试集预测
# =====================================================
if not BERT_PRED_PATH.exists():
    raise FileNotFoundError(
        f"找不到 BERT 预测文件：{BERT_PRED_PATH}"
    )

df = pd.read_csv(BERT_PRED_PATH, encoding="utf-8-sig")

required = {"review", "label", "prediction", "cat"}
missing = required - set(df.columns)
if missing:
    raise ValueError(
        f"BERT 预测文件缺少字段：{missing}；"
        f"当前字段为：{list(df.columns)}"
    )

df["review"] = df["review"].fillna("").astype(str)
df["review_norm"] = normalize_text(df["review"])
df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
df["prediction"] = pd.to_numeric(
    df["prediction"], errors="raise"
).astype(int)

df["correct"] = df["label"] == df["prediction"]
df["review_length"] = df["review"].str.len()

if {"prob_negative", "prob_positive"}.issubset(df.columns):
    df["confidence"] = df[
        ["prob_negative", "prob_positive"]
    ].max(axis=1)
    df["probability_margin"] = (
        df["prob_positive"] - df["prob_negative"]
    ).abs()
else:
    df["confidence"] = float("nan")
    df["probability_margin"] = float("nan")

print("=" * 65)
print("BERT 测试集错误分析")
print("=" * 65)
print(f"测试集样本数：{len(df)}")
print(f"错误样本数：{int((~df['correct']).sum())}")
print(f"整体准确率：{accuracy_score(df['label'], df['prediction']):.4f}")
print(
    f"整体 Macro-F1："
    f"{f1_score(df['label'], df['prediction'], average='macro'):.4f}"
)


# =====================================================
# 3. 按商品类别计算指标
# =====================================================
category_rows = []

for category, group in df.groupby("cat", dropna=False):
    metrics = get_metrics(group["label"], group["prediction"])
    category_rows.append({
        "category": str(category),
        "sample_count": len(group),
        "error_count": int((~group["correct"]).sum()),
        "error_rate": float((~group["correct"]).mean()),
        **metrics,
    })

category_report = pd.DataFrame(category_rows)
category_report = category_report.sort_values(
    "macro_f1", ascending=True
)

category_report.to_csv(
    OUT_DIR / "bert_category_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n===== 各商品类别指标（按 Macro-F1 从低到高） =====")
print(
    category_report[
        [
            "category", "sample_count", "error_count",
            "error_rate", "macro_f1",
            "negative_recall", "positive_recall",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)


# =====================================================
# 4. 输出误判评论
# =====================================================
errors = df.loc[~df["correct"]].copy()
errors["error_type"] = errors.apply(
    lambda row: (
        "负面误判为正面"
        if row["label"] == 0 and row["prediction"] == 1
        else "正面误判为负面"
    ),
    axis=1,
)

errors.sort_values(
    ["cat", "probability_margin"],
    ascending=[True, True],
).to_csv(
    OUT_DIR / "bert_misclassified_reviews.csv",
    index=False,
    encoding="utf-8-sig",
)

# 低置信度误判：模型对错误预测本身也不够确定
errors.sort_values(
    "confidence", ascending=True
).head(200).to_csv(
    OUT_DIR / "bert_low_confidence_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

# 高置信度误判：模型非常确信，但判断错误
errors.sort_values(
    "confidence", ascending=False
).head(200).to_csv(
    OUT_DIR / "bert_high_confidence_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

# 按评论长度汇总错误率
df["length_group"] = pd.cut(
    df["review_length"],
    bins=[-1, 20, 50, 100, 200, float("inf")],
    labels=["0-20", "21-50", "51-100", "101-200", "200以上"],
)

length_report = (
    df.groupby("length_group", observed=False)
    .agg(
        sample_count=("correct", "size"),
        error_count=("correct", lambda s: int((~s).sum())),
        accuracy=("correct", "mean"),
    )
    .reset_index()
)
length_report["error_rate"] = (
    1 - length_report["accuracy"]
)
length_report.to_csv(
    OUT_DIR / "bert_length_error_analysis.csv",
    index=False,
    encoding="utf-8-sig",
)


# =====================================================
# 5. 可选：与 LinearSVC 对比
# =====================================================
if not SVC_PRED_PATH.exists():
    print("\n没有找到配置的 LinearSVC 完整预测文件：")
    print(SVC_PRED_PATH)
    print(
        "已完成 BERT 的类别、长度和错误样本分析。"
        "如需公平对比，请将 SVC_PRED_PATH 改为"
        "包含完整测试集预测的 CSV 文件。"
    )
else:
    svc = pd.read_csv(SVC_PRED_PATH, encoding="utf-8-sig")

    svc_required = {"review", "label", "prediction"}
    svc_missing = svc_required - set(svc.columns)
    if svc_missing:
        raise ValueError(
            f"LinearSVC 预测文件缺少字段：{svc_missing}"
        )

    svc["review_norm"] = normalize_text(svc["review"])
    svc["label"] = pd.to_numeric(
        svc["label"], errors="raise"
    ).astype(int)
    svc["prediction"] = pd.to_numeric(
        svc["prediction"], errors="raise"
    ).astype(int)

    # 用规范化文本和真实标签匹配，避免仅按行号错配。
    keys = ["review_norm", "label"]

    if df.duplicated(keys).any() or svc.duplicated(keys).any():
        raise ValueError(
            "存在重复的“评论文本+标签”匹配键，"
            "为避免错误配对，已停止模型对比。"
        )

    merged = df.merge(
        svc[keys + ["prediction"]].rename(
            columns={"prediction": "svc_prediction"}
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(df) or len(merged) != len(svc):
        raise ValueError(
            f"两模型预测未能完整对齐："
            f"BERT={len(df)}，SVC={len(svc)}，"
            f"匹配={len(merged)}。"
            "请检查是否使用相同测试集和标签。"
        )

    merged["bert_correct"] = (
        merged["label"] == merged["prediction"]
    )
    merged["svc_correct"] = (
        merged["label"] == merged["svc_prediction"]
    )

    comparison = pd.DataFrame([
        {
            "model": "LinearSVC",
            **get_metrics(merged["label"], merged["svc_prediction"]),
        },
        {
            "model": "BERT",
            **get_metrics(merged["label"], merged["prediction"]),
        },
    ])

    comparison.to_csv(
        OUT_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # BERT 修正了 SVC 错误的样本
    bert_fixed = merged[
        (~merged["svc_correct"]) & merged["bert_correct"]
    ].copy()
    bert_fixed["comparison_type"] = "BERT修正SVC错误"

    # SVC 正确、BERT 错误的样本
    bert_regressed = merged[
        merged["svc_correct"] & (~merged["bert_correct"])
    ].copy()
    bert_regressed["comparison_type"] = "BERT新增错误"

    comparison_errors = pd.concat(
        [bert_fixed, bert_regressed],
        ignore_index=True,
    )
    comparison_errors.to_csv(
        OUT_DIR / "bert_vs_linearsvc_error_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 模型对比 =====")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"BERT 修正 SVC 错误：{len(bert_fixed)} 条")
    print(f"BERT 新增错误：{len(bert_regressed)} 条")

print("\n分析完成，输出目录：")
print(OUT_DIR)
