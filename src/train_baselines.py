
from pathlib import Path
import json
import sys
import time

import joblib
import jieba
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_split import path_of, assert_valid, sha256  # noqa: E402

MODEL_DIR = ROOT / "models"
METRIC_DIR = ROOT / "outputs" / "metrics"
PRED_DIR = ROOT / "outputs" / "predictions"

for folder in [MODEL_DIR, METRIC_DIR, PRED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def tokenize_chinese(text):
    """中文分词；暂时不删除停用词，以保留否定词等信息。"""
    return [
        word.strip()
        for word in jieba.__lcut(str(text))
        if word.strip()
    ]


def make_pipeline(classifier):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                tokenizer=tokenize_chinese,
                token_pattern=None,
                lowercase=False,
                ngram_range=(1, 2),
                min_df=2,
                max_features=150_000,
                sublinear_tf=True,
            ),
        ),
        ("classifier", classifier),
    ])


def get_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def main():
    # ============================================================
    # 统一划分：不再自行 train_test_split
    # 唯一合法划分来自 outputs/predictions/{train,val,test}.csv，
    # 与 train_bert.py 完全一致。
    # ============================================================
    assert_valid()

    train = pd.read_csv(path_of("train"), encoding="utf-8-sig")
    val = pd.read_csv(path_of("val"), encoding="utf-8-sig")
    test = pd.read_csv(path_of("test"), encoding="utf-8-sig")

    for part in (train, val, test):
        part["review"] = part["review"].astype(str)
        part["label"] = part["label"].astype(int)
        part["review_norm"] = (
            part["review"].str.replace(r"\s+", " ", regex=True).str.strip()
        )

    # 与 train_bert.py 保持一致：训练集剔除与 val/test 重复的文本，再内部去重
    protected = set(val["review_norm"]) | set(test["review_norm"])
    before = len(train)
    train = train.loc[~train["review_norm"].isin(protected)]
    leak_removed = before - len(train)
    train = train.drop_duplicates(
        subset="review_norm", keep="first"
    ).reset_index(drop=True)

    print("统一划分（outputs/predictions）：")
    print(f"训练集: {len(train)} 条（原始 {before}，剔除跨集合重复 {leak_removed}）")
    print(f"验证集: {len(val)} 条，正向比例={val['label'].mean():.4f}")
    print(f"测试集: {len(test)} 条，正向比例={test['label'].mean():.4f}")
    print("sample_id 沿用原始语料 ID，未重新编号。")

    X_train = train["review"]
    y_train = train["label"]
    X_val = val["review"]
    y_val = val["label"]
    X_test = test["review"]
    y_test = test["label"]

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "naive_bayes": MultinomialNB(alpha=0.5),
        "linear_svm": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    validation_scores = {}

    # 第一轮：训练并用验证集比较模型
    for name, classifier in models.items():
        print(f"\n========== 训练 {name} ==========")

        model = make_pipeline(classifier)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        val_pred = model.predict(X_val)
        val_metrics = get_metrics(y_val, val_pred)
        validation_scores[name] = val_metrics["f1_macro"]

        print(f"训练耗时：{train_seconds:.2f} 秒")
        print("验证集指标：")
        print(json.dumps(val_metrics, ensure_ascii=False, indent=2))

        results.append({
            "model": name,
            "train_seconds": round(train_seconds, 3),
            **{
                f"validation_{k}": v
                for k, v in val_metrics.items()
            },
        })

        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    # 按验证集 Macro-F1 选择模型，不根据测试集选型
    best_name = max(validation_scores, key=validation_scores.get)
    print(f"\n验证集最佳模型：{best_name}")

    # 第二轮：统一在测试集上评价三个已经确定的模型
    test_results = []

    for name in models:
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        pred = model.predict(X_test)
        metrics = get_metrics(y_test, pred)

        test_results.append({"model": name, **metrics})

        print(f"\n========== 测试集：{name} ==========")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(classification_report(
            y_test,
            pred,
            labels=[0, 1],
            target_names=["negative", "positive"],
            digits=4,
            zero_division=0,
        ))
        print("混淆矩阵：")
        print(confusion_matrix(y_test, pred, labels=[0, 1]))

        # 保存测试集预测和后续校准所需的分数
        output = test[
            ["sample_id", "cat", "review", "label"]
        ].copy()
        output["prediction"] = pred

        if hasattr(model, "predict_proba"):
            output["prob_negative"] = model.predict_proba(X_test)[:, 0]
            output["prob_positive"] = model.predict_proba(X_test)[:, 1]

        if hasattr(model, "decision_function"):
            output["decision_score"] = model.decision_function(X_test)

        output.to_csv(
            PRED_DIR / f"{name}_test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(results).to_csv(
        METRIC_DIR / "validation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(test_results).to_csv(
        METRIC_DIR / "test_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(METRIC_DIR / "experiment_config.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "split_source": "outputs/predictions/{train,val,test}.csv",
            "split_policy": "统一划分，与 train_bert.py 完全一致；脚本内不再重新划分",
            "split_sizes": {
                "train": len(train), "val": len(val), "test": len(test),
            },
            "split_sha256": {
                s: sha256(path_of(s))[:16] for s in ("train", "val", "test")
            },
            "train_leak_removed": leak_removed,
            "random_state": RANDOM_STATE,
            "selection_metric": "validation_macro_f1",
            "best_model_by_validation": best_name,
            "models": list(models.keys()),
            "features": "jieba + word-level TF-IDF unigram/bigram",
        }, f, ensure_ascii=False, indent=2)

    print("\n第一阶段基线实验完成。")
    print(f"指标目录：{METRIC_DIR}")
    print(f"预测目录：{PRED_DIR}")
    print(f"模型目录：{MODEL_DIR}")


if __name__ == "__main__":
    main()
