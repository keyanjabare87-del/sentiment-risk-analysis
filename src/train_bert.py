
from pathlib import Path
import csv
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm
import argparse


# =========================
# 0. 命令行参数（不传则与原脚本行为完全一致）
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--label-fixes", default="",
                    help="修正标签清单 CSV（sample_id, split, corrected_label）")
parser.add_argument("--run-tag", default="",
                    help="实验标签；给定时输出到 outputs/bert/runs/<tag>/")
parser.add_argument("--allow-split", nargs="*", default=["train"],
                    help="允许被清单修改的 split，默认只有 train")
parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
parser.add_argument("--augment-file", default="",
                    help="过采样清单 CSV（sample_id, split, repeat），实验 C 使用")
args = parser.parse_args()


# =========================
# 1. 配置
# =========================
ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"
OUT_DIR = ROOT / "outputs" / "bert"
if args.run_tag:
    OUT_DIR = OUT_DIR / "runs" / args.run_tag
MODEL_DIR = OUT_DIR / "best_model"

# 统一划分校验：任何训练开始前先确认划分唯一、无跨集合泄漏
sys.path.insert(0, str(ROOT / "src"))
from data_split import assert_valid  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 192
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 2
SEED = 42 if args.seed is None else args.seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("=" * 65)
print("中文 BERT 情感分类")
print("=" * 65)
print("设备：", DEVICE)
if torch.cuda.is_available():
    print("GPU：", torch.cuda.get_device_name(0))
print("模型：", MODEL_NAME)


# =========================
# 2. 读取数据
# =========================
def load_data(filename):
    path = PRED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"review", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{filename} 缺少必要字段：{required}")

    df = df.copy()
    df["review"] = df["review"].fillna("").astype(str)
    df["review_norm"] = (
        df["review"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df[df["review_norm"].ne("")].copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{filename} 标签必须是 0/1")

    # 保留原始标签，便于清洗前后在同一进程内对照
    df["label_orig"] = df["label"]

    return df.reset_index(drop=True)


# ---------- 标签修正（可选） ----------
def load_label_fixes(path):
    """读取修正清单，返回 {split: {sample_id: new_label}}"""
    if not path:
        return {}
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"找不到修正清单：{p}")
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        sid = str(r["sample_id"]).strip()
        sp = str(r["split"]).strip()
        new = int(float(r["corrected_label"]))
        if new not in (0, 1):
            raise ValueError(f"{sid} corrected_label 非法：{new}")
        if sp not in args.allow_split:
            raise ValueError(
                f"{sid} 属于 {sp}，但 --allow-split={args.allow_split} 未放行。"
                f"修改测试集标签会改变评测口径，必须显式放行。"
            )
        out.setdefault(sp, {})[sid] = new
    return out


def load_augment(path):
    """读取过采样清单，返回 {split: {sample_id: repeat}}"""
    if not path:
        return {}
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"找不到过采样清单：{p}")
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        sp = str(r["split"]).strip()
        if sp != "train":
            raise ValueError(f"过采样只允许作用于 train，发现 split={sp}")
        out.setdefault(sp, {})[str(r["sample_id"]).strip()] = int(float(r["repeat"]))
    return out


def sid_key(df):
    """把 sample_id 统一成字符串键，避免 pandas 把纯数字 ID 读成 int 导致匹配失败"""
    return df["sample_id"].astype(str).str.strip()


def build_augment(df, split_name, augment):
    """按 sample_id 复制训练行，返回 (命中数, 待追加的 DataFrame)"""
    m = augment.get(split_name)
    if not m:
        return 0, None
    hit = sid_key(df).isin(m.keys())
    keys = sid_key(df)
    extra = []
    for sid, k in m.items():
        rows = df[keys == sid]
        if len(rows) and k > 1:
            extra.append(rows.loc[rows.index.repeat(k - 1)])
    dup = pd.concat(extra, ignore_index=True) if extra else None
    return int(hit.sum()), dup


def apply_label_fixes(df, split_name, fixes):
    """按 sample_id 精确替换标签；未命中即跳过，绝不影响其他样本"""
    m = fixes.get(split_name)
    if not m:
        return 0, 0
    if "sample_id" not in df.columns:
        print(f"[警告] {split_name} 缺少 sample_id 列，跳过标签修正")
        return 0, 0
    keys = sid_key(df)
    hit = keys.isin(m.keys())
    changed = 0
    for idx in df.index[hit]:
        sid = str(df.at[idx, "sample_id"]).strip()
        new = m[sid]
        if int(df.at[idx, "label"]) != new:
            df.at[idx, "label"] = new
            changed += 1
    return int(hit.sum()), changed


LABEL_FIXES = load_label_fixes(args.label_fixes)

assert_valid()
train_df = load_data("train.csv")
val_df = load_data("val.csv")
test_df = load_data("test.csv")


print("\n原始数据规模：")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# =========================
# 3. 检查并处理跨集合重复评论
# =========================
# 保留验证集和测试集不动。
# 如果训练集评论与验证集/测试集重复，则只从训练集中剔除，
# 避免把验证集或测试集信息泄漏到训练阶段。

val_texts = set(val_df["review_norm"])
test_texts = set(test_df["review_norm"])

val_test_overlap = val_texts & test_texts
if val_test_overlap:
    raise ValueError(
        f"验证集与测试集存在 {len(val_test_overlap)} 条重复评论。"
        "请先检查数据划分，不能自动修改测试集。"
    )

protected_texts = val_texts | test_texts
leak_mask = train_df["review_norm"].isin(protected_texts)
leak_count = int(leak_mask.sum())

if leak_count:
    print(f"\n发现训练集与验证/测试集重复 {leak_count} 行。")
    print("将只从训练集中剔除这些行，验证集和测试集保持不变。")
    train_df = train_df.loc[~leak_mask].reset_index(drop=True)

# 删除训练集内部的重复评论，保留第一条。
before = len(train_df)
train_df = train_df.drop_duplicates(
    subset="review_norm", keep="first"
).reset_index(drop=True)
internal_duplicates = before - len(train_df)

print("训练集跨集合重复剔除：", leak_count)
print("训练集内部重复剔除：", internal_duplicates)
print("最终 Train:", len(train_df))
print("最终 Validation:", len(val_df))
print("最终 Test:", len(test_df))

# ---------- 标签修正放在去重之后：统计的是真正进入训练/评测的行 ----------
fix_report = {}
for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
    hit, changed = apply_label_fixes(df, name, LABEL_FIXES)
    fix_report[name] = {"matched": hit, "changed": changed}
    if hit:
        print(f"标签修正[{name}]：命中 {hit} 条，实际翻转 {changed} 条")
if args.label_fixes:
    print("标签修正清单：", args.label_fixes)
    print("修正统计：", fix_report)
    if sum(v["matched"] for v in fix_report.values()) == 0:
        raise SystemExit(
            "错误：指定了 --label-fixes 但一条都没命中，训练已中止。"
            "请检查 sample_id 类型、split 归属与清单路径。")

# ---------- 过采样（实验 C）：只作用于训练集 ----------
AUGMENT = load_augment(args.augment_file)
aug_report = {}
if AUGMENT:
    matched, dup = build_augment(train_df, "train", AUGMENT)
    added = 0 if dup is None else len(dup)
    if dup is not None:
        train_df = pd.concat([train_df, dup], ignore_index=True)
    aug_report = {"requested": len(AUGMENT.get("train", {})),
                  "matched": matched, "added_rows": added,
                  "train_size_after": len(train_df)}
    print(f"过采样：清单 {aug_report['requested']} 条，命中 {matched} 条，"
          f"新增 {added} 行，训练集变为 {len(train_df)} 行")
    print("过采样清单：", args.augment_file)


# =========================
# 4. Tokenizer 与 Dataset
# =========================
print("\n加载 Tokenizer 和预训练模型……")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class ReviewDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["review"].tolist()
        self.labels = dataframe["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(
            self.labels[idx], dtype=torch.long
        )
        return item


train_dataset = ReviewDataset(train_df)
val_dataset = ReviewDataset(val_df)
test_dataset = ReviewDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# =========================
# 5. 初始化模型
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)
model.to(DEVICE)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.1)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


# =========================
# 6. 评估函数
# =========================
def evaluate(model, loader, with_loss=False):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="评估", leave=False):
            labels = batch["labels"].to(DEVICE)
            inputs = {
                key: value.to(DEVICE)
                for key, value in batch.items()
                if key != "labels"
            }

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(**inputs, labels=labels)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            batch_count = labels.size(0)
            total_loss += outputs.loss.item() * batch_count
            total_count += batch_count

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_precision": precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
    }

    if with_loss:
        metrics["loss"] = total_loss / max(total_count, 1)

    return metrics, all_labels, all_preds, np.array(all_probs)


# =========================
# 7. 训练与验证
# =========================
history = []
best_val_f1 = -1.0
best_epoch = 0
bad_epochs = 0

print("\n开始训练……")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{EPOCHS}",
    )

    for batch in progress:
        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / max(len(train_loader), 1)
    val_metrics, _, _, _ = evaluate(
        model, val_loader, with_loss=True
    )

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }
    history.append(record)

    print(f"\nEpoch {epoch}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_metrics['loss']:.4f}")
    print(f"Val Acc:    {val_metrics['accuracy']:.4f}")
    print(f"Val Macro-F1: {val_metrics['macro_f1']:.4f}")

    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        best_epoch = epoch
        bad_epochs = 0

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        print("已保存当前最佳模型。")
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print("验证集 Macro-F1 连续未提升，提前停止。")
            break

pd.DataFrame(history).to_csv(
    OUT_DIR / "training_history.csv",
    index=False,
    encoding="utf-8-sig",
)

print(f"\n最佳 Epoch：{best_epoch}")
print(f"最佳验证集 Macro-F1：{best_val_f1:.4f}")


# =========================
# 8. 加载最佳模型，最终评估测试集
# =========================
print("\n加载最佳模型，进行最终测试……")

best_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
).to(DEVICE)

test_metrics, y_true, y_pred, y_probs = evaluate(
    best_model, test_loader
)

# ---------- 同一进程内同时给出「原标签口径」与「修正后口径」 ----------
# test_loader shuffle=False，顺序与 test_df 一致，可直接对齐
y_true_orig = test_df["label_orig"].astype(int).tolist()
n_flip_test = sum(
    1 for a, b in zip(y_true_orig, y_true) if int(a) != int(b)
)

test_metrics_orig = None
if n_flip_test:
    test_metrics_orig = {
        "accuracy": accuracy_score(y_true_orig, y_pred),
        "macro_precision": precision_score(
            y_true_orig, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(
            y_true_orig, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(
            y_true_orig, y_pred, average="macro", zero_division=0),
    }
    print(f"\n测试集标签被修正 {n_flip_test} 条，以下同时报告两种口径。")
    print("===== 原标签口径（与历史结果可比）=====")
    for k, v in test_metrics_orig.items():
        print(f"{k}: {v:.4f}")
    print("===== 修正后口径 =====")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

print("\n===== BERT 测试集结果 =====")
for key, value in test_metrics.items():
    print(f"{key}: {value:.4f}")

print("\n===== 分类报告 =====")
report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["负面(0)", "正面(1)"],
    digits=4,
    zero_division=0,
)
print(report)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("混淆矩阵（行是真实标签，列是预测标签）：")
print(cm)

result = {
    "model": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "seed": SEED,
    "best_epoch": best_epoch,
    "best_val_macro_f1": float(best_val_f1),
    "device": str(DEVICE),
    "run_tag": args.run_tag,
    "label_fixes_file": args.label_fixes,
    "label_fix_report": fix_report,
    "augment_file": args.augment_file,
    "augment_report": aug_report,
    "test_label_flips": n_flip_test,
    "train_size_after_dedup": len(train_df),
    "validation_size": len(val_df),
    "test_size": len(test_df),
    "train_test_overlap_removed": leak_count,
    **{f"test_{k}": float(v) for k, v in test_metrics.items()},
}
if test_metrics_orig is not None:
    result.update(
        {f"test_orig_{k}": float(v) for k, v in test_metrics_orig.items()}
    )

# 清洗前后对照表（同一次运行内）
if test_metrics_orig is not None:
    with open(OUT_DIR / "metric_compare.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "原标签口径", "修正后口径", "差值"])
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            w.writerow([
                k,
                f"{test_metrics_orig[k]:.4f}",
                f"{test_metrics[k]:.4f}",
                f"{test_metrics[k] - test_metrics_orig[k]:+.4f}",
            ])

with open(
    OUT_DIR / "test_metrics.json", "w", encoding="utf-8"
) as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

with open(
    OUT_DIR / "classification_report.txt",
    "w", encoding="utf-8"
) as f:
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(str(cm))
    f.write("\n")

predictions = test_df.copy()
predictions["prediction"] = y_pred
predictions["prob_negative"] = y_probs[:, 0]
predictions["prob_positive"] = y_probs[:, 1]
predictions["correct"] = (
    predictions["label"] == predictions["prediction"]
)

predictions.drop(
    columns=["review_norm"], errors="ignore"
).to_csv(
    OUT_DIR / "test_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(
    cm,
    index=["真实负面", "真实正面"],
    columns=["预测负面", "预测正面"],
).to_csv(
    OUT_DIR / "confusion_matrix.csv",
    encoding="utf-8-sig",
)

print("\n输出目录：", OUT_DIR)
print("训练耗时（分钟）：", round((time.time() - start_time) / 60, 2))
print("训练与评估完成。")
