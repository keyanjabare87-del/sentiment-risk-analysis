# -*- coding: utf-8 -*-
"""
阶段 6：生成报告用图表。

产出 reports/figures/*.png
"""
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 中文字体
for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

C_BASE, C_ALT, C_BAD = "#4C78A8", "#F58518", "#B0B0B0"


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------- 图1：实验对比 ----------
comp = [r for r in load(ROOT / "outputs/experiments/stage3/experiment_comparison.csv")
        if r["status"] == "已完成"]
names = [f"{r['experiment_id']}\n{r['experiment']}" for r in comp]
mf1 = [float(r["macro_f1"]) for r in comp]
nrec = [float(r["negative_recall"]) for r in comp]

fig, ax = plt.subplots(figsize=(9, 4.6))
x = range(len(comp))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], mf1, w, label="Macro-F1", color=C_BASE)
b2 = ax.bar([i + w / 2 for i in x], nrec, w, label="负面 Recall", color=C_ALT)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                f"{b.get_height():.3f}", ha="center", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0.42, 1.04)
ax.set_ylabel("得分")
ax.set_title("阶段 3 实验对比（测试集原标签口径，n=9406）")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig1_experiments.png", dpi=150)
plt.close(fig)

# ---------- 图2：噪声率 vs 提升（检验「高噪声类目更受益」假设） ----------
aud = load(ROOT / "outputs/bert/category_analysis/manual_audit_all_502_error_cases.csv")
cm = load(ROOT / "outputs/bert/category_analysis/category_metrics_and_counts.csv")
n_by_cat = {r["category"]: int(r["sample_count"]) for r in cm}
le = Counter(r["category"] for r in aud if r["audit_verdict"] == "标签错")
catm = load(ROOT / "outputs/experiments/stage3/category_macro_f1.csv")


def mf1_of(eid, cat):
    for r in catm:
        if r["experiment_id"] == eid and r["category"] == cat:
            return float(r["macro_f1"])
    return None


xs, ys, labels = [], [], []
for c, n in n_by_cat.items():
    xs.append(le[c] / n * 100)
    ys.append((mf1_of("BC", c) - mf1_of("A", c)) * 100)
    labels.append(c)

fig, ax = plt.subplots(figsize=(7.2, 5))
ax.axhline(0, color="#888", lw=1, ls="--")
ax.scatter(xs, ys, s=90, color=C_BASE, zorder=3)
for x_, y_, t in zip(xs, ys, labels):
    ax.annotate(t, (x_, y_), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.set_xlabel("该类别标签错误率（人工审计，%）")
ax.set_ylabel("BC 相对 A 的 Macro-F1 变化（百分点）")
ax.set_title("标签错误率与清洗增益：相关性很弱（r = +0.28, n = 10）")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig2_noise_vs_gain.png", dpi=150)
plt.close(fig)

# ---------- 图3：混淆矩阵 A vs BC ----------
def cm_of(path):
    d = load(path)
    m = Counter()
    for r in d:
        y, p = int(r["label"]), int(r["prediction"])
        m[(y, p)] += 1
    return [[m[(0, 0)], m[(0, 1)]], [m[(1, 0)], m[(1, 1)]]]


mats = [("A 原始 BERT", cm_of(ROOT / "outputs/bert/test_predictions.csv")),
        ("BC 清洗+过采样", cm_of(ROOT / "outputs/bert/runs/BC_clean_contrastive/test_predictions.csv"))]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
for ax, (title, m) in zip(axes, mats):
    im = ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, m[i][j], ha="center", va="center",
                    color="white" if m[i][j] > 3000 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["预测负面", "预测正面"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["真实负面", "真实正面"])
    ax.set_title(title, fontsize=11)
fig.suptitle("混淆矩阵对比", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "fig3_confusion.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- 图4：推理性能 ----------
bench = load(ROOT / "reports/inference_benchmark.csv")
fig, ax = plt.subplots(figsize=(7, 4.3))
for dev, color in (("cuda", C_BASE), ("cpu", C_ALT)):
    sub = [r for r in bench if r["device"] == dev]
    ax.plot([int(r["batch_size"]) for r in sub],
            [float(r["ms_per_sample"]) for r in sub],
            marker="o", label=dev.upper(), color=color, lw=2)
    for r in sub:
        ax.annotate(f'{float(r["ms_per_sample"]):.1f}ms',
                    (int(r["batch_size"]), float(r["ms_per_sample"])),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 8, 32]); ax.set_xticklabels(["1", "8", "32"])
ax.set_xlabel("batch size")
ax.set_ylabel("每条推理耗时 (ms)")
ax.set_title("推理耗时（max_length=192，RTX 3060 Laptop）")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig4_latency.png", dpi=150)
plt.close(fig)

print("图表已生成：")
for p in sorted(FIG.glob("*.png")):
    print("  ", p.name, f"{p.stat().st_size/1024:.0f} KB")
