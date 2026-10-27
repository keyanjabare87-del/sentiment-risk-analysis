from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ERROR_PATH = (
    ROOT / "outputs" / "error_analysis"
    / "misclassified_reviews.csv"
)
OUT_DIR = ROOT / "outputs" / "error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "label_audit_template.csv"

if not ERROR_PATH.exists():
    raise FileNotFoundError(
        f"找不到误判样本文件：{ERROR_PATH}\n"
        "请先运行 src/analyze_errors.py。"
    )

df = pd.read_csv(ERROR_PATH, encoding="utf-8-sig")

required = {
    "cat", "review", "label", "prediction",
    "decision_score", "abs_margin", "error_type"
}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"误判文件缺少字段：{missing}")

# 只审计高绝对决策分数的误判样本。
# 这些样本更适合调查：模型为何会强烈倾向错误类别？

df["abs_margin"] = pd.to_numeric(df["abs_margin"], errors="coerce").copy()

# 每个商品类别、每种误判方向最多抽取 10 条。
# 如果某组不足 10 条，则全部保留。

parts = []

for _,group in df.groupby(["cat","error_type"],dropna=False):
    selected = group.nlargest(10, "abs_margin")
    parts.append(selected)

if not parts:
    print("没有可供审计的误判样本")
    raise SystemExit(0)

audit = pd.concat(parts, ignore_index=True)
audit = audit.drop_duplicates(
    subset=["cat","review","label","prediction"]
)

# 添加人工审计字段
audit["人工判断"] = ""
audit["建议处理"] = ""
audit["审计原因"] = ""

# 将模型预测与原标签都保留，方便人工核对
columns = [
    "cat",
    "review",
    "label",
    "prediction",
    "decision_score",
    "abs_margin",
    "error_type",
    "人工判断",
    "建议处理",
    "审计原因",
]

audit = audit[columns].sort_values(
    ["cat", "abs_margin"],
    ascending=[True, False],
)

audit.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("=" * 60)
print("标签质量审计模板已生成")
print("=" * 60)
print("审计样本数：", len(audit))
print("保存位置：", OUT_PATH)
print("\n人工判断建议填写：")
print("1. 标签合理")
print("2. 疑似标签错误")
print("3. 语义模糊或无法判断")
print("\n建议处理填写：")
print("1. 保留原标签")
print("2. 待核实原始数据")
print("3. 标记为不确定")
print("\n不要仅凭模型预测直接修改真实标签。")