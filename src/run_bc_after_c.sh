#!/bin/bash
# 等待实验 C 完成后，自动接着跑 B+C，最后重生成对比表。
# 用法：bash src/run_bc_after_c.sh
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/d/anaconda3/envs/ml311/python.exe
C_DONE=outputs/bert/runs/C_contrastive/test_metrics.json
LOG=outputs/experiments/stage3/logs/BC_clean_contrastive.log

echo "[chain] 等待 C 完成：$(date '+%H:%M:%S')"
for i in $(seq 1 180); do
    if [ -f "$C_DONE" ]; then
        echo "[chain] 检测到 C 完成：$(date '+%H:%M:%S')"
        break
    fi
    sleep 60
done

if [ ! -f "$C_DONE" ]; then
    echo "[chain] 超时 180 分钟仍未等到 C，放弃"
    exit 1
fi

# 防止 C 刚写完 json 但预测文件尚未落盘
sleep 20

echo "[chain] 启动 BC：$(date '+%H:%M:%S')"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" src/train_bert.py \
    --run-tag BC_clean_contrastive \
    --label-fixes outputs/experiments/stage3/label_fixes_train.csv \
    --allow-split train \
    --augment-file outputs/experiments/stage3/contrastive_oversample.csv \
    > "$LOG" 2>&1
echo "[chain] BC 退出码 $?：$(date '+%H:%M:%S')"

echo "[chain] 重生成对比表"
"$PY" src/stage3_report.py
echo "[chain] 全部完成：$(date '+%H:%M:%S')"
