# -*- coding: utf-8 -*-
"""
阶段 5：推理性能基准（CPU vs GPU，不同 batch size）+ HTTP 端到端延迟。

产出 reports/inference_benchmark.csv
"""
import csv
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "bert" / "best_model"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 192
N_TEXT = 128
BATCHES = [1, 8, 32]
WARMUP = 2


def load_texts(n=N_TEXT):
    with open(ROOT / "outputs" / "bert" / "test_predictions.csv",
              encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))[:n]
    return [r["review"] for r in rows]


def bench(device: str, texts):
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    mdl.eval().to(device)
    rows = []
    for bs in BATCHES:
        chunks = [texts[i:i + bs] for i in range(0, len(texts), bs)]
        # warmup
        for c in chunks[:WARMUP]:
            enc = tok(c, max_length=MAX_LENGTH, truncation=True,
                      padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                mdl(**enc)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for c in chunks:
            enc = tok(c, max_length=MAX_LENGTH, truncation=True,
                      padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                mdl(**enc)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append({
            "device": device,
            "batch_size": bs,
            "n_samples": len(texts),
            "total_seconds": round(dt, 4),
            "ms_per_sample": round(dt / len(texts) * 1000, 3),
            "samples_per_second": round(len(texts) / dt, 2),
        })
        print(f"  {device:<5} bs={bs:<3} {dt:7.3f}s  "
              f"{dt/len(texts)*1000:7.2f} ms/条  {len(texts)/dt:7.2f} 条/秒")
    del mdl
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def http_latency(n=30):
    """HTTP 端到端单条延迟（含序列化与框架开销）"""
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import api

    with TestClient(api.app) as c:
        texts = load_texts(60)[:n]
        for t in texts[:3]:
            c.post("/predict", json={"review": t})
        lat = []
        for t in texts:
            t0 = time.perf_counter()
            r = c.post("/predict", json={"review": t})
            assert r.status_code == 200
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        return {
            "p50_ms": round(statistics.median(lat), 2),
            "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 2),
            "mean_ms": round(statistics.mean(lat), 2),
        }


def main():
    texts = load_texts()
    print(f"基准文本 {len(texts)} 条，max_length={MAX_LENGTH}\n")
    rows = []
    rows += bench("cuda", texts)
    rows += bench("cpu", texts)

    with open(OUT / "inference_benchmark.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nHTTP 端到端（TestClient，单条）")
    h = http_latency()
    print(f"  p50={h['p50_ms']}ms  p95={h['p95_ms']}ms  mean={h['mean_ms']}ms")
    with open(OUT / "http_latency.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["场景", "p50_ms", "p95_ms", "mean_ms"])
        w.writerow(["HTTP /predict 单条", h["p50_ms"], h["p95_ms"], h["mean_ms"]])

    print(f"\n输出：{OUT}")


if __name__ == "__main__":
    main()
