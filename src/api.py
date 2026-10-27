# -*- coding: utf-8 -*-
"""
阶段 5：情感分类推理 API（FastAPI）

启动
----
    uvicorn src.api:app --host 127.0.0.1 --port 8000
    # 或指定模型 / 设备
    MODEL_DIR=outputs/bert/best_model DEVICE=cpu uvicorn src.api:app --port 8000

接口
----
    POST /predict          单条评论
    POST /predict/batch    批量评论（1~256 条）
    GET  /health           健康检查

设计要点
--------
1. 模型在 lifespan 中只加载一次，进程内复用；不会随请求重复加载权重。
2. 端点用同步 def（跑在 FastAPI 线程池），避免阻塞事件循环；
   同时用一把锁串行化前向推理，防止多线程并发抢占 CUDA 上下文。
3. 入参校验：空文本、纯空白、超长文本、批量超限都返回 422 并给出明确原因。
4. 返回的 probability 是「预测类别的概率」，与需求示例一致；
   另外附上 prob_negative / prob_positive 便于下游使用。
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "outputs" / "bert" / "best_model"))
if not MODEL_DIR.is_absolute():
    MODEL_DIR = ROOT / MODEL_DIR
DEVICE_NAME = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 192))
MAX_CHARS = int(os.getenv("MAX_CHARS", 2000))     # 超长文本直接拒绝，避免静默截断
MAX_BATCH = int(os.getenv("MAX_BATCH", 256))

LABEL_NAME = {0: "negative", 1: "positive"}

# 进程级单例
_state: dict = {"model": None, "tokenizer": None, "device": None, "loaded_at": None}
_lock = threading.Lock()          # 串行化 GPU 前向
_start_ts = time.time()


class PredictRequest(BaseModel):
    review: str = Field(..., description="单条评论文本")

    @field_validator("review")
    @classmethod
    def _check(cls, v: str) -> str:
        if v is None or not v.strip():
            raise ValueError("review 不能为空或纯空白")
        if len(v) > MAX_CHARS:
            raise ValueError(f"review 长度 {len(v)} 超过上限 {MAX_CHARS} 字符")
        return v


class BatchRequest(BaseModel):
    reviews: List[str] = Field(..., description=f"评论列表，1~{MAX_BATCH} 条")

    @field_validator("reviews")
    @classmethod
    def _check(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("reviews 不能为空")
        if len(v) > MAX_BATCH:
            raise ValueError(f"批量条数 {len(v)} 超过上限 {MAX_BATCH}")
        for i, s in enumerate(v):
            if s is None or not s.strip():
                raise ValueError(f"第 {i} 条为空或纯空白")
            if len(s) > MAX_CHARS:
                raise ValueError(f"第 {i} 条长度 {len(s)} 超过上限 {MAX_CHARS}")
        return v


class PredictResponse(BaseModel):
    review: str
    label: int
    sentiment: str
    probability: float
    prob_negative: float
    prob_positive: float
    latency_ms: float


class BatchResponse(BaseModel):
    results: List[PredictResponse]
    count: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_dir: str
    device: str
    max_length: int
    uptime_s: float
    cuda_available: bool


def load_model(model_dir: Path = MODEL_DIR, device: str = DEVICE_NAME) -> None:
    """加载模型到进程单例（可被测试直接调用）"""
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在：{model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    mdl.eval()
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    mdl.to(dev)
    _state.update(model=mdl, tokenizer=tok, device=dev,
                  loaded_at=time.time())


def predict_texts(texts: List[str]) -> List[dict]:
    """核心推理函数；测试可直接调用它与 API 输出比对"""
    if _state["model"] is None:
        raise RuntimeError("模型尚未加载")
    mdl, tok, dev = _state["model"], _state["tokenizer"], _state["device"]

    enc = tok(texts, max_length=MAX_LENGTH, truncation=True,
              padding=True, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}

    with _lock, torch.inference_mode():
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu()

    out = []
    for text, p in zip(texts, probs):
        pn, pp = float(p[0]), float(p[1])
        label = 0 if pn >= pp else 1
        out.append({
            "review": text,
            "label": label,
            "sentiment": LABEL_NAME[label],
            "probability": round(pn if label == 0 else pp, 6),
            "prob_negative": round(pn, 6),
            "prob_positive": round(pp, 6),
        })
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    _state.update(model=None, tokenizer=None, device=None)


app = FastAPI(
    title="电商评论情感分类 API",
    description="BERT-base-chinese 二分类：0=负面，1=正面",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _state["model"] is not None else "model_not_loaded",
        model_dir=str(MODEL_DIR),
        device=str(_state["device"]),
        max_length=MAX_LENGTH,
        uptime_s=round(time.time() - _start_ts, 1),
        cuda_available=torch.cuda.is_available(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    t0 = time.perf_counter()
    res = predict_texts([req.review])[0]
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return PredictResponse(**res)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    t0 = time.perf_counter()
    res = predict_texts(req.reviews)
    total = round((time.perf_counter() - t0) * 1000, 2)
    per = round(total / len(res), 2)
    for r in res:
        r["latency_ms"] = per
    return BatchResponse(results=[PredictResponse(**r) for r in res],
                         count=len(res), total_latency_ms=total)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
