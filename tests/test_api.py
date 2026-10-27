# -*- coding: utf-8 -*-
"""
阶段 5：API 测试。

重点验证两件事：
1. 接口契约（状态码、字段、错误提示）
2. **API 输出与直接调用模型的结果一致**（防止封装层引入偏差）
"""
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


POS_REVIEW = "包装很仔细，衣服质量很好，穿着很舒服，非常满意，下次还会再买"
NEG_REVIEW = "物流太慢，客服态度差，衣服还有破损，非常失望"


# ---------- 健康检查 ----------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["max_length"] > 0
    assert d["device"] in {"cuda", "cpu"}


# ---------- 单条预测 ----------
def test_predict_schema(client):
    r = client.post("/predict", json={"review": POS_REVIEW})
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"review", "label", "sentiment", "probability",
                      "prob_negative", "prob_positive", "latency_ms"}
    assert d["review"] == POS_REVIEW
    assert d["label"] in (0, 1)
    assert d["sentiment"] == ("negative" if d["label"] == 0 else "positive")
    assert 0.0 <= d["probability"] <= 1.0
    assert abs(d["prob_negative"] + d["prob_positive"] - 1.0) < 1e-3


def test_predict_matches_direct_model(client):
    """API 结果必须与直接调用 predict_texts 完全一致"""
    for text in (POS_REVIEW, NEG_REVIEW):
        api_res = client.post("/predict", json={"review": text}).json()
        direct = api.predict_texts([text])[0]
        assert api_res["label"] == direct["label"]
        assert api_res["sentiment"] == direct["sentiment"]
        assert api_res["prob_negative"] == pytest.approx(direct["prob_negative"], abs=1e-6)


def test_predict_extremes(client):
    """极性明显的样本不应被反判（模型在测试集上 acc≈0.95，这两条属易分样本）"""
    pos = client.post("/predict", json={"review": POS_REVIEW}).json()
    neg = client.post("/predict", json={"review": NEG_REVIEW}).json()
    assert pos["label"] == 1, f"正面样本被判为负面：{pos}"
    assert neg["label"] == 0, f"负面样本被判为正面：{neg}"


# ---------- 入参校验 ----------
@pytest.mark.parametrize("payload", [
    {"review": ""},
    {"review": "   \n\t "},
    {},
])
def test_predict_reject_invalid(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_reject_too_long(client):
    r = client.post("/predict", json={"review": "好" * (api.MAX_CHARS + 1)})
    assert r.status_code == 422
    assert "超过上限" in str(r.json())


# ---------- 批量预测 ----------
def test_batch_equals_single(client):
    texts = [POS_REVIEW, NEG_REVIEW, "还可以，就是有点小贵"]
    batch = client.post("/predict/batch", json={"reviews": texts}).json()
    assert batch["count"] == len(texts)
    assert len(batch["results"]) == len(texts)
    for t, got in zip(texts, batch["results"]):
        single = client.post("/predict", json={"review": t}).json()
        assert got["label"] == single["label"]
        assert got["prob_negative"] == pytest.approx(single["prob_negative"], abs=1e-6)


def test_batch_matches_direct_model(client):
    texts = [POS_REVIEW, NEG_REVIEW] * 5
    batch = client.post("/predict/batch", json={"reviews": texts}).json()
    direct = api.predict_texts(texts)
    assert [r["label"] for r in batch["results"]] == [d["label"] for d in direct]


@pytest.mark.parametrize("payload", [
    {"reviews": []},
    {"reviews": ["正常", ""]},
    {"reviews": ["正常"] * (api.MAX_BATCH + 1)},
])
def test_batch_reject_invalid(client, payload):
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 422


# ---------- 与测试集真实标签的一致性 ----------
def test_consistency_on_test_split(client):
    """抽 20 条测试集样本，API 标签应与离线预测文件一致"""
    import csv
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    preds = root / "outputs" / "bert" / "test_predictions.csv"
    if not preds.exists():
        pytest.skip("缺少离线预测文件")
    with open(preds, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))[:20]
    batch = client.post("/predict/batch",
                        json={"reviews": [r["review"] for r in rows]}).json()
    api_labels = [x["label"] for x in batch["results"]]
    offline = [int(r["prediction"]) for r in rows]
    agree = sum(a == b for a, b in zip(api_labels, offline))
    # 允许极小差异（同一权重、同一 tokenizer，理论上应为 20/20）
    assert agree >= 19, f"在线与离线预测不一致：{agree}/20"
