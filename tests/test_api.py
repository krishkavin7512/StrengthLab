"""Smoke tests for the HTTP API (needs trained artifacts: run `python -m backend.train`)."""
from fastapi.testclient import TestClient

from backend import config
from backend.api import app

client = TestClient(app)


def test_overview():
    o = client.get("/api/overview").json()
    assert o["data"]["n_clean"] == 1005
    assert o["regression_test"]["krr"]["r2"] > o["regression_test"]["ols"]["r2"] > 0.8
    assert set(o["classification"]) == {g["grade"] for g in config.GRADES}


def test_predict_is_monotone_in_water_and_age():
    base = {**config.DEFAULT_MIX}
    wet = client.post("/api/predict", json={"mix": {**base, "water": 230}}).json()
    dry = client.post("/api/predict", json={"mix": {**base, "water": 150, "sp": 10}}).json()
    old = client.post("/api/predict", json={"mix": {**base, "age": 180}}).json()
    young = client.post("/api/predict", json={"mix": {**base, "age": 3}}).json()
    assert dry["mean"] > wet["mean"]
    assert old["mean"] > young["mean"]
    assert len(dry["curve"]["age"]) == len(dry["curve"]["mean"])
    assert set(dry["votes"]) >= {"lsq", "fisher", "generative", "logistic", "bayes_logistic", "svm", "kglm"}


def test_predict_rejects_bad_input():
    assert client.post("/api/predict", json={"mix": {"cement": 5000}}).status_code == 400
    assert client.post("/api/predict", json={"mix": {}, "grade": "M99"}).status_code == 400


def test_kernel_lab_overfits_with_huge_gamma():
    good = client.get("/api/kernel", params={"gamma": 0.03}).json()
    wild = client.get("/api/kernel", params={"gamma": 15, "log_lambda": -10}).json()
    assert wild["train_r2"] > 0.99 and wild["val_r2"] < good["val_r2"]


def test_pages_and_graphs():
    assert len(client.get("/api/graphs").json()["graphs"]) >= 30
    assert "models" in client.get("/api/boundary").json()
    assert client.get("/").status_code == 200 and client.get("/graphs").status_code == 200
