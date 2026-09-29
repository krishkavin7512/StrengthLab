"""FastAPI backend for StrengthLab. Start with `python run.py`; API docs at http://127.0.0.1:8002/docs"""
import json
import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .data import KERNEL_FEATURES, LINEAR_FEATURES, Standardizer, clean, engineer, load_raw, preview, split
from .predictor import Predictor
from .regression import KernelRidge, LeastSquares, r2, rmse

app = FastAPI(title="StrengthLab API", version="1.0",
              description="Concrete compressive strength and IS-grade compliance with from-scratch linear models, kernels and SVMs.")


def _load(name):
    path = config.ARTIFACTS / name
    if not path.exists():
        raise RuntimeError(f"{path} is missing. Run `python -m backend.train` first.")
    return json.loads(path.read_text())


REPORT = _load("report.json")
GRAPHS = _load("graphs.json")
BOUNDARY = _load("boundary.json")
PRED = Predictor()

# The cleaned data and the same split as training, for the viewer and the live kernel lab.
DF, _ = clean(load_raw())
DF = engineer(DF)
SPLIT = split(DF)
RANGES = {k: (float(DF[k].min()), float(DF[k].max())) for k in config.RAW}
_KS = Standardizer(KERNEL_FEATURES).fit(SPLIT.train)
_LS = Standardizer(LINEAR_FEATURES).fit(SPLIT.train)
KTR, KVA, KTE = (_KS.transform(p) for p in (SPLIT.train, SPLIT.val, SPLIT.test))
XTR = _LS.transform(SPLIT.train)
YTR, YVA, YTE = (p[config.TARGET].to_numpy() for p in (SPLIT.train, SPLIT.val, SPLIT.test))


# --------------------------------------------------------------------------- #
@app.get("/api/overview")
def overview():
    R, C = REPORT["regression"], REPORT["classification"]
    return {
        "data": {k: REPORT["data"][k] for k in ("n_raw", "n_clean", "duplicates", "pass_rates")},
        "sizes": REPORT["sizes"], "regression_test": R["test"], "regression_sklearn": {k: v for k, v in R["sklearn"].items() if k != "krr_sample"},
        "blr": {k: R["blr"][k] for k in ("alpha", "beta", "gamma", "noise_sd")},
        "krr": {k: R["krr"][k] for k in ("gamma", "lambda", "kernels")}, "ridge_lambda": R["ridge"]["lambda"],
        "classification": {g: {"target": v["target"], "test_pass_rate": v["test_pass_rate"], "results": v["results"]}
                           for g, v in C["grades"].items()},
        "classifier_names": C["names"], "svm": C["svm_params"], "laplace_alpha": C["laplace_evidence"]["best_alpha"],
        "classification_sklearn": C["sklearn"], "coverage": R["coverage"],
        "grades": config.GRADES, "exposure": config.EXPOSURE, "ingredients": config.INGREDIENTS,
        "presets": config.PRESETS, "default_mix": config.DEFAULT_MIX, "ranges": RANGES, "trained_at": REPORT["trained_at"],
    }


@app.get("/api/data/preview")
def data_preview(offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=200)):
    return preview(DF[config.RAW + [config.TARGET]], offset, limit)


@app.get("/api/data/summary")
def data_summary():
    return {"summary": REPORT["data"]["summary"]}


@app.get("/api/regression/lab")
def regression_lab():
    R = REPORT["regression"]
    return {"parity": R["parity"], "test": R["test"], "ridge": {"path": R["ridge"]["path"], "lambda": R["ridge"]["lambda"],
            "cv": R["ridge"]["cv"]}, "contamination": R["contamination"], "features": REPORT["features"]["linear"],
            "blr": {"m": R["blr"]["m"], "sd": R["blr"]["sd"]}}


@app.get("/api/kernel")
def kernel_lab(gamma: float = Query(0.03, gt=0, le=20), log_lambda: float = Query(-4.6, ge=-12, le=5)):
    """Refit kernel ridge live on the 704 training mixes and draw its strength-vs-age curve."""
    lam = math.exp(log_lambda)
    m = KernelRidge("rbf", gamma, lam).fit(KTR, YTR)
    ages = np.unique(np.round(np.geomspace(1, 365, 60)).astype(int))
    ref = [{**config.DEFAULT_MIX, "age": int(a)} for a in ages]
    ref_df = engineer(pd.DataFrame(ref)[config.RAW])
    ols = LeastSquares().fit(XTR, YTR)
    return {
        "gamma": gamma, "lambda": lam,
        "train_r2": r2(YTR, m.predict(KTR)), "val_r2": r2(YVA, m.predict(KVA)), "test_r2": r2(YTE, m.predict(KTE)),
        "test_rmse": rmse(YTE, m.predict(KTE)),
        "parity": {"actual": YTE.tolist(), "pred": m.predict(KTE).tolist()},
        "curve": {"age": ages.tolist(), "krr": m.predict(_KS.transform(ref_df)).tolist(),
                  "ols": ols.predict(_LS.transform(ref_df)).tolist()},
        "linear_test_r2": r2(YTE, ols.predict(_LS.transform(SPLIT.test))),
    }


@app.get("/api/boundary")
def boundary():
    return BOUNDARY


class MixRequest(BaseModel):
    mix: dict[str, float] = Field(..., description="Ingredient key -> kg/m3 (age in days)")
    grade: str = Field(config.MAIN_GRADE)
    exposure: str = Field("Moderate")


@app.post("/api/predict")
def predict(body: MixRequest):
    if body.grade not in config.GRADE_BY_NAME:
        raise HTTPException(400, f"grade must be one of {list(config.GRADE_BY_NAME)}")
    if body.exposure not in [e["exposure"] for e in config.EXPOSURE]:
        raise HTTPException(400, "unknown exposure condition")
    mix = dict(config.DEFAULT_MIX)
    for k, v in body.mix.items():
        if k not in config.ING_BY_KEY:
            raise HTTPException(400, f"unknown ingredient {k}")
        i = config.ING_BY_KEY[k]
        if not i["min"] <= v <= i["max"]:
            raise HTTPException(400, f"{k} must be between {i['min']} and {i['max']} {i['unit']}")
        mix[k] = v
    if mix["cement"] <= 0:
        raise HTTPException(400, "cement must be positive")
    return PRED.analyse(mix, body.grade, body.exposure, RANGES)


@app.get("/api/graphs")
def graphs():
    return {"graphs": GRAPHS}


@app.get("/api/graphs/{gid}")
def graph(gid: str):
    for g in GRAPHS:
        if g["id"] == gid:
            return g
    raise HTTPException(404, "no such graph")


# --------------------------------------------------------------------------- #
@app.middleware("http")
async def revalidate_assets(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/assets") or request.url.path in ("/", "/graphs"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/assets", StaticFiles(directory=config.FRONTEND / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.FRONTEND / "index.html")


@app.get("/graphs", include_in_schema=False)
def graphs_page():
    return FileResponse(config.FRONTEND / "graphs.html")
