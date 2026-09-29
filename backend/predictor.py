"""Rebuilds every trained model from artifacts/model.json and answers mix-design questions."""
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config
from .classification import sigmoid
from .data import Standardizer, engineer
from .regression import rbf_kernel

REG_KEYS = ["ols", "ridge", "huber", "blr", "krr", "kglm"]
CLS_KEYS = ["lsq", "fisher", "generative", "logistic", "bayes_logistic", "svm"]


class Predictor:
    def __init__(self, path=config.ARTIFACTS / "model.json"):
        d = json.loads(path.read_text())
        self.lin = Standardizer.from_dict(d["standardizers"]["linear"])
        self.ker = Standardizer.from_dict(d["standardizers"]["kernel"])
        self.reg = {k: {kk: (np.array(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}
                    for k, v in d["regression"].items()}
        self.cls = {g: {k: {kk: (np.array(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in m.items()}
                    for g, m in d["classification"].items()}
        self.hyper = d["hyper"]

    # ------------------------------------------------------------ features
    def frames(self, mixes: list[dict]):
        df = engineer(pd.DataFrame(mixes)[config.RAW])
        return df, self.lin.transform(df), self.ker.transform(df)

    # ---------------------------------------------------------- regression
    def strength(self, X, K):
        r = self.reg
        out = {
            "ols": (X - r["ols"]["x_mean"]) @ r["ols"]["w"] + r["ols"]["t_mean"],
            "ridge": (X - r["ridge"]["x_mean"]) @ r["ridge"]["w"] + r["ridge"]["t_mean"],
            "huber": X @ r["huber"]["w"] + r["huber"]["b"] + r["huber"]["t_mean"],
            "blr": (X - r["blr"]["x_mean"]) @ r["blr"]["m"] + r["blr"]["t_mean"],
            "krr": rbf_kernel(K, r["krr"]["X"], r["krr"]["gamma"]) @ r["krr"]["a"] + r["krr"]["t_mean"],
        }
        Xb = X - r["blr"]["x_mean"]
        out["blr_sd"] = np.sqrt(1 / r["blr"]["beta"] + np.einsum("ij,jk,ik->i", Xb, r["blr"]["S"], Xb))
        kg = r["kglm"]
        phi = rbf_kernel(K, kg["centers"], kg["gamma"]) - kg["x_mean"]
        out["kglm"] = phi @ kg["m"] + kg["t_mean"]
        out["kglm_sd"] = np.sqrt(1 / kg["beta"] + np.einsum("ij,jk,ik->i", phi, kg["S"], phi))
        return out

    # ------------------------------------------------------ classification
    def pass_probability(self, grade, X, K):
        m = self.cls[grade]
        Xb = np.column_stack([np.ones(len(X)), X])
        f = m["fisher"]
        z = X @ f["w"]
        mu, var, pr = f["proj"]["mu"], f["proj"]["var"], f["proj"]["prior"]
        lp = [math.log(pr[k]) - 0.5 * np.log(2 * np.pi * var[k]) - (z - mu[k]) ** 2 / (2 * var[k]) for k in (0, 1)]
        bl = m["bayes_logistic"]
        mu_a = Xb @ bl["w"]
        var_a = np.einsum("ij,jk,ik->i", Xb, bl["S"], Xb)
        sv = m["svm"]
        margin = rbf_kernel(K, sv["sv_X"], sv["gamma"]) @ sv["sv_coef"] - sv["rho"]
        pw = m["platt"]["w"]
        lsq = Xb @ m["lsq"]["w"]
        return {
            "lsq": {"score": lsq, "prob": None, "pass": lsq >= 0},
            "fisher": {"score": z, "prob": sigmoid(lp[1] - lp[0])},
            "generative": {"score": X @ m["generative"]["w"] + m["generative"]["w0"],
                           "prob": sigmoid(X @ m["generative"]["w"] + m["generative"]["w0"])},
            "logistic": {"score": Xb @ m["logistic"]["w"], "prob": sigmoid(Xb @ m["logistic"]["w"])},
            "bayes_logistic": {"score": mu_a, "prob": sigmoid(mu_a / np.sqrt(1 + math.pi * var_a / 8)), "sd": np.sqrt(var_a)},
            "svm": {"score": margin, "prob": sigmoid(pw[0] + pw[1] * margin)},
        }

    # ------------------------------------------------------------- answer
    def analyse(self, mix: dict, grade: str, exposure: str, ranges: dict | None = None) -> dict:
        g = config.GRADE_BY_NAME[grade]
        df, X, K = self.frames([mix])
        s = self.strength(X, K)
        cls = self.pass_probability(grade, X, K)
        mean, sd = float(s["kglm"][0]), float(s["kglm_sd"][0])
        votes = {}
        for k in CLS_KEYS:
            c = cls[k]
            p = None if c["prob"] is None else float(c["prob"][0])
            passed = bool(c["score"][0] >= 0) if k in ("lsq", "svm") else (p >= 0.5)
            votes[k] = {"prob": p, "pass": passed, "score": float(c["score"][0])}
        p_kglm = float(1 - norm.cdf((g["target"] - mean) / sd))
        votes["kglm"] = {"prob": p_kglm, "pass": p_kglm >= 0.5, "score": mean - g["target"]}

        # strength-gain curve for the same mix at every age
        ages = np.unique(np.round(np.geomspace(1, 365, 48)).astype(int))
        curve_mix = [{**mix, "age": int(a)} for a in ages]
        _, Xc, Kc = self.frames(curve_mix)
        sc = self.strength(Xc, Kc)
        reach = next((int(a) for a, m_ in zip(ages, sc["kglm"]) if m_ >= g["target"]), None)

        row = df.iloc[0]
        binder = float(mix["cement"] + mix["slag"] + mix["flyash"])
        ex = next(e for e in config.EXPOSURE if e["exposure"] == exposure)
        grade_ok = int(grade[1:]) >= int(ex["min_grade"][1:])
        checks = [
            {"rule": f"Binder content ≥ {ex['min_cement']} kg/m³", "value": f"{binder:.0f} kg/m³", "ok": binder >= ex["min_cement"]},
            {"rule": f"Water/binder ≤ {ex['max_wc']:.2f}", "value": f"{row['wb']:.2f}", "ok": float(row["wb"]) <= ex["max_wc"] + 1e-9},
            {"rule": f"Grade ≥ {ex['min_grade']}", "value": grade, "ok": grade_ok},
        ]
        density = float(sum(mix[k] for k in ("cement", "slag", "flyash", "water", "sp", "coarse", "fine")))
        co2 = float(sum(mix[k] * f for k, f in config.CO2_FACTORS.items()))
        ranges = ranges or {}
        outside = [config.LABEL[k] for k in config.RAW
                   if k in ranges and not (ranges[k][0] <= mix[k] <= ranges[k][1])]
        return {
            "grade": g, "exposure": ex,
            "strength": {k: float(s[k][0]) for k in REG_KEYS},
            "mean": mean, "sd": sd, "interval": [mean - 1.96 * sd, mean + 1.96 * sd],
            "blr_sd": float(s["blr_sd"][0]),
            "votes": votes, "n_pass_votes": int(sum(v["pass"] for v in votes.values())),
            "p_pass": p_kglm,
            "curve": {"age": ages.tolist(), "mean": sc["kglm"].tolist(), "sd": sc["kglm_sd"].tolist(),
                      "krr": sc["krr"].tolist(), "ols": sc["ols"].tolist()},
            "reaches_target_at": reach,
            "derived": {"wc": float(row["wc"]), "wb": float(row["wb"]), "binder": binder, "density": density,
                        "co2": co2, "cement_share": float(mix["cement"] / binder)},
            "durability": checks, "durability_ok": all(c["ok"] for c in checks),
            "extrapolating": outside,
        }
