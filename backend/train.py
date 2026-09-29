"""Training pipeline for StrengthLab.

Regression (strength in MPa): least squares (closed form + gradient descent), ridge, Huber,
Bayesian linear regression, kernel ridge, and a Bayesian kernel GLM.
Classification (does a mix reach the IS 10262 target mean strength for a grade?): least-squares
discriminant, Fisher's LDA, Gaussian generative model, logistic regression, Bayesian logistic
regression (Laplace), and a kernel SVM, for every grade from M20 to M40.

Run with:  python -m backend.train
"""
import json
import math
import sys
import time

import numpy as np
import pandas as pd

from . import config
from .classification import (BayesianLogisticRegression, FisherLDA, GaussianGenerative, KernelSVM,
                             LeastSquaresClassifier, LogisticRegression, PlattScaling)
from .data import (BOUNDARY_FEATURES, KERNEL_FEATURES, LINEAR_FEATURES, Standardizer, clean, engineer,
                   load_raw, split, summary_table)
from .metrics import calibration_bins, classification_summary, pr_points, roc_points
from .regression import (BayesianLinearRegression, Huber, KernelGLM, KernelRidge, LeastSquares, Ridge,
                         gradient_descent, kfold_indices, r2, rbf_kernel, regression_scores, ridge_path, rmse)

T = config.TARGET


def log(msg):
    print(f"[train] {msg}", flush=True)


def _r(x, nd=6):
    if isinstance(x, float):
        return float(f"{x:.{nd + 1}g}") if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return _r(x.tolist(), nd)
    if isinstance(x, np.floating):
        return _r(float(x), nd)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


# --------------------------------------------------------------------------- #
def data_report(raw, df, n_dupes):
    corr_cols = config.RAW + ["wc", "wb", T]
    C = np.corrcoef(df[corr_cols].to_numpy(dtype=float), rowvar=False)
    return {
        "n_raw": len(raw), "n_clean": len(df), "duplicates": n_dupes,
        "summary": summary_table(df[config.RAW + [T, "wc", "wb"]]),
        "corr": {"keys": corr_cols, "labels": [config.LABEL.get(c, c) for c in corr_cols], "matrix": C},
        "strength": df[T].to_numpy(),
        "age_counts": df["age"].value_counts().sort_index().to_dict(),
        "pass_rates": [{"grade": g["grade"], "target": g["target"], "rate_all": float((df[T] >= g["target"]).mean()),
                        "rate_28": float((df[df.age == 28][T] >= g["target"]).mean())} for g in config.GRADES],
        "scatter": {"age": df["age"].to_numpy(), "strength": df[T].to_numpy(), "wb": df["wb"].to_numpy(),
                    "cement": df["cement"].to_numpy()},
    }


# --------------------------------------------------------------------------- #
def train_regression(s, Xtr, Xva, Xte, Ktr, Kva, Kte):
    ytr, yva, yte = s.train[T].to_numpy(), s.val[T].to_numpy(), s.test[T].to_numpy()
    rep, models = {}, {}

    # Least squares / MLE
    ols = LeastSquares().fit(Xtr, ytr)
    w_gd, losses = gradient_descent(Xtr, ytr, lr=0.1, epochs=300)
    closed_mse = float(np.mean((ytr - ols.predict(Xtr)) ** 2))
    rep["ols"] = {"w": ols.w, "noise_sd": math.sqrt(ols.noise_var), "loglik": ols.loglik,
                  "gd_losses": losses, "closed_form_mse": closed_mse,
                  "gd_vs_closed_max_diff": float(np.abs(w_gd - ols.w).max())}
    models["ols"] = ols

    # Ridge: lambda by 5-fold cross-validation on the training split
    lams = np.logspace(-3, 4, 36)
    folds = kfold_indices(len(Xtr), 5, config.SEED)
    cv = []
    for lam in lams:
        errs = []
        for f in folds:
            tr = np.setdiff1d(np.arange(len(Xtr)), f)
            errs.append(rmse(ytr[f], Ridge(lam).fit(Xtr[tr], ytr[tr]).predict(Xtr[f])))
        cv.append({"lambda": float(lam), "cv_rmse": float(np.mean(errs)), "cv_se": float(np.std(errs) / math.sqrt(5))})
    best = min(cv, key=lambda r: r["cv_rmse"])
    ridge = Ridge(best["lambda"]).fit(Xtr, ytr)
    rep["ridge"] = {"lambda": best["lambda"], "cv": cv, "path": {"lambdas": lams, "coefs": ridge_path(Xtr, ytr, lams)},
                    "w": ridge.w}
    models["ridge"] = ridge

    # Huber, plus an outlier stress test: some cube results mistyped in the training data
    hub = Huber().fit(Xtr, ytr)
    rep["huber"] = {"w": hub.w, "b": hub.b, "scale": hub.scale, "weights": hub.weights, "iterations": len(hub.history),
                    "history": hub.history, "downweighted": int((hub.weights < 0.999).sum())}
    models["huber"] = hub
    contam = []
    for frac in [0, 0.025, 0.05, 0.1, 0.15, 0.2]:
        e_ols, e_hub = [], []
        for rep_i in range(6):
            rng = np.random.default_rng(100 + rep_i)
            yc = ytr.copy()
            k = int(round(frac * len(yc)))
            idx = rng.choice(len(yc), k, replace=False)
            yc[idx] = yc[idx] * rng.uniform(1.8, 3.0, k)  # e.g. a decimal slip or a wrong unit
            e_ols.append(rmse(yte, LeastSquares().fit(Xtr, yc).predict(Xte)))
            e_hub.append(rmse(yte, Huber().fit(Xtr, yc).predict(Xte)))
        contam.append({"fraction": frac, "ols": float(np.mean(e_ols)), "huber": float(np.mean(e_hub))})
    rep["contamination"] = contam

    # Bayesian linear regression with evidence maximisation
    blr = BayesianLinearRegression().fit(Xtr, ytr)
    tc = ytr - ytr.mean()
    log_alphas = np.linspace(-9, 6, 61)
    rep["blr"] = {"alpha": blr.alpha, "beta": blr.beta, "gamma": blr.gamma, "evidence": blr.evidence,
                  "noise_sd": 1 / math.sqrt(blr.beta), "trace": blr.trace, "m": blr.m,
                  "sd": np.sqrt(np.diag(blr.S)),
                  "evidence_curve": {"log_alpha": log_alphas,
                                     "log_evidence": [BayesianLinearRegression.log_evidence(Xtr, tc, math.exp(a), blr.beta)
                                                      for a in log_alphas]}}
    models["blr"] = blr

    # Kernel ridge (the kernel trick): gamma and lambda by validation RMSE
    grid = []
    for g in np.logspace(-2.3, 0.3, 14):
        for lam in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
            m = KernelRidge("rbf", g, lam).fit(Ktr, ytr)
            grid.append({"gamma": float(g), "lambda": lam, "val_rmse": rmse(yva, m.predict(Kva)),
                         "train_r2": r2(ytr, m.predict(Ktr)), "val_r2": r2(yva, m.predict(Kva))})
    kb = min(grid, key=lambda r: r["val_rmse"])
    krr = KernelRidge("rbf", kb["gamma"], kb["lambda"]).fit(Ktr, ytr)
    other_kernels = {}
    for name, m in [("linear", KernelRidge("linear", lam=kb["lambda"])), ("poly2", KernelRidge("poly", lam=1.0, degree=2)),
                    ("poly3", KernelRidge("poly", lam=3.0, degree=3)), ("rbf", krr)]:
        m.fit(Ktr, ytr)
        other_kernels[name] = regression_scores(yte, m.predict(Kte))
    rep["krr"] = {"gamma": kb["gamma"], "lambda": kb["lambda"], "grid": grid, "kernels": other_kernels}
    models["krr"] = krr
    log(f"kernel ridge gamma={kb['gamma']:.3f} lambda={kb['lambda']}  test R2 {other_kernels['rbf']['r2']:.4f}")

    # Bayesian kernel GLM (RBF basis functions + evidence-maximised Bayesian linear regression)
    kg_grid = []
    for g in [0.05, 0.08, 0.12, 0.18]:
        for nc in [160, 300]:
            m = KernelGLM(g, nc, seed=1).fit(Ktr, ytr)
            mu, sd = m.predict(Kva, return_std=True)
            kg_grid.append({"gamma": g, "centers": nc, "val_rmse": rmse(yva, mu),
                            "val_nlpd": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + (yva - mu) ** 2 / (2 * sd ** 2)))})
    kgb = min(kg_grid, key=lambda r: r["val_nlpd"])
    kglm = KernelGLM(kgb["gamma"], kgb["centers"], seed=1).fit(Ktr, ytr)
    rep["kglm"] = {"gamma": kgb["gamma"], "centers": kgb["centers"], "grid": kg_grid, "alpha": kglm.blr.alpha,
                   "beta": kglm.blr.beta, "gamma_eff": kglm.blr.gamma}
    models["kglm"] = kglm
    log(f"kernel GLM gamma={kgb['gamma']} centers={kgb['centers']}")

    # Test-set scores, parity data and interval coverage
    preds = {"ols": ols.predict(Xte), "ridge": ridge.predict(Xte), "huber": hub.predict(Xte),
             "blr": blr.predict(Xte), "krr": krr.predict(Kte), "kglm": kglm.predict(Kte)}
    rep["test"] = {k: regression_scores(yte, p) for k, p in preds.items()}
    rep["val"] = {"ols": regression_scores(yva, ols.predict(Xva)), "ridge": regression_scores(yva, ridge.predict(Xva)),
                  "huber": regression_scores(yva, hub.predict(Xva)), "blr": regression_scores(yva, blr.predict(Xva)),
                  "krr": regression_scores(yva, krr.predict(Kva)), "kglm": regression_scores(yva, kglm.predict(Kva))}
    rep["parity"] = {"actual": yte, **preds}
    mu_b, sd_b = blr.predict(Xte, return_std=True)
    mu_k, sd_k = kglm.predict(Kte, return_std=True)
    rep["kglm_test_sd"] = sd_k
    levels = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
    from scipy.stats import norm
    rep["coverage"] = {"nominal": levels,
                       "blr": [float(np.mean(np.abs(yte - mu_b) <= norm.ppf(0.5 + l / 2) * sd_b)) for l in levels],
                       "kglm": [float(np.mean(np.abs(yte - mu_k) <= norm.ppf(0.5 + l / 2) * sd_k)) for l in levels]}
    rep["residuals"] = {"ols": yte - preds["ols"], "krr": yte - preds["krr"], "pred_ols": preds["ols"], "pred_krr": preds["krr"]}

    # Learning curves: linear (OLS) vs kernel ridge
    lc = []
    for n in [40, 80, 120, 200, 300, 400, 500, len(Xtr)]:
        a = {"ols_train": [], "ols_val": [], "krr_train": [], "krr_val": []}
        for rep_i in range(5 if n < len(Xtr) else 1):
            idx = np.random.default_rng(200 + rep_i).choice(len(Xtr), n, replace=False)
            o = LeastSquares().fit(Xtr[idx], ytr[idx])
            k = KernelRidge("rbf", kb["gamma"], kb["lambda"]).fit(Ktr[idx], ytr[idx])
            a["ols_train"].append(rmse(ytr[idx], o.predict(Xtr[idx])))
            a["ols_val"].append(rmse(yva, o.predict(Xva)))
            a["krr_train"].append(rmse(ytr[idx], k.predict(Ktr[idx])))
            a["krr_val"].append(rmse(yva, k.predict(Kva)))
        lc.append({"n": n, **{k: float(np.mean(v)) for k, v in a.items()}})
    rep["learning_curve"] = lc

    # Kernel matrix for 120 training mixes sorted by strength
    idx = np.argsort(ytr)[:: max(1, len(ytr) // 120)][:120]
    rep["kernel_matrix"] = {"K": rbf_kernel(Ktr[idx], Ktr[idx], kb["gamma"]), "strength": ytr[idx]}

    # scikit-learn agreement
    from sklearn.kernel_ridge import KernelRidge as SKKR
    from sklearn.linear_model import LinearRegression as SKLin, Ridge as SKRidge
    sk_krr = SKKR(kernel="rbf", gamma=kb["gamma"], alpha=kb["lambda"]).fit(Ktr, ytr - ytr.mean())
    rep["sklearn"] = {
        "ols_max_diff": float(np.abs(SKLin().fit(Xtr, ytr).predict(Xte) - preds["ols"]).max()),
        "ridge_max_diff": float(np.abs(SKRidge(ridge.lam).fit(Xtr, ytr).predict(Xte) - preds["ridge"]).max()),
        "krr_max_diff": float(np.abs(sk_krr.predict(Kte) + ytr.mean() - preds["krr"]).max()),
        "krr_sample": {"ours": preds["krr"], "sklearn": sk_krr.predict(Kte) + ytr.mean()},
    }
    log("regression: " + ", ".join(f"{k} R2 {v['r2']:.3f}" for k, v in rep["test"].items()))
    return rep, models


# --------------------------------------------------------------------------- #
CLASSIFIERS = ["lsq", "fisher", "generative", "logistic", "bayes_logistic", "svm"]
NAMES = {"lsq": "Least-squares discriminant", "fisher": "Fisher's LDA", "generative": "Gaussian generative",
         "logistic": "Logistic regression", "bayes_logistic": "Bayesian logistic (Laplace)", "svm": "Kernel SVM (RBF)",
         "kglm": "Bayesian kernel GLM (via regression)"}


def svm_grid(Ktr, c, seed):
    folds = kfold_indices(len(Ktr), 5, seed)
    res = []
    for C in [0.3, 1, 3, 10, 30, 100]:
        for g in [0.03, 0.08, 0.15, 0.3, 0.6]:
            acc, nsv = [], []
            for f in folds:
                tr = np.setdiff1d(np.arange(len(Ktr)), f)
                m = KernelSVM(C, g).fit(Ktr[tr], c[tr])
                acc.append(float(np.mean(m.predict(Ktr[f]) == c[f])))
                nsv.append(len(m.sv_coef))
            res.append({"C": C, "gamma": g, "cv_acc": float(np.mean(acc)), "n_sv": float(np.mean(nsv))})
    return res


def fit_classifiers(Xtr, Ktr, c, Xva, Kva, cva, svm_C, svm_gamma, alpha):
    m = {"lsq": LeastSquaresClassifier().fit(Xtr, c), "fisher": FisherLDA().fit(Xtr, c),
         "generative": GaussianGenerative().fit(Xtr, c), "logistic": LogisticRegression().fit(Xtr, c),
         "bayes_logistic": BayesianLogisticRegression(alpha).fit(Xtr, c), "svm": KernelSVM(svm_C, svm_gamma).fit(Ktr, c)}
    m["platt"] = PlattScaling().fit(m["svm"].score(Kva), cva)
    return m


def scores_of(m, X, K):
    """(decision score, probability or None) for every classifier."""
    out = {}
    for k in CLASSIFIERS:
        if k == "svm":
            s = m["svm"].score(K)
            out[k] = (s, m["platt"].proba(s))
        elif k == "lsq":
            out[k] = (m[k].score(X), None)
        else:
            out[k] = (m[k].score(X), m[k].proba(X))
    return out


def train_classification(s, Xtr, Xva, Xte, Ktr, Kva, Kte, kglm):
    ytr, yva, yte = s.train[T].to_numpy(), s.val[T].to_numpy(), s.test[T].to_numpy()
    rep = {"grades": {}, "names": NAMES}
    saved = {}
    main = config.GRADE_BY_NAME[config.MAIN_GRADE]
    c_main = (ytr >= main["target"]).astype(int)

    # Hyperparameters chosen once on the main grade (M30) and reused for every grade.
    t0 = time.time()
    grid = svm_grid(Ktr, c_main, config.SEED)
    best = max(grid, key=lambda r: (round(r["cv_acc"], 4), -r["n_sv"]))
    rep["svm_grid"] = grid
    rep["svm_params"] = {"C": best["C"], "gamma": best["gamma"], "cv_acc": best["cv_acc"]}
    log(f"SVM grid ({len(grid) * 5} fits, {time.time() - t0:.1f}s): C={best['C']} gamma={best['gamma']} cv acc {best['cv_acc']:.3f}")
    alphas = np.logspace(-3, 2, 26)
    ev = [BayesianLogisticRegression(a).fit(Xtr, c_main).log_evidence for a in alphas]
    alpha = float(alphas[int(np.argmax(ev))])
    rep["laplace_evidence"] = {"alpha": alphas, "log_evidence": ev, "best_alpha": alpha}
    log(f"Bayesian logistic alpha by Laplace evidence: {alpha:.4f}")

    mu_te, sd_te = kglm.predict(Kte, return_std=True)
    mu_va, sd_va = kglm.predict(Kva, return_std=True)
    from scipy.stats import norm
    for g in config.GRADES:
        c, cva, cte = [(y >= g["target"]).astype(int) for y in (ytr, yva, yte)]
        m = fit_classifiers(Xtr, Ktr, c, Xva, Kva, cva, best["C"], best["gamma"], alpha)
        sc = scores_of(m, Xte, Kte)
        res = {}
        for k, (score, prob) in sc.items():
            res[k] = classification_summary(cte, score, prob)
        p_kglm = 1 - norm.cdf((g["target"] - mu_te) / sd_te)
        res["kglm"] = classification_summary(cte, p_kglm, p_kglm, threshold=0.5)
        rep["grades"][g["grade"]] = {"target": g["target"], "test_pass_rate": float(cte.mean()), "results": res}
        saved[g["grade"]] = m
        if g["grade"] == config.MAIN_GRADE:
            rep["main"] = main_grade_details(m, sc, cte, p_kglm, Xtr, c, Xte, Kte)
    log("M30 test accuracy: " + ", ".join(f"{k} {v['accuracy']:.3f}" for k, v in rep["grades"]["M30"]["results"].items()))

    # Generative vs discriminative as the training set grows (Ng & Jordan, 2002)
    cte_main = (yte >= main["target"]).astype(int)
    curve = []
    for n in [16, 24, 32, 48, 64, 96, 128, 200, 300, 450, len(Xtr)]:
        gen, dis = [], []
        for rep_i in range(20 if n < len(Xtr) else 1):
            rng = np.random.default_rng(300 + rep_i)
            idx = rng.choice(len(Xtr), n, replace=False)
            if c_main[idx].min() == c_main[idx].max():
                continue
            gen.append(float(np.mean((GaussianGenerative().fit(Xtr[idx], c_main[idx]).score(Xte) >= 0) == cte_main)))
            dis.append(float(np.mean((LogisticRegression(lam=1e-2).fit(Xtr[idx], c_main[idx]).score(Xte) >= 0) == cte_main)))
        curve.append({"n": n, "generative": float(np.mean(gen)), "discriminative": float(np.mean(dis))})
    rep["gen_vs_disc"] = curve

    # sklearn agreement for the classifiers that have an exact reference
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression as SKLR
    from sklearn.svm import SVC
    mm = saved[config.MAIN_GRADE]
    sk_lr = SKLR(C=1 / 1e-4, max_iter=20000, tol=1e-10).fit(Xtr, c_main)
    sk_svm = SVC(C=best["C"], gamma=best["gamma"]).fit(Ktr, c_main)
    rep["sklearn"] = {
        "logistic_max_prob_diff": float(np.abs(sk_lr.predict_proba(Xte)[:, 1] - mm["logistic"].proba(Xte)).max()),
        "lda_max_prob_diff": float(np.abs(LinearDiscriminantAnalysis().fit(Xtr, c_main).predict_proba(Xte)[:, 1]
                                          - mm["generative"].proba(Xte)).max()),
        "svm_decision_max_diff": float(np.abs(sk_svm.decision_function(Kte) - mm["svm"].score(Kte)).max()),
        "svm_n_sv": [len(mm["svm"].sv_coef), int(sk_svm.n_support_.sum())],
        "svm_same_predictions": float(np.mean(sk_svm.predict(Kte) == mm["svm"].predict(Kte))),
        "fisher_vs_generative_cosine": float(abs(mm["fisher"].w @ mm["generative"].w) / np.linalg.norm(mm["generative"].w)),
    }
    return rep, saved, {"alpha": alpha, **rep["svm_params"]}


def main_grade_details(m, sc, cte, p_kglm, Xtr, ctr, Xte, Kte):
    d = {"roc": {}, "pr": {}, "calibration": {}, "confusion": {}}
    for k, (score, prob) in list(sc.items()) + [("kglm", (p_kglm, p_kglm))]:
        d["roc"][k] = roc_points(cte, score)
        d["pr"][k] = pr_points(cte, score)
        if prob is not None:
            d["calibration"][k] = calibration_bins(cte, prob, 8)
        pred = (score >= (0.5 if k == "kglm" else 0)).astype(int)
        d["confusion"][k] = {"tp": int(((pred == 1) & (cte == 1)).sum()), "fp": int(((pred == 1) & (cte == 0)).sum()),
                             "fn": int(((pred == 0) & (cte == 1)).sum()), "tn": int(((pred == 0) & (cte == 0)).sum())}
    f = m["fisher"]
    d["fisher"] = {"proj_pass": f.project(Xtr[ctr == 1]), "proj_fail": f.project(Xtr[ctr == 0]), "params": f.proj}
    d["logistic_history"] = m["logistic"].history
    bl = m["bayes_logistic"]
    d["moderation"] = {"map": bl.proba_map(Xte), "bayes": bl.proba(Xte), "sd": np.sqrt(bl.moments(Xte)[1])}
    d["svm"] = {"n_sv": len(m["svm"].sv_coef), "n_bounded": m["svm"].n_bounded, "iterations": m["svm"].n_iter,
                "gaps": m["svm"].gaps, "dual_objective": m["svm"].dual_objective}
    d["weights"] = {"features": LINEAR_FEATURES, "logistic": m["logistic"].w[1:], "generative": m["generative"].w,
                    "fisher": m["fisher"].w, "bayes_logistic": bl.w[1:], "bayes_logistic_sd": np.sqrt(np.diag(bl.S))[1:]}
    return d


# --------------------------------------------------------------------------- #
def decision_boundaries(s, grade="M30", n=90):
    """Every classifier refitted on two features (water/binder, ln age) so its boundary can be drawn."""
    target = config.GRADE_BY_NAME[grade]["target"]
    st = Standardizer(BOUNDARY_FEATURES).fit(s.train)
    X = st.transform(s.train)
    c = (s.train[T].to_numpy() >= target).astype(int)
    Xv = st.transform(s.val)
    cv = (s.val[T].to_numpy() >= target).astype(int)
    m = fit_classifiers(X, X, c, Xv, Xv, cv, 10.0, 0.5, 1.0)
    raw_wb = np.linspace(0.23, 0.95, n)
    raw_age = np.linspace(math.log(1), math.log(365), n)
    G = np.array([[a, b] for b in raw_age for a in raw_wb])
    Gs = (G - st.mean) / st.std
    out = {"x": raw_wb, "y": np.exp(raw_age), "grade": grade, "target": target,
           "points": {"wb": s.train["wb"].to_numpy(), "age": s.train["age"].to_numpy(), "label": c}, "models": {}}
    for k, (score, prob) in scores_of(m, Gs, Gs).items():
        z = (prob if prob is not None else 1 / (1 + np.exp(-score))).reshape(n, n)
        acc = float(np.mean(((m[k].score(X) if k != "svm" else m["svm"].score(X)) >= 0) == c))
        out["models"][k] = {"z": z, "train_acc": acc}
    bl = m["bayes_logistic"]
    out["models"]["bayes_logistic"]["sd"] = np.sqrt(bl.moments(Gs)[1]).reshape(n, n)
    out["models"]["svm"]["support"] = m["svm"].sv_index
    out["models"]["svm"]["margin"] = m["svm"].score(Gs).reshape(n, n)
    return out


# --------------------------------------------------------------------------- #
def export_models(stand, reg, cls, hp):
    def clf(m):
        return {"lsq": {"w": m["lsq"].w}, "fisher": {"w": m["fisher"].w, "proj": m["fisher"].proj},
                "generative": {"w": m["generative"].w, "w0": m["generative"].w0},
                "logistic": {"w": m["logistic"].w}, "bayes_logistic": {"w": m["bayes_logistic"].w, "S": m["bayes_logistic"].S},
                "svm": {"sv_X": m["svm"].sv_X, "sv_coef": m["svm"].sv_coef, "rho": m["svm"].rho, "gamma": m["svm"].gamma},
                "platt": {"w": m["platt"].lr.w}}
    kg = reg["kglm"]
    return {
        "standardizers": {k: v.to_dict() for k, v in stand.items()},
        "regression": {
            "ols": {"w": reg["ols"].w, "t_mean": reg["ols"].t_mean, "x_mean": reg["ols"].x_mean},
            "ridge": {"w": reg["ridge"].w, "t_mean": reg["ridge"].t_mean, "x_mean": reg["ridge"].x_mean, "lambda": reg["ridge"].lam},
            "huber": {"w": reg["huber"].w, "b": reg["huber"].b, "t_mean": reg["huber"].t_mean},
            "blr": {"m": reg["blr"].m, "S": reg["blr"].S, "beta": reg["blr"].beta, "t_mean": reg["blr"].t_mean, "x_mean": reg["blr"].x_mean},
            "krr": {"X": reg["krr"].X, "a": reg["krr"].a, "gamma": reg["krr"].gamma, "t_mean": reg["krr"].t_mean},
            "kglm": {"centers": kg.centers, "gamma": kg.gamma, "m": kg.blr.m, "S": kg.blr.S, "beta": kg.blr.beta,
                     "t_mean": kg.blr.t_mean, "x_mean": kg.blr.x_mean},
        },
        "classification": {g: clf(m) for g, m in cls.items()},
        "hyper": hp,
    }


def main():
    t0 = time.time()
    config.ARTIFACTS.mkdir(exist_ok=True)
    raw = load_raw()
    df, n_dupes = clean(raw)
    df = engineer(df)
    s = split(df)
    log(f"rows {len(raw)} -> {len(df)} after removing {n_dupes} duplicates; train/val/test {len(s.train)}/{len(s.val)}/{len(s.test)}")
    stand = {"linear": Standardizer(LINEAR_FEATURES).fit(s.train), "kernel": Standardizer(KERNEL_FEATURES).fit(s.train)}
    Xtr, Xva, Xte = (stand["linear"].transform(p) for p in (s.train, s.val, s.test))
    Ktr, Kva, Kte = (stand["kernel"].transform(p) for p in (s.train, s.val, s.test))

    rep = {"data": data_report(raw, df, n_dupes), "features": {"linear": LINEAR_FEATURES, "kernel": KERNEL_FEATURES},
           "sizes": {"train": len(s.train), "val": len(s.val), "test": len(s.test)}}
    reg_rep, reg_models = train_regression(s, Xtr, Xva, Xte, Ktr, Kva, Kte)
    rep["regression"] = reg_rep
    cls_rep, cls_models, hp = train_classification(s, Xtr, Xva, Xte, Ktr, Kva, Kte, reg_models["kglm"])
    rep["classification"] = cls_rep
    boundary = decision_boundaries(s)
    rep["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    (config.ARTIFACTS / "model.json").write_text(json.dumps(_r(export_models(stand, reg_models, cls_models, hp), 12)))
    rep = _r(rep)
    (config.ARTIFACTS / "report.json").write_text(json.dumps(rep))
    (config.ARTIFACTS / "boundary.json").write_text(json.dumps(_r(boundary, 5)))
    from .charts import build_all
    graphs = build_all(rep, _r(boundary, 5))
    (config.ARTIFACTS / "graphs.json").write_text(json.dumps(graphs))
    log(f"{len(graphs)} graphs written; done in {time.time() - t0:.1f}s")


def rebuild_charts():
    from .charts import build_all
    rep = json.loads((config.ARTIFACTS / "report.json").read_text())
    boundary = json.loads((config.ARTIFACTS / "boundary.json").read_text())
    graphs = build_all(rep, boundary)
    (config.ARTIFACTS / "graphs.json").write_text(json.dumps(graphs))
    log(f"{len(graphs)} graphs rebuilt")


if __name__ == "__main__":
    rebuild_charts() if "--charts-only" in sys.argv else main()
