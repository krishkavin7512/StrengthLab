"""Every from-scratch model is checked against a reference implementation or a known identity."""
import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_ridge import KernelRidge as SKKernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression as SKLogistic, Ridge as SKRidge
from sklearn.svm import SVC

from backend.classification import (BayesianLogisticRegression, FisherLDA, GaussianGenerative, KernelSVM,
                                    LogisticRegression, PlattScaling)
from backend.regression import (BayesianLinearRegression, Huber, KernelGLM, KernelRidge, LeastSquares, Ridge,
                                gradient_descent, rmse)

rng = np.random.default_rng(3)
X = rng.normal(size=(300, 5))
w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
t = X @ w_true + 10 + rng.normal(0, 0.5, 300)
y = (X[:, 0] + X[:, 1] ** 2 - 1 + rng.normal(0, 0.5, 300) > 0).astype(int)


def test_least_squares_matches_sklearn_and_gd():
    ours = LeastSquares().fit(X, t)
    sk = LinearRegression().fit(X, t)
    assert np.allclose(ours.w, sk.coef_) and ours.predict(X[:3]) == pytest.approx(sk.predict(X[:3]))
    w_gd, losses = gradient_descent(X, t, lr=0.1, epochs=2000)
    assert np.allclose(w_gd, ours.w, atol=1e-6)
    assert losses[-1] <= losses[0]


def test_ridge_matches_sklearn():
    assert np.allclose(Ridge(5.0).fit(X, t).w, SKRidge(5.0).fit(X, t).coef_, atol=1e-10)


def test_bayesian_posterior_mean_is_ridge_with_lambda_alpha_over_beta():
    b = BayesianLinearRegression(alpha=2.0, beta=4.0).fit(X, t)
    assert np.allclose(b.m, Ridge(2.0 / 4.0).fit(X, t).w)


def test_bayesian_evidence_recovers_noise_level():
    b = BayesianLinearRegression().fit(X, t)
    assert 1 / np.sqrt(b.beta) == pytest.approx(0.5, rel=0.15)
    mean, sd = b.predict(X, return_std=True)
    assert np.all(sd >= 1 / np.sqrt(b.beta))


def test_huber_resists_outliers():
    tc = t.copy()
    tc[:30] += 60
    clean = X @ w_true + 10
    assert rmse(clean, Huber().fit(X, tc).predict(X)) < 0.5 * rmse(clean, LeastSquares().fit(X, tc).predict(X))


def test_kernel_ridge_matches_sklearn_and_linear_kernel_is_ridge():
    ours = KernelRidge("rbf", 0.3, 0.1).fit(X, t)
    sk = SKKernelRidge(kernel="rbf", gamma=0.3, alpha=0.1).fit(X, t - t.mean())
    assert np.allclose(ours.predict(X), sk.predict(X) + t.mean(), atol=1e-9)
    Xc = X - X.mean(axis=0)  # the dual/primal identity holds for centred inputs
    lin = KernelRidge("linear", lam=3.0).fit(Xc, t)
    assert np.allclose(lin.predict(Xc), Ridge(3.0).fit(Xc, t).predict(Xc), atol=1e-8)


def test_kernel_glm_fits_nonlinear_function():
    tn = np.sin(X[:, 0]) * 5 + X[:, 1] ** 2
    m = KernelGLM(gamma=0.3, n_centers=120).fit(X, tn)
    assert rmse(tn, m.predict(X)) < 0.5 * rmse(tn, LeastSquares().fit(X, tn).predict(X))


def test_logistic_irls_matches_sklearn():
    ours = LogisticRegression(lam=1e-4).fit(X, y)
    sk = SKLogistic(C=1e4, tol=1e-12, max_iter=10000).fit(X, y)
    assert np.allclose(ours.proba(X), sk.predict_proba(X)[:, 1], atol=1e-4)


def test_generative_matches_lda_and_fisher_direction():
    g = GaussianGenerative().fit(X, y)
    sk = LinearDiscriminantAnalysis().fit(X, y)
    assert np.allclose(g.proba(X), sk.predict_proba(X)[:, 1], atol=5e-3)
    f = FisherLDA().fit(X, y)
    assert abs(f.w @ g.w) / np.linalg.norm(g.w) == pytest.approx(1.0, abs=1e-10)


def test_bayesian_logistic_moderates_towards_half():
    b = BayesianLogisticRegression(alpha=1.0).fit(X[:40], y[:40])
    p_map, p_bayes = b.proba_map(X), b.proba(X)
    assert np.all(np.abs(p_bayes - 0.5) <= np.abs(p_map - 0.5) + 1e-12)
    assert np.all(np.sign(p_bayes - 0.5) == np.sign(p_map - 0.5))


def test_svm_matches_libsvm():
    ours = KernelSVM(C=3.0, gamma=0.2).fit(X, y)
    sk = SVC(C=3.0, gamma=0.2, tol=1e-3).fit(X, y)
    assert len(ours.sv_coef) == sk.n_support_.sum()
    assert np.abs(ours.score(X) - sk.decision_function(X)).max() < 0.01
    assert np.mean(ours.predict(X) == sk.predict(X)) > 0.99
    p = PlattScaling().fit(ours.score(X), y).proba(ours.score(X))
    assert np.corrcoef(p, ours.score(X))[0, 1] > 0.9
