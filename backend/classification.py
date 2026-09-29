"""Linear classifiers, kernels and the SVM, written from the equations with NumPy.

Inputs are standardised features; labels are 1 (mix reaches the target strength) or 0.
Every model exposes score(X) (a real-valued discriminant) and, where the model is
probabilistic, proba(X) = P(pass | x).
"""
import math

import numpy as np

from .regression import rbf_kernel


def sigmoid(a):
    a = np.clip(a, -40, 40)
    return 1.0 / (1.0 + np.exp(-a))


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


# --------------------------------------------------------------------------- #
# Discriminant functions (Bishop section 4.1)
# --------------------------------------------------------------------------- #
class LeastSquaresClassifier:
    """Least squares for classification (Bishop 4.1.3): regress targets t in {-1, +1}, predict sign(y(x))."""

    def fit(self, X, y):
        t = np.where(np.asarray(y) == 1, 1.0, -1.0)
        self.w, *_ = np.linalg.lstsq(add_bias(X), t, rcond=None)
        return self

    def score(self, X):
        return add_bias(X) @ self.w

    def predict(self, X):
        return (self.score(X) >= 0).astype(int)


class FisherLDA:
    """Fisher's linear discriminant (Bishop 4.1.4): w proportional to S_W^-1 (m_1 - m_0).

    The direction maximises between-class separation over within-class spread. To classify,
    each class's projection y = w^T x is modelled by a 1-D Gaussian and combined with the class
    priors through Bayes' rule.
    """

    def fit(self, X, y):
        y = np.asarray(y)
        X0, X1 = X[y == 0], X[y == 1]
        self.m0, self.m1 = X0.mean(0), X1.mean(0)
        Sw = (X0 - self.m0).T @ (X0 - self.m0) + (X1 - self.m1).T @ (X1 - self.m1)
        w = np.linalg.solve(Sw, self.m1 - self.m0)
        self.w = w / np.linalg.norm(w)
        p0, p1 = X0 @ self.w, X1 @ self.w
        self.proj = {"mu": [float(p0.mean()), float(p1.mean())], "var": [float(p0.var()), float(p1.var())],
                     "prior": [len(X0) / len(X), len(X1) / len(X)]}
        return self

    def project(self, X):
        return X @ self.w

    def proba(self, X):
        z = self.project(X)
        mu, var, pr = self.proj["mu"], self.proj["var"], self.proj["prior"]
        logp = [np.log(pr[k]) - 0.5 * np.log(2 * np.pi * var[k]) - (z - mu[k]) ** 2 / (2 * var[k]) for k in (0, 1)]
        return sigmoid(logp[1] - logp[0])

    def score(self, X):
        return np.log(self.proba(X) + 1e-12) - np.log(1 - self.proba(X) + 1e-12)


# --------------------------------------------------------------------------- #
# Probabilistic generative model (Bishop 4.2.2)
# --------------------------------------------------------------------------- #
class GaussianGenerative:
    """Class-conditional Gaussians with a shared covariance: p(x | C_k) = N(x | mu_k, Sigma).

    Bayes' rule then gives P(C_1 | x) = sigmoid(w^T x + w_0) with
        w = Sigma^-1 (mu_1 - mu_0)
        w_0 = -1/2 mu_1^T Sigma^-1 mu_1 + 1/2 mu_0^T Sigma^-1 mu_0 + ln(pi_1 / pi_0)
    All parameters are maximum-likelihood estimates (class means, pooled covariance, class fractions).
    """

    def fit(self, X, y):
        y = np.asarray(y)
        X0, X1 = X[y == 0], X[y == 1]
        self.mu0, self.mu1 = X0.mean(0), X1.mean(0)
        self.pi1 = len(X1) / len(X)
        self.Sigma = ((X0 - self.mu0).T @ (X0 - self.mu0) + (X1 - self.mu1).T @ (X1 - self.mu1)) / len(X)
        Si = np.linalg.inv(self.Sigma)
        self.w = Si @ (self.mu1 - self.mu0)
        self.w0 = float(-0.5 * self.mu1 @ Si @ self.mu1 + 0.5 * self.mu0 @ Si @ self.mu0
                        + math.log(self.pi1 / (1 - self.pi1)))
        return self

    def score(self, X):
        return X @ self.w + self.w0

    def proba(self, X):
        return sigmoid(self.score(X))


# --------------------------------------------------------------------------- #
# Probabilistic discriminative model (Bishop 4.3.3)
# --------------------------------------------------------------------------- #
class LogisticRegression:
    """Logistic regression fitted by iteratively reweighted least squares (Newton-Raphson).

    w_new = w_old - H^-1 grad E,  grad E = Phi^T (y - t),  H = Phi^T R Phi,  R = diag(y_n (1 - y_n)).
    A tiny ridge term keeps H invertible if the classes happen to be separable.
    """

    def __init__(self, lam=1e-4, max_iter=50, tol=1e-10):
        self.lam, self.max_iter, self.tol = lam, max_iter, tol

    def fit(self, X, t):
        Phi = add_bias(X)
        t = np.asarray(t, dtype=float)
        w = np.zeros(Phi.shape[1])
        P = self.lam * np.eye(Phi.shape[1])
        P[0, 0] = 0.0
        self.history = []
        for _ in range(self.max_iter):
            y = sigmoid(Phi @ w)
            nll = -float(np.sum(t * np.log(y + 1e-12) + (1 - t) * np.log(1 - y + 1e-12)))
            self.history.append(nll)
            g = Phi.T @ (y - t) + P @ w
            H = Phi.T @ (Phi * (y * (1 - y))[:, None]) + P
            step = np.linalg.solve(H, g)
            w = w - step
            if np.max(np.abs(step)) < self.tol:
                break
        self.w = w
        return self

    def score(self, X):
        return add_bias(X) @ self.w

    def proba(self, X):
        return sigmoid(self.score(X))


class BayesianLogisticRegression:
    """Bayesian logistic regression with the Laplace approximation (Bishop 4.4-4.5).

    Prior p(w) = N(0, alpha^-1 I). The posterior is approximated by a Gaussian at the MAP
    weights with covariance S_N = A^-1, A = alpha I + sum_n y_n (1 - y_n) phi_n phi_n^T.
    The predictive probability integrates over that uncertainty using the probit trick:
        P(C_1 | x) ~ sigmoid(kappa(sigma_a^2) mu_a),  kappa = (1 + pi sigma_a^2 / 8)^-1/2
    with mu_a = w_MAP^T phi and sigma_a^2 = phi^T S_N phi. alpha is chosen by the Laplace
    approximation to the model evidence.
    """

    def __init__(self, alpha=1.0, max_iter=60, tol=1e-10):
        self.alpha, self.max_iter, self.tol = alpha, max_iter, tol

    def fit(self, X, t):
        Phi = add_bias(X)
        t = np.asarray(t, dtype=float)
        M = Phi.shape[1]
        prior = self.alpha * np.eye(M)
        prior[0, 0] = 1e-6  # an essentially flat prior on the bias
        w = np.zeros(M)
        for _ in range(self.max_iter):
            y = sigmoid(Phi @ w)
            g = Phi.T @ (y - t) + prior @ w
            A = Phi.T @ (Phi * (y * (1 - y))[:, None]) + prior
            step = np.linalg.solve(A, g)
            w -= step
            if np.max(np.abs(step)) < self.tol:
                break
        y = sigmoid(Phi @ w)
        self.A = Phi.T @ (Phi * (y * (1 - y))[:, None]) + prior
        self.S = np.linalg.inv(self.A)
        self.w = w
        loglik = float(np.sum(t * np.log(y + 1e-12) + (1 - t) * np.log(1 - y + 1e-12)))
        _, logdet_A = np.linalg.slogdet(self.A)
        _, logdet_P = np.linalg.slogdet(prior)
        # ln p(t | alpha) ~ ln p(t | w_MAP) + ln p(w_MAP) + M/2 ln 2pi - 1/2 ln |A|
        self.log_evidence = float(loglik - 0.5 * w @ prior @ w + 0.5 * logdet_P - 0.5 * logdet_A)
        return self

    def moments(self, X):
        Phi = add_bias(X)
        mu = Phi @ self.w
        var = np.einsum("ij,jk,ik->i", Phi, self.S, Phi)
        return mu, var

    def score(self, X):
        return add_bias(X) @ self.w

    def proba_map(self, X):
        return sigmoid(self.score(X))

    def proba(self, X):
        mu, var = self.moments(X)
        return sigmoid(mu / np.sqrt(1 + math.pi * var / 8))


# --------------------------------------------------------------------------- #
# Support vector machine
# --------------------------------------------------------------------------- #
class KernelSVM:
    """Soft-margin kernel SVM trained by sequential minimal optimisation (the LIBSVM algorithm).

    Dual:  max_a  sum_n a_n - 1/2 sum_n sum_m a_n a_m t_n t_m k(x_n, x_m)
           s.t.   0 <= a_n <= C,  sum_n a_n t_n = 0,        t_n in {-1, +1}
    Each step picks the most violating pair (i, j) with second-order working-set selection,
    solves the two-variable sub-problem analytically, and clips it back into the box.
    Prediction: y(x) = sum_n a_n t_n k(x, x_n) - rho, with only support vectors (a_n > 0) mattering.
    """

    def __init__(self, C=10.0, gamma=0.1, tol=1e-3, max_iter=200000):
        self.C, self.gamma, self.tol, self.max_iter = C, gamma, tol, max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        t = np.where(np.asarray(y) == 1, 1.0, -1.0)
        n, C, TAU = len(X), self.C, 1e-12
        K = rbf_kernel(X, X, self.gamma)
        Q = K * np.outer(t, t)
        dK = np.diag(K).copy()
        a = np.zeros(n)
        G = -np.ones(n)  # gradient of the dual objective 1/2 a^T Q a - e^T a
        self.gaps = []
        it = 0
        for it in range(self.max_iter):
            vals = -t * G
            up = ((t > 0) & (a < C)) | ((t < 0) & (a > 0))
            low = ((t > 0) & (a > 0)) | ((t < 0) & (a < C))
            if not up.any() or not low.any():
                break
            iu = np.where(up)[0]
            i = iu[np.argmax(vals[iu])]
            gmax = vals[i]
            il = np.where(low)[0]
            gmin = vals[il].min()
            if it % 50 == 0:
                self.gaps.append(float(gmax - gmin))
            if gmax - gmin < self.tol:
                break
            cand = il[vals[il] < gmax]
            b = gmax - vals[cand]
            quad = np.maximum(dK[i] + dK[cand] - 2 * K[i, cand], TAU)
            j = cand[np.argmin(-(b * b) / quad)]
            ai_old, aj_old = a[i], a[j]
            if t[i] != t[j]:
                quad_ij = max(dK[i] + dK[j] + 2 * Q[i, j], TAU)
                delta = (-G[i] - G[j]) / quad_ij
                diff = a[i] - a[j]
                a[i] += delta
                a[j] += delta
                if diff > 0:
                    if a[j] < 0:
                        a[j], a[i] = 0.0, diff
                elif a[i] < 0:
                    a[i], a[j] = 0.0, -diff
                if diff > 0:
                    if a[i] > C:
                        a[i], a[j] = C, C - diff
                elif a[j] > C:
                    a[j], a[i] = C, C + diff
            else:
                quad_ij = max(dK[i] + dK[j] - 2 * Q[i, j], TAU)
                delta = (G[i] - G[j]) / quad_ij
                s = a[i] + a[j]
                a[i] -= delta
                a[j] += delta
                if s > C:
                    if a[i] > C:
                        a[i], a[j] = C, s - C
                    if a[j] > C:
                        a[j], a[i] = C, s - C
                else:
                    if a[j] < 0:
                        a[j], a[i] = 0.0, s
                    if a[i] < 0:
                        a[i], a[j] = 0.0, s
            G += Q[:, i] * (a[i] - ai_old) + Q[:, j] * (a[j] - aj_old)
        self.n_iter = it + 1
        free = (a > 1e-12) & (a < C - 1e-12)
        vals = t * G
        if free.any():
            self.rho = float(vals[free].mean())
        else:
            ub = np.where(((t < 0) & (a >= C)) | ((t > 0) & (a <= 0)), vals, np.inf).min()
            lb = np.where(((t > 0) & (a >= C)) | ((t < 0) & (a <= 0)), vals, -np.inf).max()
            self.rho = float((ub + lb) / 2)
        sv = a > 1e-12
        self.sv_X, self.sv_coef = X[sv], (a * t)[sv]
        self.sv_index = np.where(sv)[0]
        self.n_bounded = int((a >= C - 1e-12).sum())
        self.dual_objective = float(a.sum() - 0.5 * a @ Q @ a)
        return self

    def score(self, X):
        return rbf_kernel(np.asarray(X, dtype=float), self.sv_X, self.gamma) @ self.sv_coef - self.rho

    def predict(self, X):
        return (self.score(X) >= 0).astype(int)


class PlattScaling:
    """Turn SVM margins into probabilities: P(pass | f) = sigmoid(A f + B), fitted by 1-D logistic regression."""

    def fit(self, f, y):
        self.lr = LogisticRegression(lam=1e-6).fit(np.asarray(f).reshape(-1, 1), y)
        return self

    def proba(self, f):
        return self.lr.proba(np.asarray(f).reshape(-1, 1))
