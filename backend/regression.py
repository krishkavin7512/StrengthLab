"""Regression models, written from the equations with NumPy.

Every linear model centres its inputs and targets first (x - mean(x), t - mean(t)),
fits the weights on the centred data, and recovers the intercept afterwards, so no
penalty or prior ever touches the intercept.
"""
import math

import numpy as np


# --------------------------------------------------------------------------- #
# Maximum likelihood / least squares
# --------------------------------------------------------------------------- #
class LeastSquares:
    """MLE under Gaussian noise: w_ML = (X^T X)^-1 X^T t,  1/beta_ML = mean squared residual."""

    def fit(self, X, t):
        self.x_mean = X.mean(axis=0)
        self.t_mean = float(np.mean(t))
        Xc, tc = X - self.x_mean, np.asarray(t, dtype=float) - self.t_mean
        # lstsq solves the normal equations through an SVD, which is stable for correlated columns
        self.w, *_ = np.linalg.lstsq(Xc, tc, rcond=None)
        r = tc - Xc @ self.w
        self.noise_var = float(np.mean(r ** 2))   # sigma^2_ML (Bishop eq. 3.21)
        self.loglik = float(-0.5 * len(tc) * (math.log(2 * math.pi * self.noise_var) + 1))
        return self

    def predict(self, X):
        return (X - self.x_mean) @ self.w + self.t_mean


def gradient_descent(X, t, lr=0.1, epochs=300):
    """Batch gradient descent on the mean squared error; returns the loss after every epoch."""
    X = X - X.mean(axis=0)
    tc = np.asarray(t, dtype=float) - float(np.mean(t))
    w = np.zeros(X.shape[1])
    losses = []
    for _ in range(epochs):
        r = X @ w - tc
        losses.append(float(np.mean(r ** 2)))
        w -= lr * (X.T @ r) / len(tc)
    return w, losses


class Ridge:
    """Regularised least squares: (X^T X + lambda I) w = X^T t."""

    def __init__(self, lam=1.0):
        self.lam = float(lam)

    def fit(self, X, t):
        self.x_mean = X.mean(axis=0)
        self.t_mean = float(np.mean(t))
        Xc, tc = X - self.x_mean, np.asarray(t, dtype=float) - self.t_mean
        A = Xc.T @ Xc + self.lam * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, Xc.T @ tc)
        return self

    def predict(self, X):
        return (X - self.x_mean) @ self.w + self.t_mean


def ridge_path(X, t, lams):
    return np.array([Ridge(l).fit(X, t).w for l in lams])


class Huber:
    """Robust linear regression with the Huber loss, fitted by iteratively reweighted least squares.

    rho(u) = u^2 / 2 for |u| <= delta, delta |u| - delta^2 / 2 otherwise, with u = r / scale.
    Each IRLS step solves (X^T W X) w = X^T W t with weights w_i = min(1, delta / |u_i|),
    so points with large residuals get proportionally less say. scale = 1.4826 * MAD(r).
    """

    def __init__(self, delta=1.345, max_iter=100, tol=1e-8):
        self.delta, self.max_iter, self.tol = delta, max_iter, tol

    def fit(self, X, t):
        self.t_mean = float(np.median(t))
        tc = np.asarray(t, dtype=float) - self.t_mean
        Xb = np.column_stack([np.ones(len(X)), X])  # the robust fit also re-estimates its own intercept
        w, *_ = np.linalg.lstsq(Xb, tc, rcond=None)
        self.history = []
        for _ in range(self.max_iter):
            r = tc - Xb @ w
            scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
            u = np.abs(r / scale)
            weights = np.where(u <= self.delta, 1.0, self.delta / u)
            WX = Xb * weights[:, None]
            w_new = np.linalg.solve(Xb.T @ WX + 1e-9 * np.eye(Xb.shape[1]), WX.T @ tc)
            self.history.append(float(np.max(np.abs(w_new - w))))
            done = np.max(np.abs(w_new - w)) < self.tol
            w = w_new
            if done:
                break
        self.b, self.w = float(w[0]), w[1:]
        self.weights, self.scale = weights, float(scale)
        return self

    def predict(self, X):
        return X @ self.w + self.b + self.t_mean


class BayesianLinearRegression:
    """Gaussian prior w ~ N(0, alpha^-1 I), Gaussian noise with precision beta (Bishop section 3.3).

    Posterior:   S_N^-1 = alpha I + beta X^T X,   m_N = beta S_N X^T t
    Predictive:  N(t | m_N^T x, 1/beta + x^T S_N x)
    alpha and beta are set by maximising the evidence p(t | alpha, beta) (Bishop section 3.5.2):
        gamma = sum_i lambda_i / (alpha + lambda_i),  alpha = gamma / m_N^T m_N,
        1/beta = ||t - X m_N||^2 / (N - gamma)
    """

    def __init__(self, alpha=None, beta=None, max_iter=500, tol=1e-9):
        self.alpha, self.beta, self.max_iter, self.tol = alpha, beta, max_iter, tol

    def _posterior(self, X, tc, alpha, beta):
        A = alpha * np.eye(X.shape[1]) + beta * X.T @ X
        S = np.linalg.inv(A)
        m = beta * S @ X.T @ tc
        return A, S, m

    def fit(self, X, t):
        self.x_mean = X.mean(axis=0)
        self.t_mean = float(np.mean(t))
        X = X - self.x_mean
        tc = np.asarray(t, dtype=float) - self.t_mean
        N, M = X.shape
        eig0 = np.linalg.eigvalsh(X.T @ X)
        alpha = self.alpha or 1.0
        beta = self.beta or 1.0 / np.var(tc)
        learn = self.alpha is None
        self.trace = []
        for _ in range(self.max_iter if learn else 1):
            A, S, m = self._posterior(X, tc, alpha, beta)
            if not learn:
                break
            lam = beta * eig0
            gamma = float(np.sum(lam / (alpha + lam)))
            alpha_new = gamma / float(m @ m)
            beta_new = (N - gamma) / float(np.sum((tc - X @ m) ** 2))
            self.trace.append({"alpha": alpha_new, "beta": beta_new, "gamma": gamma,
                               "log_evidence": self.log_evidence(X, tc, alpha_new, beta_new)})
            done = abs(alpha_new - alpha) / alpha < self.tol and abs(beta_new - beta) / beta < self.tol
            alpha, beta = alpha_new, beta_new
            if done:
                break
        self.alpha, self.beta = float(alpha), float(beta)
        self.A, self.S, self.m = self._posterior(X, tc, alpha, beta)
        self.gamma = float(np.sum(beta * eig0 / (alpha + beta * eig0)))
        self.evidence = self.log_evidence(X, tc, alpha, beta)
        return self

    @staticmethod
    def log_evidence(X, tc, alpha, beta):
        """ln p(t | alpha, beta), Bishop eq. 3.86."""
        N, M = X.shape
        A = alpha * np.eye(M) + beta * X.T @ X
        m = beta * np.linalg.solve(A, X.T @ tc)
        E = beta / 2 * np.sum((tc - X @ m) ** 2) + alpha / 2 * float(m @ m)
        _, logdet = np.linalg.slogdet(A)
        return float(M / 2 * math.log(alpha) + N / 2 * math.log(beta) - E - logdet / 2 - N / 2 * math.log(2 * math.pi))

    def predict(self, X, return_std=False):
        X = X - self.x_mean
        mean = X @ self.m + self.t_mean
        if not return_std:
            return mean
        var = 1.0 / self.beta + np.einsum("ij,jk,ik->i", X, self.S, X)
        return mean, np.sqrt(var)


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def sq_dists(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def rbf_kernel(A, B, gamma):
    """k(a, b) = exp(-gamma ||a - b||^2)"""
    return np.exp(-gamma * sq_dists(A, B))


def poly_kernel(A, B, degree=2, c=1.0):
    """k(a, b) = (a^T b + c)^d"""
    return (A @ B.T + c) ** degree


def linear_kernel(A, B):
    return A @ B.T


class KernelRidge:
    """The kernel trick: ridge regression written entirely in terms of k(x, x').

    Dual solution a = (K + lambda I)^-1 t, prediction y(x) = k(x)^T a. With the linear
    kernel this reproduces ordinary ridge regression exactly; with the RBF kernel it fits a
    non-linear function without ever building the infinite-dimensional feature vector.
    """

    def __init__(self, kernel="rbf", gamma=0.1, lam=1.0, degree=2):
        self.kernel, self.gamma, self.lam, self.degree = kernel, gamma, lam, degree

    def K(self, A, B):
        if self.kernel == "rbf":
            return rbf_kernel(A, B, self.gamma)
        if self.kernel == "poly":
            return poly_kernel(A, B, self.degree)
        return linear_kernel(A, B)

    def fit(self, X, t):
        self.X = np.asarray(X, dtype=float)
        self.t_mean = float(np.mean(t))
        tc = np.asarray(t, dtype=float) - self.t_mean
        self.a = np.linalg.solve(self.K(self.X, self.X) + self.lam * np.eye(len(self.X)), tc)
        return self

    def predict(self, X):
        return self.K(np.asarray(X, dtype=float), self.X) @ self.a + self.t_mean


class KernelGLM:
    """Kernels inside a generalised linear model (Murphy section 14.3.1).

    The feature vector is phi(x) = [k(x, mu_1), ..., k(x, mu_K)], the RBF similarity of x to K
    prototype mixes from the training set, and Bayesian linear regression is fitted on phi(x).
    The model is non-linear in x but still linear in its weights, so it keeps the closed-form
    posterior and a predictive standard deviation for every mix.
    """

    def __init__(self, gamma=0.2, n_centers=160, seed=0):
        self.gamma, self.n_centers, self.seed = gamma, n_centers, seed

    def features(self, X):
        return rbf_kernel(np.asarray(X, dtype=float), self.centers, self.gamma)

    def fit(self, X, t):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), min(self.n_centers, len(X)), replace=False)
        self.centers = X[idx]
        self.blr = BayesianLinearRegression().fit(self.features(X), t)
        return self

    def predict(self, X, return_std=False):
        return self.blr.predict(self.features(X), return_std=return_std)


# --------------------------------------------------------------------------- #
def kfold_indices(n, k, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def regression_scores(y, p):
    return {"rmse": rmse(y, p), "mae": mae(y, p), "r2": r2(y, p)}
