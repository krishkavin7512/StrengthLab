# StrengthLab

Predict how strong a concrete mix will be, and whether it will pass its IS grade (M20–M40), straight from its
recipe, instead of waiting 28 days for a cube test. Six regressors and six classifiers, each written from its
equations in NumPy and trained on 1,030 laboratory tests.

## Features

- **Mix designer**: set cement, slag, fly ash, water, superplasticizer, aggregates and curing age with sliders or
  presets. It shows the predicted strength with a 95% band, the probability of passing the chosen grade (seven models
  vote), the strength-gain curve from day 1 to 365, the IS 456 durability checks for the exposure condition and a
  CO₂ estimate.
- **Labs**: six regressors compared, a kernel lab whose γ / λ sliders refit kernel ridge live on the server, and a
  decision-boundary explorer for every classifier.
- **Graphs explained** (`/graphs`): all 35 training graphs, each with what it shows, how it's computed, how to read
  it and where it's used.
- Animated, accessible frontend with no framework (HTML, CSS 3D, JavaScript modules, Plotly.js bundled locally).

A mix passes a grade when its strength reaches the IS 10262:2019 target mean strength f′ck = fck + 1.65 s
(for example 38.25 MPa for M30).

## Results (150 held-out mixes)

| Model | Test RMSE | Test R² |
|---|---|---|
| Least squares (maximum likelihood) | 6.74 MPa | 0.855 |
| Kernel ridge (RBF) | **4.33 MPa** | **0.940** |
| Bayesian kernel GLM | 4.69 MPa | 0.930 (95% band covers 95.3% of test mixes) |

Grade compliance, M30: best accuracy 90.7% (Bayesian logistic regression), best AUC 0.971 (Bayesian kernel GLM).

## How it works

| Technique | Implementation (`backend/`) |
|---|---|
| Maximum likelihood, least squares | `regression.LeastSquares` (normal equations, σ²_ML) and `gradient_descent` (loss curve) |
| Ridge regression | `regression.Ridge`, regularisation path, λ by 5-fold cross-validation |
| Robust linear regression | `regression.Huber`, fitted by IRLS, plus an outlier stress test |
| Bayesian linear regression | `regression.BayesianLinearRegression`: posterior, predictive variance, evidence maximisation for α and β |
| Discriminant functions | `classification.LeastSquaresClassifier`, `classification.FisherLDA` |
| Probabilistic generative model | `classification.GaussianGenerative` (shared-covariance Gaussians) |
| Probabilistic discriminative model | `classification.LogisticRegression` (IRLS / Newton–Raphson) |
| Laplace approximation, Bayesian logistic regression | `classification.BayesianLogisticRegression` (MAP, Hessian, probit-moderated predictions, Laplace evidence) |
| Kernel functions, the kernel trick | `regression.rbf_kernel` / `poly_kernel`, `regression.KernelRidge` (dual solution) |
| Kernels in a GLM | `regression.KernelGLM` (RBF basis functions + Bayesian linear regression) |
| Support vector machine | `classification.KernelSVM` (SMO with second-order working-set selection, as in LIBSVM) + `PlattScaling` |

Every model is implemented in NumPy. scikit-learn is used only to confirm the results match.

## Data

I-C. Yeh (1998), *Modeling of strength of high-performance concrete using artificial neural networks*, Cement and
Concrete Research 28(12). Source: [UCI Machine Learning Repository #165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
(CC BY 4.0), included in `data/`. 25 exact duplicates are removed, leaving 1,005 mixes, split 704 / 151 / 150
(seed 7) into training, validation and test.

## Run it

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python run.py
```

The app opens at http://127.0.0.1:8002 (API docs at `/docs`). The trained models are included in `artifacts/`, so it
starts immediately.

- `python run.py --retrain`: retrain everything from scratch (about 30 s).
- `python -m backend.train --charts-only`: rebuild the graphs without retraining.
- `python -m pytest`: 16 tests; the models are checked against scikit-learn or known identities.

## Layout

```
backend/   config, data, regression, classification, metrics, train, charts, predictor, api
frontend/  index.html (app), graphs.html (graphs explained), assets/ (css, js, bundled plotly.js)
artifacts/ trained models, full report, graphs and decision boundaries (JSON, written by backend/train.py)
data/      the dataset (CSV)
tests/     from-scratch vs reference checks, API tests
```

A screening and teaching tool. It does not replace the cube tests that IS 456 requires for acceptance.
