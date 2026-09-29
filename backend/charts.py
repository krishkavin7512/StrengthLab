"""Every StrengthLab training graph as a Plotly figure with a plain-language explanation card."""
import json
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config

# Validated for colour-blind separation (adjacent pairs) on the card surface #151514.
PAL = ["#e0621f", "#3987e5", "#199e70", "#9085e9", "#c98500", "#d55181"]
REF = "#a8a49a"  # neutral grey for a 7th, reference series
INK, INK2, MUTED = "#f3f1ec", "#c9c5bb", "#8b877d"
GRID, AXIS, SURF = "#25241f", "#3a3831", "#151514"
SEQ = [[0, "#1c1916"], [0.3, "#5a2a14"], [0.6, "#b4481a"], [0.85, "#e0621f"], [1, "#ffc9a8"]]
DIV = [[0, "#3987e5"], [0.25, "#2b5c99"], [0.5, "#2e2c28"], [0.75, "#a34a1d"], [1, "#e0621f"]]
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"

REG = ["ols", "ridge", "huber", "blr", "krr", "kglm"]
REG_NAMES = {"ols": "Least squares (MLE)", "ridge": "Ridge", "huber": "Huber (robust)", "blr": "Bayesian linear",
             "krr": "Kernel ridge (RBF)", "kglm": "Bayesian kernel GLM"}
CLS = ["lsq", "fisher", "generative", "logistic", "bayes_logistic", "svm"]
FEAT_LABEL = {"cement": "Cement", "slag": "Slag", "flyash": "Fly ash", "water": "Water", "sp": "Superplasticizer",
              "coarse": "Coarse agg.", "fine": "Fine agg.", "log_age": "ln(age)", "wc": "Water/cement", "wb": "Water/binder"}


def base_layout(**kw):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=INK2, size=12),
        margin=dict(l=58, r=18, t=16, b=50), colorway=PAL,
        hoverlabel=dict(bgcolor="#1f1d1a", bordercolor="#4a463d", font=dict(family=FONT, color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=axis(""), yaxis=axis(""), bargap=0.35)
    for k, v in kw.items():
        layout[k] = {**layout[k], **v} if isinstance(v, dict) and isinstance(layout.get(k), dict) else v
    return layout


def axis(title, **kw):
    return dict(title=dict(text=title, font=dict(color=MUTED, size=12)), gridcolor=GRID, linecolor=AXIS,
                zerolinecolor=AXIS, tickcolor=AXIS, ticks="outside", ticklen=4, **kw)


def graph(gid, title, cat, fig, what, how, read, where, insight, height=370):
    return {"id": gid, "title": title, "category": cat, "height": height, "figure": json.loads(fig.to_json()),
            "insight": insight, "explain": {"what": what, "how": how, "read": read, "where": where}}


def L(color, width=2.5, dash=None):
    d = dict(color=color, width=width)
    if dash:
        d["dash"] = dash
    return d


def mk(color, size=8):
    return dict(color=color, size=size, line=dict(color=SURF, width=1.5))


def build_all(R, B):
    D, RG, CL = R["data"], R["regression"], R["classification"]
    G = []
    grades = config.GRADES

    # ================================ DATA ================================
    y = np.array(D["strength"])
    fig = go.Figure(go.Histogram(x=y, xbins=dict(start=0, end=85, size=2.5), marker=dict(color=PAL[0], cornerradius=3),
                                 hovertemplate="%{x} MPa: %{y} mixes<extra></extra>", name="Mixes"))
    for i, g in enumerate(grades):
        fig.add_vline(x=g["target"], line=dict(color=INK if g["grade"] == "M30" else MUTED, width=1),
                      annotation_text=g["grade"], annotation_position="top", annotation_font=dict(color=INK2, size=10),
                      annotation_yshift=-4 - 14 * (i % 2))
    fig.update_layout(**base_layout(bargap=0.06, xaxis=axis("Compressive strength (MPa)"), yaxis=axis("Number of mixes"),
                                    showlegend=False))
    G.append(graph("strength_hist", "Distribution of compressive strength", "Data", fig,
        "How the measured compressive strength of all 1,005 unique lab mixes is spread, with the IS 10262 target mean "
        "strength of each grade marked.",
        "A histogram with 2.5 MPa bins. The vertical lines are the target mean strengths f'ck = fck + 1.65 s, where s is "
        "the assumed standard deviation from IS 10262:2019 Table 2.",
        "Everything to the right of a line passes that grade. The spread is wide (2 to 83 MPa) because the data mixes "
        "very young 3-day cubes with mature 365-day ones.",
        "Quality-control charts at ready-mix plants, where cube results are plotted against the target strength for "
        "every batch.",
        f"Median strength {np.median(y):.1f} MPa; {100 * np.mean(y >= 38.25):.0f}% of mixes reach the M30 target of 38.25 MPa."))

    sc = D["scatter"]
    fig = go.Figure(go.Scatter(x=sc["age"], y=sc["strength"], mode="markers",
                               marker=dict(size=7, color=sc["wb"], colorscale=[[0, "#ffc9a8"], [0.5, "#e0621f"], [1, "#3a1a0c"]],
                                           cmin=0.24, cmax=0.9, line=dict(color=SURF, width=1),
                                           colorbar=dict(title=dict(text="w/b"), thickness=12, outlinewidth=0)),
                               customdata=sc["wb"], hovertemplate="age %{x} d, %{y:.1f} MPa, w/b %{customdata:.2f}<extra></extra>"))
    fig.update_layout(**base_layout(xaxis=axis("Curing age (days)", type="log"), yaxis=axis("Strength (MPa)"), showlegend=False))
    G.append(graph("strength_age", "Strength grows with age, and falls with water", "Data", fig,
        "Every mix's strength against its curing age, coloured by its water-to-binder ratio.",
        "A scatter plot with a logarithmic age axis. Binder = cement + slag + fly ash.",
        "Strength climbs steeply in the first 28 days and then flattens, which is why the models use ln(age). At any age "
        "the pale dots (little water per kg of binder) sit higher. That is Abrams' law, the century-old rule behind every "
        "mix design.",
        "Mix-design charts in IS 10262 and ACI 211 plot exactly this: target strength against water/cement ratio.",
        "The strongest mixes combine low w/b with long curing."))

    corr = D["corr"]
    fig = go.Figure(go.Heatmap(z=corr["matrix"], x=corr["labels"], y=corr["labels"], colorscale=DIV, zmin=-1, zmax=1,
                               xgap=2, ygap=2, colorbar=dict(title=dict(text="ρ"), thickness=12, outlinewidth=0),
                               hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(**base_layout(margin=dict(l=150, r=10, t=10, b=130), xaxis=dict(tickangle=-40, showgrid=False),
                                    yaxis=dict(autorange="reversed", showgrid=False)))
    ki = corr["keys"].index("strength")
    G.append(graph("correlation", "Correlation matrix of the mix", "Data", fig,
        "Pearson correlation between every pair of ingredients, the engineered ratios and strength.",
        "ρ = Cov(X, Y) / (σₓ σᵧ), computed on the cleaned data.",
        "Orange cells rise together, blue cells move in opposite directions. Water, the w/c ratio and the w/b ratio are "
        "strongly correlated with each other. That collinearity makes plain least squares unstable and is why ridge "
        "regression is in the toolbox.",
        "Checking for multicollinearity before fitting any regression, and choosing which sensors or lab tests are redundant.",
        f"Cement has ρ = {corr['matrix'][ki][0]:.2f} with strength; w/b has ρ = {corr['matrix'][ki][corr['keys'].index('wb')]:.2f}."))

    pr = D["pass_rates"]
    ramp = ["#ffb38a", "#f5874f", "#e0621f", "#b4481a", "#8a3413"]
    fig = go.Figure()
    fig.add_bar(x=[p["grade"] for p in pr], y=[p["rate_all"] for p in pr], name="All ages", marker=dict(color=PAL[1], cornerradius=4),
                hovertemplate="%{x}: %{y:.0%} of all mixes<extra></extra>")
    fig.add_bar(x=[p["grade"] for p in pr], y=[p["rate_28"] for p in pr], name="Tested at 28 days", marker=dict(color=PAL[0], cornerradius=4),
                hovertemplate="%{x}: %{y:.0%} of 28-day cubes<extra></extra>")
    fig.update_layout(**base_layout(barmode="group", bargap=0.3, bargroupgap=0.12, yaxis=axis("Share that reaches the target", tickformat=".0%")))
    G.append(graph("pass_rates", "How often each grade's target is met", "Data", fig,
        "The fraction of mixes whose strength reaches each grade's target mean strength, for all ages and for 28-day cubes only.",
        "Count mixes with strength ≥ fck + 1.65 s and divide by the total.",
        "Higher grades are rarer, so the classes become imbalanced: about 70% pass M20 but only about 20% pass M40. "
        "M30 sits near 40/60, which makes it the main teaching grade.",
        "Estimating rejection rates before tendering for a project that specifies a grade.",
        f"M30 target {pr[2]['target']} MPa is met by {pr[2]['rate_28']:.0%} of 28-day cubes."))

    ac = D["age_counts"]
    fig = go.Figure(go.Bar(x=[str(k) for k in ac], y=list(ac.values()), marker=dict(color=PAL[1], cornerradius=4), width=0.6,
                           hovertemplate="%{x} days: %{y} mixes<extra></extra>"))
    fig.update_layout(**base_layout(xaxis=axis("Curing age (days)", type="category"), yaxis=axis("Mixes tested"), showlegend=False))
    G.append(graph("age_counts", "When were the cubes tested?", "Data", fig,
        "The number of lab results at each curing age.",
        "A count of the age column.",
        "The data clusters at standard test ages (3, 7, 28 days). 28 days dominates, matching IS 456's acceptance age. "
        "Ages past 100 days are sparse, so predictions there carry more uncertainty.",
        "Planning which test ages a lab schedule must cover.",
        f"{ac.get('28', ac.get(28, 0))} of {D['n_clean']} results are 28-day tests."))

    # ============================== REGRESSION ==============================
    ols = RG["ols"]
    fig = go.Figure()
    fig.add_scatter(x=list(range(1, len(ols["gd_losses"]) + 1)), y=ols["gd_losses"], mode="lines", line=L(PAL[0]),
                    name="Gradient descent", hovertemplate="epoch %{x}: MSE %{y:.2f}<extra></extra>")
    fig.add_hline(y=ols["closed_form_mse"], line=dict(color=INK, width=1), annotation_text="closed-form least squares",
                  annotation_font=dict(color=INK2), annotation_position="top right")
    fig.update_layout(**base_layout(xaxis=axis("Epoch"), yaxis=axis("Training MSE (MPa²)", type="log")))
    G.append(graph("gd_loss", "Training loss curve: gradient descent reaches the MLE", "Regression", fig,
        "The mean squared error of a linear model after each epoch of batch gradient descent.",
        "w ← w − η ∇E(w), with ∇E = Φᵀ(Φw − t)/N and η = 0.1, starting from w = 0. The flat line is "
        "the exact least-squares minimum from the normal equations w = (ΦᵀΦ)⁻¹Φᵀt.",
        "The loss falls fast and then crawls towards the closed-form minimum. Under Gaussian noise, minimising squared "
        "error is exactly maximum-likelihood estimation. Slow late progress comes from the correlated features "
        "(an ill-conditioned problem).",
        "Every neural network is trained by watching this curve. It is how you spot learning rates that are too high "
        "(the curve explodes) or too low (it barely moves).",
        f"After 300 epochs the training MSE is {ols['gd_losses'][-1]:.2f} vs the closed-form minimum {ols['closed_form_mse']:.2f}, "
        f"yet individual weights still differ by up to {ols['gd_vs_closed_max_diff']:.1f}: correlated features make the valley long and flat."))

    par = RG["parity"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[REG_NAMES[k] for k in REG], horizontal_spacing=0.07, vertical_spacing=0.17)
    for i, k in enumerate(REG):
        r, c = i // 3 + 1, i % 3 + 1
        fig.add_scatter(x=[0, 85], y=[0, 85], mode="lines", line=L(MUTED, 1), showlegend=False, hoverinfo="skip", row=r, col=c)
        fig.add_scatter(x=par["actual"], y=par[k], mode="markers", marker=mk(PAL[i], 5), showlegend=False, row=r, col=c,
                        hovertemplate="actual %{x:.1f}, predicted %{y:.1f}<extra>" + REG_NAMES[k] + "</extra>")
    fig.update_layout(**base_layout(margin=dict(l=44, r=10, t=36, b=40)))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, range=[0, 85])
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, range=[0, 85])
    fig.update_annotations(font=dict(color=INK, size=12))
    t = RG["test"]
    G.append(graph("parity", "Predicted vs actual strength for all six regressors", "Regression", fig,
        "For every test mix, the predicted strength against the measured one, for each model.",
        "Each model was trained on 704 mixes, tuned on 151 and scored once on 150 untouched test mixes.",
        "Perfect predictions sit on the grey diagonal. The four linear models (top row plus Bayesian linear) fan out "
        "because strength is not linear in the recipe. Both kernel models hug the line.",
        "Parity plots are the standard way to validate any regression model in engineering and chemistry.",
        " · ".join(f"{REG_NAMES[k]} R² {t[k]['r2']:.3f}" for k in REG), height=480))

    res = RG["residuals"]
    fig = go.Figure()
    fig.add_scatter(x=res["pred_ols"], y=res["ols"], mode="markers", marker=mk(PAL[0], 7), name="Least squares",
                    hovertemplate="pred %{x:.1f}: residual %{y:.1f}<extra>OLS</extra>")
    fig.add_scatter(x=res["pred_krr"], y=res["krr"], mode="markers", marker=mk(PAL[1], 7), name="Kernel ridge",
                    hovertemplate="pred %{x:.1f}: residual %{y:.1f}<extra>KRR</extra>")
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(**base_layout(xaxis=axis("Predicted strength (MPa)"), yaxis=axis("Residual: actual − predicted (MPa)")))
    G.append(graph("residuals", "Residual plot: what the linear model misses", "Regression", fig,
        "The error of each test prediction plotted against the prediction itself.",
        "Residual r = t − y(x). A good model leaves structureless noise around zero.",
        "The least-squares residuals curve (a smile), a sign of a missing non-linear effect. The kernel model's residuals "
        "form a tighter, flatter band.",
        "Regression diagnostics in statistics, econometrics and sensor calibration.",
        f"Residual SD: least squares {np.std(res['ols']):.2f} MPa vs kernel ridge {np.std(res['krr']):.2f} MPa."))

    r = np.array(res["ols"])
    sd = ols["noise_sd"]
    grid = np.linspace(-25, 25, 200)
    fig = go.Figure()
    fig.add_histogram(x=r, histnorm="probability density", xbins=dict(start=-25, end=25, size=2.5), name="Test residuals",
                      marker=dict(color=PAL[0], opacity=0.55, cornerradius=3))
    fig.add_scatter(x=grid, y=np.exp(-grid ** 2 / (2 * sd ** 2)) / math.sqrt(2 * math.pi * sd ** 2), mode="lines",
                    line=L(INK, 2), name=f"N(0, σ²_ML), σ = {sd:.2f}")
    fig.update_layout(**base_layout(bargap=0.05, xaxis=axis("Residual (MPa)"), yaxis=axis("Density")))
    G.append(graph("noise_model", "The Gaussian noise model behind least squares", "Regression", fig,
        "Test residuals of the least-squares model with the Gaussian noise density the model assumes.",
        "Maximum likelihood gives the noise variance 1/β_ML = (1/N) Σ (tₙ − w_MLᵀφₙ)² (Bishop eq. 3.21). The curve is that Gaussian.",
        "The residuals are roughly bell-shaped and centred at zero, so the Gaussian likelihood is a fair assumption. A "
        "few large residuals in the tails are what robust regression guards against.",
        "Checking noise assumptions before trusting confidence intervals in any regression report.",
        f"Estimated noise standard deviation: {sd:.2f} MPa."))

    rp = RG["ridge"]["path"]
    coefs = np.array(rp["coefs"])
    fig = go.Figure()
    feats = R["features"]["linear"]
    colors = PAL + [REF, "#6f6a60", "#d8d2c4", "#57524a"]
    for j, f in enumerate(feats):
        fig.add_scatter(x=rp["lambdas"], y=coefs[:, j], mode="lines", name=FEAT_LABEL[f], line=L(colors[j % len(colors)], 2),
                        hovertemplate="λ=%{x:.3g}: %{y:.2f}<extra>" + FEAT_LABEL[f] + "</extra>")
    fig.add_vline(x=RG["ridge"]["lambda"], line=dict(color=INK, width=1), annotation_text="CV choice", annotation_font=dict(color=INK))
    fig.update_layout(**base_layout(xaxis=axis("λ (log scale)", type="log"), yaxis=axis("Coefficient (per SD of feature)"),
                                    legend=dict(font=dict(size=10))))
    G.append(graph("ridge_path", "Ridge regularisation path", "Regression", fig,
        "How each coefficient shrinks as the ridge penalty λ grows.",
        "For each λ, solve (XᵀX + λI) w = Xᵀt on standardised features. A large λ pulls every weight towards zero.",
        "At small λ, the correlated water, w/c and w/b coefficients fight each other with large opposite signs. As "
        "λ grows they calm down and share the credit. Cross-validation picks the λ that predicts best.",
        "Ridge (Tikhonov) regularisation is standard in econometrics, genomics and any regression with correlated inputs.",
        f"Cross-validated λ = {RG['ridge']['lambda']:.3g}."))

    cv = RG["ridge"]["cv"]
    fig = go.Figure()
    lo = [c["cv_rmse"] - c["cv_se"] for c in cv]
    hi = [c["cv_rmse"] + c["cv_se"] for c in cv]
    lam = [c["lambda"] for c in cv]
    fig.add_scatter(x=lam + lam[::-1], y=hi + lo[::-1], fill="toself", fillcolor="rgba(224,98,31,.12)", line=dict(width=0),
                    name="±1 standard error", hoverinfo="skip")
    fig.add_scatter(x=lam, y=[c["cv_rmse"] for c in cv], mode="lines+markers", line=L(PAL[0]), marker=mk(PAL[0], 7),
                    name="5-fold CV RMSE", hovertemplate="λ=%{x:.3g}: %{y:.3f} MPa<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("λ (log scale)", type="log"), yaxis=axis("Cross-validated RMSE (MPa)")))
    G.append(graph("ridge_cv", "Choosing λ by cross-validation", "Regression", fig,
        "The average error of ridge regression on held-out folds for 36 values of λ.",
        "Split the 704 training mixes into 5 folds; for each λ train on 4 folds, test on the 5th, and average.",
        "The curve is flat for small λ (the penalty barely matters) and rises steeply once λ is so large that "
        "the model underfits. The best λ sits just before that rise.",
        "Hyperparameter tuning in every ML library (GridSearchCV, RidgeCV).",
        f"Minimum CV RMSE {min(c['cv_rmse'] for c in cv):.3f} MPa."))

    ct = RG["contamination"]
    fig = go.Figure()
    fig.add_scatter(x=[c["fraction"] for c in ct], y=[c["ols"] for c in ct], mode="lines+markers", name="Least squares",
                    line=L(PAL[0]), marker=mk(PAL[0]), hovertemplate="%{x:.1%} corrupted: RMSE %{y:.2f}<extra>OLS</extra>")
    fig.add_scatter(x=[c["fraction"] for c in ct], y=[c["huber"] for c in ct], mode="lines+markers", name="Huber (robust)",
                    line=L(PAL[1]), marker=mk(PAL[1]), hovertemplate="%{x:.1%} corrupted: RMSE %{y:.2f}<extra>Huber</extra>")
    fig.update_layout(**base_layout(xaxis=axis("Share of training results mistyped", tickformat=".0%"),
                                    yaxis=axis("Test RMSE on clean data (MPa)")))
    G.append(graph("robustness", "Robust regression under mistyped lab results", "Regression", fig,
        "Test error of least squares vs Huber regression when some training strengths are corrupted (multiplied by "
        "1.8–3×, like a decimal slip or a unit error).",
        "Corrupt a random fraction of training targets, refit both models, score on the clean test set, and average over "
        "6 random corruptions.",
        "Squared error lets a few huge residuals drag the whole fit, so least squares degrades quickly. The Huber loss grows "
        "only linearly for large residuals (IRLS weights them down), so it barely notices.",
        "Robust regression is used wherever data entry is manual (lab logs, surveys, sensor glitches) and in computer "
        "vision (RANSAC-style fitting).",
        f"At 20% corruption: least squares {ct[-1]['ols']:.1f} MPa vs Huber {ct[-1]['huber']:.1f} MPa."))

    hw = np.array(RG["huber"]["weights"])
    fig = go.Figure(go.Histogram(x=hw, xbins=dict(start=0, end=1.02, size=0.04), marker=dict(color=PAL[1], cornerradius=3),
                                 hovertemplate="weight %{x}: %{y} mixes<extra></extra>"))
    fig.update_layout(**base_layout(bargap=0.05, xaxis=axis("IRLS weight in the final Huber fit"), yaxis=axis("Training mixes", type="log"),
                                    showlegend=False))
    G.append(graph("huber_weights", "Which mixes the robust model trusted less", "Regression", fig,
        "The weight each training mix received in the last iteration of iteratively reweighted least squares.",
        "wᵢ = min(1, δ / |rᵢ/s|) with δ = 1.345 and s = 1.4826 × MAD(residuals). Weight 1 means \"treated as normal\".",
        "Most mixes keep full weight. The handful near zero are unusual results the robust fit refused to chase.",
        "Flagging suspicious records for re-testing; the same reweighting idea powers robust statistics in finance.",
        f"{RG['huber']['downweighted']} of {len(hw)} mixes were down-weighted; converged in {RG['huber']['iterations']} iterations."))

    bl = RG["blr"]
    ec = bl["evidence_curve"]
    fig = go.Figure(go.Scatter(x=ec["log_alpha"], y=ec["log_evidence"], mode="lines", line=L(PAL[3]),
                               hovertemplate="ln α = %{x:.2f}: ln evidence %{y:.1f}<extra></extra>"))
    fig.add_vline(x=math.log(bl["alpha"]), line=dict(color=INK, width=1), annotation_text="evidence maximum",
                  annotation_font=dict(color=INK))
    fig.update_layout(**base_layout(xaxis=axis("ln α (prior precision)"), yaxis=axis("ln p(t | α, β)"), showlegend=False))
    G.append(graph("blr_evidence", "Bayesian model selection: the evidence curve", "Regression", fig,
        "The marginal likelihood (evidence) of the Bayesian linear model as a function of the prior precision α.",
        "ln p(t|α,β) = (M/2) ln α + (N/2) ln β − E(mₙ) − ½ ln|A| − (N/2) ln 2π (Bishop eq. 3.86). "
        "α and β are re-estimated iteratively until they settle.",
        "Too small an α (a vague prior) wastes probability on implausible weights; too large an α forces the "
        "weights to zero. The peak balances fit against complexity using only training data, with no validation set needed.",
        "Empirical Bayes / type-II maximum likelihood tunes Gaussian processes and Bayesian neural networks.",
        f"α = {bl['alpha']:.4f}, β = {bl['beta']:.4f} (noise SD {bl['noise_sd']:.2f} MPa), "
        f"effective parameters γ = {bl['gamma']:.2f} of {len(bl['m'])}."))

    order = np.argsort(np.abs(bl["m"]))
    fig = go.Figure(go.Scatter(x=np.array(bl["m"])[order], y=[FEAT_LABEL[feats[i]] for i in order], mode="markers",
                               marker=mk(PAL[3], 10), error_x=dict(type="data", array=2 * np.array(bl["sd"])[order], color=PAL[3], thickness=2, width=0),
                               hovertemplate="%{y}: %{x:.2f} ± %{error_x.array:.2f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(**base_layout(xaxis=axis("Posterior mean ± 2 SD (MPa per SD of feature)"), yaxis=dict(showgrid=False),
                                    margin=dict(l=120, r=18, t=16, b=50), showlegend=False))
    G.append(graph("blr_posterior", "Posterior over the weights", "Regression", fig,
        "The Bayesian model's belief about each coefficient: the posterior mean with a ±2 standard-deviation band.",
        "Posterior N(w | mₙ, Sₙ) with Sₙ⁻¹ = αI + βΦᵀΦ and mₙ = βSₙΦᵀt.",
        "Bars that cross zero (water, superplasticizer) are weights the data cannot pin down, because a correlated feature "
        "carries the same information. ln(age) is large and certain: age matters and the data is sure of it.",
        "Uncertainty-aware coefficients support engineering decisions such as \"is adding fly ash definitely harmful?\"",
        "Widest bands: cement, water/binder and slag, which overlap (binder = cement + slag + fly ash); ln(age) is the most certain weight."))

    cov = RG["coverage"]
    fig = go.Figure()
    fig.add_scatter(x=[0.45, 1], y=[0.45, 1], mode="lines", line=L(MUTED, 1), name="Perfect", hoverinfo="skip")
    fig.add_scatter(x=cov["nominal"], y=cov["blr"], mode="lines+markers", line=L(PAL[3]), marker=mk(PAL[3]), name="Bayesian linear",
                    hovertemplate="%{x:.0%} interval covers %{y:.1%}<extra></extra>")
    fig.add_scatter(x=cov["nominal"], y=cov["kglm"], mode="lines+markers", line=L(PAL[5]), marker=mk(PAL[5]), name="Bayesian kernel GLM",
                    hovertemplate="%{x:.0%} interval covers %{y:.1%}<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("Nominal interval level", tickformat=".0%"), yaxis=axis("Share of test mixes inside", tickformat=".0%")))
    G.append(graph("coverage", "Are the uncertainty bands honest?", "Regression", fig,
        "For predictive intervals of 50% to 99%, how often the true strength of a test mix actually fell inside.",
        "The predictive distribution is N(mₙᵀφ(x), 1/β + φ(x)ᵀSₙφ(x)); count test mixes within ±z·σ of the mean.",
        "Points on the diagonal mean the stated uncertainty is trustworthy: a \"95% band\" really contains about 95% of outcomes.",
        "Risk decisions such as \"strip the formwork now?\" need honest intervals, not just point predictions.",
        f"95% intervals cover {cov['kglm'][4]:.1%} (kernel GLM) and {cov['blr'][4]:.1%} (linear) of test mixes."))

    kg = [g for g in RG["krr"]["grid"] if g["lambda"] == RG["krr"]["lambda"]]
    fig = go.Figure()
    fig.add_scatter(x=[g["gamma"] for g in kg], y=[g["train_r2"] for g in kg], mode="lines+markers", line=L(PAL[4]), marker=mk(PAL[4]),
                    name="Training R²", hovertemplate="γ=%{x:.3f}: %{y:.3f}<extra>train</extra>")
    fig.add_scatter(x=[g["gamma"] for g in kg], y=[g["val_r2"] for g in kg], mode="lines+markers", line=L(PAL[0]), marker=mk(PAL[0]),
                    name="Validation R²", hovertemplate="γ=%{x:.3f}: %{y:.3f}<extra>validation</extra>")
    fig.add_vline(x=RG["krr"]["gamma"], line=dict(color=INK, width=1))
    fig.update_layout(**base_layout(xaxis=axis("RBF kernel width γ (log scale)", type="log"), yaxis=axis("R²", range=[0.5, 1.01])))
    G.append(graph("kernel_validation", "Validation curve for the RBF kernel", "Kernels & SVM", fig,
        "Kernel ridge regression's fit on training and validation mixes as the kernel width γ changes.",
        "k(x, x') = exp(−γ‖x − x'‖²). Solve a = (K + λI)⁻¹t (the dual) and predict y(x) = k(x)ᵀa.",
        "Small γ means every mix looks similar (a smooth, nearly linear model). Large γ means each mix is only similar to "
        "itself: training R² reaches 1 while validation R² collapses (overfitting). The best γ sits between.",
        "The same bias–variance dial exists in every kernel method: SVMs, Gaussian processes, RBF networks.",
        f"Chosen γ = {RG['krr']['gamma']:.3f}, λ = {RG['krr']['lambda']}."))

    km = RG["kernel_matrix"]
    fig = go.Figure(go.Heatmap(z=km["K"], colorscale=SEQ, zmin=0, zmax=1, showscale=True,
                               colorbar=dict(title=dict(text="k(x,x')"), thickness=12, outlinewidth=0),
                               hovertemplate="mix %{y} vs mix %{x}: similarity %{z:.2f}<extra></extra>"))
    fig.update_layout(**base_layout(xaxis=axis("Training mix (sorted by strength →)", showticklabels=False, showgrid=False),
                                    yaxis=axis("Training mix", showticklabels=False, showgrid=False, autorange="reversed")))
    G.append(graph("kernel_matrix", "The kernel (Gram) matrix", "Kernels & SVM", fig,
        "RBF similarity between 120 training mixes, sorted from weakest to strongest.",
        "Kᵢⱼ = exp(−γ‖xᵢ − xⱼ‖²) on standardised recipes. This matrix is all the kernel trick ever needs: "
        "the model never builds feature vectors.",
        "Bright cells are similar recipes. Blocks along the diagonal show that mixes of similar strength have similar "
        "recipes, which is exactly the structure kernel regression exploits.",
        "Gram matrices appear in SVMs, Gaussian processes, spectral clustering and graph kernels for molecules.",
        "The kernel trick replaces an infinite feature space with this 704 × 704 matrix."))

    kc = RG["krr"]["kernels"]
    names = {"linear": "Linear kernel", "poly2": "Polynomial d=2", "poly3": "Polynomial d=3", "rbf": "RBF (Gaussian)"}
    fig = go.Figure(go.Bar(x=[names[k] for k in kc], y=[kc[k]["r2"] for k in kc], marker=dict(color=PAL[1], cornerradius=4), width=0.5,
                           text=[f"{kc[k]['r2']:.3f}" for k in kc], textposition="outside", textfont=dict(color=INK),
                           hovertemplate="%{x}: test R² %{y:.3f}<extra></extra>"))
    fig.update_layout(**base_layout(yaxis=axis("Test R²", range=[0, 1.05]), showlegend=False))
    G.append(graph("kernel_choice", "Choosing the kernel", "Kernels & SVM", fig,
        "Test R² of kernel ridge regression with four kernels on the same raw recipe features.",
        "Identical dual solver a = (K + λI)⁻¹t; only the similarity function k(x, x') changes.",
        "The linear kernel is ordinary ridge regression in disguise. Polynomial kernels add interactions (cement × age); "
        "the RBF kernel can bend in any direction and wins.",
        "Choosing kernels is feature engineering for kernel machines, e.g. string kernels for DNA or graph kernels for chemistry.",
        f"RBF {kc['rbf']['r2']:.3f} vs linear {kc['linear']['r2']:.3f}."))

    lc = RG["learning_curve"]
    fig = go.Figure()
    for key, name, col, dash in (("ols_train", "Linear, training", PAL[0], "dot"), ("ols_val", "Linear, validation", PAL[0], None),
                                 ("krr_train", "Kernel, training", PAL[1], "dot"), ("krr_val", "Kernel, validation", PAL[1], None)):
        fig.add_scatter(x=[l["n"] for l in lc], y=[l[key] for l in lc], mode="lines+markers", name=name, line=L(col, 2.5, dash),
                        marker=mk(col, 7), hovertemplate="n=%{x}: RMSE %{y:.2f}<extra>" + name + "</extra>")
    fig.update_layout(**base_layout(xaxis=axis("Training mixes"), yaxis=axis("RMSE (MPa)")))
    G.append(graph("learning_curve", "Learning curves: linear vs kernel", "Regression", fig,
        "Training and validation error of the linear and kernel models as the training set grows.",
        "Train on random subsets of 40 to 704 mixes (averaged over 5 draws) and score each on the validation set.",
        "The linear model's two curves meet early at a high error: it is limited by bias, and more data will not help. "
        "The kernel model keeps improving with data, a sign it is limited by variance and worth feeding more lab results.",
        "Deciding whether to buy more data or a more flexible model.",
        f"With all data: linear {lc[-1]['ols_val']:.2f} MPa vs kernel {lc[-1]['krr_val']:.2f} MPa validation RMSE."))

    fig = go.Figure(go.Bar(y=[REG_NAMES[k] for k in REG][::-1], x=[t[k]["rmse"] for k in REG][::-1], orientation="h",
                           marker=dict(color=[PAL[i] for i in range(6)][::-1], cornerradius=4), width=0.55,
                           text=[f"{t[k]['rmse']:.2f}" for k in REG][::-1], textposition="outside", textfont=dict(color=INK),
                           hovertemplate="%{y}: RMSE %{x:.2f} MPa<extra></extra>"))
    fig.update_layout(**base_layout(xaxis=axis("Test RMSE (MPa)"), yaxis=dict(showgrid=False), margin=dict(l=160, r=40, t=16, b=50), showlegend=False))
    G.append(graph("model_comparison", "Regression scoreboard", "Regression", fig,
        "Root-mean-square error on the 150 held-out test mixes for every regression model.",
        "RMSE = √(mean((t − y)²)), in MPa, computed once on data no model ever saw during training or tuning.",
        "The four linear models are within a few hundredths of each other: regularisation and robustness do not fix a "
        "wrong functional form. Kernels cut the error by roughly a third.",
        "Model-selection tables like this are the core of every ML paper's results section.",
        f"Best: {REG_NAMES[min(REG, key=lambda k: t[k]['rmse'])]} ({min(t[k]['rmse'] for k in REG):.2f} MPa)."))

    sk = RG["sklearn"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 85], y=[0, 85], mode="lines", line=L(MUTED, 1), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=sk["krr_sample"]["sklearn"], y=sk["krr_sample"]["ours"], mode="markers", marker=mk(PAL[2], 7),
                    name="Test mix", hovertemplate="sklearn %{x:.4f}<br>ours %{y:.4f}<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("scikit-learn KernelRidge (MPa)"), yaxis=axis("Our from-scratch kernel ridge (MPa)")))
    G.append(graph("sklearn_check", "From-scratch vs scikit-learn", "Regression", fig,
        "Predictions from the hand-written kernel ridge against scikit-learn's implementation with identical settings.",
        "Same kernel, γ and λ, same training data; compare predictions on every test mix.",
        "All points on the diagonal: the implementations agree to floating-point precision.",
        "Testing ML code against a reference library is standard engineering practice.",
        f"Max differences: least squares {sk['ols_max_diff']:.1e}, ridge {sk['ridge_max_diff']:.1e}, kernel ridge {sk['krr_max_diff']:.1e} MPa."))

    # ============================ CLASSIFICATION ============================
    M = CL["main"]
    fp = M["fisher"]
    fig = go.Figure()
    fig.add_histogram(x=fp["proj_fail"], name="Fails M30", marker=dict(color=PAL[1], opacity=0.6, cornerradius=3), nbinsx=36,
                      histnorm="probability density")
    fig.add_histogram(x=fp["proj_pass"], name="Passes M30", marker=dict(color=PAL[0], opacity=0.6, cornerradius=3), nbinsx=36,
                      histnorm="probability density")
    fig.update_layout(**base_layout(barmode="overlay", bargap=0.03, xaxis=axis("Projection y = wᵀx onto Fisher's direction"),
                                    yaxis=axis("Density")))
    G.append(graph("fisher_projection", "Fisher's linear discriminant: one number per mix", "Classification", fig,
        "Every training mix squashed onto the single direction that best separates passing from failing M30 mixes.",
        "w ∝ S_W⁻¹(m₁ − m₀): maximise (between-class separation)² / (within-class spread) (Bishop 4.1.4). "
        "Each class's projection is then modelled by a 1-D Gaussian.",
        "Two humps with little overlap mean the 10-dimensional recipe collapses into one very informative score. The "
        "decision threshold sits where the class posteriors cross.",
        "Face recognition (Fisherfaces), credit scoring and early spectroscopy classifiers.",
        f"Fisher's w is parallel to the generative model's weights (cosine {CL['sklearn']['fisher_vs_generative_cosine']:.4f}), as theory predicts."))

    Bm = B["models"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[CL["names"][k] for k in CLS], horizontal_spacing=0.06, vertical_spacing=0.14)
    pts = B["points"]
    lab = np.array(pts["label"])
    for i, k in enumerate(CLS):
        rr, cc = i // 3 + 1, i % 3 + 1
        fig.add_heatmap(x=B["x"], y=B["y"], z=Bm[k]["z"], colorscale=DIV, zmin=0, zmax=1, zsmooth="best", showscale=False,
                        opacity=0.9, row=rr, col=cc, showlegend=False,
                        hovertemplate="w/b %{x:.2f}, age %{y:.0f} d: P(pass) %{z:.2f}<extra>" + CL["names"][k] + "</extra>")
        fig.add_contour(x=B["x"], y=B["y"], z=Bm[k]["z"], showscale=False, contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                        line=dict(color=INK, width=2), hoverinfo="skip", row=rr, col=cc, showlegend=False)
        for lv, col in ((0, "#8fb8f0"), (1, "#ffb38a")):
            m_ = lab == lv
            fig.add_scatter(x=np.array(pts["wb"])[m_], y=np.array(pts["age"])[m_], mode="markers", showlegend=False,
                            marker=dict(size=4, color=col, line=dict(color=SURF, width=0.5)), hoverinfo="skip", row=rr, col=cc)
    fig.update_layout(**base_layout(margin=dict(l=48, r=10, t=36, b=40)))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, range=[0.23, 0.95])
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, type="log", range=[0, math.log10(365)])
    fig.update_annotations(font=dict(color=INK, size=12))
    G.append(graph("decision_boundaries", "Decision boundaries of all six classifiers", "Classification", fig,
        "Each classifier retrained on just two features, water/binder ratio (x) and curing age (y, log scale), with its "
        "P(passes M30) shaded and its decision boundary drawn in white.",
        "Every model is fitted from scratch on the 704 training mixes; the shading is its output on a 90 × 90 grid.",
        "Orange = likely to pass, blue = likely to fail. Five boundaries are straight lines, because those models are "
        "linear in the features. Only the RBF SVM can curve its boundary around the data.",
        "Visualising boundaries is how practitioners sanity-check a classifier before trusting it.",
        " · ".join(f"{CL['names'][k].split(' (')[0]} {Bm[k]['train_acc']:.0%}" for k in CLS), height=520))

    h = M["logistic_history"]
    fig = go.Figure(go.Scatter(x=list(range(len(h))), y=h, mode="lines+markers", line=L(PAL[3]), marker=mk(PAL[3]),
                               hovertemplate="iteration %{x}: −ln L = %{y:.3f}<extra></extra>"))
    fig.update_layout(**base_layout(xaxis=axis("IRLS (Newton–Raphson) iteration", dtick=1), yaxis=axis("Negative log-likelihood", type="log"),
                                    showlegend=False))
    G.append(graph("irls", "Training logistic regression by IRLS", "Classification", fig,
        "The cross-entropy loss of logistic regression after each Newton–Raphson step.",
        "w⁽ⁿᵉʷ⁾ = w⁽ᵒˡᵈ⁾ − (ΦᵀRΦ)⁻¹Φᵀ(y − t), with R = diag(yₙ(1 − yₙ)), which is a weighted least-squares problem re-solved "
        "each step (Bishop 4.3.3).",
        "Newton's method uses curvature, so it converges in a handful of iterations where gradient descent would need thousands.",
        "IRLS is how R's glm() and many statistics packages fit logistic and Poisson regression.",
        f"Converged in {len(h)} iterations from {h[0]:.1f} to {h[-1]:.1f}."))

    le = CL["laplace_evidence"]
    fig = go.Figure(go.Scatter(x=le["alpha"], y=le["log_evidence"], mode="lines+markers", line=L(PAL[4]), marker=mk(PAL[4], 7),
                               hovertemplate="α=%{x:.3g}: %{y:.2f}<extra></extra>"))
    fig.add_vline(x=le["best_alpha"], line=dict(color=INK, width=1), annotation_text="chosen α", annotation_font=dict(color=INK))
    fig.update_layout(**base_layout(xaxis=axis("Prior precision α (log scale)", type="log"), yaxis=axis("Laplace log-evidence"), showlegend=False))
    G.append(graph("laplace_evidence", "Laplace approximation to the evidence", "Classification", fig,
        "The approximate marginal likelihood of Bayesian logistic regression for different prior strengths α.",
        "ln p(t|α) ≈ ln p(t|w_MAP) + ln p(w_MAP) + (M/2) ln 2π − ½ ln|A|, where A is the Hessian of the negative "
        "log-posterior at the MAP (Bishop 4.4.1).",
        "The Laplace approximation replaces an intractable integral with a Gaussian around the peak; its evidence still "
        "trades fit against complexity, so the peak picks α without a validation set.",
        "Laplace approximations are used in Bayesian deep learning (e.g. uncertainty for pretrained networks) and INLA.",
        f"Best α = {le['best_alpha']:.3f}."))

    mo = M["moderation"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=L(MUTED, 1), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=mo["map"], y=mo["bayes"], mode="markers", name="Test mix",
                    marker=dict(size=8, color=mo["sd"], colorscale=[[0, "#3a1a0c"], [1, "#ffb38a"]], line=dict(color=SURF, width=1),
                                colorbar=dict(title=dict(text="σₐ"), thickness=12, outlinewidth=0)),
                    hovertemplate="MAP %{x:.3f} → Bayesian %{y:.3f}<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("P(pass) using only w_MAP"), yaxis=axis("P(pass) averaged over the posterior"), showlegend=False))
    G.append(graph("moderation", "Bayesian averaging moderates confidence", "Classification", fig,
        "Each test mix's pass probability from the single best weights vs from the full Laplace posterior.",
        "Predictive p ≈ σ(κ(σₐ²) μₐ) with κ = (1 + πσₐ²/8)⁻¹‧² (the probit approximation, Bishop 4.5.2). "
        "Colour shows σₐ, the uncertainty in the activation.",
        "Averaging over the posterior pulls every probability towards 50% by an amount that grows with σₐ, and it never "
        "moves the decision boundary. Here the points hug the diagonal: with 704 training mixes the posterior is narrow, "
        "so the correction is small. With only a few dozen mixes the same plot bends visibly.",
        "Calibrated, uncertainty-aware probabilities matter in medicine, credit and active learning (querying where σ is large).",
        f"Largest change from averaging: {np.max(np.abs(np.array(mo['map']) - np.array(mo['bayes']))):.3f} in probability."))

    sg = CL["svm_grid"]
    Cs = sorted({g["C"] for g in sg})
    gs = sorted({g["gamma"] for g in sg})
    Z = [[next(r["cv_acc"] for r in sg if r["C"] == C_ and r["gamma"] == g_) for C_ in Cs] for g_ in gs]
    fig = go.Figure(go.Heatmap(z=Z, x=[str(c) for c in Cs], y=[str(g) for g in gs], colorscale=SEQ, xgap=3, ygap=3,
                               text=[[f"{v:.1%}" for v in row] for row in Z], texttemplate="%{text}", textfont=dict(color=INK, size=11),
                               colorbar=dict(title=dict(text="CV acc"), thickness=12, outlinewidth=0),
                               hovertemplate="C=%{x}, γ=%{y}: %{z:.1%}<extra></extra>"))
    fig.update_layout(**base_layout(xaxis=axis("C (margin softness)", type="category", showgrid=False),
                                    yaxis=axis("γ (kernel width)", type="category", showgrid=False)))
    sp = CL["svm_params"]
    G.append(graph("svm_grid", "SVM hyperparameter search (C × γ)", "Kernels & SVM", fig,
        "5-fold cross-validated accuracy of the RBF SVM for 30 combinations of C and γ (150 SMO trainings).",
        "For each pair, train the from-scratch SMO solver on 4 folds and test on the 5th.",
        "C controls how much margin violation is tolerated; γ controls how local the kernel is. The good region is a "
        "diagonal ridge, because large C with a very local kernel overfits.",
        "Grid search over (C, γ) is the standard recipe for training SVMs (the LIBSVM guide recommends exactly this).",
        f"Best: C = {sp['C']}, γ = {sp['gamma']} ({sp['cv_acc']:.1%} CV accuracy)."))

    sv = M["svm"]
    fig = go.Figure(go.Scatter(x=[i * 50 for i in range(len(sv["gaps"]))], y=sv["gaps"], mode="lines", line=L(PAL[5]),
                               hovertemplate="iteration %{x}: gap %{y:.4f}<extra></extra>"))
    fig.add_hline(y=1e-3, line=dict(color=INK, width=1), annotation_text="stopping tolerance", annotation_font=dict(color=INK))
    fig.update_layout(**base_layout(xaxis=axis("SMO iteration"), yaxis=axis("KKT violation (max − min)", type="log"), showlegend=False))
    G.append(graph("smo_convergence", "SMO convergence: solving the SVM dual", "Kernels & SVM", fig,
        "How far the SVM solution is from optimal (the largest KKT violation) as SMO works through pairs of multipliers.",
        "Each step picks the most violating pair (aᵢ, aⱼ) by second-order working-set selection, solves the 2-variable "
        "quadratic program analytically and clips to 0 ≤ a ≤ C.",
        "The violation drops by orders of magnitude; training stops below 10⁻³, the same default tolerance LIBSVM uses.",
        "SMO is the algorithm inside LIBSVM, which powers scikit-learn's SVC.",
        f"{sv['iterations']} iterations; {sv['n_sv']} support vectors ({sv['n_bounded']} at the bound C). "
        f"scikit-learn finds {CL['sklearn']['svm_n_sv'][1]}."))

    gd = CL["gen_vs_disc"]
    fig = go.Figure()
    fig.add_scatter(x=[g["n"] for g in gd], y=[g["generative"] for g in gd], mode="lines+markers", name="Generative (Gaussian, shared Σ)",
                    line=L(PAL[2]), marker=mk(PAL[2]), hovertemplate="n=%{x}: %{y:.1%}<extra>generative</extra>")
    fig.add_scatter(x=[g["n"] for g in gd], y=[g["discriminative"] for g in gd], mode="lines+markers", name="Discriminative (logistic)",
                    line=L(PAL[3]), marker=mk(PAL[3]), hovertemplate="n=%{x}: %{y:.1%}<extra>logistic</extra>")
    fig.update_layout(**base_layout(xaxis=axis("Training mixes (log scale)", type="log"), yaxis=axis("Test accuracy", tickformat=".0%")))
    G.append(graph("gen_vs_disc", "Generative vs discriminative learning", "Classification", fig,
        "Test accuracy of the Gaussian generative classifier and logistic regression as the training set grows.",
        "Both fitted from scratch on random subsets (20 draws each) and scored on the same test set.",
        "Theory (Ng & Jordan, 2002) says a generative model can learn faster from very few examples, because it also "
        "models how the data is spread. Here the opposite happens at the smallest sizes: the generative model must estimate "
        "a full 10 × 10 covariance matrix, which is unreliable from a handful of mixes, so logistic regression leads. "
        "From about 100 mixes the two are level, and both finish at the same 90%.",
        "Choosing between Naive Bayes / LDA-style models and logistic models when labelled data is scarce.",
        f"At n = {gd[0]['n']}: generative {gd[0]['generative']:.1%} vs logistic {gd[0]['discriminative']:.1%}; "
        f"at n = {gd[-1]['n']} both reach {gd[-1]['generative']:.0%}."))

    names7 = CL["names"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=L(MUTED, 1), name="Random", hoverinfo="skip")
    res30 = CL["grades"]["M30"]["results"]
    for i, k in enumerate(CLS + ["kglm"]):
        col = PAL[i] if i < 6 else REF
        fig.add_scatter(x=M["roc"][k]["fpr"], y=M["roc"][k]["tpr"], mode="lines", line=L(col, 2.2),
                        name=f"{names7[k].split(' (')[0]} ({res30[k]['auc']:.3f})",
                        hovertemplate="FPR %{x:.2f}, TPR %{y:.2f}<extra>" + names7[k] + "</extra>")
    fig.update_layout(**base_layout(xaxis=axis("False-positive rate"), yaxis=axis("True-positive rate"), legend=dict(font=dict(size=10))))
    G.append(graph("roc", "ROC curves: will the mix pass M30?", "Classification", fig,
        "True-positive vs false-positive rate for every threshold, for all seven ways of predicting M30 compliance.",
        "Sort test mixes by each model's score and sweep the threshold; AUC comes from the trapezoid rule.",
        "All models sit far above the diagonal. AUC is the chance a random passing mix scores higher than a random failing one.",
        "ROC analysis compares diagnostic and screening systems; here it compares ways of screening mix designs.",
        f"Best AUC: {names7[max(res30, key=lambda k: res30[k]['auc'])]} ({max(v['auc'] for v in res30.values()):.3f})."))

    fig = go.Figure()
    for i, k in enumerate(CLS + ["kglm"]):
        col = PAL[i] if i < 6 else REF
        fig.add_scatter(x=M["pr"][k]["recall"], y=M["pr"][k]["precision"], mode="lines", line=L(col, 2.2), name=names7[k].split(" (")[0],
                        hovertemplate="recall %{x:.2f}, precision %{y:.2f}<extra>" + names7[k] + "</extra>")
    fig.update_layout(**base_layout(xaxis=axis("Recall"), yaxis=axis("Precision", range=[0, 1.02]), legend=dict(font=dict(size=10))))
    G.append(graph("pr_curve", "Precision–recall curves", "Classification", fig,
        "For every threshold, the share of predicted passes that truly pass (precision) against the share of true passes found (recall).",
        "Precision = TP/(TP+FP), recall = TP/(TP+FN), evaluated at every distinct score.",
        "For a site engineer precision matters most: a mix predicted to pass had better really pass.",
        "Quality inspection, defect detection and search ranking.",
        f"M30 test pass rate is {CL['grades']['M30']['test_pass_rate']:.0%}, the precision of random guessing."))

    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=L(MUTED, 1), name="Perfect", hoverinfo="skip")
    for i, k in enumerate(CLS + ["kglm"]):
        if k not in M["calibration"]:
            continue
        col = PAL[i] if i < 6 else REF
        cb = M["calibration"][k]
        fig.add_scatter(x=cb["mean_predicted"], y=cb["observed"], mode="lines+markers", line=L(col, 2), marker=mk(col, 7),
                        name=names7[k].split(" (")[0], hovertemplate="predicted %{x:.2f} → observed %{y:.2f}<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("Predicted P(pass)"), yaxis=axis("Observed pass rate"), legend=dict(font=dict(size=10))))
    G.append(graph("calibration", "Calibration of the probabilistic classifiers", "Classification", fig,
        "Whether \"70% chance to pass\" really means 7 in 10 such mixes pass.",
        "Test mixes sorted by predicted probability, cut into 8 equal groups; mean prediction vs observed pass rate.",
        "Curves near the diagonal are trustworthy probabilities. The SVM has no native probability; its curve uses Platt "
        "scaling (a 1-D logistic regression fitted on the validation margins).",
        "Any decision based on expected cost (reject a batch? re-test?) needs calibrated probabilities.",
        f"Lowest log-loss on M30: {names7[min((k for k in res30 if 'log_loss' in res30[k]), key=lambda k: res30[k]['log_loss'])]}."))

    fig = go.Figure()
    for i, k in enumerate(CLS):
        fig.add_bar(x=[g["grade"] for g in grades], y=[CL["grades"][g["grade"]]["results"][k]["accuracy"] for g in grades],
                    name=names7[k].split(" (")[0], marker=dict(color=PAL[i], cornerradius=3),
                    hovertemplate="%{x}: %{y:.1%}<extra>" + names7[k] + "</extra>")
    fig.update_layout(**base_layout(barmode="group", bargap=0.25, bargroupgap=0.08,
                                    yaxis=axis("Test accuracy", tickformat=".0%", range=[0.6, 1]), legend=dict(font=dict(size=10))))
    G.append(graph("accuracy_by_grade", "Accuracy for every grade, M20 to M40", "Classification", fig,
        "Test accuracy of each classifier, retrained for each grade's target mean strength.",
        "Same features and hyperparameters; only the label (strength ≥ fck + 1.65s) changes per grade.",
        "Accuracy looks higher for M40 partly because few mixes pass, so predicting \"fail\" is easy. Always read accuracy "
        "next to the pass rate.",
        "Specification compliance checks in concrete QC, where each project specifies its own grade.",
        "The app lets you pick any grade and uses that grade's trained models."))

    w = M["weights"]
    fig = go.Figure()
    for i, (k, lab_) in enumerate((("logistic", "Logistic regression"), ("bayes_logistic", "Bayesian logistic (MAP)"),
                                   ("generative", "Gaussian generative"))):
        vec = np.array(w[k])
        vec = vec / np.max(np.abs(vec))
        fig.add_bar(y=[FEAT_LABEL[f] for f in w["features"]], x=vec, orientation="h", name=lab_,
                    marker=dict(color=PAL[[3, 4, 2][i]], cornerradius=3), hovertemplate="%{y}: %{x:.2f}<extra>" + lab_ + "</extra>")
    fig.update_layout(**base_layout(barmode="group", bargap=0.25, xaxis=axis("Weight (scaled to max |w| = 1)"),
                                    yaxis=dict(showgrid=False), margin=dict(l=120, r=18, t=16, b=50), legend=dict(font=dict(size=10))))
    G.append(graph("weights", "What each linear classifier learned", "Classification", fig,
        "The weight each linear classifier puts on each (standardised) feature for M30, scaled for comparison.",
        "Logistic weights come from IRLS, Bayesian weights from the MAP under a Gaussian prior, and generative weights "
        "from w = Σ⁻¹(μ₁ − μ₀).",
        "Positive = pushes towards passing. ln(age) dominates on the positive side and water/binder on the negative side. "
        "Smaller weights such as slag and fly ash change sign between models, a symptom of correlated inputs. The Bayesian "
        "prior shrinks the most extreme weights.",
        "Interpreting linear models is how engineers check that an ML model agrees with physics (Abrams' law).",
        "All three agree on the two biggest effects: ln(age) up, water/binder down."))
    return G
