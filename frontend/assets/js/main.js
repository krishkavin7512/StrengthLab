import { reduced, whenVisible, tween, fmt, countTo, initReveals, splitWords, magnetic, tilt, spotlight, cursor, marquee,
  scramble, slidingPill, setRangeFill, debounce, toast } from "./fx.js";
import { heroCube } from "./cube.js";
import { C, PAL, layout, axis, render, CONFIG } from "./charts.js";

// ---------------------------------------------------------------- capture mode (for report screenshots)
// ?capture=<section id|hero|footer>&set=<selector>=<value>&click=<selector>|<selector>
const Q = new URLSearchParams(location.search);
const CAPTURE = Q.get("capture");
if (CAPTURE) {
  document.documentElement.classList.add("capture");
  const note = (m) => { document.documentElement.dataset.err = (document.documentElement.dataset.err || "") + " | " + m; };
  addEventListener("error", (e) => note(e.message));
  addEventListener("unhandledrejection", (e) => note(String(e.reason && (e.reason.stack || e.reason))));
  try { sessionStorage.setItem("sl-intro", "1"); } catch {}
  const t = CAPTURE === "hero" ? document.querySelector(".hero") : CAPTURE === "footer" ? document.querySelector("footer") : document.getElementById(CAPTURE);
  t?.classList.add("cap-target");
}
async function runCaptureActions() {
  if (!CAPTURE) return;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  await wait(700);
  for (const s of Q.getAll("set")) {
    const i = s.lastIndexOf("=");
    const el = document.querySelector(s.slice(0, i));
    if (el) { el.value = s.slice(i + 1); el.dispatchEvent(new Event("input")); el.dispatchEvent(new Event("change")); await wait(500); }
  }
  for (const sel of (Q.get("click") || "").split("|").filter(Boolean)) { document.querySelector(sel)?.click(); await wait(900); }
  document.documentElement.classList.add("capture-ready");
}

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = (path, opts) => fetch(path, opts).then((r) => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json(); });

const REG = ["ols", "ridge", "huber", "blr", "krr", "kglm"];
const REG_NAMES = { ols: "Least squares (MLE)", ridge: "Ridge", huber: "Huber (robust)", blr: "Bayesian linear", krr: "Kernel ridge (RBF)", kglm: "Bayesian kernel GLM" };
const CLS = ["lsq", "fisher", "generative", "logistic", "bayes_logistic", "svm"];
const FEAT = { cement: "Cement", slag: "Slag", flyash: "Fly ash", water: "Water", sp: "Superplasticizer", coarse: "Coarse agg.", fine: "Fine agg.", log_age: "ln(age)", wc: "Water/cement", wb: "Water/binder" };
const REG_EQ = {
  ols: "Maximum likelihood under Gaussian noise ⇒ minimise Σ(tₙ − wᵀφₙ)²<br><b>w_ML = (ΦᵀΦ)⁻¹Φᵀt</b>,  1/β_ML = mean squared residual",
  ridge: "Penalise large weights:  E(w) = ½Σ(tₙ − wᵀφₙ)² + (λ/2)‖w‖²<br><b>w = (ΦᵀΦ + λI)⁻¹Φᵀt</b>,  λ chosen by 5-fold CV",
  huber: "Huber loss: quadratic for small residuals, linear for large ones<br><b>IRLS: w ← (ΦᵀWΦ)⁻¹ΦᵀWt,  Wₙ = min(1, δ/|rₙ/s|)</b>",
  blr: "Prior w ~ N(0, α⁻¹I), noise precision β, both set by the evidence<br><b>S_N⁻¹ = αI + βΦᵀΦ,  m_N = βS_NΦᵀt</b>",
  krr: "The kernel trick: never build φ(x), only k(x,x') = exp(−γ‖x−x'‖²)<br><b>a = (K + λI)⁻¹t,  y(x) = k(x)ᵀa</b>",
  kglm: "Kernels inside a GLM: φ(x) = [k(x,μ₁) … k(x,μ_K)], then Bayesian LR<br><b>p(t|x) = N(m_Nᵀφ(x), 1/β + φ(x)ᵀS_Nφ(x))</b>",
};
const CLS_EQ = {
  lsq: "Least squares on targets ±1 (Bishop 4.1.3)<br><b>w = Φ†t,  class = sign(wᵀφ)</b><br>Simple, but it chases outliers.",
  fisher: "Fisher's linear discriminant (Bishop 4.1.4)<br><b>w ∝ S_W⁻¹(m₁ − m₀)</b><br>The best 1-D projection, then 1-D Gaussians + Bayes.",
  generative: "Generative model: p(x|C_k) = N(μ_k, Σ) with a shared Σ<br><b>P(C₁|x) = σ(wᵀx + w₀),  w = Σ⁻¹(μ₁ − μ₀)</b>",
  logistic: "Discriminative model fitted by IRLS (Newton–Raphson)<br><b>w ← w − (ΦᵀRΦ)⁻¹Φᵀ(y − t)</b>",
  bayes_logistic: "Laplace approximation: posterior ≈ N(w_MAP, S_N)<br><b>P(C₁|x) ≈ σ(κ(σₐ²)·μₐ),  κ = (1 + πσₐ²/8)^−½</b>",
  svm: "Soft-margin SVM with an RBF kernel, trained by SMO<br><b>y(x) = Σ aₙtₙk(x,xₙ) − ρ,  0 ≤ aₙ ≤ C</b><br>Only the support vectors matter.",
};

// ---------------------------------------------------------------- boot
const data = Promise.all([api("/api/overview"), api("/api/regression/lab"), api("/api/data/summary")]);
runLoader(data).then(() => { document.body.classList.remove("is-loading"); $("#heroTitle").classList.add("in"); });
splitWords($("#heroTitle"), 250);
$$("[data-split]").forEach((h) => splitWords(h));
$$(".split em .w > span, .split .grad .w > span").forEach((s) => s.classList.add("grad-text"));
cursor();
$$(".magnet, .nav__cta .btn").forEach((b) => magnetic(b));
$$(".card").forEach(spotlight);
initNav();
buildMarquee();
heroCube($("#cubeStage"));

data.then(([ov, lab, summary]) => {
  initReveals();
  $("#heroR2").textContent = ov.regression_test.krr.r2.toFixed(2);
  $("#heroLinR2").textContent = ov.regression_test.ols.r2.toFixed(2);
  $("#heroRmse").textContent = ov.regression_test.krr.rmse.toFixed(1);
  initData(ov, summary);
  initRegression(ov, lab);
  initKernel();
  initClassifiers(ov);
  initDesigner(ov);
  initReport(ov);
  runCaptureActions();
}).catch((e) => { console.error(e); toast("Could not reach the backend. Is `python run.py` running?"); });

// ---------------------------------------------------------------- loader
async function runLoader(ready) {
  const loader = $("#loader");
  let seen = false;
  try { seen = sessionStorage.getItem("sl-intro") === "1"; } catch {}
  if (reduced || seen) { loader.remove(); await ready.catch(() => {}); return; }
  const ram = $("#ram"), val = $("#lpVal"), msg = $("#lpMsg"), press = $("#press"), cube = $("#pcube");
  msg.textContent = "loading ram";
  await tween(-40, 0, 650, (v) => { ram.style.transform = `translateY(${v}px)`; });
  msg.textContent = "applying load";
  await tween(0, 38.3, 1500, (v) => {
    val.textContent = v.toFixed(1);
    cube.style.transform = `scaleY(${1 - v * 0.0006})`;
    ram.style.transform = `translateY(${v * 0.02}px)`;
  });
  await ready.catch(() => {});
  press.classList.add("cracked", "press-shake");
  msg.textContent = "failure load reached";
  await new Promise((r) => setTimeout(r, 650));
  loader.classList.add("done");
  try { sessionStorage.setItem("sl-intro", "1"); } catch {}
  setTimeout(() => loader.remove(), 1100);
  await new Promise((r) => setTimeout(r, 300));
}

// ---------------------------------------------------------------- nav + chrome
function initNav() {
  const nav = $("#nav"), links = $("#navLinks"), pill = $(".nav__pill", links), progress = $("#progress");
  let lastY = scrollY;
  const movePill = () => {
    const a = $("a.active", links);
    if (!a) { pill.style.width = "0px"; return; }
    pill.style.width = a.offsetWidth + "px";
    pill.style.transform = `translateX(${a.offsetLeft}px)`;
  };
  addEventListener("scroll", () => {
    const y = scrollY, max = document.documentElement.scrollHeight - innerHeight;
    progress.style.transform = `scaleX(${max > 0 ? y / max : 0})`;
    nav.classList.toggle("hidden", y > lastY && y > 400);
    lastY = y;
  }, { passive: true });
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      $$("a", links).forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
      movePill();
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  $$("main section[id]").forEach((s) => io.observe(s));
  addEventListener("resize", movePill);
}

function buildMarquee() {
  const words = ["Least squares", "Ridge", "Huber", "Bayesian regression", "Kernel trick", "Fisher", "Logistic", "Laplace", "SVM"];
  const track = $("#marquee");
  const html = words.map((w) => `<span>${w} <b>&#9632;</b></span>`).join("");
  track.innerHTML = html + html;
  marquee(track);
}

// ---------------------------------------------------------------- 01 data
function initData(ov, summary) {
  const tabs = $("#dataTabs");
  const move = slidingPill(tabs, $(".tabs__pill", tabs));
  $$("button", tabs).forEach((b) => b.addEventListener("click", () => {
    $$("button", tabs).forEach((x) => x.classList.toggle("active", x === b));
    move();
    $("#tab-rows").hidden = b.dataset.tab !== "rows";
    $("#tab-summary").hidden = b.dataset.tab !== "summary";
  }));
  let offset = 0, total = 1005;
  const limit = 14;
  const load = async () => {
    const p = await api(`/api/data/preview?offset=${offset}&limit=${limit}`);
    total = p.total;
    const t = document.createElement("table");
    t.className = "data";
    const h = t.createTHead().insertRow();
    ["#", "Cement", "Slag", "Fly ash", "Water", "SP", "Coarse", "Fine", "Age (d)", "Strength (MPa)"].forEach((c) => { const th = document.createElement("th"); th.textContent = c; h.appendChild(th); });
    const body = t.createTBody();
    p.rows.forEach((row, i) => {
      const tr = body.insertRow();
      tr.style.setProperty("--r", i);
      tr.insertCell().textContent = offset + i + 1;
      row.forEach((v, j) => {
        const td = tr.insertCell();
        if (j === row.length - 1) td.innerHTML = `<b style="color:${v >= 38.25 ? "var(--pass)" : "var(--ink)"}">${v.toFixed(2)}</b>`;
        else td.textContent = v;
      });
    });
    $("#rowsTable").replaceChildren(t);
    $("#pagerInfo").textContent = `Mixes ${offset + 1}–${offset + p.rows.length} of ${total} (green = passes M30)`;
  };
  $("#prevPage").onclick = () => { offset = Math.max(0, offset - limit); load(); };
  $("#nextPage").onclick = () => { offset = Math.min(total - limit, offset + limit); load(); };
  $("#randPage").onclick = () => { offset = Math.floor(Math.random() * (total - limit)); load(); };
  load();

  const t = document.createElement("table");
  t.className = "data";
  const h = t.createTHead().insertRow();
  ["Column", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max", "Zeros", "Skew"].forEach((c) => { const th = document.createElement("th"); th.textContent = c; h.appendChild(th); });
  const b = t.createTBody();
  summary.summary.forEach((r, i) => {
    const tr = b.insertRow();
    tr.style.setProperty("--r", i);
    tr.insertCell().innerHTML = `<b style="color:var(--ink)">${r.column}</b><br><span class="subtle">${r.label}</span>`;
    [r.mean, r.std, r.min, r.q25, r.median, r.q75, r.max].forEach((v) => { tr.insertCell().textContent = v >= 100 ? v.toFixed(1) : v.toFixed(3); });
    tr.insertCell().textContent = r.zeros;
    tr.insertCell().textContent = r.skewness.toFixed(2);
  });
  $("#summaryTable").replaceChildren(t);

  $("#grades").innerHTML = ov.grades.map((g, i) => {
    const rate = ov.data.pass_rates[i];
    return `<div class="grade" style="--i:${i};--w:${(g.target / 50) * 100}%"><span class="spec">Grade</span><b>${g.grade}</b>
      <div class="bar"><i></i></div><div class="sub">fck ${g.fck} + 1.65×${g.s} = <b style="color:var(--acc)">${g.target}</b> MPa</div>
      <div class="sub">${(rate.rate_28 * 100).toFixed(0)}% of 28-day cubes pass</div></div>`;
  }).join("");
}

// ---------------------------------------------------------------- 02 regression
function initRegression(ov, lab) {
  const list = $("#regModels"), el = $("#parityPlot");
  let current = "ols", drawn = false;
  REG.forEach((k, i) => {
    const b = document.createElement("button");
    b.innerHTML = `<i style="background:${PAL[i]}"></i>${REG_NAMES[k]}<span class="v">R² ${lab.test[k].r2.toFixed(3)}</span>`;
    b.onclick = () => select(k);
    list.appendChild(b);
  });
  const lay = layout({ xaxis: axis("Measured strength (MPa)", { range: [0, 85] }), yaxis: axis("Predicted strength (MPa)", { range: [0, 85] }),
    showlegend: false, margin: { l: 56, r: 14, t: 14, b: 48 } });
  const select = async (k) => {
    current = k;
    const i = REG.indexOf(k);
    $$("button", list).forEach((b, j) => b.classList.toggle("active", j === i));
    $("#regEq").innerHTML = REG_EQ[k];
    countTo($("#regR2"), lab.test[k].r2, { decimals: 3, duration: 700 });
    countTo($("#regRmse"), lab.test[k].rmse, { decimals: 2, duration: 700 });
    if (!drawn) return;
    await Plotly.animate(el, { data: [{ y: lab.parity[k], marker: { color: PAL[i], size: 8, line: { color: C.surface, width: 1.5 } } }], traces: [1] },
      { transition: { duration: 800, easing: "cubic-in-out" }, frame: { duration: 800, redraw: false } });
  };
  select("ols");
  whenVisible(el, async () => {
    await render(el, { data: [
      { type: "scatter", mode: "lines", x: [0, 85], y: [0, 85], line: { color: C.muted, width: 1 }, hoverinfo: "skip" },
      { type: "scatter", mode: "markers", x: lab.parity.actual, y: lab.parity[current], marker: { color: PAL[REG.indexOf(current)], size: 8, line: { color: C.surface, width: 1.5 } },
        hovertemplate: "measured %{x:.1f} → predicted %{y:.1f} MPa<extra></extra>" },
    ], layout: lay });
    drawn = true;
    select(current);
  });

  // ridge path
  const path = lab.ridge.path, feats = lab.features;
  const lamS = $("#lamSlider");
  lamS.max = path.lambdas.length - 1;
  const cvIdx = path.lambdas.reduce((best, l, i) => (Math.abs(Math.log(l) - Math.log(lab.ridge.lambda)) < Math.abs(Math.log(path.lambdas[best]) - Math.log(lab.ridge.lambda)) ? i : best), 0);
  lamS.value = cvIdx;
  const bars = $("#coefBars");
  bars.innerHTML = feats.map((f) => `<div class="hbar"><span class="n">${FEAT[f]}</span><span class="t"><i></i></span><span class="v"></span></div>`).join("");
  const maxAbs = Math.max(...path.coefs[0].map(Math.abs));
  const drawRidge = () => {
    const i = +lamS.value;
    setRangeFill(lamS);
    $("#lamVal").textContent = Math.log(path.lambdas[i]).toFixed(1);
    $("#lamNote").textContent = i === cvIdx ? `λ = ${path.lambdas[i].toPrecision(3)}: the value 5-fold cross-validation picked.` :
      `λ = ${path.lambdas[i].toPrecision(3)}${i > cvIdx ? ": stronger than CV's choice; weights shrink towards 0." : ": weaker than CV's choice; correlated weights grow and fight."}`;
    $$(".hbar", bars).forEach((row, j) => {
      const v = path.coefs[i][j];
      const w = Math.min(50, (Math.abs(v) / maxAbs) * 50);
      const bar = $("i", row);
      bar.style.left = v >= 0 ? "50%" : 50 - w + "%";
      bar.style.width = w + "%";
      bar.style.background = v >= 0 ? "var(--acc)" : "var(--steel)";
      $(".v", row).textContent = v.toFixed(2);
    });
  };
  lamS.addEventListener("input", drawRidge);
  drawRidge();

  const conS = $("#conSlider"), con = lab.contamination;
  const drawCon = () => {
    const c = con[+conS.value];
    setRangeFill(conS);
    $("#conVal").textContent = (c.fraction * 100).toFixed(1).replace(".0", "") + "%";
    countTo($("#conOls"), c.ols, { decimals: 2, duration: 600 });
    countTo($("#conHub"), c.huber, { decimals: 2, duration: 600 });
  };
  conS.addEventListener("input", drawCon);
  drawCon();
}

// ---------------------------------------------------------------- 03 kernels
function initKernel() {
  const gS = $("#gSlider"), lS = $("#klSlider"), el = $("#kernelPlot");
  let first = true, busy = false, pending = false;
  const lay = layout({
    xaxis: axis("Curing age (days)", { type: "log", domain: [0, 0.56] }),
    yaxis: axis("Predicted strength (MPa)", { range: [0, 80] }),
    xaxis2: axis("Measured (test)", { domain: [0.66, 1], range: [0, 85] }),
    yaxis2: axis("Predicted", { anchor: "x2", range: [0, 85] }),
    legend: { orientation: "h", y: 1.1, x: 0 }, margin: { l: 56, r: 14, t: 30, b: 48 },
  });
  const refresh = async () => {
    if (busy) { pending = true; return; }
    busy = true;
    const g = Math.pow(10, +gS.value), l = +lS.value;
    setRangeFill(gS); setRangeFill(lS);
    $("#gVal").textContent = g < 0.1 ? g.toFixed(3) : g.toFixed(2);
    $("#klVal").textContent = l.toFixed(1);
    try {
      const r = await api(`/api/kernel?gamma=${g}&log_lambda=${l}`);
      const traces = [
        { type: "scatter", mode: "lines", x: r.curve.age, y: r.curve.ols, name: "Linear model", line: { color: C.blue, width: 2.5 }, hovertemplate: "%{x} d: %{y:.1f} MPa<extra>linear</extra>" },
        { type: "scatter", mode: "lines", x: r.curve.age, y: r.curve.krr, name: "Kernel ridge", line: { color: C.orange, width: 3 }, hovertemplate: "%{x} d: %{y:.1f} MPa<extra>kernel</extra>" },
        { type: "scatter", mode: "lines", x: [0, 85], y: [0, 85], xaxis: "x2", yaxis: "y2", line: { color: C.muted, width: 1 }, showlegend: false, hoverinfo: "skip" },
        { type: "scatter", mode: "markers", x: r.parity.actual, y: r.parity.pred, xaxis: "x2", yaxis: "y2", showlegend: false,
          marker: { color: C.orange, size: 6, line: { color: C.surface, width: 1 } }, hovertemplate: "%{x:.1f} → %{y:.1f}<extra>test mix</extra>" },
      ];
      if (first) { first = false; await render(el, { data: traces, layout: lay }); }
      else await Plotly.animate(el, { data: traces.map((t) => ({ x: t.x, y: t.y })), traces: [0, 1, 2, 3] },
        { transition: { duration: 450, easing: "cubic-out" }, frame: { duration: 450, redraw: false } });
      countTo($("#kTrain"), r.train_r2, { decimals: 3, duration: 450 });
      countTo($("#kVal"), r.val_r2, { decimals: 3, duration: 450 });
      countTo($("#kTest"), r.test_r2, { decimals: 3, duration: 450 });
      $("#kLin").textContent = r.linear_test_r2.toFixed(3);
      const st = $("#kStatus");
      let cls = "good", txt = "Good fit";
      if (r.train_r2 - r.val_r2 > 0.08) { cls = "over"; txt = "Overfitting: memorising mixes"; }
      else if (r.val_r2 < 0.85) { cls = "under"; txt = "Underfitting: too smooth"; }
      st.className = "status " + cls;
      $("span", st).textContent = txt;
    } finally {
      busy = false;
      if (pending) { pending = false; refresh(); }
    }
  };
  gS.addEventListener("input", refresh);
  lS.addEventListener("input", refresh);
  whenVisible(el, refresh);
}

// ---------------------------------------------------------------- 04 classifiers
function initClassifiers(ov) {
  const el = $("#boundaryPlot"), list = $("#clsModels");
  let B = null, current = "logistic", showSV = false, showUnc = false;
  CLS.forEach((k, i) => {
    const b = document.createElement("button");
    b.innerHTML = `<i style="background:${PAL[i]}"></i>${ov.classifier_names[k]}<span class="v"></span>`;
    b.onclick = () => select(k);
    list.appendChild(b);
  });
  const blueOrange = [[0, "#3987e5"], [0.25, "#2b5c99"], [0.5, "#2e2c28"], [0.75, "#a34a1d"], [1, "#e0621f"]];
  const seq = [[0, "#1c1916"], [0.4, "#5a2a14"], [0.75, "#e0621f"], [1, "#ffc9a8"]];
  const fillTrace = () => {
    const m = B.models[current];
    const unc = showUnc && current === "bayes_logistic";
    return { z: unc ? m.sd : m.z, colorscale: unc ? seq : blueOrange, zmin: unc ? undefined : 0, zmax: unc ? undefined : 1,
      hovertemplate: unc ? "w/b %{x:.2f}, %{y:.0f} d: σₐ %{z:.2f}<extra></extra>" : "w/b %{x:.2f}, %{y:.0f} d: P(pass) %{z:.2f}<extra></extra>" };
  };
  const svTrace = () => {
    const idx = B.models.svm.support;
    return { x: idx.map((i) => B.points.wb[i]), y: idx.map((i) => B.points.age[i]), visible: showSV && current === "svm" };
  };
  const draw = async () => {
    const pts = B.points, pass = pts.label.map((v, i) => v ? i : -1).filter((i) => i >= 0), fail = pts.label.map((v, i) => v ? -1 : i).filter((i) => i >= 0);
    const f = fillTrace();
    const traces = [
      { type: "heatmap", x: B.x, y: B.y, z: f.z, colorscale: f.colorscale, zmin: 0, zmax: 1, zsmooth: "best", showscale: false, opacity: 0.92, hovertemplate: f.hovertemplate, name: "P(pass)" },
      { type: "contour", x: B.x, y: B.y, z: B.models[current].z, showscale: false, contours: { start: 0.5, end: 0.5, size: 1, coloring: "none" }, line: { color: "#ffffff", width: 2.5 }, hoverinfo: "skip", name: "Decision boundary", showlegend: true },
      { type: "scatter", mode: "markers", x: fail.map((i) => pts.wb[i]), y: fail.map((i) => pts.age[i]), name: "Fails M30", marker: { color: "#8fb8f0", size: 6, line: { color: "#0b0b0a", width: 1 } }, hovertemplate: "w/b %{x:.2f}, %{y} d: failed<extra></extra>" },
      { type: "scatter", mode: "markers", x: pass.map((i) => pts.wb[i]), y: pass.map((i) => pts.age[i]), name: "Passes M30", marker: { color: "#ffb38a", size: 6, line: { color: "#0b0b0a", width: 1 } }, hovertemplate: "w/b %{x:.2f}, %{y} d: passed<extra></extra>" },
      { type: "scatter", mode: "markers", ...svTrace(), name: "Support vectors", marker: { color: "rgba(0,0,0,0)", size: 12, line: { color: "#ffffff", width: 1.5 } }, hoverinfo: "skip" },
    ];
    await render(el, { data: traces, layout: layout({ xaxis: axis("Water / binder ratio", { range: [0.23, 0.95] }), yaxis: axis("Curing age (days, log)", { type: "log", range: [0, Math.log10(365)] }),
      legend: { orientation: "h", y: 1.08, x: 0 }, margin: { l: 60, r: 14, t: 30, b: 48 } }) });
  };
  const update = () => {
    const f = fillTrace();
    Plotly.restyle(el, { z: [f.z], colorscale: [f.colorscale], zmin: [f.zmin ?? null], zmax: [f.zmax ?? null], zauto: [f.zmin === undefined], hovertemplate: [f.hovertemplate] }, [0]);
    Plotly.restyle(el, { z: [B.models[current].z] }, [1]);
    const sv = svTrace();
    Plotly.restyle(el, { x: [sv.x], y: [sv.y], visible: [sv.visible] }, [4]);
    el.animate([{ filter: "brightness(1.6) saturate(1.4)" }, { filter: "none" }], { duration: 600, easing: "ease-out" });
  };
  const select = (k) => {
    current = k;
    $$("button", list).forEach((b, j) => b.classList.toggle("active", CLS[j] === k));
    $("#clsEq").innerHTML = CLS_EQ[k] + (B ? `<br>Training accuracy on these 2 features: <b>${(B.models[k].train_acc * 100).toFixed(1)}%</b>` : "");
    if (el.data) update();
  };
  const sw = (id, fn) => {
    const s = $(id);
    const flip = () => { s.classList.toggle("on"); fn(s.classList.contains("on")); };
    s.addEventListener("click", flip);
    s.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); flip(); } });
  };
  sw("#swSV", (v) => { showSV = v; if (v && current !== "svm") select("svm"); else if (el.data) update(); });
  sw("#swUnc", (v) => { showUnc = v; if (v && current !== "bayes_logistic") select("bayes_logistic"); else if (el.data) update(); });
  whenVisible(el, async () => {
    B = await api("/api/boundary");
    CLS.forEach((k, j) => { $$(".v", list)[j].textContent = (B.models[k].train_acc * 100).toFixed(0) + "%"; });
    select(current);
    await draw();
  }, 0.15);
  select(current);
}

// ---------------------------------------------------------------- 05 designer
function initDesigner(ov) {
  const mix = { ...ov.default_mix };
  let grade = "M30", exposure = "Moderate", lastVerdict = null;
  const groups = [["Binder", ["cement", "slag", "flyash"]], ["Water & admixture", ["water", "sp"]], ["Aggregates", ["coarse", "fine"]], ["Curing", ["age"]]];
  const ing = Object.fromEntries(ov.ingredients.map((i) => [i.key, i]));
  const ctrls = {};
  const ageToV = (a) => Math.round((Math.log(a) / Math.log(365)) * 100);
  const vToAge = (v) => Math.max(1, Math.round(Math.exp((v / 100) * Math.log(365))));
  const box = $("#ingredients");
  groups.forEach(([title, keys]) => {
    const g = document.createElement("div");
    g.className = "ing-group";
    g.innerHTML = `<h4>${title}</h4>`;
    keys.forEach((k) => {
      const i = ing[k];
      const row = document.createElement("div");
      row.className = "ing";
      const isAge = k === "age";
      row.innerHTML = `<label for="in-${k}">${i.label}</label><input type="range" id="in-${k}"><output></output>`;
      const r = $("input", row), out = $("output", row);
      if (isAge) { r.min = 0; r.max = 100; r.step = 1; r.value = ageToV(mix.age); } else { r.min = i.min; r.max = i.max; r.step = i.step; r.value = mix[k]; }
      const show = () => { out.innerHTML = `${mix[k]}<small> ${i.unit}</small>`; setRangeFill(r); };
      r.addEventListener("input", () => { mix[k] = isAge ? vToAge(+r.value) : +r.value; show(); changed(); });
      ctrls[k] = { set: (v) => { mix[k] = v; r.value = isAge ? ageToV(v) : v; show(); }, isAge };
      show();
      g.appendChild(row);
    });
    box.appendChild(g);
  });
  // presets
  const pre = $("#mixPresets");
  Object.entries(ov.presets).forEach(([id, p]) => {
    const b = document.createElement("button");
    b.className = "btn btn--sm";
    b.textContent = p.label;
    b.onclick = () => {
      Object.entries(p.mix).forEach(([k, v]) => {
        const from = mix[k];
        tween(from, v, 700, (x) => ctrls[k].set(k === "sp" ? Math.round(x * 2) / 2 : Math.round(x)));
      });
      setTimeout(changed, 720);
      toast(`Loaded "${p.label}"`);
    };
    pre.appendChild(b);
  });
  const segs = (id, items, getLabel, onPick, active) => {
    const s = $(id);
    items.forEach((it) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = getLabel(it);
      b.classList.toggle("active", getLabel(it) === active);
      b.onclick = () => { $$("button", s).forEach((x) => x.classList.toggle("active", x === b)); onPick(it); changed(); };
      s.appendChild(b);
    });
  };
  segs("#gradeSeg", ov.grades, (g) => g.grade, (g) => { grade = g.grade; }, grade);
  segs("#expSeg", ov.exposure, (e) => e.exposure, (e) => { exposure = e.exposure; }, exposure);

  const colors = { cement: C.orange, slag: C.blue, flyash: C.haz, water: "#8fb8f0", sp: C.magenta, coarse: "#6f6a60", fine: "#a8a49a" };
  $("#mixBar").innerHTML = Object.keys(colors).map((k) => `<i data-k="${k}" style="background:${colors[k]}"></i>`).join("");
  $("#mixLegend").innerHTML = Object.keys(colors).map((k) => `<span style="--c:${colors[k]}">${ing[k].label}</span>`).join("");
  const drawBar = () => {
    const tot = Object.keys(colors).reduce((a, k) => a + mix[k], 0);
    $$("#mixBar i").forEach((i) => { i.style.width = (mix[i.dataset.k] / tot) * 100 + "%"; i.title = `${ing[i.dataset.k].label}: ${mix[i.dataset.k]} kg/m³`; });
  };

  // result widgets
  const votes = $("#votes");
  const voteKeys = [...CLS, "kglm"];
  votes.innerHTML = voteKeys.map((k) => `<div class="vote"><span class="n">${ov.classifier_names[k]}</span><span class="t"><i></i></span><span class="p"></span><span class="ic"></span></div>`).join("");
  $("#allReg").innerHTML = REG.map((k, i) => `<div class="hbar"><span class="n">${REG_NAMES[k]}</span><span class="t" style="background:var(--surface-3)"><i style="left:0;background:${PAL[i]}"></i></span><span class="v"></span></div>`).join("");
  $$("#allReg .t").forEach((t) => t.classList.add("noaxis"));
  const curveEl = $("#curvePlot");
  let curveDrawn = false;
  const gaugeMax = 90;

  const predict = debounce(async () => {
    const r = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mix, grade, exposure }) });
    show(r);
  }, 90);
  function changed() { drawBar(); predict(); }

  function show(r) {
    const g = r.grade, pass = r.p_pass >= 0.5;
    $("#ageTag").textContent = `${mix.age} day${mix.age > 1 ? "s" : ""}`;
    countTo($("#mpa"), r.mean, { decimals: 1, duration: 900 });
    $("#pm").textContent = `± ${(1.96 * r.sd).toFixed(1)}  (95%: ${Math.max(0, r.interval[0]).toFixed(1)}–${r.interval[1].toFixed(1)})`;
    const pct = (v) => Math.max(0, Math.min(100, (v / gaugeMax) * 100));
    $("#gFill").style.height = pct(r.mean) + "%";
    $("#gFill").style.background = pass ? "linear-gradient(0deg, #1f8a4a, var(--pass))" : "linear-gradient(0deg, var(--acc-deep), var(--acc))";
    $("#gBand").style.bottom = pct(r.interval[0]) + "%";
    $("#gBand").style.height = pct(r.interval[1]) - pct(r.interval[0]) + "%";
    $("#gTarget").style.bottom = pct(g.target) + "%";
    $("#gTargetLbl").textContent = `${g.grade} · ${g.target}`;
    // the ram pulses down onto the cube; cracks appear when the mix fails
    const ram = $("#ramG");
    ram.animate([{ transform: "translateY(0)" }, { transform: "translateY(52px)" }, { transform: "translateY(44px)" }],
      { duration: 700, easing: "cubic-bezier(.3,1.4,.5,1)", fill: "forwards" });
    const cracks = $$("#cracks path");
    cracks.forEach((p, i) => {
      const len = p.getTotalLength();
      p.style.strokeDasharray = len;
      p.animate([{ strokeDashoffset: p.style.strokeDashoffset || len }, { strokeDashoffset: pass ? len : 0 }],
        { duration: 500, delay: pass ? 0 : 450 + i * 90, fill: "forwards", easing: "ease-out" });
      p.style.strokeDashoffset = pass ? len : 0;
    });
    const verdict = pass ? "pass" : "fail";
    if (verdict !== lastVerdict) {
      const st = $("#stamp");
      st.className = "stamp " + verdict;
      st.textContent = pass ? "PASS" : "FAIL";
      void st.offsetWidth;
      st.classList.add("slam");
      if (!pass && lastVerdict !== null) $("#rigCard").animate([{ transform: "translate(0,0)" }, { transform: "translate(-4px,2px)" }, { transform: "translate(3px,-2px)" }, { transform: "translate(0,0)" }], { duration: 320, delay: 350 });
      lastVerdict = verdict;
    }
    countTo($("#pPass"), r.p_pass * 100, { decimals: 0, suffix: "%", duration: 800 });
    $("#pPassText").textContent = `chance this mix reaches the ${g.grade} target of ${g.target} MPa at ${mix.age} days (Bayesian kernel GLM)`;
    const w = $("#warn");
    w.classList.toggle("show", r.extrapolating.length > 0);
    w.textContent = r.extrapolating.length ? `Outside the training data for: ${r.extrapolating.join(", ")}. Treat the prediction with extra caution; the band is wider for a reason.` : "";

    // strength-gain curve
    const cv = r.curve;
    const up = cv.mean.map((m, i) => m + 1.96 * cv.sd[i]), lo = cv.mean.map((m, i) => Math.max(0, m - 1.96 * cv.sd[i]));
    const traces = [
      { type: "scatter", mode: "lines", x: [...cv.age, ...cv.age.slice().reverse()], y: [...up, ...lo.slice().reverse()], fill: "toself", fillcolor: "rgba(224,98,31,.14)", line: { width: 0 }, name: "95% band", hoverinfo: "skip" },
      { type: "scatter", mode: "lines", x: cv.age, y: cv.ols, name: "Linear model", line: { color: C.blue, width: 1.8 }, hovertemplate: "%{x} d: %{y:.1f} MPa<extra>linear</extra>" },
      { type: "scatter", mode: "lines", x: cv.age, y: cv.mean, name: "Bayesian kernel GLM", line: { color: C.orange, width: 3 }, hovertemplate: "%{x} d: %{y:.1f} MPa<extra>kernel GLM</extra>" },
      { type: "scatter", mode: "markers", x: [mix.age], y: [r.mean], name: "This mix", marker: { color: "#ffffff", size: 11, line: { color: C.orange, width: 3 } }, hovertemplate: "%{x} d: %{y:.1f} MPa<extra>selected age</extra>" },
    ];
    const lay = layout({ xaxis: axis("Curing age (days, log)", { type: "log" }), yaxis: axis("Strength (MPa)", { range: [0, Math.max(90, ...up) * 1.02] }),
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: g.target, y1: g.target, line: { color: C.haz, width: 1.5 } }],
      annotations: [{ xref: "paper", x: 0.01, y: g.target, yanchor: "bottom", text: `${g.grade} target ${g.target} MPa`, showarrow: false, font: { color: C.haz, size: 11 } }],
      legend: { orientation: "h", y: 1.12, x: 0 }, margin: { l: 50, r: 12, t: 30, b: 46 }, height: 320 });
    if (!curveDrawn) { curveDrawn = true; render(curveEl, { data: traces, layout: lay }); }
    else Plotly.react(curveEl, traces, lay, CONFIG);
    $("#curveSub").textContent = r.reaches_target_at ? `Expected to reach ${g.target} MPa at about day ${r.reaches_target_at}.` : `Not expected to reach ${g.target} MPa within a year.`;

    // votes
    const rows = $$(".vote", votes);
    voteKeys.forEach((k, i) => {
      const v = r.votes[k], row = rows[i];
      const p = v.prob ?? (v.pass ? 1 : 0);
      const bar = $("i", row);
      bar.style.width = (v.prob == null ? (v.pass ? 100 : 0) : p * 100) + "%";
      bar.style.background = v.pass ? "var(--pass)" : "var(--fail)";
      $(".p", row).textContent = v.prob == null ? (v.score >= 0 ? "+" : "−") + Math.abs(v.score).toFixed(2) : (p * 100).toFixed(0) + "%";
      const ic = $(".ic", row);
      ic.className = "ic " + (v.pass ? "y" : "n");
      ic.textContent = v.pass ? "✓" : "✕";
    });
    $("#voteSub").innerHTML = `<b style="color:${r.n_pass_votes >= 4 ? "var(--pass)" : "var(--fail)"}">${r.n_pass_votes} of 7</b> models predict this mix passes ${g.grade}. Least squares gives a score, not a probability.`;

    // durability + derived
    $("#expSub").textContent = `${r.exposure.exposure} exposure: min. binder ${r.exposure.min_cement} kg/m³, max. w/b ${r.exposure.max_wc}, min. grade ${r.exposure.min_grade}.`;
    $("#checks").innerHTML = r.durability.map((c) => `<div class="check ${c.ok ? "ok" : "bad"}"><span class="ic">${c.ok ? "✓" : "✕"}</span><span>${c.rule}</span><span class="v">${c.value}</span></div>`).join("");
    const d = r.derived;
    $("#derived").innerHTML = [["w/c", d.wc.toFixed(2)], ["w/b", d.wb.toFixed(2)], ["Binder", d.binder.toFixed(0) + " kg"], ["Fresh density", d.density.toFixed(0)], ["≈ CO₂", d.co2.toFixed(0) + " kg"], ["Cement in binder", (d.cement_share * 100).toFixed(0) + "%"]]
      .map(([l, v]) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join("");
    $$("#allReg .hbar").forEach((row, i) => {
      const v = r.strength[REG[i]];
      $("i", row).style.width = Math.max(0, Math.min(100, (v / gaugeMax) * 100)) + "%";
      $(".v", row).textContent = v.toFixed(1);
    });
  }
  drawBar();
  predict();
}

// ---------------------------------------------------------------- 06 report
function initReport(ov) {
  const t = ov.regression_test, sk = ov.regression_sklearn, cs = ov.classification_sklearn;
  const m30 = ov.classification.M30.results;
  const bestCls = Object.entries(m30).filter(([k]) => k !== "kglm").sort((a, b) => b[1].accuracy - a[1].accuracy)[0];
  $("#metrics").innerHTML = `
    <div class="metric" style="--glow:rgba(255,90,31,.3)"><div class="metric__v" data-count="${t.krr.r2}" data-decimals="3">0</div><div class="metric__l">Kernel ridge test R²</div><div class="metric__s">linear least squares: ${t.ols.r2.toFixed(3)}</div></div>
    <div class="metric" style="--glow:rgba(122,162,214,.3)"><div class="metric__v" data-count="${ov.coverage.kglm[4] * 100}" data-decimals="1" data-suffix="%">0</div><div class="metric__l">95% bands that contain the truth</div><div class="metric__s">Bayesian kernel GLM, 150 test mixes</div></div>
    <div class="metric" style="--glow:rgba(255,194,26,.3)"><div class="metric__v" data-count="${bestCls[1].accuracy * 100}" data-decimals="1" data-suffix="%">0</div><div class="metric__l">Best M30 pass/fail accuracy</div><div class="metric__s">${ov.classifier_names[bestCls[0]]}</div></div>
    <div class="metric" style="--glow:rgba(60,207,110,.3)"><div class="metric__v">${Math.max(sk.krr_max_diff, sk.ols_max_diff).toExponential(0)}</div><div class="metric__l">max |ours − scikit-learn| (MPa)</div><div class="metric__s">SVM: same ${cs.svm_n_sv[0]} support vectors, ${(cs.svm_same_predictions * 100).toFixed(0)}% same labels</div></div>`;
  $$("#metrics .metric").forEach((m) => tilt(m, 8));
  initReveals($("#metrics").parentElement);

  const best = REG.reduce((a, k) => (t[k].rmse < t[a].rmse ? k : a), REG[0]);
  const rt = document.createElement("table");
  rt.className = "data";
  rt.innerHTML = `<thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>${REG.map((k, i) =>
    `<tr style="--r:${i}"><td><span class="dot" style="background:${PAL[i]}"></span>${REG_NAMES[k]}</td><td class="${k === best ? "best" : ""}">${t[k].rmse.toFixed(2)}</td><td>${t[k].mae.toFixed(2)}</td><td class="${k === best ? "best" : ""}">${t[k].r2.toFixed(3)}</td></tr>`).join("")}</tbody>`;
  $("#regTable").replaceChildren(rt);

  const seg = $("#repGrade");
  const drawCls = (g) => {
    const res = ov.classification[g].results;
    const keys = [...CLS, "kglm"];
    const bestK = keys.reduce((a, k) => (res[k].accuracy > res[a].accuracy ? k : a), keys[0]);
    const ct = document.createElement("table");
    ct.className = "data";
    ct.innerHTML = `<thead><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>AUC</th><th>Log-loss</th></tr></thead><tbody>${keys.map((k, i) =>
      `<tr style="--r:${i}"><td>${ov.classifier_names[k]}</td><td class="${k === bestK ? "best" : ""}">${(res[k].accuracy * 100).toFixed(1)}%</td><td>${res[k].f1.toFixed(3)}</td><td>${res[k].auc.toFixed(3)}</td><td>${res[k].log_loss != null ? res[k].log_loss.toFixed(3) : "—"}</td></tr>`).join("")}</tbody>`;
    $("#clsTable").replaceChildren(ct);
  };
  ov.grades.forEach((g) => {
    const b = document.createElement("button");
    b.textContent = g.grade;
    b.classList.toggle("active", g.grade === "M30");
    b.onclick = () => { $$("button", seg).forEach((x) => x.classList.toggle("active", x === b)); drawCls(g.grade); };
    seg.appendChild(b);
  });
  drawCls("M30");
}
