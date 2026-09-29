// Plotly helpers shared by the landing page and the graphs page.
import { reduced } from "./fx.js";

export const C = { orange: "#e0621f", blue: "#3987e5", aqua: "#199e70", violet: "#9085e9", amber: "#c98500", magenta: "#d55181",
  ref: "#a8a49a", ink: "#f3f1ec", ink2: "#c9c5bb", muted: "#8b877d", grid: "#25241f", axis: "#3a3831", surface: "#151514",
  haz: "#ffc21a", pass: "#3ccf6e", fail: "#ff4d3d" };
export const PAL = [C.orange, C.blue, C.aqua, C.violet, C.amber, C.magenta];
const FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif";

export const CONFIG = { displayModeBar: false, responsive: true };
export const CONFIG_TOOLS = { displayModeBar: "hover", displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"] };

export function axis(title, extra = {}) {
  return { title: { text: title, font: { color: C.muted, size: 12 } }, gridcolor: C.grid, linecolor: C.axis,
    zerolinecolor: C.axis, tickcolor: C.axis, ticks: "outside", ticklen: 4, ...extra };
}

export function layout(extra = {}) {
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: FONT, color: C.ink2, size: 12 },
    margin: { l: 56, r: 18, t: 16, b: 48 },
    colorway: PAL,
    hoverlabel: { bgcolor: "#1f1d1a", bordercolor: "#4a463d", font: { family: FONT, color: C.ink, size: 12 } },
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "left", x: 0, bgcolor: "rgba(0,0,0,0)" },
    xaxis: axis(""), yaxis: axis(""),
    ...extra,
  };
}

const clone = (o) => JSON.parse(JSON.stringify(o));

/** Render a figure, then play an entrance: lines draw themselves, bars grow, markers fade in. */
export async function render(el, fig, { animate = true, config = CONFIG } = {}) {
  await Plotly.newPlot(el, clone(fig.data), clone(fig.layout), config);
  if (animate && !reduced) entrance(el);
  return el;
}

const OUT = "cubic-bezier(.16,1,.3,1)";

/** Entrance choreography on the rendered SVG, using the Web Animations API (no Plotly transitions). */
export function entrance(el) {
  el.querySelectorAll(".scatterlayer .trace").forEach((tr, ti) => {
    tr.querySelectorAll("path.js-line").forEach((p) => {
      const dashed = p.style.strokeDasharray && p.style.strokeDasharray !== "none";
      const len = p.getTotalLength?.() || 0;
      if (dashed || !len) {
        p.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 900, delay: 200 + ti * 120, fill: "backwards" });
        return;
      }
      p.animate([{ strokeDasharray: `${len} ${len}`, strokeDashoffset: len }, { strokeDasharray: `${len} ${len}`, strokeDashoffset: 0 }],
        { duration: 1500, delay: ti * 150, easing: OUT, fill: "backwards" });
    });
    tr.querySelectorAll("path.js-fill").forEach((p) => p.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 1200, delay: 300, fill: "backwards" }));
    // Markers are positioned with a transform attribute, so only their opacity is animated.
    tr.querySelectorAll(".points path").forEach((p, i) =>
      p.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500, delay: 250 + ti * 140 + Math.min(i, 150) * 7, fill: "backwards" }));
    tr.querySelectorAll(".textpoint").forEach((t, i) =>
      t.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500, delay: 700 + i * 60, fill: "backwards" }));
  });
  el.querySelectorAll(".barlayer .trace").forEach((tr, ti) => {
    const horizontal = tr.__data__?.[0]?.trace?.orientation === "h";
    tr.querySelectorAll(".point").forEach((pt, i) => {
      const p = pt.querySelector("path");
      if (!p) return;
      const neg = (pt.__data__?.s ?? 0) < 0;
      p.style.transformBox = "fill-box";
      p.style.transformOrigin = horizontal ? (neg ? "right center" : "left center") : (neg ? "center top" : "center bottom");
      p.animate([{ transform: horizontal ? "scaleX(0)" : "scaleY(0)" }, { transform: "none" }],
        { duration: 950, delay: ti * 110 + Math.min(i, 60) * 16, easing: OUT, fill: "backwards" });
      pt.querySelectorAll("text").forEach((t) => t.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500, delay: 700 + i * 30, fill: "backwards" }));
    });
  });
  el.querySelectorAll(".heatmaplayer, .boxlayer, .contourlayer").forEach((l) =>
    l.animate([{ opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0)" }], { duration: 1000, easing: OUT, fill: "backwards" }));
  el.querySelectorAll(".annotation, .shapelayer path").forEach((a, i) =>
    a.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 600, delay: 900 + i * 80, fill: "backwards" }));
}

/** Plain HTML table of every trace, the accessible twin of a chart. */
export function tableView(fig) {
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  table.className = "data";
  const thead = table.createTHead().insertRow();
  const tbody = table.createTBody();
  const traces = fig.data.filter((t) => t.x || t.y || t.z);
  const heat = traces.find((t) => t.type === "heatmap");
  if (heat) {
    thead.insertCell().textContent = "";
    heat.x.forEach((x) => { const th = document.createElement("th"); th.textContent = x; thead.appendChild(th); });
    heat.y.forEach((y, i) => {
      const r = tbody.insertRow();
      r.insertCell().textContent = y;
      heat.z[i].forEach((v) => { r.insertCell().textContent = typeof v === "number" ? v.toFixed(4) : v; });
    });
  } else {
    ["Series", "x", "y"].forEach((h) => { const th = document.createElement("th"); th.textContent = h; thead.appendChild(th); });
    for (const t of traces) {
      const xs = t.x || [], ys = t.y || [];
      const n = Math.max(xs.length || 0, ys.length || 0);
      const step = Math.max(1, Math.ceil(n / 60));
      for (let i = 0; i < n; i += step) {
        const r = tbody.insertRow();
        r.insertCell().textContent = t.name || t.type || "trace";
        r.insertCell().textContent = fmtCell(xs[i]);
        r.insertCell().textContent = fmtCell(ys[i]);
      }
    }
  }
  wrap.appendChild(table);
  return wrap;
}

function fmtCell(v) {
  if (typeof v === "number") return Math.abs(v) >= 1000 ? v.toLocaleString() : +v.toFixed(5);
  return v ?? "";
}
