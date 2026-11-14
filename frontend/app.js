"use strict";

const $ = (id) => document.getElementById(id);
const state = { info: null, icType: "pdebench_test", u0: null, testIndex: null, result: null };

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- API helpers
async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    throw new Error("Cannot reach the server. Is it running?");
  }
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
  if (!res.ok) throw new Error((data && data.detail) || `Request failed (HTTP ${res.status})`);
  return data;
}

function setStatus(kind, msg) {
  const el = $("status");
  el.hidden = !kind;
  el.className = "status " + (kind || "");
  el.textContent = msg || "";
}

function busy(on) {
  for (const id of ["run", "preview"]) $(id).disabled = on;
}

// ---------------------------------------------------------------- inputs
function addTerm(amp = 1, k = 1, phase = 0) {
  const row = document.createElement("div");
  row.className = "term";
  row.innerHTML = `
    <label>A <input type="number" step="0.1" value="${amp}" data-f="amp"></label>
    <label>k <input type="number" step="1" min="1" value="${k}" data-f="k"></label>
    <label>&phi; <input type="number" step="0.1" value="${phase}" data-f="phase"></label>
    <button type="button" class="ghost" aria-label="remove term">&times;</button>`;
  row.querySelector("button").onclick = () => {
    if ($("sine-terms").children.length > 1) row.remove();
  };
  $("sine-terms").appendChild(row);
}

function parseCustom() {
  const txt = $("custom-values").value.trim();
  if (!txt) return [];
  return txt.split(/[\s,;]+/).filter(Boolean).map(Number);
}

function readNumber(id, label) {
  const v = Number($(id).value);
  if ($(id).value === "" || !Number.isFinite(v)) throw new Error(`${label} must be a number`);
  return v;
}

function icSpec() {
  switch (state.icType) {
    case "pdebench_test":
      return { type: "pdebench_test", index: Math.trunc(readNumber("test-index", "Test sample index")) };
    case "sines": {
      const terms = [...$("sine-terms").children].map((row) => {
        const get = (f) => Number(row.querySelector(`[data-f="${f}"]`).value);
        return { amp: get("amp"), k: get("k"), phase: get("phase") };
      });
      if (terms.some((t) => Object.values(t).some((v) => !Number.isFinite(v)))) {
        throw new Error("All sine-term fields must be numbers");
      }
      return { type: "sines", terms };
    }
    case "gaussian":
      return { type: "gaussian", amp: readNumber("g-amp", "Amplitude"),
               center: readNumber("g-center", "Centre"), width: readNumber("g-width", "Width") };
    default:
      return null;
  }
}

async function buildU0() {
  const nu = Number($("nu").value);
  if (state.icType === "custom") {
    const vals = parseCustom();
    if (vals.some((v) => !Number.isFinite(v))) throw new Error("Custom values contain something that is not a number");
    const n = currentModel().grid_size;
    if (vals.length !== n) throw new Error(`Custom input needs exactly ${n} values, got ${vals.length}`);
    const x = vals.map((_, j) => (j + 0.5) / n);
    return { x, u0: vals, testIndex: null };
  }
  const r = await api("/api/initial-condition", { nu, ic: icSpec() });
  return { x: r.x, u0: r.u0, testIndex: r.test_index ?? null };
}

function currentModel() {
  return state.info.models[String(Number($("nu").value))];
}

// ---------------------------------------------------------------- plotting
function baseLayout(title, xTitle, yTitle) {
  return {
    title: { text: title, font: { size: 13, color: css("--ink") }, x: 0, xanchor: "left" },
    paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", color: css("--ink-2"), size: 11 },
    margin: { l: 52, r: 16, t: 36, b: 44 },
    xaxis: { title: xTitle, gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false },
    yaxis: { title: yTitle, gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false },
    legend: { orientation: "h", y: -0.25, font: { color: css("--ink-2") } },
  };
}
const plotConfig = { displaylogo: false, responsive: true };

function divergingScale() {
  return [[0, css("--div-neg")], [0.25, css("--div-neg-2")], [0.5, css("--div-mid")],
          [0.75, css("--div-pos-2")], [1, css("--div-pos")]];
}
function sequentialScale() {
  return [[0, css("--seq-0")], [0.25, css("--seq-1")], [0.5, css("--seq-2")],
          [0.75, css("--seq-3")], [1, css("--seq-4")]];
}

function heatmap(el, title, x, t, z, opts) {
  const layout = baseLayout(title, "x", "t");
  if (opts.tMax !== undefined && opts.tMax < t.at(-1)) {
    // supported time window: snapshots above the line are low confidence
    layout.shapes = [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: opts.tMax, y1: opts.tMax,
                       line: { color: css("--ink"), width: 1.5, dash: "dash" } }];
    layout.annotations = [{ xref: "paper", x: 0.01, y: opts.tMax, yanchor: "bottom", xanchor: "left",
                            text: `supported t ≤ ${opts.tMax}; above: low confidence`, showarrow: false,
                            font: { size: 10, color: css("--ink") }, bgcolor: css("--surface") }];
  }
  const trace = {
    type: "heatmap", x, y: t, z, colorscale: opts.scale, zmin: opts.zmin, zmax: opts.zmax,
    hovertemplate: `x=%{x:.3f}<br>t=%{y:.2f}<br>${opts.label}=%{z:.4f}<extra></extra>`,
    colorbar: { title: { text: opts.label, side: "right" }, thickness: 10, outlinewidth: 0 },
  };
  Plotly.react(el, [trace], layout, plotConfig);
}

function plotIC(x, u0) {
  if (!window.Plotly) return;
  const layout = baseLayout("Initial condition u(x, 0)", "x", "u");
  layout.margin.b = 36;
  Plotly.react("ic-plot", [{ x, y: u0, mode: "lines", line: { color: css("--s1"), width: 2 },
    hovertemplate: "x=%{x:.3f}<br>u=%{y:.4f}<extra></extra>", name: "u0" }], layout, plotConfig);
}

function referenceOf(r) {
  if (r.pdebench_reference) return { z: r.pdebench_reference, name: "PDEBench reference", color: "--s2" };
  if (r.solver) return { z: r.solver, name: "Reference solver", color: "--s3" };
  return null;
}

function plotResults(r) {
  const all = [r.fno, r.pdebench_reference, r.solver].filter(Boolean).flat(2);
  const lim = Math.max(...all.map(Math.abs)) || 1;
  const div = { scale: divergingScale(), zmin: -lim, zmax: lim, label: "u", tMax: r.reliable_t_max };
  heatmap("hm-fno", "FNO prediction u(x, t)", r.x, r.t, r.fno, div);

  const ref = referenceOf(r);
  if (ref) {
    heatmap("hm-ref", `${ref.name} u(x, t)`, r.x, r.t, ref.z, div);
    const err = r.fno.map((row, i) => row.map((v, j) => Math.abs(v - ref.z[i][j])));
    const emax = Math.max(...err.flat()) || 1e-12;
    heatmap("hm-err", `|FNO − ${ref.name}|`, r.x, r.t, err,
            { scale: sequentialScale(), zmin: 0, zmax: emax, label: "|err|", tMax: r.reliable_t_max });
  } else {
    Plotly.purge("hm-ref"); Plotly.purge("hm-err");
  }
  $("t-slider").max = r.t.length - 1;
  $("t-slider").value = r.t.length - 1;
  $("slider-wrap").hidden = false;
  plotSlice();
}

function plotSlice() {
  const r = state.result;
  if (!r) return;
  const k = Number($("t-slider").value);
  const outside = r.t[k] > r.reliable_t_max + 1e-9;
  $("t-label").textContent = `t = ${r.t[k].toFixed(2)}` +
    (outside ? ` (outside supported window t ≤ ${r.reliable_t_max}: low confidence)` : "");
  const traces = [{ x: r.x, y: r.fno[k], name: "FNO", mode: "lines", line: { color: css("--s1"), width: 2 } }];
  if (r.pdebench_reference) traces.push({ x: r.x, y: r.pdebench_reference[k], name: "PDEBench reference",
    mode: "lines", line: { color: css("--s2"), width: 2, dash: "dash" } });
  if (r.solver) traces.push({ x: r.x, y: r.solver[k], name: "Reference solver",
    mode: "lines", line: { color: css("--s3"), width: 2, dash: "dot" } });
  traces.forEach((tr) => { tr.hovertemplate = `${tr.name}: %{y:.4f}<extra></extra>`; });
  const layout = baseLayout(`Solution at t = ${r.t[k].toFixed(2)}${outside ? " (low confidence)" : ""}`, "x", "u");
  layout.hovermode = "x unified";
  Plotly.react("slice", traces, layout, plotConfig);
}

// ---------------------------------------------------------------- results UI
const fmt = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4 ? v.toExponential(2) : v.toPrecision(3));

function tile(k, v, s) {
  return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
}

function renderTiles(r) {
  let html = "";
  const add = (m, against) => {
    html += tile(`Relative L2 vs ${against}`, `${(100 * m.rel_l2).toFixed(2)}%`, "over full space-time field");
    html += tile(`RMSE vs ${against}`, fmt(m.rmse), `MSE ${fmt(m.mse)}`);
  };
  if (r.metrics_vs_pdebench) add(r.metrics_vs_pdebench, "PDEBench");
  if (r.metrics_vs_solver) add(r.metrics_vs_solver, "solver");
  html += tile("FNO time", `${r.timing_ms.fno.toFixed(1)} ms`,
               r.timing_ms.solver ? `solver ${r.timing_ms.solver.toFixed(0)} ms` : "solver not run");
  if (r.source.type === "pdebench_test") {
    html += tile("Input", `test #${r.source.test_index}`, `PDEBench sample ${r.source.pdebench_sample}`);
  }
  $("tiles").innerHTML = html;
}

function renderWarnings(list) {
  const el = $("warnings");
  el.hidden = !list.length;
  el.innerHTML = "";
  for (const w of list) {
    const p = document.createElement("p");
    p.textContent = w;
    el.appendChild(p);
  }
}

// ---------------------------------------------------------------- downloads
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function resultCSV(r) {
  const cols = ["t", "x", "u_fno"];
  if (r.pdebench_reference) cols.push("u_pdebench");
  if (r.solver) cols.push("u_solver");
  const lines = [`# Burgers FNO prediction, nu=${r.nu}, source=${r.source.type}`, cols.join(",")];
  r.t.forEach((t, i) => r.x.forEach((x, j) => {
    const row = [t, x, r.fno[i][j]];
    if (r.pdebench_reference) row.push(r.pdebench_reference[i][j]);
    if (r.solver) row.push(r.solver[i][j]);
    lines.push(row.join(","));
  }));
  return lines.join("\n");
}

// ---------------------------------------------------------------- actions
async function preview() {
  busy(true); setStatus("loading", "Building initial condition…");
  try {
    const ic = await buildU0();
    Object.assign(state, { u0: ic.u0, testIndex: ic.testIndex });
    plotIC(ic.x, ic.u0);
    setStatus(null);
    return ic;
  } catch (e) {
    setStatus("error", e.message);
    return null;
  } finally { busy(false); }
}

async function run() {
  const ic = await preview();
  if (!ic) return;
  busy(true); setStatus("loading", $("run-solver").checked ? "Running FNO and reference solver…" : "Running FNO…");
  try {
    const body = { nu: Number($("nu").value), run_solver: $("run-solver").checked };
    if (ic.testIndex !== null) body.test_index = ic.testIndex; else body.u0 = ic.u0;
    const r = await api("/api/predict", body);
    state.result = r;
    $("empty").hidden = true;
    renderWarnings(r.warnings);
    renderTiles(r);
    plotResults(r);
    $("dl-json").disabled = $("dl-csv").disabled = false;
    setStatus(null);
  } catch (e) {
    setStatus("error", e.message);
  } finally { busy(false); }
}

function selectTab(type) {
  state.icType = type;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.ic === type));
  document.querySelectorAll(".ic-form").forEach((f) => { f.hidden = f.dataset.form !== type; });
}

function renderInfo(info) {
  $("nu").innerHTML = "";
  for (const nu of Object.keys(info.models)) {
    const o = document.createElement("option");
    o.value = nu; o.textContent = `ν = ${nu}`;
    $("nu").appendChild(o);
  }
  const m = currentModel();
  if (!m) {
    $("nu-hint").textContent = "No trained model is loaded on this server.";
    $("run").disabled = $("preview").disabled = true;
    return;
  }
  $("nu-hint").textContent = `Grid N = ${m.grid_size}, outputs at ${m.t_out.length - 1} times up to t = ${m.t_out.at(-1)}; ` +
    `supported time window t ≤ ${m.reliable_t_max}. Supported |u₀| ≤ ${m.max_abs_u0}.`;
  $("test-index").max = Math.max(0, m.n_test_samples - 1);
  $("custom-values").placeholder = `exactly ${m.grid_size} numbers on x_j = (j + 0.5)/${m.grid_size}`;
  const tm = m.test_metrics;
  $("metrics-info").innerHTML = tm
    ? `<b>${tm.label}</b> (${tm.split_sizes.test} samples, disjoint from ${tm.split_sizes.train} training samples): ` +
      `relative L2 ${(100 * tm.overall.rel_l2).toFixed(2)}% (median ${(100 * tm.per_sample_rel_l2.median).toFixed(2)}%, ` +
      `worst ${(100 * tm.per_sample_rel_l2.max).toFixed(2)}%), RMSE ${fmt(tm.overall.rmse)}, MSE ${fmt(tm.overall.mse)}.`
    : "No test metrics file found on the server.";
}

async function init() {
  addTerm(1, 1, 0); addTerm(0.5, 3, 1);
  document.querySelectorAll(".tabs button").forEach((b) => { b.onclick = () => selectTab(b.dataset.ic); });
  $("add-term").onclick = () => { if ($("sine-terms").children.length < 8) addTerm(0.3, 2, 0); };
  $("custom-values").oninput = () => { $("custom-count").textContent = `${parseCustom().length} values`; };
  $("preview").onclick = preview;
  $("run").onclick = run;
  $("t-slider").oninput = plotSlice;
  $("dl-json").onclick = () => download("burgers_prediction.json", JSON.stringify(state.result), "application/json");
  $("dl-csv").onclick = () => download("burgers_prediction.csv", resultCSV(state.result), "text/csv");

  if (!window.Plotly) {
    setStatus("error", "Plotting library failed to load (needs internet access to cdn.plot.ly).");
  }
  busy(true);
  try {
    state.info = await api("/api/info");
    renderInfo(state.info);
    if (currentModel()) await preview();
  } catch (e) {
    setStatus("error", e.message);
  } finally {
    if (state.info && currentModel()) busy(false);
  }
}

window.addEventListener("load", init);
