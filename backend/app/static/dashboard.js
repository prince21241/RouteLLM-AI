import {
  apiQuery,
  costSeries,
  actionReason,
  detailModel,
  detailState,
  formatCost,
  formatRate,
  panelState,
  qualityLabel,
  recordedOrUnknown,
  resolvedFinal,
  routingLines,
} from "./dashboard_view.mjs";
import { renderMarkdown } from "./markdown_dom.mjs";

const API_BASE = window.ROUTELLM_API_BASE || "";
const pageSize = 20;

const filtersForm = document.querySelector("#filters");
const statusLine = document.querySelector("#dashboard-status");
const retry = document.querySelector("#retry");
const empty = document.querySelector("#empty");
const content = document.querySelector("#content");
const demoBanner = document.querySelector("#demo-banner");
const cards = document.querySelector("#cards");
const secondary = document.querySelector("#secondary-metrics");
const costSummary = document.querySelector("#cost-summary");
const daysBody = document.querySelector("#days");
const notes = document.querySelector("#notes");
const costNotes = document.querySelector("#cost-notes");
const modelsBody = document.querySelector("#models");
const judgesBody = document.querySelector("#judges");
const breakdownNotes = document.querySelector("#breakdown-notes");
const historyBody = document.querySelector("#history");
const recentBody = document.querySelector("#recent");
const pageLabel = document.querySelector("#page-label");
const filterScope = document.querySelector("#filter-scope");
const prev = document.querySelector("#prev");
const next = document.querySelector("#next");
const volumeChart = document.querySelector("#volume-chart");
const costChart = document.querySelector("#cost-chart");
const volumeTip = document.querySelector("#volume-tip");
const costTip = document.querySelector("#cost-tip");
const detail = document.querySelector("#detail");
const detailStatus = document.querySelector("#detail-status");
const detailRetry = document.querySelector("#detail-retry");
const detailBody = document.querySelector("#detail-body");
const detailClose = document.querySelector("#detail-close");

const demoMode = new URLSearchParams(location.search).get("demo");
let offset = 0;
let latestTotal = 0;
let demoBundle = null;
let openRequestId = null;
let detailOpener = null;

filtersForm.addEventListener("submit", (event) => {
  event.preventDefault();
  offset = 0;
  load();
});

document.querySelector("#reset-filters").addEventListener("click", () => {
  filtersForm.reset();
  offset = 0;
  load();
});

retry.addEventListener("click", () => load());
prev.addEventListener("click", () => {
  offset = Math.max(0, offset - pageSize);
  load();
});
next.addEventListener("click", () => {
  offset += pageSize;
  load();
});
detailRetry.addEventListener("click", () => {
  if (openRequestId) {
    loadDetail(openRequestId);
  }
});
detailClose.addEventListener("click", closeDetail);
detail.addEventListener("keydown", (event) => {
  if (detail.hidden) {
    return;
  }
  if (event.key === "Escape") {
    event.preventDefault();
    closeDetail();
    return;
  }
  if (event.key !== "Tab") {
    return;
  }
  const focusable = [...detail.querySelectorAll("button, a, input, select, textarea")].filter(
    (node) => !node.hidden && !node.disabled,
  );
  if (!focusable.length) {
    return;
  }
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
});

content.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-id]");
  if (!button) {
    return;
  }
  loadDetail(button.dataset.id, button);
});

load();

async function load() {
  applyPanel({ loading: true, error: "", requestCount: 0 });
  try {
    const bundle = await loadBundle();
    latestTotal = bundle.history.total;
    renderOverview(bundle.overview);
    renderBreakdown(bundle.breakdown);
    renderHistory(bundle.history);
    renderRecent(bundle.recent.items);
    applyPanel({
      loading: false,
      error: "",
      requestCount: bundle.overview.request_count,
    });
    const requested = requestIdFromHash();
    if (requested) {
      loadDetail(requested);
    }
  } catch (error) {
    clearPanels();
    applyPanel({
      loading: false,
      error: error instanceof Error ? error.message : "Dashboard data is unavailable.",
      requestCount: 0,
    });
  }
}

async function loadBundle() {
  if (demoMode === "error") {
    showDemo("Demo error state. This is not a failed database query.");
    throw new Error("Demo error state. The sample dashboard did not load.");
  }
  if (demoMode === "empty") {
    showDemo("Demo empty state. This is not your database.");
    return emptyBundle();
  }
  if (demoMode === "1") {
    showDemo("Demo data. This is not your database.");
    if (!demoBundle) {
      const response = await fetch("/static/dashboard_demo.json");
      if (!response.ok) {
        throw new Error("The demo fixture could not be loaded.");
      }
      demoBundle = await response.json();
    }
    const history = pageDemoHistory(demoBundle.history);
    return {
      overview: demoBundle.overview,
      breakdown: demoBundle.breakdown,
      history,
      recent: {
        items: demoBundle.history.items.slice(0, 5),
        total: demoBundle.history.items.length,
      },
    };
  }
  demoBanner.hidden = true;
  const [overview, breakdown, history, recent] = await Promise.all([
    getJson(`/api/v1/dashboard/overview${currentQuery()}`),
    getJson(`/api/v1/dashboard/breakdown${currentQuery()}`),
    getJson(`/api/v1/dashboard/requests${currentQuery()}`),
    getJson(`/api/v1/dashboard/requests${currentQuery({ limit: 5, offset: 0 })}`),
  ]);
  return { overview, breakdown, history, recent };
}

function currentQuery(overrides = {}) {
  const data = new FormData(filtersForm);
  const extra = [];
  const model = String(data.get("model") || "").trim();
  const provider = String(data.get("provider") || "");
  const status = String(data.get("status") || "");
  if (model) {
    extra.push(`initial model ${model}`);
  }
  if (provider) {
    extra.push(`initial provider ${provider}`);
  }
  if (status) {
    extra.push(`status ${status}`);
  }
  filterScope.textContent =
    extra.length && document.body.dataset.route === "overview"
      ? `Date range also uses ${extra.join(", ")}. Change those on Requests, Costs & Usage, or Providers & Models.`
      : "";
  return apiQuery({
    from: String(data.get("from") || ""),
    to: String(data.get("to") || ""),
    model,
    provider,
    status,
    limit: overrides.limit ?? pageSize,
    offset: overrides.offset ?? offset,
  });
}

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(detailText(payload));
  }
  return payload;
}

function applyPanel(state) {
  const mode = panelState(state);
  statusLine.textContent =
    mode === "loading"
      ? "Loading dashboard…"
      : mode === "error"
        ? state.error
        : mode === "empty"
          ? "No stored requests match these filters."
          : "Showing stored chat requests. Offline evaluation runs are not included.";
  retry.hidden = mode !== "error";
  empty.hidden = mode !== "empty";
  content.hidden = mode !== "populated";
  statusLine.className = mode === "error" ? "error" : "";
}

function clearPanels() {
  cards.replaceChildren();
  secondary.replaceChildren();
  costSummary.replaceChildren();
  daysBody.replaceChildren();
  notes.replaceChildren();
  costNotes.replaceChildren();
  modelsBody.replaceChildren();
  judgesBody.replaceChildren();
  breakdownNotes.replaceChildren();
  historyBody.replaceChildren();
  recentBody.replaceChildren();
  clearChart(volumeChart);
  clearChart(costChart);
  volumeTip.textContent = "";
  costTip.textContent = "";
}

function renderOverview(overview) {
  const latency = overview.request_latency_ms;
  const attemptLatency = overview.attempt_latency_ms;
  fillMetrics(cards, [
    ["Requests", String(overview.request_count), "Logical chat requests"],
    ["Recorded cost", formatCost(overview.request_cost), "Unknown amounts are not shown as zero"],
    ["Error rate", formatRate(overview.error_rate), "Failed requests. Pending is not an error."],
    [
      "Median latency",
      latency.p50 === null ? "unknown" : `${latency.p50} ms`,
      `End-to-end p50. ${latency.known_count} recorded, ${latency.missing_count} missing.`,
    ],
  ]);
  fillMetrics(secondary, [
    ["Attempts", String(overview.attempt_count), "Generation calls, counted separately"],
    ["Fallback rate", formatRate(overview.fallback_rate), "Rows with a recorded fallback flag"],
    ["Escalation rate", formatRate(overview.escalation_rate), "Quality escalation, not provider fallback"],
    ["Judge cost", formatCost(overview.judge_cost), "Not added again onto the request total"],
    [
      "Savings estimate",
      overview.savings_estimate === null ? "unknown" : `${overview.savings_estimate} USD`,
      "Same-token-volume estimate. The premium model was not called.",
    ],
    [
      "Attempt latency",
      attemptLatency.p50 === null ? "unknown" : `${attemptLatency.p50} ms`,
      `Provider-call p50. ${attemptLatency.known_count} recorded, ${attemptLatency.missing_count} missing.`,
    ],
  ]);
  fillMetrics(costSummary, [
    ["Recorded request cost", formatCost(overview.request_cost), "Complete and estimated stay separate"],
    ["Judge cost", formatCost(overview.judge_cost), "Identified separately and not added again"],
    [
      "Savings estimate",
      overview.savings_estimate === null ? "unknown" : `${overview.savings_estimate} USD`,
      `${overview.savings_known_requests} known, ${overview.savings_unknown_requests} unknown. Basis: ${overview.savings_basis}.`,
    ],
  ]);
  const series = costSeries(overview.by_day);
  drawVolume(volumeChart, series, volumeTip);
  drawCost(costChart, series, costTip);
  daysBody.replaceChildren(
    ...overview.by_day.map((day) => {
      const row = document.createElement("tr");
      row.append(
        cell(day.day),
        cell(String(day.requests), true),
        cell(blankMoney(day.complete), true),
        cell(blankMoney(day.estimated), true),
        cell(String(day.unknown_count), true),
      );
      return row;
    }),
  );
  notes.replaceChildren(...overview.notes.map((note) => el("li", note)));
  costNotes.replaceChildren(...overview.notes.map((note) => el("li", note)));
}

function fillMetrics(container, rows) {
  container.replaceChildren(
    ...rows.map(([label, value, hint]) => {
      const card = el("div", "", "card");
      card.append(el("span", label), el("strong", value), el("span", hint, "hint"));
      return card;
    }),
  );
}

function renderBreakdown(breakdown) {
  modelsBody.replaceChildren(
    ...(breakdown.models.length
      ? breakdown.models.map((item) => {
          const row = document.createElement("tr");
          const model = el("td");
          model.append(el("span", item.model, "model-name"));
          row.append(
            cell(item.provider),
            model,
            cell(String(item.attempt_count), true),
            cell(String(item.request_count), true),
            cell(String(item.failed_attempts), true),
            cell(blankMoney(item.cost_complete), true),
            cell(blankMoney(item.cost_estimated), true),
            cell(String(item.cost_unknown_attempts), true),
            cell(item.latency_p50_ms === null ? "unknown" : String(item.latency_p50_ms), true),
          );
          return row;
        })
      : [emptyRow(9, "No generation attempts match these filters.")]),
  );
  judgesBody.replaceChildren(
    ...(breakdown.judges.length
      ? breakdown.judges.map((item) => {
          const row = document.createElement("tr");
          row.append(
            cell(item.judge_model || "Not recorded"),
            cell(String(item.evaluations), true),
            cell(blankMoney(item.cost_complete), true),
            cell(blankMoney(item.cost_estimated), true),
            cell(String(item.cost_unknown_evaluations), true),
          );
          return row;
        })
      : [emptyRow(5, "No judge calls match these filters.")]),
  );
  breakdownNotes.replaceChildren(...breakdown.notes.map((note) => el("li", note)));
}

function renderHistory(history) {
  const start = history.total === 0 ? 0 : history.offset + 1;
  const end = Math.min(history.offset + history.items.length, history.total);
  pageLabel.textContent = `${start}–${end} of ${history.total} requests. Prompt previews are truncated.`;
  prev.disabled = history.offset === 0;
  next.disabled = history.offset + history.items.length >= history.total;
  historyBody.replaceChildren(...history.items.map((item) => requestRow(item)));
}

function renderRecent(items) {
  recentBody.replaceChildren(
    ...(items.length
      ? items.map((item) => requestRow(item))
      : [emptyRow(6, "No recent requests in this filter.")]),
  );
}

function requestRow(item) {
  const row = document.createElement("tr");
  const preview = el("button", item.prompt_preview, "linkish");
  preview.type = "button";
  preview.dataset.id = item.request_id;
  const previewCell = document.createElement("td");
  previewCell.append(preview);
  if (item.prompt_truncated) {
    previewCell.append(el("span", " truncated", "note"));
  }
  const statusCell = document.createElement("td");
  statusCell.append(statusBadge(item.status));
  const modelCell = document.createElement("td");
  modelCell.append(el("span", finalModel(item), "model-name"));
  row.append(
    timeCell(item.created_at),
    previewCell,
    statusCell,
    modelCell,
    cell(item.total_cost === null ? `unknown (${item.cost_completeness})` : `${item.total_cost} USD (${item.cost_completeness})`, true),
    cell(item.end_to_end_latency_ms === null ? "unknown" : `${item.end_to_end_latency_ms} ms`, true),
  );
  return row;
}

async function loadDetail(requestId, opener) {
  if (opener) {
    detailOpener = opener;
  }
  openRequestId = requestId;
  const nextUrl = `/requests#request=${encodeURIComponent(requestId)}`;
  const currentUrl = `${location.pathname}${location.hash}`;
  if (currentUrl !== nextUrl) {
    const method = location.hash.startsWith("#request=") ? "replaceState" : "pushState";
    history[method](null, "", nextUrl);
    window.routellmApplyRoute?.();
  }
  detail.hidden = false;
  detailBody.replaceChildren();
  detailRetry.hidden = true;
  detailStatus.className = "";
  detailStatus.textContent = "Loading request…";
  detailClose.focus();
  const state = detailState({ loading: true, error: "", request: null });
  if (state !== "loading") {
    return;
  }
  try {
    const payload = demoMode === "1" ? demoDetail(requestId) : await getJson(`/api/v1/dashboard/requests/${encodeURIComponent(requestId)}`);
    renderDetail(payload);
  } catch (error) {
    detailBody.replaceChildren();
    detailStatus.className = "error";
    detailStatus.textContent = error instanceof Error ? error.message : "The stored request could not be loaded.";
    detailRetry.hidden = false;
  }
}

function closeDetail() {
  detail.hidden = true;
  openRequestId = null;
  if (location.hash.startsWith("#request=")) {
    history.replaceState(null, "", `${location.pathname}${location.search}`);
    window.routellmApplyRoute?.();
  }
  if (detailOpener && typeof detailOpener.focus === "function") {
    detailOpener.focus();
  }
  detailOpener = null;
}

function renderDetail(request) {
  const model = detailModel(request);
  const finals = resolvedFinal(request);
  const mode = detailState({ loading: false, error: "", request });
  if (mode !== "populated") {
    detailStatus.textContent = "Request detail is empty.";
    return;
  }
  detailStatus.className = "";
  detailStatus.textContent = model.persistence;
  const body = document.createElement("div");
  body.append(definitionList([
    ["Created", readableUtc(request.created_at)],
    ["Exact created time", exactUtc(request.created_at)],
    ["Status", model.status],
    ["Initial provider", model.initialProvider],
    ["Initial model", model.initialModel],
    ["Final provider", finals.provider || "Not recorded"],
    ["Final model", finals.model || "Not recorded"],
    ["Attempts", String(model.attempts.length)],
    ["Escalated", model.escalated ? "yes" : "no"],
    ["Escalation reason", actionReason(model.escalated, model.escalationReason)],
    ["Escalation error", actionReason(model.escalated, model.escalationError)],
    ["Fallback", fallbackLabel(model.fallbackUsed)],
    ["Fallback reason", actionReason(model.fallbackUsed, model.fallbackReason)],
    ["Quality", qualityLabel(model.qualityVerdict, model.evaluations)],
    ["Cost", model.cost === null ? `unknown (${model.costCompleteness || "unknown"})` : `${model.cost} USD (${model.costCompleteness})`],
    ["Savings estimate", model.savings === null ? "unknown" : `${model.savings} USD`],
    ["Savings basis", model.savingsBasis || "same_token_volume"],
    ["End-to-end latency", model.latencyMs === null ? "unknown" : `${model.latencyMs} ms`],
    ...routingLines(model.routingMetadata),
  ]));
  body.append(el("h3", "Prompt"), el("p", model.prompt, "prompt"));
  if (model.systemPrompt) {
    body.append(el("h3", "System prompt"), el("p", model.systemPrompt, "prompt"));
  }
  body.append(el("h3", "Answer"));
  if (model.answer === null) {
    body.append(el("p", "Not recorded", "answer"));
  } else {
    const answer = document.createElement("div");
    answer.className = "markdown answer";
    renderMarkdown(answer, model.answer);
    body.append(answer);
  }
  if (model.qualityReasons.length) {
    body.append(el("h3", "Quality reasons"));
    body.append(list(model.qualityReasons));
  }
  if (request.fallback_skips && request.fallback_skips.length) {
    body.append(el("h3", "Fallback skips"));
    body.append(list(request.fallback_skips.map((skip) => `${skip.model_id}: ${skip.reason}`)));
  }
  body.append(el("h3", "Generation attempts"));
  body.append(attemptTable(model.attempts));
  body.append(el("h3", "Evaluations"));
  body.append(model.evaluations.length ? evaluationTable(model.evaluations) : el("p", "No quality check was stored.", "empty"));
  detailBody.replaceChildren(body);
}

function attemptTable(attempts) {
  const table = document.createElement("table");
  const head = document.createElement("tr");
  ["#", "Purpose", "Provider", "Model", "Status", "In", "Out", "Cost", "Completeness", "Latency (ms)", "Error category", "Error"].forEach((label) => {
    head.append(el("th", label));
  });
  const thead = document.createElement("thead");
  thead.append(head);
  const tbody = document.createElement("tbody");
  attempts.forEach((attempt) => {
    const row = document.createElement("tr");
    row.append(
      cell(String(attempt.attempt_number), true),
      cell(attempt.purpose || "unknown"),
      cell(attempt.provider),
      cell(attempt.configured_model_id),
      cell(attempt.status),
      cell(recordedOrUnknown(attempt.input_tokens), true),
      cell(recordedOrUnknown(attempt.output_tokens), true),
      cell(attempt.estimated_cost === null ? "unknown" : `${attempt.estimated_cost} USD`, true),
      cell(attempt.cost_completeness),
      cell(attempt.provider_latency_ms === null ? "unknown" : String(attempt.provider_latency_ms), true),
      cell(attempt.error_category || ""),
      cell(attempt.error_message || ""),
    );
    tbody.append(row);
  });
  table.append(thead, tbody);
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.append(table);
  return wrap;
}

function evaluationTable(evaluations) {
  const table = document.createElement("table");
  const head = document.createElement("tr");
  ["Attempt", "Verdict", "Reasons", "Judge model", "Judge cost", "Completeness", "Latency (ms)"].forEach((label) => {
    head.append(el("th", label));
  });
  const thead = document.createElement("thead");
  thead.append(head);
  const tbody = document.createElement("tbody");
  evaluations.forEach((item) => {
    const row = document.createElement("tr");
    row.append(
      cell(String(item.attempt_number), true),
      cell(item.verdict),
      cell((item.reasons || []).join("; ")),
      cell(item.judge_model || "Not recorded"),
      cell(item.judge_cost === null ? "unknown" : `${item.judge_cost} USD`, true),
      cell(item.judge_cost_completeness || "unknown"),
      cell(item.judge_latency_ms === null ? "unknown" : String(item.judge_latency_ms), true),
    );
    tbody.append(row);
  });
  table.append(thead, tbody);
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.append(table);
  return wrap;
}

function definitionList(pairs) {
  const listNode = document.createElement("dl");
  listNode.className = "facts";
  pairs.forEach(([label, value]) => {
    listNode.append(el("dt", label), el("dd", value));
  });
  return listNode;
}

function list(items) {
  const node = document.createElement("ul");
  items.forEach((item) => node.append(el("li", item)));
  return node;
}

function drawVolume(svg, series, tip) {
  drawChart(svg, series, [{ key: "requests", color: "#0f766e", label: "Requests" }], tip);
}

function drawCost(svg, series, tip) {
  drawChart(
    svg,
    series,
    [
      { key: "complete", color: "#0f766e", label: "Complete USD" },
      { key: "estimated", color: "#b45309", label: "Estimated USD" },
    ],
    tip,
  );
}

function drawChart(svg, series, marks, tip) {
  clearChart(svg);
  const width = 720;
  const height = 280;
  const padLeft = 52;
  const padRight = 12;
  const padTop = 16;
  const padBottom = 36;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  if (!series.length) {
    svg.append(svgText(padLeft, 40, "No days in this range."));
    if (tip) {
      tip.textContent = "No days in this range.";
    }
    return;
  }
  const values = series.flatMap((point) => marks.map((mark) => point[mark.key])).filter((value) => value !== null);
  if (!values.length) {
    svg.append(svgText(padLeft, 40, "No recorded values in this range."));
    if (tip) {
      tip.textContent = "No recorded values in this range. Unknown costs are omitted.";
    }
    return;
  }
  const max = Math.max(...values, 0);
  const top = max === 0 ? 1 : max;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;
  const baseline = padTop + plotHeight;
  axisLine(svg, padLeft, padTop, padLeft, baseline);
  axisLine(svg, padLeft, baseline, width - padRight, baseline);
  [0, 0.5, 1].forEach((step) => {
    const value = top * step;
    const y = baseline - step * plotHeight;
    svg.append(svgText(4, y + 4, axisLabel(value)));
  });
  const groupWidth = plotWidth / series.length;
  const barWidth = Math.min(18, Math.max(groupWidth / (marks.length + 1), 4));
  const labelStep = series.length > 12 ? Math.ceil(series.length / 8) : 1;
  series.forEach((point, index) => {
    marks.forEach((mark, markIndex) => {
      const value = point[mark.key];
      if (value === null) {
        return;
      }
      const barHeight = (value / top) * plotHeight;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      const x = padLeft + index * groupWidth + markIndex * (barWidth + 3) + 4;
      rect.setAttribute("x", String(x));
      rect.setAttribute("y", String(baseline - Math.max(barHeight, value === 0 ? 1 : 0)));
      rect.setAttribute("width", String(Math.max(barWidth - 2, 2)));
      rect.setAttribute("height", String(Math.max(barHeight, value === 0 ? 1 : 0)));
      rect.setAttribute("fill", mark.color);
      rect.setAttribute("tabindex", "0");
      const description = `${point.day} ${mark.label}: ${value}`;
      rect.append(svgTitle(description));
      const show = () => {
        if (tip) {
          tip.textContent = description;
        }
      };
      rect.addEventListener("pointerenter", show);
      rect.addEventListener("focus", show);
      svg.append(rect);
    });
    if (index % labelStep === 0) {
      svg.append(svgText(padLeft + index * groupWidth, height - 12, String(point.day).slice(5)));
    }
  });
}

function axisLine(svg, x1, y1, x2, y2) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", String(x1));
  line.setAttribute("y1", String(y1));
  line.setAttribute("x2", String(x2));
  line.setAttribute("y2", String(y2));
  line.setAttribute("stroke", "#cbd5e1");
  svg.append(line);
}

function axisLabel(value) {
  if (value === 0) {
    return "0";
  }
  if (value >= 100) {
    return String(Math.round(value));
  }
  return String(Number(value.toPrecision(3)));
}

function clearChart(svg) {
  svg.replaceChildren();
}

function svgText(x, y, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "text");
  node.setAttribute("x", String(x));
  node.setAttribute("y", String(y));
  node.setAttribute("fill", "#526072");
  node.setAttribute("font-size", "12");
  node.textContent = text;
  return node;
}

function svgTitle(text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "title");
  node.textContent = text;
  return node;
}

function demoDetail(requestId) {
  if (!demoBundle || demoBundle.detail.request_id !== requestId) {
    throw new Error("That request is not in the demo fixture.");
  }
  return demoBundle.detail;
}

function pageDemoHistory(history) {
  return {
    ...history,
    items: history.items.slice(offset, offset + pageSize),
    limit: pageSize,
    offset,
    total: history.items.length,
  };
}

function emptyBundle() {
  return {
    overview: {
      request_count: 0,
      attempt_count: 0,
      error_rate: null,
      escalation_rate: null,
      fallback_rate: null,
      request_latency_ms: { p50: null, average: null, known_count: 0, missing_count: 0 },
      attempt_latency_ms: { p50: null, average: null, known_count: 0, missing_count: 0 },
      request_cost: { complete: null, estimated: null, unknown_count: 0 },
      judge_cost: { complete: null, estimated: null, unknown_count: 0 },
      savings_estimate: null,
      savings_known_requests: 0,
      savings_unknown_requests: 0,
      savings_basis: "same_token_volume",
      by_day: [],
      notes: ["Demo empty state. Missing costs stay blank."],
    },
    breakdown: { models: [], judges: [], notes: [] },
    history: { items: [], limit: pageSize, offset: 0, total: 0 },
    recent: { items: [], total: 0 },
  };
}

function showDemo(message) {
  demoBanner.hidden = false;
  demoBanner.textContent = message;
}

function requestIdFromHash() {
  const match = location.hash.match(/^#request=(.+)$/);
  return match ? decodeURIComponent(match[1]) : "";
}

function finalModel(item) {
  return item.final_model || "Not recorded";
}

function fallbackLabel(value) {
  if (value === null || value === undefined) {
    return "not recorded";
  }
  return value ? "yes" : "no";
}

function blankMoney(value) {
  return value === null || value === undefined ? "unknown" : String(value);
}

function statusBadge(status) {
  const kind = status === "succeeded" ? "ok" : status === "failed" ? "bad" : status === "pending" ? "warn" : "neutral";
  return el("span", status, `badge badge-${kind}`);
}

function timeCell(value) {
  const td = document.createElement("td");
  const stack = document.createElement("div");
  stack.className = "time-stack";
  stack.append(el("span", readableUtc(value)));
  const exact = document.createElement("time");
  exact.className = "exact-time";
  const iso = exactUtc(value);
  exact.dateTime = iso;
  exact.textContent = iso;
  stack.append(exact);
  td.append(stack);
  return td;
}

function readableUtc(value) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return String(value ?? "unknown");
  }
  const formatted = new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(parsed);
  return `${formatted} UTC`;
}

function exactUtc(value) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return String(value ?? "unknown");
  }
  return parsed.toISOString().replace(".000Z", "Z");
}

function detailText(payload) {
  if (!payload || payload.detail === undefined) {
    return "Dashboard data is unavailable.";
  }
  if (typeof payload.detail === "string") {
    return payload.detail;
  }
  return "The dashboard query was rejected.";
}

function cell(text, numeric) {
  const node = el("td", text);
  if (numeric) {
    node.className = "num";
  }
  return node;
}

function emptyRow(columns, text) {
  const row = document.createElement("tr");
  const node = el("td", text);
  node.colSpan = columns;
  row.append(node);
  return row;
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined && text !== null) {
    node.textContent = String(text);
  }
  return node;
}
