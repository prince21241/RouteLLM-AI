import {
  apiQuery,
  costSeries,
  detailModel,
  detailState,
  formatCost,
  formatRate,
  panelState,
  recordedOrUnknown,
} from "./dashboard_view.mjs";

const API_BASE = window.ROUTELLM_API_BASE || "";
const pageSize = 20;

const filtersForm = document.querySelector("#filters");
const statusLine = document.querySelector("#dashboard-status");
const retry = document.querySelector("#retry");
const empty = document.querySelector("#empty");
const content = document.querySelector("#content");
const demoBanner = document.querySelector("#demo-banner");
const cards = document.querySelector("#cards");
const daysBody = document.querySelector("#days");
const notes = document.querySelector("#notes");
const modelsBody = document.querySelector("#models");
const judgesBody = document.querySelector("#judges");
const breakdownNotes = document.querySelector("#breakdown-notes");
const historyBody = document.querySelector("#history");
const pageLabel = document.querySelector("#page-label");
const prev = document.querySelector("#prev");
const next = document.querySelector("#next");
const volumeChart = document.querySelector("#volume-chart");
const costChart = document.querySelector("#cost-chart");
const detail = document.querySelector("#detail");
const detailStatus = document.querySelector("#detail-status");
const detailRetry = document.querySelector("#detail-retry");
const detailBody = document.querySelector("#detail-body");

const demoMode = new URLSearchParams(location.search).get("demo");
let offset = 0;
let latestTotal = 0;
let demoBundle = null;
let openRequestId = null;

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

historyBody.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-id]");
  if (!button) {
    return;
  }
  loadDetail(button.dataset.id);
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
    return {
      overview: demoBundle.overview,
      breakdown: demoBundle.breakdown,
      history: pageDemoHistory(demoBundle.history),
    };
  }
  demoBanner.hidden = true;
  const query = currentQuery();
  const [overview, breakdown, history] = await Promise.all([
    getJson(`/api/v1/dashboard/overview${query}`),
    getJson(`/api/v1/dashboard/breakdown${query}`),
    getJson(`/api/v1/dashboard/requests${query}`),
  ]);
  return { overview, breakdown, history };
}

function currentQuery() {
  const data = new FormData(filtersForm);
  return apiQuery({
    from: String(data.get("from") || ""),
    to: String(data.get("to") || ""),
    model: String(data.get("model") || "").trim(),
    provider: String(data.get("provider") || ""),
    status: String(data.get("status") || ""),
    limit: pageSize,
    offset,
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
  daysBody.replaceChildren();
  notes.replaceChildren();
  modelsBody.replaceChildren();
  judgesBody.replaceChildren();
  breakdownNotes.replaceChildren();
  historyBody.replaceChildren();
  clearChart(volumeChart);
  clearChart(costChart);
}

function renderOverview(overview) {
  const latency = overview.request_latency_ms;
  const attemptLatency = overview.attempt_latency_ms;
  const cardRows = [
    ["Requests", String(overview.request_count), "Logical chat requests"],
    ["Attempts", String(overview.attempt_count), "Generation calls, counted separately"],
    ["Recorded cost", formatCost(overview.request_cost), "Request totals, split by completeness"],
    ["Judge cost", formatCost(overview.judge_cost), "Recorded judge component, not added again"],
    ["Error rate", formatRate(overview.error_rate), "Failed requests. Pending is not an error."],
    ["Escalation rate", formatRate(overview.escalation_rate), "Quality escalation, not provider fallback"],
    ["Fallback rate", formatRate(overview.fallback_rate), "Uses rows with a recorded fallback flag"],
    [
      "Request latency",
      latency.p50 === null ? "unknown" : `${latency.p50} ms p50`,
      `${latency.known_count} recorded, ${latency.missing_count} missing. End-to-end.`,
    ],
    [
      "Attempt latency",
      attemptLatency.p50 === null ? "unknown" : `${attemptLatency.p50} ms p50`,
      `${attemptLatency.known_count} recorded, ${attemptLatency.missing_count} missing. Provider calls.`,
    ],
    [
      "Savings estimate",
      overview.savings_estimate === null ? "unknown" : `${overview.savings_estimate} USD`,
      "Same-token-volume estimate. The premium model was not called.",
    ],
  ];
  cards.replaceChildren(
    ...cardRows.map(([label, value, note]) => {
      const card = el("article", "", "card");
      card.append(el("span", label), el("strong", value), el("span", note));
      return card;
    }),
  );
  const series = costSeries(overview.by_day);
  drawVolume(volumeChart, series);
  drawCost(costChart, series);
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
}

function renderBreakdown(breakdown) {
  modelsBody.replaceChildren(
    ...(breakdown.models.length
      ? breakdown.models.map((item) => {
          const row = document.createElement("tr");
          row.append(
            cell(item.provider),
            cell(item.model),
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
  historyBody.replaceChildren(
    ...history.items.map((item) => {
      const row = document.createElement("tr");
      const preview = el("button", item.prompt_preview, "linkish");
      preview.type = "button";
      preview.dataset.id = item.request_id;
      const previewCell = document.createElement("td");
      previewCell.append(preview);
      if (item.prompt_truncated) {
        previewCell.append(el("span", " truncated"));
      }
      row.append(
        cell(formatTime(item.created_at), true),
        previewCell,
        cell(item.status),
        cell(`${item.provider} / ${item.model}`),
        cell(finalRoute(item)),
        cell(String(item.attempt_count), true),
        cell(item.total_cost === null ? `unknown (${item.cost_completeness})` : `${item.total_cost} USD (${item.cost_completeness})`, true),
        cell(item.end_to_end_latency_ms === null ? "unknown" : String(item.end_to_end_latency_ms), true),
        cell(item.quality_verdict || "unknown"),
        cell(item.escalated ? "yes" : "no"),
        cell(fallbackLabel(item.fallback_used)),
      );
      return row;
    }),
  );
}

async function loadDetail(requestId) {
  openRequestId = requestId;
  location.hash = `request=${requestId}`;
  detail.hidden = false;
  detailBody.replaceChildren();
  detailRetry.hidden = true;
  detailStatus.className = "";
  detailStatus.textContent = "Loading request…";
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

function renderDetail(request) {
  const model = detailModel(request);
  const mode = detailState({ loading: false, error: "", request });
  if (mode !== "populated") {
    detailStatus.textContent = "Request detail is empty.";
    return;
  }
  detailStatus.className = "";
  detailStatus.textContent = model.persistence;
  const body = document.createElement("div");
  body.append(definitionList([
    ["Status", model.status],
    ["Initial provider", model.initialProvider],
    ["Initial model", model.initialModel],
    ["Final provider", recordedOrUnknown(model.finalProvider)],
    ["Final model", recordedOrUnknown(model.finalModel)],
    ["Escalated", model.escalated ? "yes" : "no"],
    ["Escalation reason", recordedOrUnknown(model.escalationReason)],
    ["Escalation error", recordedOrUnknown(model.escalationError)],
    ["Fallback", fallbackLabel(model.fallbackUsed)],
    ["Fallback reason", recordedOrUnknown(model.fallbackReason)],
    ["Quality", recordedOrUnknown(model.qualityVerdict)],
    ["Cost", model.cost === null ? `unknown (${model.costCompleteness || "unknown"})` : `${model.cost} USD (${model.costCompleteness})`],
    ["Savings estimate", model.savings === null ? "unknown" : `${model.savings} USD`],
    ["Savings basis", model.savingsBasis || "same_token_volume"],
    ["End-to-end latency", model.latencyMs === null ? "unknown" : `${model.latencyMs} ms`],
  ]));
  body.append(el("h3", "Prompt"), el("p", model.prompt, "prompt"));
  if (model.systemPrompt) {
    body.append(el("h3", "System prompt"), el("p", model.systemPrompt, "prompt"));
  }
  body.append(el("h3", "Answer"), el("p", model.answer === null ? "Not recorded" : model.answer, "answer"));
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
  detail.scrollIntoView({ block: "start" });
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

function drawVolume(svg, series) {
  drawChart(svg, series, [
    { key: "requests", color: "#1f6b4a", label: "Requests" },
  ]);
}

function drawCost(svg, series) {
  drawChart(svg, series, [
    { key: "complete", color: "#1f6b4a", label: "Complete" },
    { key: "estimated", color: "#8a6a2f", label: "Estimated" },
  ]);
}

function drawChart(svg, series, marks) {
  clearChart(svg);
  const width = Math.max(series.length * 48, 280);
  const height = 180;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!series.length) {
    svg.append(svgText(12, 24, "No days in this range."));
    return;
  }
  const values = series.flatMap((point) => marks.map((mark) => point[mark.key])).filter((value) => value !== null);
  if (!values.length) {
    svg.append(svgText(12, 24, "No recorded values in this range."));
    return;
  }
  const max = Math.max(...values, 0);
  const top = max === 0 ? 1 : max;
  const groupWidth = width / series.length;
  const barWidth = Math.min(18, Math.max(groupWidth / (marks.length + 1), 6));
  series.forEach((point, index) => {
    marks.forEach((mark, markIndex) => {
      const value = point[mark.key];
      if (value === null) {
        return;
      }
      const barHeight = top === 0 ? 0 : (value / top) * 120;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", String(index * groupWidth + markIndex * (barWidth + 4) + 8));
      rect.setAttribute("y", String(140 - barHeight));
      rect.setAttribute("width", String(barWidth - 2));
      rect.setAttribute("height", String(Math.max(barHeight, value === 0 ? 1 : 0)));
      rect.setAttribute("fill", mark.color);
      svg.append(rect);
    });
    svg.append(svgText(index * groupWidth + 6, 168, String(point.day).slice(5)));
  });
}

function clearChart(svg) {
  svg.replaceChildren();
}

function svgText(x, y, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "text");
  node.setAttribute("x", String(x));
  node.setAttribute("y", String(y));
  node.setAttribute("fill", "#5c564c");
  node.setAttribute("font-size", "11");
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
  const filters = { start: null, end: null, model: null, provider: null, status: null };
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
      by_day: [],
      notes: ["Demo empty state. Missing costs stay blank."],
    },
    breakdown: { models: [], judges: [], notes: [] },
    history: { items: [], limit: pageSize, offset: 0, total: 0 },
    filters,
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

function finalRoute(item) {
  if (!item.final_provider && !item.final_model) {
    return "Not recorded";
  }
  return `${item.final_provider || "Not recorded"} / ${item.final_model || "Not recorded"}`;
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

function formatTime(value) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
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
