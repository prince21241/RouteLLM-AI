import {
  attemptRows,
  chatRequestBody,
  featureLines,
  liveDetails,
  providerChoices,
  providerLabel,
  reuseDraft,
  storedDetails,
  submissionState,
} from "./chat_view.mjs?v=3";
import { renderMarkdown } from "./markdown_dom.mjs?v=3";

const API_BASE = window.ROUTELLM_API_BASE || "";
const EXAMPLES = document.querySelector("#examples");

const form = document.querySelector("#chat-form");
const promptField = document.querySelector("#prompt");
const systemField = document.querySelector("#system-prompt");
const providerField = document.querySelector("#chat-provider");
const providerStatus = document.querySelector("#provider-status");
const featureStatus = document.querySelector("#feature-status");
const contextNote = document.querySelector("#context-note");
const sendButton = document.querySelector("#send");
const pendingStatus = document.querySelector("#pending-status");
const formStatus = document.querySelector("#form-status");
const errorBox = document.querySelector("#error");
const empty = document.querySelector("#chat-empty");
const thread = document.querySelector("#thread");
const messages = document.querySelector("#messages");
const optionsToggle = document.querySelector("#options-toggle");
const optionsPanel = document.querySelector("#options-panel");
const pageChat = document.querySelector("#page-chat");
const usePromptButton = document.querySelector("#use-prompt");
const historyList = document.querySelector("#chat-history");
const historyStatus = document.querySelector("#history-status");
const historyToggle = document.querySelector("#history-toggle");
const historyPanel = document.querySelector("#chat-history-panel");
const historyClose = document.querySelector("#history-close");
const availability = document.querySelector("#availability");
const ready = document.querySelector("#ready");

const turns = [];
let turnSerial = 0;
let requestState = { pending: false, prompt: "", systemPrompt: "", provider: "" };
let savedDetail = null;
let composing = false;
let following = true;
let ignoreScroll = false;

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void sendPrompt();
});

promptField.addEventListener("compositionstart", () => {
  composing = true;
});

promptField.addEventListener("compositionend", () => {
  composing = false;
});

promptField.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) {
    return;
  }
  if (event.isComposing || composing || event.keyCode === 229) {
    return;
  }
  event.preventDefault();
  void sendPrompt();
});

promptField.addEventListener("input", resizePrompt);
promptField.addEventListener("focus", () => {
  window.setTimeout(fitChatToViewport, 50);
});

EXAMPLES.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-example]");
  if (!button) {
    return;
  }
  promptField.value = button.dataset.example || "";
  formStatus.textContent = "";
  resizePrompt();
  promptField.focus();
});

optionsToggle.addEventListener("click", () => {
  setOptionsOpen(optionsPanel.hidden);
});

usePromptButton.addEventListener("click", () => {
  if (!savedDetail) {
    return;
  }
  applyDraft(reuseDraft(savedDetail));
});

historyToggle.addEventListener("click", () => setHistoryOpen(historyPanel.hidden));
historyClose.addEventListener("click", () => {
  setHistoryOpen(false);
  historyToggle.focus();
});

historyList.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-id]");
  if (!button) {
    return;
  }
  const requestId = button.dataset.id;
  if (button.dataset.action === "reuse") {
    void reuseSaved(requestId);
    return;
  }
  void openSaved(requestId);
});

messages.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-copy]");
  if (!button) {
    return;
  }
  const turn = turns.find((item) => item.key === button.dataset.copy);
  if (!turn) {
    return;
  }
  void copyAnswer(button, turn.answer);
});

thread.addEventListener("scroll", () => {
  if (ignoreScroll) {
    return;
  }
  following = thread.scrollHeight - thread.scrollTop - thread.clientHeight <= 80;
});

thread.addEventListener("click", (event) => {
  const summary = event.target instanceof Element ? event.target.closest("summary") : null;
  const details = summary ? summary.parentElement : null;
  if (!(details instanceof HTMLDetailsElement) || !details.classList.contains("routing")) {
    return;
  }
  window.requestAnimationFrame(() => {
    if (!details.open) {
      return;
    }
    const threadBox = thread.getBoundingClientRect();
    const box = details.getBoundingClientRect();
    const room = threadBox.height - 16;
    const delta = box.height > room ? box.top - threadBox.top - 8 : box.bottom - (threadBox.bottom - 8);
    if (delta > 0) {
      ignoreScroll = true;
      thread.scrollTop += delta;
      ignoreScroll = false;
    }
  });
});

async function sendPrompt() {
  hideError();
  const next = submissionState(requestState, {
    type: "start",
    prompt: promptField.value,
    systemPrompt: systemField.value,
    provider: providerField.value,
  });
  if (!next.accepted) {
    if (next.error) {
      showError(next.error);
    }
    return;
  }
  const submittedPrompt = promptField.value;
  requestState = next;
  formStatus.textContent = "";
  promptField.value = "";
  resizePrompt();
  const turn = {
    key: newKey(),
    prompt: submittedPrompt,
    status: "pending",
    answer: "",
    error: "",
    details: null,
    attempts: [],
    attemptsNote: "",
    saved: false,
    requestId: "",
  };
  turns.push(turn);
  setPending(true);
  renderMessages({ scroll: following });
  try {
    const response = await fetch(`${API_BASE}/api/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(chatRequestBody(requestState)),
    });
    const payload = await readJson(response);
    if (!response.ok) {
      failTurn(turn, submittedPrompt, detailText(payload));
      return;
    }
    savedDetail = null;
    usePromptButton.hidden = true;
    turn.status = "answered";
    turn.answer = payload.response || "";
    turn.details = liveDetails(payload);
    turn.requestId = payload.request_id || "";
    renderMessages({ scroll: following });
    if (payload.persistence_status === "stored" && payload.request_id) {
      await attachStoredAttempts(turn.key, payload.request_id);
    }
    await loadHistory();
  } catch (error) {
    failTurn(turn, submittedPrompt, "The API could not be reached.");
  } finally {
    requestState = submissionState(requestState, { type: "finish" });
    setPending(false);
  }
}

function failTurn(turn, submittedPrompt, message) {
  turn.status = "error";
  turn.error = message;
  if (!promptField.value) {
    promptField.value = submittedPrompt;
    resizePrompt();
  }
  showError(message);
  renderMessages({ scroll: following });
}

async function openSaved(requestId) {
  hideError();
  try {
    const payload = await fetchRequest(requestId);
    savedDetail = payload;
    const existing = turns.find((turn) => turn.requestId && turn.requestId === payload.request_id);
    const turn = existing || {
      key: newKey(),
      prompt: "",
      status: "answered",
      answer: "",
      error: "",
      details: null,
      attempts: [],
      attemptsNote: "",
      saved: true,
      requestId: "",
    };
    turn.prompt = payload.prompt || "";
    turn.status = payload.response ? "answered" : "error";
    turn.answer = payload.response || "";
    turn.error = payload.response ? "" : "No answer was stored.";
    turn.details = storedDetails(payload);
    turn.attempts = payload.attempts || [];
    turn.saved = true;
    turn.requestId = payload.request_id || requestId;
    if (!existing) {
      turns.push(turn);
    }
    usePromptButton.hidden = false;
    renderMessages({ scroll: true });
    setHistoryOpen(false);
  } catch (error) {
    showError(error instanceof Error ? error.message : "The stored request could not be loaded.");
  }
}

async function reuseSaved(requestId) {
  hideError();
  try {
    const payload = await fetchRequest(requestId);
    savedDetail = payload;
    applyDraft(reuseDraft(payload));
    usePromptButton.hidden = false;
    setHistoryOpen(false);
  } catch (error) {
    showError(error instanceof Error ? error.message : "The stored prompt could not be loaded.");
  }
}

function setHistoryOpen(open) {
  historyPanel.hidden = !open;
  historyToggle.setAttribute("aria-expanded", open ? "true" : "false");
  if (open) {
    historyClose.focus();
  }
}

function setOptionsOpen(open) {
  optionsPanel.hidden = !open;
  optionsToggle.setAttribute("aria-expanded", open ? "true" : "false");
}

function applyDraft(draft) {
  promptField.value = draft.prompt;
  systemField.value = draft.systemPrompt;
  formStatus.textContent = "Prompt copied into the editor. It has not been sent.";
  resizePrompt();
  promptField.focus();
}

async function attachStoredAttempts(key, requestId) {
  const turn = turns.find((item) => item.key === key);
  if (!turn) {
    return;
  }
  try {
    const payload = await fetchRequest(requestId);
    turn.attempts = payload.attempts || [];
    turn.attemptsNote = "";
  } catch (error) {
    turn.attemptsNote = "The answer was saved, and the attempt list could not be loaded.";
  }
  renderMessages();
}

async function fetchRequest(requestId) {
  const response = await fetch(`${API_BASE}/api/v1/requests/${requestId}`);
  const payload = await readJson(response);
  if (!response.ok) {
    throw new Error(detailText(payload));
  }
  return payload;
}

function renderMessages(options = {}) {
  const previous = thread.scrollTop;
  empty.hidden = turns.length > 0;
  messages.replaceChildren();
  for (const turn of turns) {
    messages.append(renderTurn(turn));
  }
  if (options.scroll) {
    scrollLatestIntoView();
    return;
  }
  ignoreScroll = true;
  thread.scrollTop = previous;
  ignoreScroll = false;
}

function renderTurn(turn) {
  const item = document.createElement("li");
  item.className = "message";
  item.dataset.key = turn.key;

  const user = document.createElement("div");
  user.className = "user-bubble";
  user.textContent = turn.prompt;
  item.append(user);

  const assistant = document.createElement("div");
  assistant.className = "assistant";
  if (turn.status === "pending") {
    const pending = document.createElement("div");
    pending.className = "pending-row";
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    spinner.setAttribute("aria-hidden", "true");
    const label = document.createElement("span");
    label.textContent = "Routing this prompt…";
    pending.append(spinner, label);
    assistant.append(pending);
  } else if (turn.status === "error") {
    const error = document.createElement("p");
    error.className = "answer-error";
    error.textContent = turn.error;
    assistant.append(error);
  } else {
    if (turn.details && turn.details.banner) {
      const banner = document.createElement("p");
      banner.className = "saved-banner";
      banner.textContent = turn.details.banner;
      assistant.append(banner);
    }
    const body = document.createElement("div");
    body.className = "markdown";
    renderMarkdown(body, turn.answer);
    const actions = document.createElement("div");
    actions.className = "answer-actions";
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "secondary";
    copy.dataset.copy = turn.key;
    copy.textContent = "Copy";
    actions.append(copy);
    assistant.append(body, actions, renderRouting(turn));
  }
  item.append(assistant);
  return item;
}

function renderRouting(turn) {
  const details = document.createElement("details");
  details.className = "routing";
  const summary = document.createElement("summary");
  summary.textContent = routingSummary(turn.details);
  details.append(summary);
  if (turn.details) {
    const facts = document.createElement("dl");
    facts.className = "facts";
    for (const [label, value] of turn.details.rows) {
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = value;
      facts.append(term, description);
    }
    details.append(facts);
  }
  const heading = document.createElement("h3");
  heading.textContent = "Attempts";
  details.append(heading);
  const rows = attemptRows(turn.attempts);
  if (turn.attemptsNote) {
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = turn.attemptsNote;
    details.append(note);
  } else if (rows.length === 0) {
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = turn.saved ? "No attempt rows were stored." : "Attempt rows appear when the saved request is loaded.";
    details.append(note);
  }
  if (rows.length > 0) {
    details.append(renderAttempts(rows));
  }
  const savings = document.createElement("p");
  savings.className = "note";
  savings.textContent =
    "Savings are an estimate. The premium price book is applied to the returned answer’s tokens. That baseline model is not called.";
  details.append(savings);
  if (turn.details && turn.details.dashboardHref) {
    const linkRow = document.createElement("p");
    const link = document.createElement("a");
    link.href = turn.details.dashboardHref;
    link.textContent = "Open this request";
    linkRow.append(link);
    details.append(linkRow);
  }
  return details;
}

function renderAttempts(rows) {
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  table.className = "attempts";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of ["#", "Purpose", "Provider", "Model", "Status", "Tokens", "Latency", "Cost"]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    headRow.append(cell);
  }
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const row of rows) {
    const line = document.createElement("tr");
    for (const value of [row.number, row.purpose, row.provider, row.model, row.status, row.tokens, row.latency, row.cost]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      line.append(cell);
    }
    body.append(line);
  }
  table.append(head, body);
  wrap.append(table);
  return wrap;
}

function routingSummary(details) {
  if (!details) {
    return "Routing details";
  }
  const model = details.rows.find(([label]) => label === "Final model");
  const cost = details.rows.find(([label]) => label === "Recorded cost");
  return ["Routing details", model && model[1], cost && cost[1]].filter(Boolean).join(" · ");
}

function scrollLatestIntoView() {
  const latest = messages.lastElementChild;
  if (!latest) {
    return;
  }
  ignoreScroll = true;
  const top = latest.getBoundingClientRect().top - thread.getBoundingClientRect().top + thread.scrollTop;
  thread.scrollTop = Math.max(0, top - 12);
  ignoreScroll = false;
  following = true;
}

function resizePrompt() {
  promptField.style.height = "auto";
  const max = 160;
  const next = Math.min(promptField.scrollHeight, max);
  promptField.style.height = `${Math.max(next, 24)}px`;
  promptField.style.overflowY = promptField.scrollHeight > max ? "auto" : "hidden";
}

function syncChatScrollbar() {
  if (!thread || document.body.dataset.route !== "chat") {
    document.documentElement.style.removeProperty("--chat-scrollbar");
    return;
  }
  const width = Math.max(0, thread.offsetWidth - thread.clientWidth);
  document.documentElement.style.setProperty("--chat-scrollbar", `${width}px`);
}

function fitChatToViewport() {
  syncChatScrollbar();
  if (!pageChat || document.body.dataset.route !== "chat") {
    if (pageChat) {
      pageChat.style.height = "";
    }
    return;
  }
  const viewport = window.visualViewport;
  const main = document.querySelector("#main");
  if (!viewport || !main) {
    return;
  }
  const layoutBottom = window.innerHeight;
  const visibleBottom = viewport.offsetTop + viewport.height;
  const keyboardOpen = viewport.offsetTop > 0 || layoutBottom - visibleBottom > 40;
  if (!keyboardOpen) {
    pageChat.style.height = "";
    return;
  }
  const height = Math.min(main.clientHeight, visibleBottom - main.getBoundingClientRect().top);
  pageChat.style.height = `${Math.max(180, height)}px`;
}

async function copyAnswer(button, text) {
  let copied = false;
  try {
    if (navigator.clipboard && window.isSecureContext && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      copied = true;
    }
  } catch (error) {
    copied = false;
  }
  if (!copied) {
    try {
      copyWithSelection(text);
      copied = true;
    } catch (error) {
      copied = false;
    }
  }
  if (!copied && selectAnswer(button)) {
    button.textContent = "Selected";
  } else {
    button.textContent = copied ? "Copied" : "Copy failed";
  }
  window.setTimeout(() => {
    if (button.isConnected) {
      button.textContent = "Copy";
    }
  }, 1500);
}

function selectAnswer(button) {
  const markdown = button.closest(".assistant")?.querySelector(".markdown");
  if (!markdown) {
    return false;
  }
  const range = document.createRange();
  range.selectNodeContents(markdown);
  const selection = window.getSelection();
  if (!selection) {
    return false;
  }
  selection.removeAllRanges();
  selection.addRange(range);
  return true;
}

function copyWithSelection(text) {
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.top = "0";
  area.style.left = "0";
  area.style.width = "2rem";
  area.style.height = "2rem";
  area.style.opacity = "0";
  document.body.append(area);
  area.focus();
  area.select();
  area.setSelectionRange(0, area.value.length);
  const copied = document.execCommand("copy");
  area.remove();
  if (!copied) {
    throw new Error("copy failed");
  }
}

function newKey() {
  turnSerial += 1;
  return `turn-${turnSerial}`;
}

async function loadHistory() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/dashboard/requests?limit=20`);
    const payload = await readJson(response);
    if (!response.ok) {
      historyStatus.hidden = false;
      historyStatus.textContent = detailText(payload);
      historyList.replaceChildren();
      return;
    }
    historyList.replaceChildren();
    if (!payload.items || payload.items.length === 0) {
      historyStatus.hidden = false;
      historyStatus.textContent = "No stored requests yet.";
      return;
    }
    historyStatus.hidden = true;
    for (const item of payload.items) {
      const row = document.createElement("li");
      const open = document.createElement("button");
      open.type = "button";
      open.className = "linkish";
      open.dataset.action = "open";
      open.dataset.id = item.request_id;
      open.textContent = item.prompt_preview || "No preview";
      const badge = document.createElement("span");
      badge.className = `badge badge-${item.status === "succeeded" ? "ok" : item.status === "failed" ? "bad" : item.status === "pending" ? "warn" : "neutral"}`;
      badge.textContent = item.status;
      const meta = document.createElement("small");
      const cost = item.total_cost == null || item.total_cost === "" ? "cost unknown" : `${item.total_cost} USD`;
      meta.textContent = `${item.final_model || item.model} · ${cost}`;
      const reuse = document.createElement("button");
      reuse.type = "button";
      reuse.className = "secondary";
      reuse.dataset.action = "reuse";
      reuse.dataset.id = item.request_id;
      reuse.textContent = "Use this prompt";
      row.append(open, badge, meta, reuse);
      historyList.append(row);
    }
  } catch (error) {
    historyStatus.hidden = false;
    historyStatus.textContent = "History is unavailable.";
  }
}

async function loadOptions() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/chat/options`);
    const payload = await readJson(response);
    if (!response.ok) {
      providerStatus.textContent = detailText(payload);
      renderAvailability(null, detailText(payload));
      return;
    }
    contextNote.textContent = payload.context_note || "";
    const choices = providerChoices(payload);
    const selected = providerField.value;
    providerField.replaceChildren();
    for (const choice of choices) {
      const option = document.createElement("option");
      option.value = choice.value;
      option.textContent = choice.label;
      option.disabled = !choice.enabled;
      providerField.append(option);
    }
    if ([...providerField.options].some((option) => option.value === selected && !option.disabled)) {
      providerField.value = selected;
    }
    providerStatus.textContent = choices
      .filter((choice) => choice.value)
      .map((choice) => choice.detail)
      .join(" ");
    featureStatus.replaceChildren();
    for (const line of featureLines(payload)) {
      const item = document.createElement("li");
      item.textContent = line;
      featureStatus.append(item);
    }
    renderAvailability(payload.models);
  } catch (error) {
    providerStatus.textContent = "Routing choices could not be loaded. Automatic routing is still available.";
    renderAvailability(null, "Availability could not be loaded.");
  }
}

function renderAvailability(models, failure) {
  availability.replaceChildren();
  if (failure) {
    const item = document.createElement("li");
    item.textContent = failure;
    availability.append(item);
    return;
  }
  if (!models || models.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No catalog models were returned.";
    availability.append(item);
    return;
  }
  for (const model of models) {
    const item = document.createElement("li");
    const badge = document.createElement("span");
    let label = "Not reported";
    let kind = "neutral";
    if (model.enabled === true) {
      label = "Enabled";
      kind = "ok";
    } else if (model.enabled === false) {
      label = "Not enabled";
    }
    badge.className = `badge badge-${kind}`;
    badge.textContent = label;
    const name = document.createElement("span");
    name.className = "model-name";
    name.textContent = `${providerLabel(model.provider)} · ${model.model_id}`;
    item.append(badge, name);
    availability.append(item);
  }
}

async function loadReady() {
  try {
    const health = await fetch(`${API_BASE}/health`);
    if (!health.ok) {
      ready.dataset.state = "down";
      ready.textContent = "API health check failed. Local use only.";
      return;
    }
    const database = await fetch(`${API_BASE}/ready`);
    if (database.ok) {
      ready.dataset.state = "up";
      ready.textContent = "API connected. Database ready. Local use only.";
      return;
    }
    ready.dataset.state = "warn";
    ready.textContent = "API connected. Database is not ready. Local use only.";
  } catch (error) {
    ready.dataset.state = "down";
    ready.textContent = "API is unreachable. Local use only.";
  }
}

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") {
    return;
  }
  const requestDetail = document.querySelector("#detail");
  if (requestDetail && !requestDetail.hidden) {
    return;
  }
  if (!optionsPanel.hidden) {
    setOptionsOpen(false);
    optionsToggle.focus();
    return;
  }
  if (historyPanel.hidden) {
    return;
  }
  setHistoryOpen(false);
  historyToggle.focus();
});

function setPending(pending) {
  form.setAttribute("aria-busy", pending ? "true" : "false");
  sendButton.disabled = pending;
  sendButton.textContent = pending ? "Sending…" : "Send";
  sendButton.setAttribute("aria-label", pending ? "Sending" : "Send message");
  pendingStatus.hidden = !pending;
}

async function readJson(response) {
  try {
    return await response.json();
  } catch (error) {
    return null;
  }
}

function detailText(payload) {
  if (!payload || payload.detail === undefined) {
    return "The request failed.";
  }
  if (typeof payload.detail === "string") {
    return payload.detail;
  }
  return "The request was rejected.";
}

function showError(message) {
  errorBox.hidden = false;
  errorBox.textContent = message;
}

function hideError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

if (window.visualViewport) {
  window.visualViewport.addEventListener("resize", fitChatToViewport);
  window.visualViewport.addEventListener("scroll", fitChatToViewport);
}
window.addEventListener("resize", fitChatToViewport);
new MutationObserver(fitChatToViewport).observe(document.body, {
  attributes: true,
  attributeFilter: ["data-route"],
});

resizePrompt();
fitChatToViewport();
loadReady();
loadOptions();
loadHistory();
