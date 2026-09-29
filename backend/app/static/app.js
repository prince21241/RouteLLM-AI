import {
  attemptRows,
  chatRequestBody,
  featureLines,
  liveDetails,
  markdownBlocks,
  providerChoices,
  reuseDraft,
  storedDetails,
  submissionState,
} from "./chat_view.mjs";

const API_BASE = window.ROUTELLM_API_BASE || "";

const form = document.querySelector("#chat-form");
const promptField = document.querySelector("#prompt");
const systemField = document.querySelector("#system-prompt");
const providerField = document.querySelector("#provider");
const providerStatus = document.querySelector("#provider-status");
const featureStatus = document.querySelector("#feature-status");
const independence = document.querySelector("#independence");
const sendButton = document.querySelector("#send");
const pendingStatus = document.querySelector("#pending-status");
const formStatus = document.querySelector("#form-status");
const errorBox = document.querySelector("#error");
const empty = document.querySelector("#empty");
const answer = document.querySelector("#answer");
const answerBanner = document.querySelector("#answer-banner");
const answerText = document.querySelector("#answer-text");
const facts = document.querySelector("#facts");
const attemptsEmpty = document.querySelector("#attempts-empty");
const attemptsTable = document.querySelector("#attempts");
const attemptsBody = document.querySelector("#attempts tbody");
const savingsNote = document.querySelector("#savings-note");
const dashboardLink = document.querySelector("#dashboard-link");
const usePromptButton = document.querySelector("#use-prompt");
const historyList = document.querySelector("#history");
const historyStatus = document.querySelector("#history-status");
const ready = document.querySelector("#ready");

let requestState = { pending: false, prompt: "", systemPrompt: "", provider: "" };
let savedDetail = null;

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void sendPrompt();
});

promptField.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    void sendPrompt();
  }
});

usePromptButton.addEventListener("click", () => {
  if (!savedDetail) {
    return;
  }
  applyDraft(reuseDraft(savedDetail));
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
  requestState = next;
  setPending(true);
  try {
    const response = await fetch(`${API_BASE}/api/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(chatRequestBody(requestState)),
    });
    const payload = await readJson(response);
    if (!response.ok) {
      showError(detailText(payload));
      return;
    }
    savedDetail = null;
    renderDetails(liveDetails(payload), payload.response || "");
    usePromptButton.hidden = true;
    if (payload.persistence_status === "stored" && payload.request_id) {
      await attachStoredAttempts(payload.request_id);
    }
    await loadHistory();
  } catch (error) {
    showError("The API could not be reached.");
  } finally {
    requestState = submissionState(requestState, { type: "finish" });
    setPending(false);
  }
}

async function openSaved(requestId) {
  hideError();
  try {
    const payload = await fetchRequest(requestId);
    savedDetail = payload;
    const text = payload.response || "No answer was stored.";
    renderDetails(storedDetails(payload), text);
    renderAttempts(payload.attempts);
    usePromptButton.hidden = false;
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
  } catch (error) {
    showError(error instanceof Error ? error.message : "The stored prompt could not be loaded.");
  }
}

function applyDraft(draft) {
  promptField.value = draft.prompt;
  systemField.value = draft.systemPrompt;
  formStatus.textContent = "Prompt copied into the editor. It has not been sent.";
  promptField.focus();
}

async function attachStoredAttempts(requestId) {
  try {
    const payload = await fetchRequest(requestId);
    renderAttempts(payload.attempts);
  } catch (error) {
    attemptsEmpty.hidden = false;
    attemptsEmpty.textContent = "The answer was saved, and the attempt list could not be loaded.";
  }
}

async function fetchRequest(requestId) {
  const response = await fetch(`${API_BASE}/api/v1/requests/${requestId}`);
  const payload = await readJson(response);
  if (!response.ok) {
    throw new Error(detailText(payload));
  }
  return payload;
}

function renderDetails(details, text) {
  empty.hidden = true;
  answer.hidden = false;
  answerBanner.textContent = details.banner;
  renderMarkdown(answerText, markdownBlocks(text));
  fillFacts(details.rows);
  renderAttempts([]);
  attemptsEmpty.hidden = false;
  attemptsEmpty.textContent =
    details.mode === "saved"
      ? "No attempt rows were stored."
      : "Attempt rows appear when the saved request is loaded.";
  savingsNote.textContent =
    "Savings are an estimate. The premium price book is applied to the returned answer’s tokens. That baseline model is not called.";
  if (details.dashboardHref) {
    dashboardLink.hidden = false;
    dashboardLink.href = details.dashboardHref;
  } else {
    dashboardLink.hidden = true;
    dashboardLink.removeAttribute("href");
  }
}

function renderAttempts(attempts) {
  const rows = attemptRows(attempts);
  attemptsBody.replaceChildren();
  attemptsTable.hidden = rows.length === 0;
  if (rows.length === 0) {
    return;
  }
  attemptsEmpty.hidden = true;
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of [row.number, row.purpose, row.provider, row.model, row.status, row.tokens, row.latency, row.cost]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      tr.append(cell);
    }
    attemptsBody.append(tr);
  }
}

function renderMarkdown(container, blocks) {
  container.replaceChildren();
  for (const block of blocks) {
    if (block.type === "code") {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = block.text;
      if (block.language) {
        const label = document.createElement("span");
        label.className = "code-label";
        label.textContent = block.language;
        pre.append(label);
      }
      pre.append(code);
      container.append(pre);
      continue;
    }
    if (block.type === "heading") {
      const heading = document.createElement(block.level === 1 ? "h3" : "h4");
      appendInlines(heading, block.inlines);
      container.append(heading);
      continue;
    }
    if (block.type === "list") {
      const list = document.createElement(block.ordered ? "ol" : "ul");
      for (const item of block.items) {
        const li = document.createElement("li");
        appendInlines(li, item);
        list.append(li);
      }
      container.append(list);
      continue;
    }
    const paragraph = document.createElement("p");
    appendInlines(paragraph, block.inlines);
    container.append(paragraph);
  }
}

function appendInlines(parent, tokens) {
  for (const token of tokens) {
    if (token.type === "text") {
      parent.append(document.createTextNode(token.text));
    } else if (token.type === "code") {
      const code = document.createElement("code");
      code.textContent = token.text;
      parent.append(code);
    } else if (token.type === "strong") {
      const strong = document.createElement("strong");
      strong.textContent = token.text;
      parent.append(strong);
    } else if (token.type === "em") {
      const em = document.createElement("em");
      em.textContent = token.text;
      parent.append(em);
    } else if (token.type === "link") {
      const link = document.createElement("a");
      link.href = token.href;
      link.rel = "noopener noreferrer";
      link.target = "_blank";
      link.textContent = token.text;
      parent.append(link);
    }
  }
}

function fillFacts(rows) {
  facts.replaceChildren();
  for (const [label, value] of rows) {
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = value;
    facts.append(term, description);
  }
}

async function loadHistory() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/requests?limit=20`);
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
      open.dataset.action = "open";
      open.dataset.id = item.request_id;
      open.textContent = `${item.model} · ${item.status}`;
      const meta = document.createElement("small");
      meta.textContent = `${item.provider} · ${item.total_cost == null || item.total_cost === "" ? "cost unknown" : item.total_cost}`;
      open.append(meta);
      const reuse = document.createElement("button");
      reuse.type = "button";
      reuse.dataset.action = "reuse";
      reuse.dataset.id = item.request_id;
      reuse.textContent = "Use this prompt";
      row.append(open, reuse);
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
      return;
    }
    independence.textContent = payload.context_note;
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
  } catch (error) {
    providerStatus.textContent = "Routing choices could not be loaded. Automatic routing is still available.";
  }
}

async function loadReady() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    ready.textContent = response.ok ? "API is up. Local use only." : "API health check failed";
  } catch (error) {
    ready.textContent = "API is unreachable";
  }
}

function setPending(pending) {
  form.setAttribute("aria-busy", pending ? "true" : "false");
  sendButton.disabled = pending;
  sendButton.textContent = pending ? "Sending…" : "Send";
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

loadReady();
loadOptions();
loadHistory();
