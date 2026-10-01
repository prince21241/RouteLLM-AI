/** Pure helpers for the chat page. DOM rendering stays in app.js. */

const PROVIDER_LABELS = {
  openai: "OpenAI",
  anthropic: "Claude",
  ollama: "Ollama",
};

export function providerLabel(provider) {
  return PROVIDER_LABELS[provider] || String(provider || "Unknown");
}

export function providerChoices(options) {
  const groups = new Map();
  for (const model of options?.models || []) {
    const group = groups.get(model.provider) || [];
    group.push(model);
    groups.set(model.provider, group);
  }
  const choices = [{ value: "", label: "Automatic routing", enabled: true, detail: "The server chooses an enabled model." }];
  for (const [provider, models] of groups) {
    const enabled = models.filter((model) => model.enabled);
    const names = (enabled.length ? enabled : models).map((model) => model.model_id).join(", ");
    choices.push({
      value: provider,
      label: enabled.length
        ? `${providerLabel(provider)} only`
        : `${providerLabel(provider)} only (not available)`,
      enabled: enabled.length > 0,
      detail: enabled.length ? `Available: ${names}` : `Not available: ${names}`,
    });
  }
  return choices;
}

export function featureLines(options) {
  return [
    featureLine("Quality checks", "are", options?.quality_evaluation),
    featureLine("Escalation", "is", options?.escalation),
    featureLine("Provider fallback", "is", options?.fallback),
    "Routing stays automatic. You can limit the provider. This page cannot name a model.",
  ];
}

export function submissionState(current, action) {
  if (action.type === "finish") {
    return { ...current, pending: false };
  }
  if (current.pending) {
    return { ...current, accepted: false };
  }
  const prompt = String(action.prompt ?? "");
  if (prompt.trim() === "") {
    return {
      ...current,
      pending: false,
      accepted: false,
      error: "Prompt must not be blank",
    };
  }
  return {
    pending: true,
    accepted: true,
    error: "",
    prompt,
    systemPrompt: action.systemPrompt ?? "",
    provider: action.provider || "",
  };
}

export function chatRequestBody(state) {
  const body = { prompt: state.prompt };
  const systemPrompt = String(state.systemPrompt ?? "").trim();
  if (systemPrompt) {
    body.system_prompt = systemPrompt;
  }
  if (state.provider) {
    body.provider = state.provider;
  }
  return body;
}

export function reuseDraft(detail) {
  return {
    prompt: detail?.prompt || "",
    systemPrompt: detail?.system_prompt || "",
    submit: false,
  };
}

export function markdownBlocks(source) {
  const lines = String(source ?? "").split(/\r?\n/);
  const blocks = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    const fence = line.match(/^```([A-Za-z0-9_-]*)$/);
    if (fence) {
      const code = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      blocks.push({ type: "code", language: fence[1], text: code.join("\n") });
      continue;
    }
    if (line.trim() === "") {
      index += 1;
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.*)$/);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, inlines: inlineTokens(heading[2]) });
      index += 1;
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index])) {
        items.push(inlineTokens(lines[index].replace(/^[-*]\s+/, "")));
        index += 1;
      }
      blocks.push({ type: "list", ordered: false, items });
      continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index])) {
        items.push(inlineTokens(lines[index].replace(/^\d+\.\s+/, "")));
        index += 1;
      }
      blocks.push({ type: "list", ordered: true, items });
      continue;
    }
    if (/^>\s?/.test(line)) {
      const quote = [];
      while (index < lines.length && /^>\s?/.test(lines[index])) {
        quote.push(lines[index].replace(/^>\s?/, ""));
        index += 1;
      }
      blocks.push({ type: "quote", inlines: inlineTokens(quote.join("\n")) });
      continue;
    }
    if (/^(?:---|\*\*\*|___)\s*$/.test(line)) {
      blocks.push({ type: "rule" });
      index += 1;
      continue;
    }
    const paragraph = [];
    while (index < lines.length && lines[index].trim() !== "" && !isBlockStart(lines[index])) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push({ type: "paragraph", inlines: inlineTokens(paragraph.join("\n")) });
  }
  return blocks;
}

export function liveDetails(payload) {
  const routing = payload?.routing || {};
  const metrics = payload?.metrics || {};
  const status = payload?.persistence_status;
  return {
    mode: "live",
    banner: liveBanner(status),
    rows: [
      ["Initial model", recorded(routing.model)],
      ["Initial provider", recorded(routing.provider)],
      ["Final model", recorded(payload?.returned_model)],
      ["Final provider", recorded(payload?.final_provider)],
      ["Routing reason", recorded(routing.selection_reason)],
      ["Complexity", complexityText(routing)],
      ["Quality", verdictText(payload?.quality_verdict)],
      ["Quality reasons", listText(payload?.quality_reasons)],
      ["Escalated", boolText(routing.escalated)],
      ["Escalation reason", recorded(payload?.escalation_reason)],
      ["Escalation error", recorded(payload?.escalation_error)],
      ["Fallback", boolText(payload?.fallback_used)],
      ["Fallback reason", recorded(payload?.fallback_reason)],
      ["Fallback skips", skipText(payload?.fallback_skips)],
      ["Input tokens", numberText(metrics.input_tokens)],
      ["Output tokens", numberText(metrics.output_tokens)],
      ["Provider latency", latencyText(metrics.latency_ms)],
      ["End-to-end latency", latencyText(payload?.end_to_end_latency_ms)],
      ["Recorded cost", moneyText(metrics.cost, metrics.cost_completeness)],
      ["Savings estimate", savingsText(payload?.estimated_savings, payload?.savings_basis)],
      ["Persistence", persistenceLabel(status, payload?.persistence_warning)],
    ],
    dashboardHref: status === "stored" && payload?.request_id ? dashboardHref(payload.request_id) : null,
  };
}

export function storedDetails(payload) {
  return {
    mode: "saved",
    banner: "Saved request. This is not a new draft. Send the editor to make another request.",
    rows: [
      ["Initial model", recorded(payload?.model)],
      ["Initial provider", recorded(payload?.provider)],
      ["Final model", recorded(payload?.final_model)],
      ["Final provider", recorded(payload?.final_provider)],
      ["Routing reason", recorded(payload?.selection_reason)],
      ["Complexity", complexityText(payload)],
      ["Quality", verdictText(payload?.quality_verdict)],
      ["Quality reasons", listText(payload?.quality_reasons)],
      ["Escalated", boolText(payload?.escalated)],
      ["Escalation reason", recorded(payload?.escalation_reason)],
      ["Escalation error", recorded(payload?.escalation_error)],
      ["Fallback", boolText(payload?.fallback_used)],
      ["Fallback reason", recorded(payload?.fallback_reason)],
      ["Fallback skips", skipText(payload?.fallback_skips)],
      ["End-to-end latency", latencyText(payload?.end_to_end_latency_ms)],
      ["Recorded cost", moneyText(payload?.total_cost, payload?.cost_completeness)],
      ["Savings estimate", savingsText(payload?.estimated_savings, payload?.savings_basis)],
      ["Persistence", "Read from history"],
      ["Status", recorded(payload?.status)],
    ],
    dashboardHref: payload?.request_id ? dashboardHref(payload.request_id) : null,
  };
}

export function attemptRows(attempts) {
  return [...(attempts || [])]
    .map((attempt, index) => ({ attempt, index }))
    .sort((left, right) => {
      const leftNumber = left.attempt.attempt_number ?? Number.MAX_SAFE_INTEGER;
      const rightNumber = right.attempt.attempt_number ?? Number.MAX_SAFE_INTEGER;
      return leftNumber - rightNumber || left.index - right.index;
    })
    .map(({ attempt }) => ({
      number: numberText(attempt.attempt_number),
      purpose: attempt.purpose || "not recorded",
      provider: recorded(attempt.provider),
      model: recorded(attempt.configured_model_id),
      status: recorded(attempt.status),
      tokens: `${numberText(attempt.input_tokens)} in / ${numberText(attempt.output_tokens)} out`,
      latency: latencyText(attempt.provider_latency_ms),
      cost: moneyText(attempt.estimated_cost, attempt.cost_completeness),
    }));
}

export function dashboardHref(requestId) {
  return `/dashboard#request=${encodeURIComponent(requestId)}`;
}

function featureLine(name, verb, feature) {
  const state = feature?.enabled ? "on" : "off";
  return `${name} ${verb} ${state}. The server controls this. This page does not switch it.`;
}

function liveBanner(status) {
  if (status === "stored") {
    return "New answer. This request was saved.";
  }
  if (status === "failed") {
    return "New answer. It was not saved.";
  }
  if (status === "not_configured") {
    return "New answer. History is not configured, so it was not saved.";
  }
  return "New answer.";
}

function persistenceLabel(status, warning) {
  if (status === "stored") {
    return "Saved in history";
  }
  if (status === "failed") {
    return warning || "The answer was returned, and saving it failed";
  }
  if (status === "not_configured") {
    return "History is not configured, so this answer was not saved";
  }
  return "not recorded";
}

function recorded(value) {
  if (value === null || value === undefined || value === "") {
    return "not recorded";
  }
  return String(value);
}

function boolText(value) {
  if (value === true) {
    return "yes";
  }
  if (value === false) {
    return "no";
  }
  return "not recorded";
}

function verdictText(value) {
  if (value === null || value === undefined || value === "") {
    return "not recorded";
  }
  return String(value);
}

function listText(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return "not recorded";
  }
  return values.join("; ");
}

function skipText(skips) {
  if (!Array.isArray(skips) || skips.length === 0) {
    return "not recorded";
  }
  return skips.map((skip) => `${skip.model_id}: ${skip.reason}`).join("; ");
}

function complexityText(routing) {
  if (!routing || routing.complexity_tier === null || routing.complexity_tier === undefined || routing.complexity_tier === "") {
    return "not recorded";
  }
  if (routing.complexity_score === null || routing.complexity_score === undefined || routing.complexity_score === "") {
    return String(routing.complexity_tier);
  }
  return `${routing.complexity_tier} (${routing.complexity_score})`;
}

export function numberText(value) {
  if (value === null || value === undefined || value === "") {
    return "unknown";
  }
  return String(value);
}

export function latencyText(value) {
  if (value === null || value === undefined || value === "") {
    return "unknown";
  }
  return `${value} ms`;
}

export function moneyText(amount, completeness) {
  if (amount === null || amount === undefined || amount === "") {
    return "unknown";
  }
  return completeness ? `${amount} USD (${completeness})` : `${amount} USD`;
}

function savingsText(amount, basis) {
  const money = moneyText(amount);
  const label = basis || "same_token_volume";
  if (money === "unknown") {
    return `unknown (${label} estimate)`;
  }
  return `${money} (${label} estimate)`;
}

function isBlockStart(line) {
  return /^```/.test(line) || /^#{1,3}\s+/.test(line) || /^[-*]\s+/.test(line) || /^\d+\.\s+/.test(line) || /^>\s?/.test(line) || /^(?:---|\*\*\*|___)\s*$/.test(line);
}

export function inlineTokens(input) {
  const text = String(input ?? "");
  const tokens = [];
  let buffer = "";
  let index = 0;
  const pushText = () => {
    if (buffer) {
      tokens.push({ type: "text", text: buffer });
      buffer = "";
    }
  };
  while (index < text.length) {
    if (text[index] === "\n") {
      pushText();
      tokens.push({ type: "break" });
      index += 1;
      continue;
    }
    if (text.startsWith("**", index)) {
      const end = text.indexOf("**", index + 2);
      if (end !== -1) {
        pushText();
        tokens.push({ type: "strong", text: text.slice(index + 2, end) });
        index = end + 2;
        continue;
      }
    }
    if (text[index] === "`") {
      const end = text.indexOf("`", index + 1);
      if (end !== -1) {
        pushText();
        tokens.push({ type: "code", text: text.slice(index + 1, end) });
        index = end + 1;
        continue;
      }
    }
    if (text[index] === "[") {
      const labelEnd = text.indexOf("]", index + 1);
      if (labelEnd !== -1 && text[labelEnd + 1] === "(") {
        const hrefEnd = text.indexOf(")", labelEnd + 2);
        if (hrefEnd !== -1) {
          const href = text.slice(labelEnd + 2, hrefEnd).trim();
          if (/^https?:\/\//i.test(href)) {
            pushText();
            tokens.push({ type: "link", text: text.slice(index + 1, labelEnd), href });
            index = hrefEnd + 1;
            continue;
          }
        }
      }
    }
    if (text[index] === "*" && text[index + 1] !== "*") {
      const end = text.indexOf("*", index + 1);
      if (end !== -1) {
        pushText();
        tokens.push({ type: "em", text: text.slice(index + 1, end) });
        index = end + 1;
        continue;
      }
    }
    buffer += text[index];
    index += 1;
  }
  pushText();
  return tokens;
}
