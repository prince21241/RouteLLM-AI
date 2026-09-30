export function panelState({ loading, error, requestCount }) {
  if (loading) {
    return "loading";
  }
  if (error) {
    return "error";
  }
  if (!requestCount) {
    return "empty";
  }
  return "populated";
}

export function detailState({ loading, error, request }) {
  if (loading) {
    return "loading";
  }
  if (error) {
    return "error";
  }
  if (!request) {
    return "empty";
  }
  return "populated";
}

export function inclusiveEnd(day) {
  const [year, month, date] = day.split("-").map(Number);
  const next = new Date(Date.UTC(year, month - 1, date + 1));
  return next.toISOString().slice(0, 10);
}

export function apiQuery(filters) {
  const params = new URLSearchParams();
  if (filters.from) {
    params.set("from", `${filters.from}T00:00:00Z`);
  }
  if (filters.to) {
    params.set("to", `${inclusiveEnd(filters.to)}T00:00:00Z`);
  }
  if (filters.model) {
    params.set("model", filters.model);
  }
  if (filters.provider) {
    params.set("provider", filters.provider);
  }
  if (filters.status) {
    params.set("status", filters.status);
  }
  if (filters.limit) {
    params.set("limit", String(filters.limit));
  }
  if (filters.offset) {
    params.set("offset", String(filters.offset));
  }
  const text = params.toString();
  return text ? `?${text}` : "";
}

export function formatRate(rate) {
  if (rate === null || rate === undefined || rate === "") {
    return "unknown";
  }
  const value = Number(rate);
  if (!Number.isFinite(value)) {
    return "unknown";
  }
  return `${(value * 100).toFixed(1)}%`;
}

export function formatCost(bucket) {
  if (!bucket) {
    return "No recorded cost";
  }
  const parts = [];
  if (bucket.complete !== null && bucket.complete !== undefined) {
    parts.push(`${bucket.complete} USD complete`);
  }
  if (bucket.estimated !== null && bucket.estimated !== undefined) {
    parts.push(`${bucket.estimated} USD estimated`);
  }
  if (bucket.unknown_count > 0) {
    parts.push(`${bucket.unknown_count} unknown`);
  }
  return parts.length ? parts.join(", ") : "No recorded cost";
}

export function moneyNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function costSeries(days) {
  return (days || []).map((day) => ({
    day: day.day,
    requests: day.requests,
    complete: moneyNumber(day.complete),
    estimated: moneyNumber(day.estimated),
  }));
}

export function orderedAttempts(attempts) {
  return [...(attempts || [])].sort((left, right) => left.attempt_number - right.attempt_number);
}

export function persistenceLine(request) {
  if (request.status === "pending") {
    return "Stored before a final provider result was written.";
  }
  return "Stored in request history. A response that failed to save does not appear here.";
}

export function detailModel(request) {
  return {
    prompt: request.prompt ?? "",
    systemPrompt: request.system_prompt ?? null,
    answer: request.response ?? null,
    persistence: persistenceLine(request),
    status: request.status,
    initialProvider: request.provider,
    initialModel: request.model,
    finalProvider: request.final_provider ?? null,
    finalModel: request.final_model ?? null,
    escalated: Boolean(request.escalated),
    escalationReason: request.escalation_reason ?? null,
    escalationError: request.escalation_error ?? null,
    fallbackUsed: request.fallback_used ?? null,
    fallbackReason: request.fallback_reason ?? null,
    qualityVerdict: request.quality_verdict ?? null,
    qualityReasons: [...(request.quality_reasons || [])],
    cost: request.total_cost ?? null,
    costCompleteness: request.cost_completeness ?? null,
    savings: request.estimated_savings ?? null,
    savingsBasis: request.savings_basis ?? null,
    latencyMs: request.end_to_end_latency_ms ?? null,
    attempts: orderedAttempts(request.attempts),
    evaluations: [...(request.evaluations || [])],
    routingMetadata: request.routing_metadata ?? null,
  };
}

export function recordedOrUnknown(value, suffix) {
  if (value === null || value === undefined || value === "") {
    return "unknown";
  }
  return suffix ? `${value} ${suffix}` : String(value);
}

export function answeringAttempt(attempts) {
  const succeeded = (attempts || []).filter((attempt) => attempt.status === "succeeded");
  if (succeeded.length === 0) {
    return null;
  }
  return [...succeeded].sort((left, right) => right.attempt_number - left.attempt_number)[0];
}

export function resolvedFinal(request) {
  const answered = answeringAttempt(request?.attempts);
  return {
    provider: request?.final_provider || answered?.provider || null,
    model: request?.final_model || answered?.configured_model_id || null,
  };
}

export function qualityLabel(verdict, evaluations) {
  if (verdict === null || verdict === undefined || verdict === "") {
    return evaluations && evaluations.length ? "unknown" : "Not evaluated";
  }
  return String(verdict);
}

export function actionReason(used, reason) {
  if (reason !== null && reason !== undefined && reason !== "") {
    return String(reason);
  }
  return used === true ? "unknown" : "Not applicable";
}

export function routingLines(metadata) {
  if (!metadata) {
    return [];
  }
  return [
    ["Requested routing", metadata.requested_strategy || "unknown"],
    ["Effective routing", metadata.effective_strategy || "unknown"],
    ["ML artifact", metadata.artifact_version || "Not applicable"],
    ["ML prediction", metadata.predicted_model || "Not applicable"],
    ["ML confidence", metadata.confidence == null ? "Not applicable" : String(metadata.confidence)],
    ["Rules used", metadata.rules_used ? "yes" : "no"],
    ["Rules reason", metadata.rules_reason || "Not applicable"],
  ];
}
