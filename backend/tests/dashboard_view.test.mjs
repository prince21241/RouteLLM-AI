import assert from "node:assert/strict";
import test from "node:test";

import {
  apiQuery,
  costSeries,
  actionReason,
  detailModel,
  detailState,
  formatCost,
  formatRate,
  inclusiveEnd,
  panelState,
  qualityLabel,
  resolvedFinal,
  routingLines,
} from "../app/static/dashboard_view.mjs";

test("panel and detail states", () => {
  assert.equal(panelState({ loading: true, error: "", requestCount: 3 }), "loading");
  assert.equal(panelState({ loading: false, error: "down", requestCount: 3 }), "error");
  assert.equal(panelState({ loading: false, error: "", requestCount: 0 }), "empty");
  assert.equal(panelState({ loading: false, error: "", requestCount: 2 }), "populated");
  assert.equal(detailState({ loading: true, error: "", request: null }), "loading");
  assert.equal(detailState({ loading: false, error: "missing", request: null }), "error");
  assert.equal(detailState({ loading: false, error: "", request: null }), "empty");
  assert.equal(detailState({ loading: false, error: "", request: { prompt: "x" } }), "populated");
});

test("an inclusive end date becomes the next UTC midnight", () => {
  assert.equal(inclusiveEnd("2026-09-29"), "2026-09-30");
  const query = apiQuery({
    from: "2026-09-29",
    to: "2026-09-29",
    model: "gpt-5-nano",
    provider: "openai",
    status: "failed",
    limit: 20,
    offset: 0,
  });
  assert.match(query, /from=2026-09-29T00%3A00%3A00Z/);
  assert.match(query, /to=2026-09-30T00%3A00%3A00Z/);
  assert.match(query, /model=gpt-5-nano/);
});

test("unknown money stays unknown", () => {
  assert.equal(formatCost({ complete: null, estimated: null, unknown_count: 2 }), "2 unknown");
  assert.equal(formatCost({ complete: null, estimated: null, unknown_count: 0 }), "No recorded cost");
  assert.equal(formatCost({ complete: "0", estimated: null, unknown_count: 0 }), "0 USD complete");
  assert.equal(formatRate(null), "unknown");
  assert.equal(formatRate("0"), "0.0%");
  const series = costSeries([{ day: "2026-09-29", requests: 1, complete: null, estimated: "0.01" }]);
  assert.equal(series[0].complete, null);
  assert.equal(series[0].estimated, 0.01);
});

test("request detail keeps untrusted text and attempt order", () => {
  const detail = detailModel({
    prompt: "<script>alert(1)</script>",
    response: "answer",
    status: "succeeded",
    provider: "openai",
    model: "gpt-5-nano",
    final_provider: null,
    final_model: "claude-sonnet-4-6",
    total_cost: null,
    cost_completeness: "unknown",
    estimated_savings: null,
    savings_basis: "same_token_volume",
    end_to_end_latency_ms: null,
    quality_verdict: "fail",
    quality_reasons: ["missed the reference"],
    escalated: true,
    escalation_reason: "explicit fail",
    fallback_used: null,
    fallback_reason: "timeout",
    attempts: [
      { attempt_number: 2, purpose: "fallback" },
      { attempt_number: 1, purpose: "routing" },
    ],
    evaluations: [{ verdict: "fail", reasons: ["missed the reference"] }],
  });

  assert.equal(detail.prompt, "<script>alert(1)</script>");
  assert.equal(detail.cost, null);
  assert.equal(detail.latencyMs, null);
  assert.equal(detail.fallbackUsed, null);
  assert.equal(detail.finalProvider, null);
  assert.deepEqual(
    detail.attempts.map((attempt) => attempt.purpose),
    ["routing", "fallback"],
  );
  assert.match(detail.persistence, /Stored in request history/);
});

test("a successful attempt fills a missing final route", () => {
  const request = {
    final_provider: null,
    final_model: null,
    quality_verdict: null,
    evaluations: [],
    escalated: false,
    escalation_reason: null,
    escalation_error: null,
    fallback_used: false,
    fallback_reason: null,
    attempts: [
      { attempt_number: 1, status: "failed", provider: "openai", configured_model_id: "gpt-5-nano" },
      { attempt_number: 2, status: "succeeded", provider: "anthropic", configured_model_id: "claude-sonnet-4-6" },
    ],
  };

  assert.deepEqual(resolvedFinal(request), {
    provider: "anthropic",
    model: "claude-sonnet-4-6",
  });
  assert.equal(qualityLabel(null, []), "Not evaluated");
  assert.equal(qualityLabel("unknown", []), "unknown");
  assert.equal(actionReason(false, null), "Not applicable");
  assert.equal(actionReason(true, null), "unknown");
  assert.equal(actionReason(false, "timeout"), "timeout");
});

test("routing metadata is shown and a missing trace is omitted", () => {
  assert.deepEqual(routingLines(null), []);
  const lines = routingLines({
    requested_strategy: "ml",
    effective_strategy: "rule_based",
    artifact_version: "abc",
    predicted_model: "gpt-5-nano",
    confidence: 0.42,
    rules_used: true,
    rules_reason: "confidence below validation threshold",
  });
  assert.equal(lines[0][1], "ml");
  assert.equal(lines[1][1], "rule_based");
  assert.equal(lines[4][1], "0.42");
  assert.equal(lines[5][1], "yes");
});
