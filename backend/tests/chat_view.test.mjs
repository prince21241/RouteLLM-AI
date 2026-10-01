import assert from "node:assert/strict";
import test from "node:test";

import {
  attemptRows,
  chatRequestBody,
  featureLines,
  liveDetails,
  markdownBlocks,
  moneyText,
  providerChoices,
  reuseDraft,
  storedDetails,
  submissionState,
} from "../app/static/chat_view.mjs";

const options = {
  models: [
    { model_id: "gpt-5-nano", provider: "openai", enabled: true },
    { model_id: "claude-sonnet-4-6", provider: "anthropic", enabled: false },
    { model_id: "llama3.2", provider: "ollama", enabled: false },
  ],
  quality_evaluation: { enabled: false },
  escalation: { enabled: true },
  fallback: { enabled: false },
};

test("a blank prompt is rejected and a second submit is ignored while pending", () => {
  const idle = { pending: false, prompt: "keep me", systemPrompt: "", provider: "" };
  const blank = submissionState(idle, { type: "start", prompt: "  ", systemPrompt: "", provider: "" });
  assert.equal(blank.accepted, false);
  assert.equal(blank.prompt, "keep me");

  const started = submissionState(idle, {
    type: "start",
    prompt: "Hello",
    systemPrompt: "Be brief",
    provider: "openai",
  });
  assert.equal(started.accepted, true);
  assert.equal(started.pending, true);
  assert.deepEqual(chatRequestBody(started), {
    prompt: "Hello",
    system_prompt: "Be brief",
    provider: "openai",
  });

  const duplicate = submissionState(started, {
    type: "start",
    prompt: "A different prompt",
    systemPrompt: "",
    provider: "",
  });
  assert.equal(duplicate.accepted, false);
  assert.equal(duplicate.prompt, "Hello");
  assert.equal(duplicate.pending, true);
});

test("automatic routing omits the provider and unavailable providers stay disabled", () => {
  const automatic = submissionState(
    { pending: false },
    { type: "start", prompt: "Hello", systemPrompt: "", provider: "" },
  );
  assert.deepEqual(chatRequestBody(automatic), { prompt: "Hello" });

  const choices = providerChoices(options);
  assert.equal(choices[0].label, "Automatic routing");
  assert.equal(choices.find((choice) => choice.value === "openai").enabled, true);
  assert.equal(choices.find((choice) => choice.value === "anthropic").enabled, false);
  assert.match(featureLines(options).join(" "), /Escalation is on/);
  assert.match(featureLines(options).join(" "), /cannot name a model/);
});

test("markdown keeps untrusted text and separates code blocks", () => {
  const blocks = markdownBlocks("See <script>alert(1)</script>\n\n```js\nconst x = 1;\n```\n\n**Bold** and `code`");
  assert.equal(blocks[0].type, "paragraph");
  assert.equal(blocks[0].inlines[0].text, "See <script>alert(1)</script>");
  assert.equal(blocks[1].type, "code");
  assert.equal(blocks[1].language, "js");
  assert.equal(blocks[1].text, "const x = 1;");
  assert.equal(blocks[2].inlines[0].type, "strong");
  assert.equal(JSON.stringify(blocks).includes("innerHTML"), false);
});

test("markdown keeps line breaks, quotes, and rules as structured text", () => {
  const blocks = markdownBlocks("> quoted <b>\n\n---\n\nLine one\nLine two");
  assert.equal(blocks[0].type, "quote");
  assert.equal(blocks[0].inlines[0].text, "quoted <b>");
  assert.equal(blocks[1].type, "rule");
  assert.equal(blocks[2].inlines.some((token) => token.type === "break"), true);
  assert.equal(JSON.stringify(blocks).includes("innerHTML"), false);
});

test("live details keep unknown cost, fallback, escalation, and failed persistence", () => {
  const details = liveDetails({
    request_id: "req-1",
    response: "answer",
    persistence_status: "failed",
    persistence_warning: "The database write failed",
    routing: {
      provider: "openai",
      model: "gpt-5-nano",
      selection_reason: "selected gpt-5-nano from the requested low tier",
      complexity_tier: "low",
      complexity_score: 0.1,
      escalated: true,
    },
    metrics: { input_tokens: 3, output_tokens: 0, latency_ms: 12, cost: null, cost_completeness: "unknown" },
    quality_verdict: "unknown",
    fallback_used: true,
    fallback_reason: "timeout",
    fallback_skips: [{ model_id: "llama3.2", reason: "same provider" }],
    returned_model: "claude-sonnet-4-6",
    final_provider: null,
    estimated_savings: null,
    savings_basis: "same_token_volume",
    end_to_end_latency_ms: null,
  });

  const rows = Object.fromEntries(details.rows);
  assert.equal(rows["Recorded cost"], "unknown");
  assert.equal(rows["Output tokens"], "0");
  assert.equal(rows["End-to-end latency"], "unknown");
  assert.equal(rows.Quality, "unknown");
  assert.equal(rows.Escalated, "yes");
  assert.equal(rows.Fallback, "yes");
  assert.equal(rows["Final provider"], "not recorded");
  assert.equal(rows["Final model"], "claude-sonnet-4-6");
  assert.match(rows.Persistence, /database write failed/);
  assert.match(details.banner, /not saved/);
  assert.equal(details.dashboardHref, null);
  assert.match(rows["Savings estimate"], /unknown \(same_token_volume estimate\)/);
});

test("stored details do not invent missing history fields", () => {
  const details = storedDetails({
    request_id: "req-2",
    prompt: "Remember this",
    system_prompt: "Be brief",
    response: null,
    provider: "openai",
    model: "gpt-5-nano",
    final_provider: null,
    final_model: null,
    total_cost: null,
    cost_completeness: "unknown",
    estimated_savings: null,
    quality_verdict: null,
    escalated: null,
    fallback_used: null,
    end_to_end_latency_ms: null,
    attempts: [
      { attempt_number: 2, purpose: "fallback", provider: "anthropic", configured_model_id: "claude-sonnet-4-6", status: "succeeded", input_tokens: null, output_tokens: 4, provider_latency_ms: null, estimated_cost: null, cost_completeness: "unknown" },
      { attempt_number: 1, purpose: null, provider: "openai", configured_model_id: "gpt-5-nano", status: "failed", input_tokens: 2, output_tokens: 0, provider_latency_ms: 8, estimated_cost: "0", cost_completeness: "complete" },
    ],
  });

  const rows = Object.fromEntries(details.rows);
  assert.equal(rows["Final model"], "not recorded");
  assert.equal(rows["Recorded cost"], "unknown");
  assert.equal(rows.Quality, "not recorded");
  assert.equal(rows.Fallback, "not recorded");
  assert.equal(rows.Persistence, "Read from history");
  assert.equal(details.dashboardHref, "/dashboard#request=req-2");
  assert.match(details.banner, /not a new draft/);

  const attempts = attemptRows(details && [
    { attempt_number: 2, purpose: "fallback", provider: "anthropic", configured_model_id: "claude-sonnet-4-6", status: "succeeded", input_tokens: null, output_tokens: 4, provider_latency_ms: null, estimated_cost: null, cost_completeness: "unknown" },
    { attempt_number: 1, purpose: null, provider: "openai", configured_model_id: "gpt-5-nano", status: "failed", input_tokens: 2, output_tokens: 0, provider_latency_ms: 8, estimated_cost: "0", cost_completeness: "complete" },
  ]);
  assert.equal(attempts[0].number, "1");
  assert.equal(attempts[0].purpose, "not recorded");
  assert.equal(attempts[0].cost, "0 USD (complete)");
  assert.equal(attempts[1].tokens, "unknown in / 4 out");
  assert.equal(attempts[1].latency, "unknown");
  assert.equal(attempts[1].cost, "unknown");
  assert.equal(moneyText(null, "complete"), "unknown");
});

test("reusing a saved prompt does not submit it", () => {
  const draft = reuseDraft({ prompt: "Remember this", system_prompt: "Be brief" });
  assert.deepEqual(draft, { prompt: "Remember this", systemPrompt: "Be brief", submit: false });
});
