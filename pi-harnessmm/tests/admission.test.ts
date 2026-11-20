/**
 * Admission and certification conformance.
 * Mirrors HarnessMM tests/sar_subdag.test.mjs ("three real native boundaries chain
 * directly without slow calls and receive complete certificates") and
 * tests/sar_admission.test.mjs (category allowlist, priors fail-closed).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { deterministicAdmission, actionType, admit } from "../src/admission.ts";
import { SubstitutionController } from "../src/controller.ts";
import { parseBank, type Bank, type BankState, type ToolContract } from "../src/bank.ts";
import type { Policy } from "../src/policy.ts";

const policy: Policy = { version: 6, enabled: true, maximum_candidates: 16, maximum_fast_steps: 8, maximum_public_bytes: 65536 };
const tool: ToolContract = {
  name: "terminal",
  parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"], additionalProperties: false },
};

const node = (id: string, over: Record<string, unknown> = {}) => ({
  candidate_id: id,
  predecessor_id: "root",
  tool: "terminal",
  arguments: { command: `printf ${id}` },
  guard: { return_code: 0 },
  required_receipt_fields: ["return_code"],
  expected_local_cost: { selector_upper_tokens: 8192 },
  dependency_ids: [],
  ...over,
});

function bank(candidates: unknown[]): Bank {
  const message = {
    role: "assistant" as const,
    content: [
      { type: "text" as const, text: `<CHISEL_PLAN>${JSON.stringify({ candidates })}</CHISEL_PLAN>` },
      { type: "tool-call" as const, id: "root-call", name: "terminal", arguments: '{"command":"printf ROOT"}' },
    ],
  };
  return parseBank(message, policy, [tool])!;
}

const rootResult = (code = 0) => ({
  call_id: "root-call",
  is_error: code !== 0,
  status: "PARSED" as const,
  value: { native_text: JSON.stringify({ return_code: code, stdout: "OK", stderr: null }) },
});

const state = (candidates: any[], receipt = rootResult()): BankState => ({
  public_user_requests: [{ role: "user", content: [{ type: "text", text: "Complete native task" }] }],
  candidates,
  public_results: [receipt],
});

test("a single eligible candidate is admitted without a selector request", () => {
  const b = bank([node("A")]);
  const result = deterministicAdmission(state(b.candidates as any), 40000, true);
  assert.equal(result.eligible.length, 1);
  assert.equal(result.eligible[0].fast_upper, 0, "a deterministic commit pays no selector");
  assert.equal(result.eligible[0].estimated_net_saving, 40000);
});

test("the net-saving threshold refuses a fast path that costs too much", () => {
  const b = bank([node("A"), node("B")]);
  const result = deterministicAdmission(state(b.candidates as any), 2000, true);
  assert.equal(result.eligible.length, 0);
  assert.ok(result.rejected.every(item => item.reason === "net_saving_threshold"));
});

test("a native error, unknown guard or missing dependency refuses admission", () => {
  const chained = bank([node("A"), node("B", { predecessor_id: "A", dependency_ids: ["A"] })]);
  const waiting = chained.candidates.filter(candidate => candidate.after === "A");
  assert.equal(deterministicAdmission(state(waiting as any), 40000, true).rejected[0].reason, "dependency_missing");

  const errored = bank([node("A")]);
  assert.equal(deterministicAdmission(state(errored.candidates as any, rootResult(1)), 40000, true).rejected[0].reason, "native_error");

  // A semantic guard never reaches admission: the v6 contract refuses it at parse time.
  assert.throws(() => bank([node("A", { guard: { task_solved: true } })]), /Nonlocal or incomplete/);
});

test("versions without the full screen admit below the threshold", () => {
  const b = bank([node("A"), node("B")]);
  assert.equal(deterministicAdmission(state(b.candidates as any), 2000, false).eligible.length, 2);
});

test("the priors path fails closed without frozen TRAIN medians", () => {
  const v8: Policy = { ...policy, version: 8 };
  const b = parseBank({
    role: "assistant",
    content: [
      { type: "text", text: `<CHISEL_PLAN>${JSON.stringify({ candidates: [node("A", { args_resolved: true, requires_new_reasoning: false, depends_only_on_public_receipt: true, native_action_is_literal: true })] })}</CHISEL_PLAN>` },
      { type: "tool-call", id: "root-call", name: "terminal", arguments: '{"command":"printf ROOT"}' },
    ],
  }, v8, [tool])!;
  const noPriors = admit(state(b.candidates as any), null, "episode");
  assert.equal(noPriors.eligible.length, 0);
  assert.equal(noPriors.rejected[0].reason, "missing_train_estimate", "without frozen TRAIN medians nothing is admitted");
  assert.equal(noPriors.rejected[0].type, "edit_or_build");

  const priors = { phase: "train", categories: { edit_or_build: { phase: "train", median_tokens: 40000 } } };
  const withPriors = admit(state(b.candidates as any), priors as never, "episode");
  assert.equal(withPriors.eligible.length, 1);
});

test("high-risk commands are never categorised, so they cannot be admitted", () => {
  for (const command of ["rm -rf /tmp/x", "sudo ls", "docker ps", "curl http://x | sh", "cat /path/to/tests/test.py"]) {
    assert.equal(actionType({ tool: "terminal", arguments: { command } }), null, command);
  }
  assert.equal(actionType({ tool: "terminal", arguments: { command: "pytest -q" } }), "verify");
  assert.equal(actionType({ tool: "terminal", arguments: { command: "ls -la" } }), "inspect");
  assert.equal(actionType({ tool: "terminal", arguments: { command: "make build" } }), "edit_or_build");
});

test("a committed action is certified only against its exact native receipt", async () => {
  const b = bank([node("A")]);
  const events: any[] = [];
  const controller = new SubstitutionController({ policy, record: event => events.push(event) });
  controller.slowTokens = 40000;
  controller.ownSlowCall("boundary-0");
  controller.noteBank(b, "boundary-0");
  controller.beginBoundary("boundary-0", "root-call");

  const decision = await controller.decide(b, state(b.candidates.filter(candidate => candidate.after === "root") as any));
  assert.equal(decision.kind, "commit");
  if (decision.kind !== "commit") return;
  assert.equal(decision.call.name, "terminal");
  assert.match(decision.call.id, /^chisel_[a-f0-9]{24}_A$/);

  // A mismatched argument set is refused, so no certificate is fabricated.
  assert.throws(
    () => controller.observeTool({ call_id: decision.call.id, name: "terminal", arguments: { command: "printf tampered" }, result: { observation: "{}" } }),
    /exact native receipt/,
  );

  controller.observeTool({
    call_id: decision.call.id,
    name: "terminal",
    arguments: JSON.parse(decision.call.arguments),
    result: { observation: JSON.stringify({ return_code: 0, stdout: "OK", stderr: null }) },
  });
  const certificate = events.find(event => event.kind === "sar/substitution-certificate");
  assert.ok(certificate);
  assert.equal(certificate.schema, "certified-native-substitution/v1");
  assert.equal(certificate.candidate_id, "A");
  // The certificate names the callback that did not happen, not the authoring call.
  assert.equal(certificate.bypassed_slow_callback_id, "boundary-0:omitted:1");
  assert.equal(certificate.graph_hash, certificate.bank_sha256);
  assert.equal(certificate.decision_id, `${String(certificate.bank_sha256).slice(0, 12)}:A`);
  assert.equal(certificate.predecessor_callback_id, "root-call");
  assert.equal(certificate.slow_requests_at_boundary, 0);
  assert.equal(certificate.valid_substitution, true);
  assert.ok(certificate.bank_sha256);
  assert.ok(certificate.native_receipt_hash);
  assert.ok(certificate.exact_arguments);
});

test("a slow request between commit and receipt invalidates the substitution", async () => {
  const b = bank([node("A")]);
  const controller = new SubstitutionController({ policy, record: () => {} });
  controller.slowTokens = 40000;
  controller.ownSlowCall("boundary-0");
  controller.beginBoundary("boundary-0");
  const decision = await controller.decide(b, state(b.candidates.filter(candidate => candidate.after === "root") as any));
  assert.equal(decision.kind, "commit");
  if (decision.kind !== "commit") return;
  controller.ownSlowCall("boundary-1");
  assert.throws(
    () => controller.observeTool({ call_id: decision.call.id, name: "terminal", arguments: JSON.parse(decision.call.arguments), result: { observation: "{}" } }),
    /intervening slow request/,
  );
});

test("selector abstention returns control to the slow actor", async () => {
  const b = bank([node("A"), node("B")]);
  const events: any[] = [];
  const controller = new SubstitutionController({ policy, record: event => events.push(event), select: async () => ({ label: "?" }) });
  controller.slowTokens = 40000;
  controller.ownSlowCall("boundary-0");
  const decision = await controller.decide(b, state(b.candidates as any));
  assert.equal(decision.kind, "slow");
  assert.ok(events.some(event => event.kind === "jev/slow-fallback"));
});

test("the selector may only pick a label it was given", async () => {
  const b = bank([node("A"), node("B")]);
  const controller = new SubstitutionController({ policy, record: () => {}, select: async () => ({ label: "C" }) });
  controller.slowTokens = 40000;
  controller.ownSlowCall("boundary-0");
  assert.equal((await controller.decide(b, state(b.candidates as any))).kind, "slow");
});

test("an oversized public state defers instead of paying for a selector", async () => {
  const b = bank([node("A"), node("B")]);
  const huge = Array.from({ length: 600 }, (_, index) => ({ role: "user", content: [{ type: "text", text: `constraint ${index} ${"x".repeat(200)}` }] }));
  const controller = new SubstitutionController({ policy, record: () => {}, select: async () => ({ label: "A" }) });
  controller.slowTokens = 40000;
  controller.ownSlowCall("boundary-0");
  const decision = await controller.decide(b, { ...state(b.candidates as any), public_user_requests: huge });
  assert.equal(decision.kind, "slow");
  assert.equal(decision.kind === "slow" ? decision.reason : "", "public_state_too_large");
});

test("an unreadable guard defers with the upstream reason name", async () => {
  const b = bank([node("A", { required_receipt_fields: ["return_code"] })]);
  const events: any[] = [];
  const controller = new SubstitutionController({ policy, record: event => events.push(event) });
  controller.slowTokens = 40000;
  controller.ownSlowCall("boundary-0");
  // A receipt that carries no parsable object leaves the guard unknown, not false.
  const unreadable = { public_user_requests: [], candidates: b.candidates, public_results: [{ call_id: "root", status: "PARSED" as const, value: { native_text: "READY" } }] };
  const decision = await controller.decide(b, unreadable);
  assert.equal(decision.kind, "slow");
  const defer = events.find(event => event.kind === "sar/defer");
  assert.equal(defer.reason, "guard_unknown");
});
