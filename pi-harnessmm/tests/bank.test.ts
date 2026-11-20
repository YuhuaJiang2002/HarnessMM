/**
 * Bank-contract conformance.
 *
 * These cases mirror HarnessMM tests/deepseek_cot_jev.test.mjs:
 *   - "banks reject future dependencies, duplicates, unknown native arguments and extra tools"
 *   - "native JSON-string observations decode without changing their original receipt"
 *   - "ambiguous native result shapes abstain without fabricating record fields"
 * plus the v6/v8 node contract that sar_subdag.test.mjs exercises.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseBank, publicReceipt, type AuthoringMessage, type ToolContract } from "../src/bank.ts";
import type { Policy } from "../src/policy.ts";

const v1: Policy = { version: 1, enabled: true, maximum_candidates: 16, maximum_fast_steps: 16, maximum_public_bytes: 32768 };
const v6: Policy = { version: 6, enabled: true, maximum_candidates: 16, maximum_fast_steps: 8, maximum_public_bytes: 65536 };
const v8: Policy = { version: 8, enabled: true, maximum_candidates: 16, maximum_fast_steps: 8, maximum_public_bytes: 65536 };

const tool: ToolContract = {
  name: "terminal",
  parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"], additionalProperties: false },
};

const candidate = { label: "A", after: "root", condition: "The root returned READY.", tool: "terminal", arguments: { command: "printf DONE" } };
const root = { type: "tool-call" as const, id: "native-root", name: "terminal", arguments: '{"command":"printf READY"}' };

const slowMessage = (candidates: unknown[] = [candidate]): AuthoringMessage => ({
  role: "assistant",
  content: [
    { type: "text", text: `<CHISEL_PLAN>${JSON.stringify({ candidates })}</CHISEL_PLAN>` },
    { ...root },
  ],
});
const node = (over: Record<string, unknown> = {}) => ({
  candidate_id: "A",
  predecessor_id: "root",
  tool: "terminal",
  arguments: { command: "printf A" },
  guard: { return_code: 0 },
  required_receipt_fields: ["return_code"],
  expected_local_cost: { selector_upper_tokens: 8192 },
  dependency_ids: [],
  ...over,
});

const v6Message = (candidates: unknown[]): AuthoringMessage => ({
  role: "assistant",
  content: [
    { type: "text", text: `<CHISEL_PLAN>${JSON.stringify({ candidates })}</CHISEL_PLAN>` },
    { ...root },
  ],
});
const v8Message = v6Message;

test("v1 banks reject future dependencies, duplicates, unknown tools, extra arguments and long labels", () => {
  for (const candidates of [
    [{ ...candidate, after: "B" }],
    [candidate, candidate],
    [{ ...candidate, tool: "invented-tool" }],
    [{ ...candidate, arguments: { command: "printf OK", invented: true } }],
    [{ ...candidate, arguments: { command: "printf OK", timeout_sec: true } }],
    [{ ...candidate, label: "AB" }],
  ]) {
    assert.throws(() => parseBank(slowMessage(candidates), v1, [tool]), /candidate/, JSON.stringify(candidates));
  }
});

test("v6 refuses nodes whose arguments are not fully resolved", () => {
  for (const bad of [
    node({ guard: { task_solved: true } }),
    node({ guard: {} }),
    node({ dependency_ids: ["B"] }),
    node({ expected_local_cost: {} }),
    node({ tool: "invented-tool" }),
  ]) {
    assert.throws(() => parseBank(v6Message([bad]), v6, [tool]), (error: unknown) => error instanceof Error, JSON.stringify(bad));
  }
  // A shell placeholder is legal for this tool schema, so the frozen contract cannot
  // reject it here; the authoring instruction forbids it and admission refuses the result.
  assert.ok(parseBank(v6Message([node({ arguments: { command: "printf ${HOME}" } })]), v6, [tool]));
});

test("a node that repeats the root command is refused", () => {
  const sameAsRoot = node({ arguments: { command: "printf READY" } });
  assert.throws(() => parseBank(v6Message([sameAsRoot]), v6, [tool]), /cannot repeat the already executed native root/);
});

test("v8 nodes must declare resolved literal public-receipt arguments", () => {
  const resolved = node({
    args_resolved: true,
    requires_new_reasoning: false,
    depends_only_on_public_receipt: true,
    native_action_is_literal: true,
  });
  assert.equal(parseBank(v8Message([resolved]), v8, [tool])!.candidates.length, 1);
  for (const flag of ["args_resolved", "requires_new_reasoning", "depends_only_on_public_receipt", "native_action_is_literal"]) {
    const broken = { ...resolved, [flag]: flag === "requires_new_reasoning" ? true : false };
    assert.throws(() => parseBank(v8Message([broken]), v8, [tool]), /resolved literal/);
  }
  assert.throws(() => parseBank(v8Message([{ ...resolved, guard: { return_code: 1 } }]), v8, [tool]), /resolved literal/);
});

test("two banks in one turn, or a bank without exactly one root, is refused", () => {
  const twoBanks: AuthoringMessage = {
    role: "assistant",
    content: [
      { type: "text", text: `<CHISEL_PLAN>${JSON.stringify({ candidates: [candidate] })}</CHISEL_PLAN>` },
      { type: "text", text: `<CHISEL_PLAN>${JSON.stringify({ candidates: [candidate] })}</CHISEL_PLAN>` },
      { ...root },
    ],
  };
  assert.throws(() => parseBank(twoBanks, v1, [tool]), /only one action bank/);
  const noRoot: AuthoringMessage = {
    role: "assistant",
    content: [{ type: "text", text: `<CHISEL_PLAN>${JSON.stringify({ candidates: [candidate] })}</CHISEL_PLAN>` }],
  };
  assert.throws(() => parseBank(noRoot, v1, [tool]), /exactly one registered native root/);
});

test("a turn without a bank parses to null instead of throwing", () => {
  const plain: AuthoringMessage = { role: "assistant", content: [{ type: "text", text: "no plan here" }, { ...root }] };
  assert.equal(parseBank(plain, v1, [tool]), null);
});

test("native JSON-string observations decode without changing the original receipt", () => {
  const value = { stdout: "READY", stderr: "", exit_code: 0, timed_out: false };
  const message = { role: "tool" as const, toolCallId: "native-root", isError: false, content: [{ type: "text" as const, text: JSON.stringify(JSON.stringify(value)) }] };
  const original = structuredClone(message);
  assert.deepEqual(publicReceipt(message), { call_id: "native-root", is_error: false, status: "PARSED", value });
  assert.deepEqual(message, original);
});

test("ambiguous native result shapes abstain without fabricating record fields", () => {
  for (const text of ["READY", '"READY"', "null", "[]", '{"stdout":']) {
    const message = { role: "tool" as const, toolCallId: "native-root", content: [{ type: "text" as const, text }] };
    assert.equal(publicReceipt(message).status, "UNKNOWN", text);
  }
  const multi = { role: "tool" as const, toolCallId: "n", content: [{ type: "text" as const, text: "{}" }, { type: "text" as const, text: "{}" }] };
  assert.equal(publicReceipt(multi).status, "UNKNOWN");
});

test("single-encoding mode refuses a rewritten numeric literal", () => {
  const message = { role: "tool" as const, toolCallId: "n", content: [{ type: "text" as const, text: JSON.stringify({ return_code: 0, ratio: "1.0" }) }] };
  assert.equal(publicReceipt(message, true).status, "PARSED");
  const rewritten = { role: "tool" as const, toolCallId: "n", content: [{ type: "text" as const, text: '{"return_code":01}' }] };
  assert.equal(publicReceipt(rewritten, true).status, "UNKNOWN");
});
