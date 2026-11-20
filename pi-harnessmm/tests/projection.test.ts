/**
 * Bank projection: the authored <CHISEL_PLAN> leaves the slow context while every
 * native action, receipt and user constraint stays byte-identical.
 * Mirrors HarnessMM tests/sar_v2.test.mjs ("native session extraction preserves
 * authentic reasoning and receipts while removing bank from slow context").
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { compactMessages, projectionBinding, slowTokensOf, toAuthoringMessage, userRequests } from "../src/pi-messages.ts";
import { parseBank } from "../src/bank.ts";
import type { Policy } from "../src/policy.ts";

const policy: Policy = { version: 6, enabled: true, maximum_candidates: 16, maximum_fast_steps: 8, maximum_public_bytes: 65536 };
const tool = { name: "terminal", parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"], additionalProperties: false } };

const authoring = {
  role: "assistant",
  content: [
    { type: "thinking", thinking: "authentic reasoning" },
    {
      type: "text",
      text: '<CHISEL_PLAN>{"candidates":[]}</CHISEL_PLAN>Now run the build.',
    },
    { type: "toolCall", id: "call-1", name: "terminal", arguments: { command: "npm run build" } },
  ],
};

test("a pi assistant turn converts to the authoring shape the contract parses", () => {
  const message = toAuthoringMessage(authoring);
  assert.ok(message);
  assert.deepEqual(message!.content.map(part => part.type), ["reasoning", "text", "tool-call"]);
  assert.equal(message!.content[2]!.type === "tool-call" && message!.content[2]!.arguments, '{"command":"npm run build"}');
});

test("the bank is removed, reasoning and native actions survive", () => {
  const { messages, replacements, originals } = compactMessages([{ role: "user", content: "do it" }, authoring]);
  assert.equal(replacements, 1);
  const next = messages[1];
  const text = (next.content as any[]).find(part => part.type === "text");
  assert.equal(text.text, "Now run the build.");
  assert.deepEqual((next.content as any[]).find(part => part.type === "thinking"), authoring.content[0]);
  assert.deepEqual((next.content as any[]).find(part => part.type === "toolCall"), authoring.content[2]);
  assert.deepEqual(originals.get("call-1"), authoring);
  assert.equal(projectionBinding(authoring, next).nativeActionsPreserved, true);
});

test("a message without a bank is left untouched", () => {
  const plain = { role: "assistant", content: [{ type: "text", text: "no bank" }] };
  const { messages, replacements } = compactMessages([plain]);
  assert.equal(replacements, 0);
  assert.equal(messages[0], plain);
});

test("a turn whose text is only the bank keeps its tool call and drops the empty text part", () => {
  const onlyBank = { role: "assistant", content: [{ type: "text", text: '<CHISEL_PLAN>{"candidates":[]}</CHISEL_PLAN>' }, authoring.content[2]] };
  const { messages } = compactMessages([onlyBank]);
  assert.deepEqual((messages[0].content as any[]).map(part => part.type), ["toolCall"]);
});

test("user constraints and the slow call's token cost are read from the transcript", () => {
  const messages = [
    { role: "user", content: "task one" },
    { role: "assistant", content: [{ type: "text", text: "working" }], usage: { totalTokens: 4321 } },
    { role: "user", content: "correction" },
    authoring,
  ];
  assert.deepEqual(userRequests(messages, 3).map(message => message.content), ["task one", "correction"]);
  assert.equal(slowTokensOf(messages[1] as never), 4321);
  assert.equal(slowTokensOf(undefined), 0);
});

test("a projected turn still parses to no bank, so nothing re-executes", () => {
  const { messages } = compactMessages([authoring]);
  assert.equal(parseBank(toAuthoringMessage(messages[0] as never), policy, [tool]), null);
});
