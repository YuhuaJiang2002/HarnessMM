/**
 * pi adapters: convert pi's session messages into the shapes the upstream
 * contracts expect, and strip the authored bank from the slow context.
 *
 * Port of the message-shaping half of cot_jev.mjs (decisionState inputs) and of
 * sar_v2.mjs (compactSession / controllerProjection): the bank text is removed
 * before the next slow request, while reasoning, tool calls, receipts and user
 * constraints are preserved byte for byte.
 */
import { hash } from "./policy.ts";
import type { AuthoringMessage } from "./bank.ts";

/** Subset of pi's message shapes this extension reads. */
export interface PiMessage {
  role: string;
  content?: unknown;
  usage?: { totalTokens?: number; inputTokens?: number; outputTokens?: number };
  toolCallId?: string;
  isError?: boolean;
}

/**
 * Convert a pi assistant message to the upstream authoring shape.
 * pi stores tool arguments as objects; the contract parses JSON strings.
 */
export function toAuthoringMessage(message: PiMessage | undefined): AuthoringMessage | null {
  if (!message || message.role !== "assistant" || !Array.isArray(message.content)) return null;
  const content: AuthoringMessage["content"] = [];
  for (const part of message.content as any[]) {
    if (part?.type === "text" && typeof part.text === "string") content.push({ type: "text", text: part.text });
    else if (part?.type === "thinking" && typeof part.thinking === "string") content.push({ type: "reasoning", text: part.thinking });
    else if (part?.type === "toolCall" && typeof part.name === "string") {
      content.push({ type: "tool-call", id: String(part.id), name: part.name, arguments: JSON.stringify(part.arguments ?? {}) });
    }
  }
  return { role: "assistant", content };
}

/** Total tokens of a paid slow call, used as the admission's saving estimate. */
export function slowTokensOf(message: PiMessage | undefined): number {
  const usage = message?.usage;
  if (!usage) return 0;
  const total = usage.totalTokens;
  return typeof total === "number" && Number.isFinite(total) && total > 0 ? total : 0;
}

/** User constraints authored before the bank, verbatim. */
export function userRequests(messages: readonly PiMessage[], beforeIndex: number): Array<{ role: string; content: unknown }> {
  return messages.slice(0, beforeIndex).filter(message => message.role === "user").map(message => ({ role: "user", content: message.content }));
}

/**
 * Remove the `<CHISEL_PLAN>` block from assistant text.
 * Reasoning and tool calls are preserved, and the replacement is only returned
 * when the original carried a bank, so unrelated messages stay untouched.
 */
export function compactMessages(messages: PiMessage[]): { messages: PiMessage[]; replacements: number; originals: Map<string, unknown> } {
  const originals = new Map<string, unknown>();
  let replacements = 0;
  const next = messages.map(message => {
    if (message.role !== "assistant" || !Array.isArray(message.content)) return message;
    const content = message.content as any[];
    const hasBank = content.some(part => part?.type === "text" && typeof part.text === "string" && part.text.includes("<CHISEL_PLAN>"));
    if (!hasBank) return message;
    const calls = content.filter(part => part?.type === "toolCall");
    if (calls[0]) originals.set(String(calls[0].id), structuredClone(message));
    const stripped = content.flatMap(part => {
      if (part?.type !== "text" || typeof part.text !== "string") return [part];
      const text = part.text.replace(/<CHISEL_PLAN>[\s\S]*?<\/CHISEL_PLAN>/g, "").trim();
      return text ? [{ ...part, text }] : [];
    });
    if (JSON.stringify(stripped) === JSON.stringify(content)) return message;
    replacements += 1;
    return { ...message, content: stripped };
  });
  return { messages: next, replacements, originals };
}

/** Stable binding for a projection, so a rewrite cannot silently change actions. */
export function projectionBinding(before: PiMessage, after: PiMessage): { originalMessageSha256: string; replacementSha256: string; nativeActionsPreserved: boolean } {
  const calls = (message: PiMessage) =>
    Array.isArray(message.content) ? (message.content as any[]).filter(part => part?.type === "toolCall") : [];
  return {
    originalMessageSha256: hash(before),
    replacementSha256: hash(after),
    nativeActionsPreserved: hash(calls(before)) === hash(calls(after)),
  };
}
