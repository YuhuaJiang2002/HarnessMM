/**
 * pi-harnessmm — HarnessMM (arXiv 2610.32813) as a pi extension.
 *
 * The slow actor authors an Executable Decision Graph in `<CHISEL_PLAN>` before
 * one root `terminal` call. The controller admits each successor whose guard
 * holds on the immediately preceding native receipt, executes it through the same
 * terminal interface, and certifies the reasoning callback it replaced. Control
 * returns to the model on exhaustion, unknown evidence, native error or abstention.
 *
 * Ported from https://github.com/YuhuaJiang2002/HarnessMM:
 *   integrations/deepseek_harness/cot_jev.mjs, sar_v2.mjs, sar_admission.mjs
 */
import { AsyncLocalStorage } from "node:async_hooks";
import type { ExtensionAPI, ExtensionContext, ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import type { Context as ModelContext } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { parseBank, type Bank, type BankState, type PublicResult, type ToolContract } from "../src/bank.ts";
import { BankContractError, planInstruction, validateCotJev, type Policy } from "../src/policy.ts";
import { SubstitutionController, type ControllerEvent } from "../src/controller.ts";
import type { TrainPriors } from "../src/admission.ts";
import { compactMessages, projectionBinding, slowTokensOf, toAuthoringMessage, userRequests, type PiMessage } from "../src/pi-messages.ts";
import { modelReceiptText, receiptText, runTerminal, type TerminalReceipt } from "../src/terminal.ts";

const LEDGER = "harnessmm-ledger";

/**
 * Set while the controller dispatches a substitute. A re-entered `terminal` call is
 * one admitted action going through the native interface, so it must not re-parse
 * the bank and must not recurse.
 */
const dispatch = new AsyncLocalStorage<boolean>();

/** Terminal tool contract: the only tool a bank may name. */
const TERMINAL_PARAMETERS = Type.Object({
  command: Type.String({ description: "Shell command to run" }),
  cwd: Type.Optional(Type.String({ description: "Working directory; defaults to the session cwd" })),
  timeout_sec: Type.Optional(Type.Integer({ description: "Kill the command after this many seconds" })),
});

/** Policy from the environment, validated before any use. */
function loadPolicy(): Policy {
  const raw = process.env.HARNESSMM_POLICY;
  const base = { version: 6, enabled: true, maximum_candidates: 16, maximum_fast_steps: 8, maximum_public_bytes: 65536 };
  const merged = raw ? { ...base, ...(JSON.parse(raw) as Partial<Policy>) } : base;
  return validateCotJev(merged);
}

/** Frozen TRAIN priors for the priors-based admission (policy versions 8 and 10). */
function loadPriors(): TrainPriors | null {
  const raw = process.env.HARNESSMM_TRAIN_PRIORS;
  if (!raw) return null;
  const parsed = JSON.parse(raw) as TrainPriors;
  return parsed?.phase === "train" ? parsed : null;
}

export default function harnessmm(pi: ExtensionAPI) {
  const policy = loadPolicy();
  const priors = loadPriors();
  const ledger: ControllerEvent[] = [];
  const tool: ToolContract = { name: "terminal", parameters: TERMINAL_PARAMETERS };

  const restore = (ctx: ExtensionContext) => {
    ledger.length = 0;
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type === "custom" && entry.customType === LEDGER) ledger.push(entry.data as ControllerEvent);
    }
  };
  pi.on("session_start", async (_event, ctx) => restore(ctx));
  pi.on("session_tree", async (_event, ctx) => restore(ctx));

  // The authoring instruction is the upstream text for this policy version. It goes
  // into a prompt section so pi patches only that section instead of replacing the
  // whole system prompt, which keeps the cached prefix stable across turns.
  pi.on("before_agent_start", async (event) => {
    if (!policy.enabled) return;
    event.systemPromptOptions.sections = {
      ...event.systemPromptOptions.sections,
      "harnessmm-authoring": planInstruction(policy),
    };
  });

  // Remove the authored bank from the slow context, preserving every native action.
  pi.on("context", async (event) => {
    if (!policy.enabled) return;
    const messages = event.messages as unknown as PiMessage[];
    const { messages: compacted, replacements } = compactMessages(messages);
    if (!replacements) return;
    const before = messages.length;
    const after = compacted.length;
    for (let index = 0; index < before; index++) {
      const original = messages[index];
      const next = compacted[index];
      if (!next || original === next) continue;
      const binding = projectionBinding(original, next);
      if (!binding.nativeActionsPreserved) return;
      ledger.push({ kind: "sar/controller-bank-extracted", ...binding, policy_version: policy.version });
    }
    return { messages: compacted as never };
  });

  pi.registerCommand("harnessmm", {
    description: "Report guarded decision reuse for this session",
    handler: async (_args, ctx) => {
      const certificates = ledger.filter(event => event.kind === "sar/substitution-certificate");
      const slowTokens = ledger.filter(event => event.kind === "sar/slow-call").reduce((total, event) => total + Number(event.tokens ?? 0), 0);
      const selectorTokens = ledger.filter(event => event.kind === "jev/decision").reduce((total, event) => total + Number((event.usage as { totalTokens?: number } | undefined)?.totalTokens ?? 0), 0);
      const deferred = ledger.filter(event => event.kind === "sar/defer");
      const authored = ledger.filter(event => event.kind === "sar/bank-authored");
      const extract = ledger.filter(event => event.kind === "sar/controller-bank-extracted");
      const fallbacks = ledger.filter(event => event.kind === "jev/slow-fallback" || event.kind === "sar/fallback-slow-call");
      ctx.ui.notify([
        `harnessmm policy v${policy.version} (${policy.enabled ? "enabled" : "disabled"})`,
        `  banks authored: ${authored.length}`,
        `  banks removed from slow context: ${extract.length}`,
        `  certified substitutions: ${certificates.length}`,
        `  deferred to the model: ${deferred.length}, selector abstentions: ${fallbacks.length}`,
        `  slow tokens seen: ${slowTokens}, selector tokens spent: ${selectorTokens}`,
        certificates.length
          ? `  last: ${certificates.slice(-3).map(event => String(event.candidate_id)).join(", ")}`
          : "  last: none",
      ].join("\n"), "info");
    },
  });

  pi.registerTool({
    name: "terminal",
    label: "Terminal",
    description: [
      "Run one shell command and return its native receipt {return_code, stdout, stderr}.",
      "Before calling it, optionally author a continuation bank for the calls you already know:",
      '<CHISEL_PLAN>{"candidates":[{"candidate_id":"A","predecessor_id":"root","tool":"terminal",',
      '"arguments":{"command":"literal command"},"guard":{"return_code":0},"required_receipt_fields":["return_code"],',
      '"expected_local_cost":{"selector_upper_tokens":8192},"dependency_ids":[]}]}</CHISEL_PLAN>',
      "Every argument must already be fully determined. Candidates whose guard holds on the real receipt",
      "execute without another model call; a guard that cannot be read, a native error, or an exhausted",
      "bank returns control to you.",
    ].join(" "),
    promptSnippet: "Run a shell command, optionally with a pre-authored bank of follow-up commands",
    promptGuidelines: [
      "Use terminal for shell work; when several dependent commands are already fully determined, author the bank in <CHISEL_PLAN> before the root call so they run without another turn.",
      "Never place ${...}, {{...}} or $(...) placeholders in bank arguments: the bank is literal, so an unresolved argument must wait for a later turn.",
      "A bank candidate may not repeat the root command, and every candidate's tool must be terminal.",
    ],
    parameters: TERMINAL_PARAMETERS,

    async execute(toolCallId, params, signal, _onUpdate, ctx) {
      if (dispatch.getStore() === true) {
        // A bank-authorized action arriving through the original tool interface.
        const receipt = await runTerminal(params, { cwd: ctx.cwd, signal });
        return { content: [{ type: "text", text: modelReceiptText(receipt) }], details: { returnCode: receipt.return_code, substitutedByController: true } };
      }
      const branch = ctx.sessionManager.getBranch();
      const messages = branch.filter(entry => entry.type === "message").map(entry => (entry as { message: PiMessage }).message);
      const authoring = toAuthoringMessage(messages.at(-1));
      const authoringIndex = messages.length - 1;

      let bank: Bank | null = null;
      let bankError: string | null = null;
      try {
        bank = parseBank(authoring, policy, [tool]);
      } catch (error) {
        if (!(error instanceof BankContractError)) throw error;
        bankError = error.message;
        ledger.push({ kind: "jev/invalid-bank", error: error.message });
      }

      const receipts: Array<{ id: string; label: string; receipt: TerminalReceipt }> = [];
      const events: ControllerEvent[] = [];
      const record = (event: ControllerEvent) => {
        events.push(event);
        ledger.push(event);
      };

      const runRoot = async (): Promise<TerminalReceipt> =>
        runTerminal({ command: params.command, ...(params.cwd ? { cwd: params.cwd } : {}), ...(params.timeout_sec ? { timeout_sec: params.timeout_sec } : {}) }, { cwd: ctx.cwd, signal });

      if (!bank || !bank.candidates.length) {
        const receipt = await runRoot();
        pi.appendEntry(LEDGER, bankError ? { kind: "jev/invalid-bank", error: bankError } : { kind: "sar/no-bank", root_id: toolCallId });
        return {
          content: [{ type: "text", text: modelReceiptText(receipt) }],
          details: { outcome: "plain", returnCode: receipt.return_code, substituted: 0, receipts: [receipt], events, bankError },
          ...(receipt.return_code !== 0 ? { isError: false } : {}),
        };
      }

      const controller = new SubstitutionController({
        policy,
        priors,
        record,
        select: async input => selectLabel(ctx, input.system, input.state, input.labels),
        episodeId: ctx.sessionManager.getSessionId?.() ?? "pi-session",
      });
      const authoringTokens = slowTokensOf(messages[authoringIndex]);
      controller.ownSlowCall(toolCallId, authoringTokens);
      if (authoringTokens > 0) ledger.push({ kind: "sar/slow-call", native_attempt_id: toolCallId, tokens: authoringTokens });
      controller.noteBank(bank, toolCallId);
      controller.beginBoundary(toolCallId, previousToolCallId(messages, authoringIndex));

      const root = await runRoot();
      receipts.push({ id: toolCallId, label: "root", receipt: root });
      let predecessorCallId = toolCallId;
      const results: PublicResult[] = [
        { call_id: toolCallId, is_error: root.return_code !== 0, status: "PARSED", value: { native_text: receiptText(root) } },
      ];

      let previous = "root";
      let committed = 0;
      while (committed < policy.maximum_fast_steps) {
        const candidates = bank.candidates.filter(candidate => candidate.after === previous);
        if (!candidates.length) {
          record({ kind: "jev/full-cot", bank_sha256: bank.sha256, reason: "bank_exhausted" });
          break;
        }
        controller.precede(predecessorCallId);
        const state: BankState = {
          public_user_requests: userRequests(messages, authoringIndex),
          candidates,
          public_results: results,
        };
        const decision = await controller.decide(bank, state);
        if (decision.kind === "slow") break;
        const args = JSON.parse(decision.call.arguments) as { command: string; cwd?: string; timeout_sec?: number };
        // Through pi's own tool interface, so every hook and permission check sees it.
        const receipt = await dispatch.run(true, () => callNative(ctx, args, signal));
        const text = receiptText(receipt);
        controller.observeTool({ call_id: decision.call.id, name: decision.call.name, arguments: args, result: { observation: text } });
        receipts.push({ id: decision.call.id, label: decision.candidate.label!, receipt });
        results.push({ call_id: decision.call.id, is_error: receipt.return_code !== 0, status: "PARSED", value: { native_text: text } });
        previous = decision.candidate.label!;
        predecessorCallId = decision.call.id;
        committed += 1;
        if (receipt.return_code !== 0) break;
      }

      const certificates = events.filter(event => event.kind === "sar/substitution-certificate");
      const selectorTokens = events
        .filter(event => event.kind === "jev/decision")
        .reduce((total, event) => total + Number((event.usage as { totalTokens?: number } | undefined)?.totalTokens ?? 0), 0);
      const outcome = committed === 0
        ? (events.some(event => event.kind === "sar/defer" || event.kind === "jev/full-cot") ? "deferred" : "completed")
        : (receipts.at(-1)!.receipt.return_code !== 0 ? "failed" : "completed");
      const report = [
        `harnessmm: ${outcome}; ${committed} substitution(s) certified`,
        ...receipts.map(item => `${item.label}: exit=${item.receipt.return_code}`),
        certificates.length ? `callbacks bypassed: ${certificates.length}` : "callbacks bypassed: 0",
        `complete-task cost: authoring turn ${authoringTokens} tokens, selector ${selectorTokens} tokens`,
      ].join("\n");
      for (const item of receipts) pi.appendEntry(LEDGER, { kind: "harnessmm/receipt", id: item.id, label: item.label, return_code: item.receipt.return_code });

      return {
        content: [{ type: "text", text: `${report}\n\nreceipts:\n${receipts.map(item => modelReceiptText(item.receipt)).join("\n")}` }],
        details: {
          outcome,
          returnCode: root.return_code,
          substituted: certificates.length,
          graphHash: bank.sha256,
          receipts,
          certificates,
          events,
        },
      };
    },
  });
}

/** Dispatch one admitted action through the registered `terminal` tool. */
async function callNative(ctx: ExtensionToolContext, args: { command: string; cwd?: string; timeout_sec?: number }, signal?: AbortSignal): Promise<TerminalReceipt> {
  const outcome = await ctx.executeTool("terminal", args, { signal });
  const text = outcome.result.content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map(part => part.text)
    .join("\n");
  try {
    const parsed = JSON.parse(text) as TerminalReceipt;
    if (typeof parsed.return_code === "number" && typeof parsed.stdout === "string") return parsed;
  } catch {
    /* fall through to an unreadable receipt */
  }
  return { return_code: outcome.isError ? 1 : 0, stdout: text, stderr: outcome.isError ? text : null };
}

/** The id of the tool call that preceded the authoring turn, if any. */
function previousToolCallId(messages: PiMessage[], authoringIndex: number): string | undefined {
  for (let index = authoringIndex - 1; index >= 0; index--) {
    const message = messages[index];
    if (message.role !== "assistant" || !Array.isArray(message.content)) continue;
    const call = (message.content as any[]).find(part => part?.type === "toolCall");
    if (call) return String(call.id);
  }
  return undefined;
}

/**
 * One-token style selector: ask the session model for a single label.
 * pi exposes no logprobs, so this verifies the reply shape instead of a
 * probability, and any other reply abstains back to slow reasoning.
 */
async function selectLabel(ctx: ExtensionToolContext, system: string, state: unknown, labels: string[]): Promise<{ label: string; usage?: { inputTokens: number; outputTokens: number; totalTokens: number } }> {
  const model = ctx.model;
  if (!model) return { label: "?" };
  const context: ModelContext = {
    systemPrompt: system,
    messages: [{ role: "user", content: [{ type: "text", text: JSON.stringify(state) }], timestamp: Date.now() }],
  };
  const answer = await ctx.modelRegistry.complete(model, context, { maxTokens: 1, temperature: 0 });
  const reply = answer.content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map(part => part.text)
    .join("")
    .trim();
  const usage = answer.usage
    ? { inputTokens: answer.usage.input ?? 0, outputTokens: answer.usage.output ?? 0, totalTokens: answer.usage.totalTokens ?? 0 }
    : undefined;
  return labels.includes(reply) ? { label: reply, ...(usage ? { usage } : {}) } : { label: "?", ...(usage ? { usage } : {}) };
}
