/**
 * HarnessMM for the DeepSeek Harness desktop app.
 *
 * The paper's controller lives between the model and the tool surface: a slow
 * actor writes a frozen action bank once, later steps read only public receipts
 * decoded from the *native* tool output, and a one-token selector (or a
 * deterministic commit) picks an already-decided step. This plugin rebuilds
 * that boundary on top of the stock desktop agent loop:
 *
 *   agent/pre-step        → (optional) remind the model the graph is live
 *   llm/stream            → after the slow turn is settled, replace the next
 *                           reasoning request with a certified native tool call
 *   tools/execute         → accept that call only on exact literal equality
 *   tools/result          → decode the real result and settle a certificate
 *
 * Nothing here replaces the native executor, the scorer, or provider
 * accounting: the substituted action runs through the ordinary tool pipeline,
 * so the loop logs its result, renders its card, and bills its usage itself.
 */

import { appendFileSync, mkdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import z from '@deepseek-ai/schemastery';
import {
  BankContractError,
  LABELS,
  PLAN_CLOSE,
  PLAN_OPEN,
  SHELL_TOOLS,
  SELECTOR_SYSTEM,
  decide,
  hash,
  hashJson,
  localGuard,
  makeCertificate,
  nativeCallId,
  parseBank,
  projectTurn,
  publicReceipt,
  violateDispatch,
} from './controller.mjs';

export const name = 'harnessmm';
export const inject = ['systemPrompt', 'tools'];

const SECTION_NAME = 'harnessmm:plan';
const STATUS_EVENTS = new Map([
  ['bank', 'banks_authored'],
  ['substitution-issued', 'substitutions_issued'],
  ['substitution', 'substitutions_certified'],
  ['boundary-error', 'boundary_errors'],
  ['dispatch-rejected', 'dispatch_rejections'],
]);
const STATUS_COUNTERS = Object.fromEntries([...STATUS_EVENTS.values()].map(name => [name, 0]));
// The selector is a memoryless re-read of a frozen plan, so it is the one place a
// strong model is worth paying for: a flash-tier model tends to answer "?" instead
// of picking the step the slow thinker already committed to.
const DEFAULT_ROUTE = { provider: 'qwen-token-plan-cn', model: 'glm-5.3' };

/** schemastery < 3.18.5 has no `.volatile()`; never call it unconditionally. */
function live(field) {
  return typeof field.volatile === 'function' ? field.volatile() : field;
}

/** A config field may be a plain value or a live source (`{ get() }`) from the loader. */
function read(field, fallback) {
  let value = field;
  // Live sources may nest: a group's field can be a source inside a resolved group.
  for (let depth = 0; depth < 4; depth++) {
    if (value === undefined || value === null) return fallback;
    if (typeof value === 'object' && typeof value.get === 'function') {
      try {
        value = value.get();
        continue;
      } catch {
        return fallback;
      }
    }
    break;
  }
  return value === undefined || value === null ? fallback : value;
}

// The default is the policy's own default, never the lower bound: a bound is a
// validation limit, so deriving the default from it silently narrows the policy.
const intField = (fallback, min, max) => live(z.number().default(fallback).min(min).max(max));

export const Config = z.object({
  policy: z.object({
    enabled: live(z.boolean().default(false)),
    maximum_candidates: intField(16, 1, 16),
    maximum_fast_steps: intField(8, 1, 16),
    maximum_public_bytes: intField(65536, 1024, 65536),
    admission_threshold_tokens: intField(1024, 0, 65536),
  }).default({}),
  selector: z.object({
    enabled: live(z.boolean().default(true)),
    provider: live(z.string().default('')),
    model: live(z.string().default('')),
    // A reasoning model spends tokens on hidden thinking before it can emit the
    // single label, so the paper's one-token budget starves the answer. Keep a
    // comfortable ceiling and rely on the selector prompt to emit one character.
    maxTokens: live(z.number().default(1024)),
    timeoutMs: live(z.number().default(20000)),
  }).default({}),
  shellTools: live(z.array(z.string()).default([])),
  planSection: live(z.boolean().default(true)),
  receiptsFile: live(z.string().default('')),
}).description('HarnessMM guarded decision reuse for the native desktop tool surface.');

/**
 * Read one field of a config group. schemastery resolves a declared group into a
 * plain object whose *fields* are live sources (`{ get() }`), so a group must be
 * indexed directly rather than passed through {@link read}.
 */
function field(group, key, fallback) {
  if (!group || typeof group !== 'object') return fallback;
  return read(group[key], fallback);
}

/**
 * Resolve the `policy` group from either the resolved config object or the group
 * itself. schemastery resolves a declared group into a plain object whose
 * *fields* are live sources (`{ get() }`), so a group is indexed directly rather
 * than passed through {@link read}.
 */
function policyGroup(config) {
  if (config && typeof config === 'object' && typeof config.get !== 'function') {
    // `apply` hands over the group; a resolved config hands over the whole object.
    if ('policy' in config && typeof config.policy === 'object') return config.policy;
    if ('enabled' in config) return config;
  }
  return read(config, {});
}

function policyOf(config) {
  const group = policyGroup(config);
  return {
    enabled: field(group, 'enabled', false) === true,
    version: 6,
    maximum_candidates: clampInt(field(group, 'maximum_candidates', 16), 1, 16),
    maximum_fast_steps: clampInt(field(group, 'maximum_fast_steps', 8), 1, 16),
    maximum_public_bytes: clampInt(field(group, 'maximum_public_bytes', 65536), 1024, 65536),
    admission_threshold_tokens: clampInt(field(group, 'admission_threshold_tokens', 1024), 0, 65536),
  };
}

function clampInt(value, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) return min;
  return Math.min(max, Math.max(min, Math.trunc(number)));
}

function shellToolsOf(config, ctx) {
  const group = config?.shellTools && typeof config.shellTools === 'object' && typeof config.shellTools.get !== 'function'
    ? config.shellTools
    : read(config?.shellTools, []);
  const explicit = Array.isArray(group) ? group : read(group, []);
  if (Array.isArray(explicit) && explicit.length) {
    return explicit.filter(tool => typeof tool === 'string' && SHELL_TOOLS.has(tool));
  }
  let names = [];
  try {
    names = (ctx.tools?.schemas?.() ?? []).map(tool => tool?.name).filter(tool => SHELL_TOOLS.has(tool));
  } catch {
    names = [];
  }
  return names.length ? names : [...SHELL_TOOLS];
}

/* ------------------------------------------------------------------------- *
 * Plan section — the dialect the slow actor has to speak
 * ------------------------------------------------------------------------- */

function planText(policy, shellTools) {
  const shellList = shellTools.length === 1
    ? `\`${shellTools[0]}\``
    : `${shellTools.slice(0, -1).map(tool => `\`${tool}\``).join(', ')} or \`${shellTools.at(-1)}\``;
  const example = JSON.stringify({
    candidates: [{
      candidate_id: 'A',
      predecessor_id: 'root',
      tool: shellTools[0],
      arguments: { description: 'Run the test suite', command: 'python -m pytest -q' },
      guard: { exit_code: 0 },
      required_receipt_fields: ['exit_code'],
      expected_local_cost: { selector_upper_tokens: 8192 },
      dependency_ids: [],
    }],
  }, null, 2);
  return [
    '## Executable action graph (HarnessMM)',
    '',
    'HarnessMM is active for this session. When you are about to act, you may also declare a frozen graph of *future* native calls next to the single native call you are making now. Later steps reuse that graph: a receipt-local guard is checked against the real tool result, and a surviving candidate then runs through the same native tool instead of asking you to reason again. When the graph is exhausted, a guard fails, or an action would need new judgement, control returns to you — write a fresh graph from the current state.',
    '',
    '### Declaring a graph',
    '',
    `Put exactly one \`${PLAN_OPEN}...${PLAN_CLOSE}\` block in the assistant text of a turn whose single tool call is the first ("root") action of the graph, and write the block before that tool call. Example:`,
    '',
    '```',
    PLAN_OPEN,
    example,
    PLAN_CLOSE,
    '```',
    '',
    `A declaration must be bare assistant text: a \`${PLAN_OPEN}\` body inside a code fence is documentation, not a declaration, and is ignored.`,
    `- \`tool\` is ${shellList}, or any tool in your visible catalog that the graph continues.`,
    `- \`candidate_id\` is a unique letter in A-${LABELS[policy.maximum_candidates - 1]}; \`predecessor_id\` is \`root\` or an id declared earlier in the same list.`,
    '- `arguments` is the complete literal argument object for that native tool, decided now. Never defer a decision into a placeholder: an argument that depends on output you have not seen means the action does not belong in the graph.',
    '- `guard` is a conjunction over fields of the immediate predecessor receipt: `exit_code` (shell status), `stdout_contains`, `stdout_equals`, `stderr_empty`, `json_equals` (`{"path":[keys],"value":…}` resolved against `stdout` parsed as JSON), and `ok`. Only receipt-local, syntactic conditions are allowed — no judgement about what the output means.',
    '- `required_receipt_fields` lists every receipt field the guard reads: `exit_code`, `stdout`, `stderr`, `ok`.',
    '- `expected_local_cost.selector_upper_tokens` is your upper bound on what the selector step itself may cost. The controller charges the smaller of that bound and its shipped fast-call ceiling, and admits the step only when the slow turn it replaces exceeded that charge by more than the admission threshold — so declaring more than the selector actually needs makes admission less likely, never more. Declare a realistic bound; do not inflate it.',
    '- `dependency_ids` lists earlier candidates this action relies on.',
    '',
    `Declare 2-8 genuinely certain future calls; at most ${policy.maximum_candidates}. When nothing is safely predictable write \`${PLAN_OPEN}{"candidates":[]}${PLAN_CLOSE}\` and continue normally — the graph is optional and an empty one costs nothing.`,
    '',
    '### What you will see afterwards',
    '',
    'A substituted step appears in the transcript as an assistant tool call whose id starts with `chisel_`. You did not author that step: the controller selected a candidate from your graph and replayed it with exactly the arguments you froze. Treat the result as your own and continue. Only a new root call asks for a fresh decision — do not "repair" a graph that already ran.',
    '',
    'A zero exit code never proves the task is complete. A prepared action that deletes, moves, or overwrites must carry a verified absolute path in its own `arguments`.',
  ].join('\n');
}

/* ------------------------------------------------------------------------- *
 * Plugin
 * ------------------------------------------------------------------------- */

export function apply(ctx, config = {}) {
  const policyConfig = config?.policy ?? {};
  const selectorConfig = config?.selector ?? {};

  const shellTools = shellToolsOf(config, ctx);
  const resolvePolicy = () => policyOf(policyConfig);

  const receiptsPath = () => {
    const explicit = read(config?.receiptsFile, '');
    if (typeof explicit === 'string' && explicit.trim()) return explicit;
    // The desktop host does not export DSH_HOME, so fall back to the profile's
    // own home directory rather than silently disabling every receipt.
    const home = process.env.DSH_HOME || join(homedir(), '.dsh');
    return join(home, 'harnessmm', 'receipts.jsonl');
  };

  const record = (entry) => {
    try {
      const path = receiptsPath();
      mkdirSync(dirname(path), { recursive: true });
      appendFileSync(path, `${JSON.stringify({ at: new Date().toISOString(), ...entry })}\n`, { mode: 0o600 });
    } catch {
      // Receipt durability must never break a model turn.
    }
  };

  /**
   * Per-session, per-turn bookkeeping.
   *
   * Keyed by a plain string session id, not by an agent object: the
   * `llm/stream` waterfall receives a frozen request that carries `sessionId`
   * and no agent reference at all, while `tools/execute` / `tools/result`
   * carry `exec.agent.id`. A string key is the one identity every mount point
   * can produce.
   */
  const sessions = new Map();
  const STORE_LIMIT = 64;

  const storeFor = (sessionId) => {
    const key = String(sessionId ?? 'session');
    let store = sessions.get(key);
    if (!store) {
      store = { turn: null, counters: { ...STATUS_COUNTERS } };
      sessions.set(key, store);
      while (sessions.size > STORE_LIMIT) sessions.delete(sessions.keys().next().value);
    }
    return store;
  };

  const recordFor = store => entry => {
    const counter = STATUS_EVENTS.get(entry.type);
    if (counter) store.counters[counter]++;
    record(entry);
  };

  ctx.tools.register({
    name: 'harnessmm_status',
    description: 'Read HarnessMM activity counters for this session. Use when asked about HarnessMM status.',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
    output: {
      schema: {
        type: 'object',
        properties: {
          enabled: { type: 'boolean' },
          ...Object.fromEntries(Object.keys(STATUS_COUNTERS).map(name => [name, { type: 'integer' }])),
        },
        required: ['enabled', ...Object.keys(STATUS_COUNTERS)],
        additionalProperties: false,
      },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
    },
    execute: async (_args, exec) => ({
      enabled: resolvePolicy().enabled,
      ...(sessions.get(sessionIdOfExecution(exec))?.counters ?? STATUS_COUNTERS),
    }),
  });

  if (read(config?.planSection, true) !== false) {
    ctx.systemPrompt.section({
      name: SECTION_NAME,
      order: 10300,
      interpolate: false,
      text: () => {
        const policy = resolvePolicy();
        return policy.enabled ? planText(policy, shellTools) : '';
      },
    });
  }

  // ---- llm/stream: the model-to-tool boundary ----------------------------

  ctx.on('llm/stream', (options, next) => {
    const policy = resolvePolicy();
    if (!policy.enabled) return next();
    // The request assembled by the agent loop is frozen and carries no agent
    // reference; `sessionId` is the only identity it offers. The selector's own
    // completion is a plain chat request with no session, so it falls through
    // here and never re-enters the boundary it was asked to decide.
    const sessionId = sessionIdOf(options);
    if (!sessionId) return next();
    const store = storeFor(sessionId);
    return boundaryStream({
      ctx, options, next, sessionId, policy, selectorConfig, shellTools, store, record: recordFor(store),
    });
  });

  // ---- tools/execute: the substituted call is accepted only verbatim ------

  ctx.on('tools/execute', async (payload, next) => {
    const policy = resolvePolicy();
    if (!policy.enabled) return next();
    const call = callOf(payload);
    if (!call || typeof call.id !== 'string' || !call.id.startsWith('chisel_')) return next();
    const sessionId = sessionIdOfExecution(payload);
    if (!sessionId) return next();
    const store = storeFor(sessionId);
    const turn = store.turn;
    // `tools/execute` carries no message history, so the accepted graph comes
    // from the turn the authoring step recorded, not from a fresh projection.
    const violation = violateDispatch({ projected: null, turn, call, policy });
    if (violation) {
      recordFor(store)({ type: 'dispatch-rejected', call_id: call.id, reason: violation.violation });
      throw new Error(`HarnessMM rejected a substituted call: ${violation.violation}`);
    }
    if (turn) turn.steps = (turn.steps ?? 0) + 1;
    return next();
  });

  // ---- tools/result: decode the native receipt and settle the certificate -

  ctx.on('tools/result', (exec, result) => {
    const policy = resolvePolicy();
    if (!policy.enabled) return;
    const call = callOf(exec);
    if (!call || typeof call.id !== 'string') return;
    const sessionId = sessionIdOfExecution(exec);
    if (!sessionId) return;
    const store = storeFor(sessionId);
    const turn = store.turn;
    if (!turn) return;
    // `tools/result` hands the frozen result as a second argument; the receipt
    // must come from it, never from the execution object, or every stdout guard
    // would read an empty string and a failed command would look like success.
    const receipt = publicReceipt({
      name: call.name,
      isError: result?.isError === true,
      text: textOf(result),
      canonical: canonicalOf(result),
    });
    // The boundary request that follows sees only the rendered message, so the
    // frozen receipt is remembered here by call id. Without it a failed root
    // whose rendered text lost its `[exit code: N]` trailer would be decoded as
    // exit 0 and could authorize a substitution.
    rememberNativeReceipt(turn, call.id, {
      canonical: canonicalOf(result),
      isError: result?.isError === true,
      text: textOf(result),
    });
    if (!call.id.startsWith('chisel_') || !turn.pending?.has(call.id)) return;
    turn.pending.delete(call.id);
    const meta = turn.pendingMeta?.get(call.id);
    // One certificate shape lives in `controller.mjs`; the host adds the two
    // evidence fields the desktop transcript wants (the decoded receipt itself
    // and the count of bypassed slow callbacks). Everything else — schema,
    // receipt/action hashes, selector usage and the author-time slow tokens —
    // comes from `makeCertificate`, so the pure protocol and the live receipt
    // can no longer drift apart.
    const certificate = {
      ...makeCertificate({
        turnId: turn.turnId,
        decisionId: meta?.decisionId ?? null,
        bankSha: turn.bankSha,
        candidateId: meta?.candidateId ?? null,
        predecessorId: meta?.predecessorId ?? null,
        bypassedCallbackId: turn.bypassedCallbackId ?? null,
        tool: call.name,
        argumentsText: call.arguments,
        receiptHash: hashJson(receipt),
        selector: meta?.selector ?? null,
        probability: meta?.probability ?? null,
        usage: meta?.usage ?? null,
        slowTokens: turn.slowTokens ?? 0,
      }),
      native_receipt: receipt,
      amount_substituted_slow_callbacks: 1,
    };
    recordFor(store)({ type: 'substitution', ...certificate });
    turn.certificates = (turn.certificates ?? 0) + 1;
  });

  return { planText: () => planText(resolvePolicy(), shellTools) };
}

/* ------------------------------------------------------------------------- *
 * Boundary stream
 * ------------------------------------------------------------------------- */

async function* boundaryStream({ ctx, options, next, sessionId, policy, selectorConfig, shellTools, store, record }) {
  const key = sessionId;
  let stockStarted = false;
  let emitted = false;
  const startStock = () => {
    stockStarted = true;
    return next();
  };
  try {
    const inner = boundaryStreamInner({
      ctx, options, next: startStock, sessionId, policy, selectorConfig, shellTools, store, key, record,
    });
    for await (const chunk of inner) {
      emitted = true;
      yield chunk;
    }
  } catch (error) {
    // The host owns provider failures. Re-entering next() could repeat a paid
    // request, and appending a fresh stream to emitted chunks corrupts the turn.
    if (stockStarted) throw error;
    recordBoundaryError(record, key, error);
    if (emitted) throw error;
    yield* stockStream(next);
  }
}

function recordBoundaryError(record, turnId, error) {
  record({ type: 'boundary-error', turn_id: turnId, reason: String(error?.message ?? error) });
}

function stockStream(next) {
  const produced = next();
  return produced && typeof produced[Symbol.asyncIterator] === 'function'
    ? produced
    : produced();
}

function boundaryStreamInner({ ctx, options, next, sessionId, policy, selectorConfig, shellTools, store, key, record }) {
  const projected = projectTurn(options.messages ?? [], options.tools ?? [], policy);
  let turn = store.turn;

  if (!projected) {
    // A slow request: fix the boundary facts before any chunk exists, so a
    // mid-turn projection can never be mistaken for an authoring step.
    turn = {
      turnId: `${key}:${Date.now().toString(36)}`,
      steps: 0,
      certificates: 0,
      pending: new Set(),
      pendingMeta: new Map(),
      nativeReceipts: new Map(),
      gate: 'slow',
      bypassedCallbackId: options?.metadata?.attemptId ?? options?.requestId ?? null,
    };
    store.turn = turn;
  }
  if (!turn) {
    // A projected turn whose session was not observed while it was slow — the
    // host restarted in the middle of it, or the plugin mounted late. The turn
    // still needs a stored record, or the issued call never reaches the guard
    // and the certificate sink.
    turn = {
      turnId: `${key}:resumed:${Date.now().toString(36)}`,
      steps: 0,
      certificates: 0,
      pending: new Set(),
      pendingMeta: new Map(),
      nativeReceipts: new Map(),
      gate: 'resumed',
      bypassedCallbackId: options?.metadata?.attemptId ?? options?.requestId ?? null,
    };
    store.turn = turn;
  }

  if (projected) {
    // The issued call id is derived from the bank hash, so a projected turn must
    // carry it even when this process never authored the bank — the host can
    // restart in the middle of a session, or mount the plugin late. Without this
    // the id falls back to the turn id and the certificate loses `bank_sha256`.
    turn.bankSha = projected.bank.sha256;
    turn.bankSize = projected.bank.bytes;
    // Restoring the executable graph matters as much as restoring the bank id:
    // `tools/execute` receives no transcript, so a resumed turn whose guard state
    // was lost with the process must carry the authored candidates again. The
    // projection re-derived them from the session log, which is exactly the
    // content the slow author declared.
    if (!turn.pendingBank || !Array.isArray(turn.pendingCandidates)) {
      turn.pendingBank = { sha256: projected.bank.sha256 };
      turn.pendingCandidates = projected.bank.candidates;
    }
    // The receipt is the last tool result in the turn; its tool name is the one
    // this controller dispatched, not the raw message's (which carries none).
    const receiptMessage = options.messages.at(-1);
    // The frozen `tools/result` payload is the authoritative receipt. The
    // rendered message text is only a fallback for a fresh mount or a restart
    // that happened before the result was observed.
    const native = receiptMessage?.toolCallId
      ? turn.nativeReceipts?.get(receiptMessage.toolCallId)
      : null;
    const receipt = publicReceipt(native
      ? {
        name: projected.tool ?? 'native_tool',
        isError: native.isError === true,
        text: native.text,
        // Exit status is shell semantics; a non-shell predecessor keeps the
        // field unknown instead of fabricating success.
        canonical: SHELL_TOOLS.has(projected.tool) ? native.canonical : undefined,
      }
      : {
        name: projected.tool ?? 'native_tool',
        isError: receiptMessage?.isError === true,
        text: textOfMessage(receiptMessage),
        canonical: canonicalOfMessage(receiptMessage),
      });
    // "Deterministic" means the decision needs no model call at all, which is true
    // exactly when one candidate survives. It is a property of the surviving set,
    // not of whether a selector happens to be configured: gating it on
    // selector.enabled made a single-candidate step pay the selector anyway and
    // then fail the net-saving threshold.
    const surviving = projected.candidates.filter(candidate => localGuard(candidate.node ?? candidate, receipt).eligible
      && !(candidate.node ?? candidate).dependency_ids?.some?.(id => id !== 'root' && id !== projected.position));
    const deterministic = surviving.length === 1;
    const decision = decide({
      projected,
      receipt,
      slowTokens: turn.slowTokens ?? 0,
      deterministic,
      policy,
    });
    record({
      type: 'decision',
      turn_id: turn.turnId,
      kind: decision.kind,
      reason: decision.reason ?? null,
      eligible: decision.admission.eligible.map(item => item.candidate.candidate_id),
      rejected: decision.admission.rejected.map(item => ({ id: item.candidate.candidate_id, reason: item.reason })),
    });
    if (decision.kind === 'commit') {
      return commitStream({ ctx, options, sessionId, turn, decision, policy, record });
    }
    if (decision.kind === 'select') {
      if (field(selectorConfig, 'enabled', true) === false) {
        // A switched-off selector is a deliberate choice: never spend a model call
        // on the letter, hand the turn straight back to the slow actor.
        record({ type: 'selector-abstain', turn_id: turn.turnId, reason: 'selector-disabled', usage: null });
      } else {
        return selectStream({ ctx, options, next, sessionId, turn, decision, policy, selectorConfig, record });
      }
    }
  }

  return authoringStream({ options, next, turn, policy, record });
}

/** The ordinary slow stream, plus bank extraction when it settled. */
async function* authoringStream({ options, next, turn, policy, record }) {
  const chunks = [];
  for await (const chunk of stockStream(next)) {
    chunks.push(chunk);
    yield chunk;
  }
  // The stock response has already been delivered. Bank bookkeeping can fail
  // without changing that response or asking the provider to generate it again.
  try {
    turn.slowTokens = usageOf(chunks);
    turn.bypassedCallbackId = options?.metadata?.attemptId ?? options?.requestId ?? turn.bypassedCallbackId;
    const bank = parseBank(messageOf(chunks), options.tools ?? [], { maxCandidates: policy.maximum_candidates });
    if (!bank || !bank.candidates.length) {
      turn.gate = bank ? 'empty-bank' : 'no-bank';
      return;
    }
    turn.bankSha = bank.sha256;
    turn.bankSize = bank.bytes;
    turn.gate = 'bank-authored';
    // The dispatch guard reads these because `tools/execute` has no message
    // history to re-project from.
    turn.pendingBank = { sha256: bank.sha256 };
    turn.pendingCandidates = bank.candidates;
    record({
      type: 'bank',
      turn_id: turn.turnId,
      bank_sha256: bank.sha256,
      candidates: bank.candidates.length,
      public_bytes: bank.bytes,
    });
  } catch (error) {
    if (error instanceof BankContractError) {
      turn.gate = 'invalid-bank';
      record({ type: 'invalid-bank', turn_id: turn.turnId, reason: error.message });
    } else {
      recordBoundaryError(record, turn.turnId, error);
    }
  }
}

function commitStream({ ctx, options, sessionId, turn, decision, policy, record }) {
  return emitSubstitution({
    ctx, options, sessionId, turn, policy, record,
    candidate: decision.candidate,
    payload: decision.payload,
    decisionId: decision.decisionId,
    selector: 'deterministic',
    probability: null,
    usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
  });
}

function selectStream({ ctx, options, next, sessionId, turn, decision, policy, selectorConfig, record }) {
  return (async function* stream() {
    const completion = await oneTokenSelection(ctx, selectorConfig, decision.payload, options);
    const fallback = () => authoringStream({ options, next, turn, policy, record });
    if (!completion || completion.label === '?') {
      record({
        type: 'selector-abstain',
        turn_id: turn.turnId,
        usage: completion?.usage ?? null,
        reason: completion ? (completion.failure ? 'selector-error' : 'abstained') : 'no-completion',
        failure: completion?.failure ?? null,
        raw: completion?.raw ?? null,
      });
      yield* fallback();
      return;
    }
    const candidate = decision.admission.eligible
      .map(item => item.candidate)
      .find(item => item.candidate_id === completion.label);
    if (!candidate) {
      record({ type: 'selector-unmapped', turn_id: turn.turnId, label: completion.label });
      yield* fallback();
      return;
    }
    yield* emitSubstitution({
      ctx, options, sessionId, turn, policy, record, candidate,
      payload: decision.payload,
      decisionId: decision.decisionId,
      selector: 'fast-selector',
      probability: completion.probability,
      usage: completion.usage,
    });
  })();
}

async function* emitSubstitution({
  ctx, options, sessionId, turn, policy, record, candidate, payload, decisionId, selector, probability, usage,
}) {
  const argumentsText = JSON.stringify(candidate.arguments);
  const callId = nativeCallId(turn.bankSha ?? hashOf(`${turn.turnId}`), candidate.candidate_id);
  const guard = localGuard(candidate, payload.receipt);
  if (!guard.eligible) {
    record({ type: 'substitution-aborted', turn_id: turn.turnId, reason: guard.reason });
    return;
  }
  turn.pending.add(callId);
  turn.pendingMeta.set(callId, {
    decisionId,
    candidateId: candidate.candidate_id,
    predecessorId: candidate.predecessor_id,
    selector,
    probability,
    usage,
  });
  record({
    type: 'substitution-issued',
    turn_id: turn.turnId,
    decision_id: decisionId,
    call_id: callId,
    candidate_id: candidate.candidate_id,
    tool: candidate.tool,
    selector,
    probability,
    control_usage: usage,
  });
  // A substituted step is a turn no model authored, so it carries no native
  // thinking. DeepSeek's Anthropic-compatible endpoint runs in thinking mode and
  // rejects any assistant turn whose `tool_use` has no sibling `thinking` block
  // ("The `content[].thinking` in the thinking mode must be passed back to the
  // API"), which killed the next request with a 400 after the first live
  // substitution. That endpoint accepted an unsigned block for a substituted
  // turn, so this receipt-bound note is the sibling; the host adapter replays it
  // verbatim on every later request, exactly like native reasoning.
  const thinking = substitutionThinking(candidate);
  yield { type: 'block-start', index: 0, blockType: 'reasoning' };
  yield { type: 'reasoning-delta', index: 0, text: thinking };
  yield { type: 'block-end', index: 0, block: { type: 'reasoning', text: thinking } };
  yield { type: 'block-start', index: 1, blockType: 'tool-call' };
  yield {
    type: 'tool-call-delta',
    index: 1,
    id: callId,
    name: candidate.tool,
    argumentsDelta: argumentsText,
  };
  yield { type: 'block-end', index: 1, block: { type: 'tool-call', id: callId, name: candidate.tool, arguments: argumentsText } };
  yield { type: 'usage', usage };
  yield { type: 'finish', reason: { kind: 'tool-calls' } };
  void ctx; void policy; void sessionId; void options;
}

/**
 * The `thinking` sibling a substituted assistant turn needs in thinking mode.
 *
 * It states what actually happened — a frozen candidate matched its receipt
 * guard and was replayed — and never quotes a plan tag, so the transcript replay
 * cannot mistake it for an authored bank. Kept short because it is replayed in
 * the history of every later request.
 */
function substitutionThinking(candidate) {
  return `HarnessMM replay: frozen candidate ${candidate.candidate_id} matched its receipt guard, `
    + `so ${candidate.tool} ran without a new model decision.`;
}

/* ------------------------------------------------------------------------- *
 * Selector — one forced token, host-owned route
 * ------------------------------------------------------------------------- */

function selectorRoute(ctx, selectorConfig, options) {
  const provider = field(selectorConfig, 'provider', '') || process.env.HARNESSMM_SELECTOR_PROVIDER || '';
  const model = field(selectorConfig, 'model', '') || process.env.HARNESSMM_SELECTOR_MODEL || '';
  if (provider && model) return { provider, model };
  // Default to the route the boundary request itself rides. It is the one route
  // this host has already proven answerable under the live profile, which matters
  // because a thinking-only gateway refuses a request that carries no reasoning
  // control at all — the selector would then die before it ever reads the receipt.
  if (options?.provider && options?.model) return { provider: options.provider, model: options.model };
  return DEFAULT_ROUTE;
}

/**
 * Ask the host's own model runtime for one label.
 *
 * The request carries only what `dsh-llm`'s documented example carries (provider,
 * model, messages, a cap, and — on a thinking-only route — the boundary's own
 * reasoning level). It deliberately sends no `temperature` and no `purpose`: both
 * are optional, both are interpreted by provider adapters, and a route that rejects
 * either rejects the whole call — which threw away the substitution for a step the
 * gate had already admitted. Two shapes are tried so a route that refuses an output
 * cap still gets a chance to answer, and the terminal failure of the last attempt
 * is reported to the receipt.
 */
async function oneTokenSelection(ctx, selectorConfig, payload, options) {
  const route = selectorRoute(ctx, selectorConfig, options);
  if (!route) return { label: '?', usage: null, failure: { code: 'NO_ROUTE' } };
  const llm = ctx.get('llm');
  if (!llm || typeof llm.stream !== 'function') return { label: '?', usage: null, failure: { code: 'NO_LLM' } };
  const maxTokens = clampInt(field(selectorConfig, 'maxTokens', 1024), 1, 4096);
  const timeoutMs = clampInt(field(selectorConfig, 'timeoutMs', 20000), 1000, 120000);
  const messages = [
    { role: 'system', content: [{ type: 'text', text: SELECTOR_SYSTEM_TEXT }] },
    { role: 'user', content: [{ type: 'text', text: JSON.stringify(payload) }] },
  ];
  const reasoningEffort = await selectionReasoningEffort(llm, route, options);
  const base = reasoningEffort ? { reasoningEffort } : {};
  let last = { label: '?', usage: null, failure: { code: 'NO_ATTEMPT' } };
  for (const shape of [{ maxTokens, ...base }, { ...base }]) {
    last = await oneSelectionAttempt(llm, route, messages, shape, timeoutMs);
    // A completed answer — including a genuine `?` — ends the ladder. Only a
    // failed call is retried in the more conservative shape.
    if (!last.failure) return last;
  }
  return last;
}

/**
 * The reasoning level a selector call must carry.
 *
 * A gateway that only serves thinking requests refuses a call that carries no
 * reasoning control at all (`400 ... "The value of the enable_thinking parameter
 * is restricted to True."`), which is exactly how the first live multi-candidate
 * step died before it read a single token of the receipt. The selector therefore
 * inherits the boundary request's own level — that request just succeeded, so the
 * level is valid for this route — and falls back to the model's greatest declared
 * effort when the boundary carried none.
 *
 * The selector dispatches plain `stream()` calls rather than a prepared one on
 * purpose: `prepareCall()` freezes a config that the later dispatch must match
 * exactly, and a mismatch is rejected before the adapter ever sees the request
 * ("prepared LLM call config changed before adapter dispatch").
 */
async function selectionReasoningEffort(llm, route, options) {
  const inherited = options?.reasoningEffort;
  if (typeof inherited === 'string' && inherited.length > 0) return inherited;
  if (typeof llm?.resolveModelInfo !== 'function') return null;
  try {
    const info = await llm.resolveModelInfo(route.provider, route.model);
    const efforts = Array.isArray(info?.reasoning?.efforts) ? info.reasoning.efforts : [];
    if (efforts.length === 0) return null;
    return efforts[efforts.length - 1]?.id ?? null;
  } catch {
    return null;
  }
}

async function oneSelectionAttempt(llm, route, messages, shape, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const chunks = [];
    for await (const chunk of llm.stream({
      provider: route.provider,
      model: route.model,
      messages,
      signal: controller.signal,
      ...shape,
    })) {
      chunks.push(chunk);
    }
    const usage = usageObject(chunks);
    const terminal = chunks.findLast(chunk => chunk.type === 'finish');
    const kind = terminal?.reason?.kind;
    if (terminal && (kind === 'error' || kind === 'aborted')) {
      return { label: '?', usage, failure: terminal.reason.failure ?? { code: kind } };
    }
    const text = messageOf(chunks).content
      .filter(part => part.type === 'text')
      .map(part => part.text)
      .join('')
      .trim();
    const label = [...text].find(character => LABELS.includes(character) || character === '?') ?? '?';
    return { label, probability: null, usage, raw: text.slice(0, 32), failure: null };
  } catch (error) {
    return { label: '?', usage: null, failure: { code: 'THREW', message: String(error?.message ?? error) } };
  } finally {
    clearTimeout(timer);
  }
}

const SELECTOR_SYSTEM_TEXT = SELECTOR_SYSTEM;

/* ------------------------------------------------------------------------- *
 * Chunk / payload utilities
 * ------------------------------------------------------------------------- */

/**
 * The session identity of an `llm/stream` request.
 *
 * The agent loop hands the waterfall a frozen `{...header.config, messages,
 * toolHistory, tools, sessionId, signal}` object: there is no `agent` field,
 * so the agent object the earlier draft looked for never existed. `sessionId`
 * is the documented field and is present on every request.
 */
function sessionIdOf(options) {
  const id = options?.sessionId;
  if (typeof id === 'string' && id.length > 0) return id;
  if (id !== null && typeof id === 'object' && typeof id.toString === 'function') {
    const text = String(id);
    if (text && text !== '[object Object]') return text;
  }
  // A bare harness may drive the boundary without a session; keep one bucket
  // rather than dropping every substitution on the floor.
  const fallback = options?.metadata?.sessionId;
  return typeof fallback === 'string' && fallback.length > 0 ? fallback : null;
}

/**
 * The session identity of a tool dispatch/result.
 *
 * `tools/execute` and `tools/result` receive a `ToolExecution`, whose agent is
 * `{ id: SessionId }` — the id alone, with no `session` property.
 */
function sessionIdOfExecution(exec) {
  const agent = exec?.agent ?? exec?.exec?.agent;
  const id = agent?.id ?? agent?.session?.id ?? exec?.sessionId;
  if (typeof id === 'string' && id.length > 0) return id;
  if (id !== null && typeof id === 'object' && typeof id.toString === 'function') {
    const text = String(id);
    if (text && text !== '[object Object]') return text;
  }
  return null;
}

function safeSchemas(ctx) {
  try {
    return ctx.tools?.schemas?.() ?? [];
  } catch {
    return [];
  }
}

function messageOf(chunks) {
  const parts = [];
  for (const chunk of chunks) {
    if (chunk.type === 'block-end' && chunk.block) parts.push(chunk.block);
  }
  return { role: 'assistant', content: parts };
}

function usageOf(chunks) {
  const usage = usageObject(chunks);
  return usage.totalTokens;
}

function usageObject(chunks) {
  let usage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
  for (const chunk of chunks) if (chunk.type === 'usage' && chunk.usage) usage = chunk.usage;
  return usage;
}

function callOf(payload) {
  const candidates = [payload, payload?.exec, payload?.call, payload?.execution];
  for (const entry of candidates) {
    if (!entry || typeof entry !== 'object') continue;
    if (typeof entry.name === 'string' && typeof entry.callId === 'string' && entry.arguments !== undefined) {
      return normalizeCall(entry.name, entry.callId, entry.arguments);
    }
  }
  return null;
}

function normalizeCall(name, id, args) {
  let text;
  if (typeof args === 'string') text = args;
  else {
    try {
      text = JSON.stringify(args);
    } catch {
      text = '';
    }
  }
  return { id, name, arguments: text };
}

function textOf(payload) {
  const blocks = payload?.result?.content ?? payload?.content ?? payload?.result?.value?.content;
  if (Array.isArray(blocks)) {
    const text = blocks.filter(block => block?.type === 'text').map(block => block.text ?? '').join('\n');
    if (text) return text;
  }
  if (typeof payload?.text === 'string') return payload.text;
  if (typeof payload?.result?.text === 'string') return payload.result.text;
  return '';
}

function textOfMessage(message) {
  if (!message) return '';
  if (Array.isArray(message.content)) {
    return message.content.filter(part => part?.type === 'text').map(part => part.text ?? '').join('\n');
  }
  if (typeof message.content === 'string') return message.content;
  return '';
}

function canonicalOf(payload) {
  return payload?.result?.value ?? payload?.value ?? payload?.canonical ?? undefined;
}

/**
 * Remember the receipt decoded from a frozen `tools/result` payload so the next
 * boundary request can decode the rendered tool message authoritatively. Bounded
 * per turn: only the newest receipts can matter to a projection.
 */
function rememberNativeReceipt(turn, callId, native) {
  if (!(turn?.nativeReceipts instanceof Map)) return;
  turn.nativeReceipts.set(callId, native);
  while (turn.nativeReceipts.size > 32) {
    turn.nativeReceipts.delete(turn.nativeReceipts.keys().next().value);
  }
}

/**
 * Fallback for a tool result message that was never observed through
 * `tools/result` (late mount / restart): accept a structured receipt if the
 * message happens to carry one; otherwise the rendered text is decoded.
 */
function canonicalOfMessage(message) {
  if (!message || typeof message !== 'object') return undefined;
  const parts = Array.isArray(message.content) ? message.content : [];
  for (const part of parts) {
    if (!part || typeof part !== 'object') continue;
    const candidate = part.result?.value ?? part.value ?? part.canonical;
    if (candidate && typeof candidate === 'object') return candidate;
  }
  const candidate = message.result?.value ?? message.value ?? message.canonical;
  return candidate && typeof candidate === 'object' ? candidate : undefined;
}

function hashOf(value) {
  return hash(String(value));
}