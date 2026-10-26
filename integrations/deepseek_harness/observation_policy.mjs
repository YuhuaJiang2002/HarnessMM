import { createHash } from 'node:crypto';
import {
  appendFileSync, closeSync, constants, fstatSync, mkdirSync, openSync, readFileSync, writeFileSync,
} from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

export const name = 'phiagent-native-observation-policy';
export const inject = ['tools', 'tokenMeter'];
export const RECALL_NAME = 'CHISELMemory_recall';
export const RECEIPT_PREFIX = '[Archived native observation]';
export const RECALL_PREFIX = '[Exact native observation page]';
export const POLICY_KEYS = [
  'version', 'minimum_characters', 'full_exposures', 'keep_recent',
  'preview_characters', 'page_characters',
];
export const RECALL_PARAMETERS = {
  type: 'object',
  properties: {
    observation_id: { type: 'string', pattern: '^obs_[0-9a-f]{64}$' },
    offset: { type: 'integer', minimum: 0 },
    limit: { type: 'integer', minimum: 1, maximum: 8000 },
  },
  required: ['observation_id'],
  additionalProperties: false,
};

const digest = text => createHash('sha256').update(text, 'utf8').digest('hex');

export function validatePolicy(policy) {
  const checkpoint = policy?.version === 2 ? policy.action_evidence_checkpoint : undefined;
  const keys = checkpoint ? [...POLICY_KEYS, 'action_evidence_checkpoint'] : POLICY_KEYS;
  if (!policy || Array.isArray(policy) || typeof policy !== 'object'
      || Object.keys(policy).sort().join() !== [...keys].sort().join()
      || POLICY_KEYS.some(key => !Number.isSafeInteger(policy[key]) || policy[key] < 1)
      || ![1, 2].includes(policy.version) || (policy.version === 2 && !checkpoint)
      || policy.minimum_characters < 1024
      || policy.full_exposures < 2 || policy.page_characters > 8000
      || policy.preview_characters * 2 >= policy.minimum_characters) {
    throw new Error('Invalid frozen native observation policy');
  }
  return Object.freeze({ ...policy,
    ...(checkpoint ? { action_evidence_checkpoint: Object.freeze({ ...checkpoint }) } : {}) });
}

export function completePaidUsage(usage) {
  if (!usage || !['inputTokens', 'outputTokens'].every(
    key => Number.isSafeInteger(usage[key]) && usage[key] >= 0,
  )) return false;
  const cacheKeys = ['cacheReadTokens', 'cacheWriteTokens'];
  if (cacheKeys.some(key => Object.hasOwn(usage, key)
      && (!Number.isSafeInteger(usage[key]) || usage[key] < 0))) return false;
  const known = usage.inputTokens + usage.outputTokens
    + cacheKeys.reduce((sum, key) => sum + (usage[key] ?? 0), 0);
  if (Object.hasOwn(usage, 'totalTokens')) {
    return Number.isSafeInteger(usage.totalTokens) && usage.totalTokens > 0
      && usage.totalTokens >= known
      && (!cacheKeys.every(key => Object.hasOwn(usage, key)) || usage.totalTokens === known);
  }
  return Number.isSafeInteger(known) && known > 0
    && cacheKeys.every(key => Object.hasOwn(usage, key));
}

function plainText(message) {
  if (message?.role !== 'tool' || !Array.isArray(message.content)
      || message.content.length !== 1 || message.content[0].type !== 'text'
      || typeof message.content[0].text !== 'string') return null;
  return message.content[0].text;
}

function eventKey(message) {
  const callId = message.toolCallId ?? message.source?.callId;
  const text = plainText(message);
  if (typeof callId !== 'string' || !callId || text === null) return null;
  return digest(JSON.stringify([callId, text]));
}

function failedObservation(message, text) {
  if (message.isError) return true;
  let value = text;
  for (let depth = 0; depth < 2 && typeof value === 'string'; depth++) {
    try {
      value = JSON.parse(value);
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      break;
    }
  }
  if (value && typeof value === 'object'
      && (value.error || value.is_error || value.timed_out
        || (Object.hasOwn(value, 'return_code') && value.return_code !== 0)
        || (Object.hasOwn(value, 'exit_code') && value.exit_code !== 0))) return true;
  return /(?:\berror:|traceback \(|\bassertionerror\b|\bFAILED\b)/i.test(text);
}

export class NativeObservationPolicy {
  constructor({ policy, directory, sessionId, record }) {
    this.policy = validatePolicy(policy);
    if (typeof sessionId !== 'string' || !sessionId || typeof record !== 'function') {
      throw new Error('Observation policy requires a session identity and audit sink');
    }
    mkdirSync(directory, { mode: 0o700, recursive: false });
    this.directory = directory;
    this.sessionId = sessionId;
    this.record = record;
    this.exposures = new Map();
    this.known = new Set();
    this.recalledCalls = new Set();
    this.paidRequests = 0;
    this.replacements = 0;
  }

  observe(messages, usage) {
    if (!completePaidUsage(usage)) {
      this.record({ kind: 'context/exposure-uncommitted', reason: 'incomplete_provider_usage' });
      return;
    }
    const keys = new Set();
    for (const message of messages ?? []) {
      const text = plainText(message);
      const key = eventKey(message);
      if (key && !text.startsWith(RECEIPT_PREFIX) && !text.startsWith(RECALL_PREFIX)) {
        keys.add(key);
      }
    }
    for (const key of keys) this.exposures.set(key, (this.exposures.get(key) ?? 0) + 1);
    this.paidRequests++;
    this.record({ kind: 'context/paid-exposure', observations: keys.size,
      paid_requests: this.paidRequests });
  }

  archive(text) {
    const id = `obs_${digest(text)}`;
    if (this.known.has(id)) {
      this.read(id);
      return id;
    }
    const value = { session_id: this.sessionId, sha256: id.slice(4), text };
    writeFileSync(join(this.directory, `${id}.json`), JSON.stringify(value), {
      flag: 'wx', mode: 0o600,
    });
    this.known.add(id);
    return id;
  }

  read(id) {
    if (typeof id !== 'string' || !/^obs_[0-9a-f]{64}$/.test(id) || !this.known.has(id)) {
      throw new Error('Unknown observation in this session');
    }
    const fd = openSync(join(this.directory, `${id}.json`),
      constants.O_RDONLY | constants.O_NOFOLLOW);
    let value;
    try {
      if (!fstatSync(fd).isFile()) throw new Error('Observation archive is not a regular file');
      value = JSON.parse(readFileSync(fd, 'utf8'));
    } finally {
      closeSync(fd);
    }
    if (value.session_id !== this.sessionId || typeof value.text !== 'string'
        || value.sha256 !== id.slice(4) || digest(value.text) !== id.slice(4)) {
      throw new Error('Observation archive identity or checksum mismatch');
    }
    return value.text;
  }

  recall(args, callId) {
    if (!args || Array.isArray(args) || typeof args !== 'object'
        || Object.keys(args).some(key => !['observation_id', 'offset', 'limit'].includes(key))) {
      throw new Error('Invalid exact-recall arguments');
    }
    const offset = Object.hasOwn(args, 'offset') ? args.offset : 0;
    const limit = Object.hasOwn(args, 'limit') ? args.limit : this.policy.page_characters;
    if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(limit)
        || limit < 1 || limit > this.policy.page_characters) {
      throw new Error('Invalid exact-recall page');
    }
    const points = Array.from(this.read(args.observation_id));
    if (offset > points.length) throw new Error('Recall offset exceeds observation length');
    const end = Math.min(points.length, offset + limit);
    this.recalledCalls.add(callId);
    this.record({ kind: 'context/recall', observation_id: args.observation_id, offset, end });
    return RECALL_PREFIX + '\n' + JSON.stringify({
      observation_id: args.observation_id, offset,
      next_offset: end < points.length ? end : null, total_characters: points.length,
      offset_unit: 'unicode_code_points', text: points.slice(offset, end).join(''),
    });
  }

  compact(session, tokenMeter, freezeMessage) {
    if (session.id !== this.sessionId || !session.surface?.nodes
        || typeof session.eventAt !== 'function' || typeof session.append !== 'function'
        || typeof session.deriveEventMessage !== 'function'
        || typeof tokenMeter?.estimateMessage !== 'function' || typeof freezeMessage !== 'function') {
      throw new Error('Pinned SDK replay-safe session surface API is unavailable');
    }
    const candidates = [...session.surface.nodes].map(seq => ({ seq, event: session.eventAt(seq) }))
      .filter(({ event }) => event?.type === 'tool/result');
    const recent = new Set(candidates.slice(-this.policy.keep_recent).map(item => item.seq));
    for (const { seq, event } of candidates) {
      const original = session.deriveEventMessage(event);
      const text = plainText(original);
      if (text === null || recent.has(seq) || text.startsWith(RECEIPT_PREFIX)
          || text.startsWith(RECALL_PREFIX) || this.recalledCalls.has(original.source?.callId)
          || failedObservation(original, text)) continue;
      const points = Array.from(text);
      const key = eventKey(original);
      if (!key || points.length < this.policy.minimum_characters
          || (this.exposures.get(key) ?? 0) < this.policy.full_exposures) continue;
      let id;
      try {
        id = this.archive(text);
      } catch (error) {
        this.record({ kind: 'context/original-preserved', source_seq: seq,
          reason: 'archive_error', error: { name: error.name, message: error.message } });
        continue;
      }
      const receipt = `${RECEIPT_PREFIX}\nid=${id}; characters=${points.length}; `
        + `sha256=${id.slice(4)}\n`
        + 'Previously supplied in full. This preview is incomplete and is not current state. '
        + 'Recall omitted values before using them; recall never reruns the original command.\n'
        + `Call ${RECALL_NAME} with ${JSON.stringify({ observation_id: id, offset: 0,
          limit: this.policy.page_characters })}; follow next_offset.\n`
        + `Verbatim head:\n${points.slice(0, this.policy.preview_characters).join('')}\n`
        + `Verbatim tail:\n${points.slice(-this.policy.preview_characters).join('')}`;
      const replacement = freezeMessage({
        ...original, content: [{ ...original.content[0], text: receipt }],
      });
      const before = tokenMeter.estimateMessage(original);
      const after = tokenMeter.estimateMessage(replacement);
      if (![before, after].every(value => Number.isFinite(value) && value >= 0)) {
        throw new Error('Native token-meter estimate is invalid');
      }
      if (after >= before || Array.from(receipt).length >= points.length) continue;
      session.append('compaction/prune', {
        shadowedRange: { start: seq, end: seq }, shadowedSeqs: [seq], shadowedTokenCount: before,
      });
      const landed = session.append('tool/result', { ...event.data, message: replacement }, {
        surfaceOp: { op: 'replace', startSeq: seq, endSeq: seq }, sourceEventSeqs: [seq],
      });
      this.replacements++;
      this.record({ kind: 'context/replacement', source_seq: seq, replacement_seq: landed.seq,
        observation_id: id, full_paid_exposures: this.exposures.get(key),
        estimated_tokens_before: before, estimated_tokens_after: after,
        estimates_are_not_provider_savings: true, original_event_retained: true,
        native_request_rewritten: false });
    }
  }
}

export function apply(ctx, config) {
  const job = JSON.parse(readFileSync(config.jobFile, 'utf8'));
  const record = value => appendFileSync(config.receiptsFile, JSON.stringify({
    at: new Date().toISOString(), session_id: job.session_id, ...value,
  }) + '\n', { mode: 0o600 });
  const policy = new NativeObservationPolicy({
    policy: job.observation_policy, directory: config.archiveDirectory,
    sessionId: job.session_id, record,
  });
  if (typeof ctx.tokenMeter?.estimateMessage !== 'function') {
    throw new Error('Pinned SDK token-meter APIs are required');
  }
  let readiness;
  function ready() {
    readiness ??= (async () => {
      const { freezeMessage } = await import(pathToFileURL(config.messageModule).href);
      if (typeof freezeMessage !== 'function') {
        throw new Error('Pinned SDK freezeMessage API is required');
      }
      let checkpoint;
      if (policy.policy.action_evidence_checkpoint) {
        const { checkpointFactory } = await import('./action_evidence_checkpoint.mjs');
        checkpoint = await checkpointFactory(
          policy.policy.action_evidence_checkpoint, policy, record, config.regionModule,
        );
      }
      return { freezeMessage, checkpoint };
    })();
    return readiness;
  }
  ctx.tools.register({
    name: RECALL_NAME, description: 'Read an exact page of an archived native tool observation.',
    parameters: { ...RECALL_PARAMETERS, properties: { ...RECALL_PARAMETERS.properties,
      limit: { type: 'integer', minimum: 1, maximum: policy.policy.page_characters } } },
    output: { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: value }] },
    execute(args, exec) {
      if (exec.agent.session.id !== job.session_id) throw new Error('Recall session mismatch');
      return policy.recall(args, exec.callId);
    },
  });
  ctx.on('agent/pre-step', async ({ agent, signal }, next) => {
    if (!signal.aborted) {
      const { freezeMessage, checkpoint } = await ready();
      if (checkpoint) await checkpoint.compact(agent, ctx.tokenMeter, signal, RECALL_NAME);
      policy.compact(agent.session, ctx.tokenMeter, freezeMessage);
    }
    return next();
  });
  ctx.on('llm/stream', (options, next) => (async function* () {
    let usage;
    for await (const chunk of next()) {
      if (chunk.type === 'usage') usage = chunk.usage;
      yield chunk;
    }
    if ((options.sessionId ?? job.session_id) === job.session_id
        && (!options.purpose || options.purpose === 'conversation')) {
      policy.observe(options.messages, usage);
      const { checkpoint } = await ready();
      if (checkpoint && completePaidUsage(usage)) checkpoint.observe(options.messages ?? []);
    }
  })());
  record({ kind: 'context/ready', policy: policy.policy, recall_tool: RECALL_NAME,
    provider_calls_for_routing: 0, native_loop_preserved: true });
}
