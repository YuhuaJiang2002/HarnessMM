import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';

export const CHECKPOINT_PREFIX = '[Exact action-evidence checkpoint]';
export const CHECKPOINT_KEYS = [
  'version', 'retain_tokens', 'minimum_saved_tokens', 'minimum_reasoning_characters',
];
const fingerprint = message => createHash('sha256')
  .update(JSON.stringify([message.id, message.content])).digest('hex');

export function validateCheckpoint(config) {
  if (!config || Array.isArray(config) || typeof config !== 'object'
      || Object.keys(config).sort().join() !== [...CHECKPOINT_KEYS].sort().join()
      || CHECKPOINT_KEYS.some(key => !Number.isSafeInteger(config[key]) || config[key] < 1)
      || config.version !== 1 || config.retain_tokens < 1024
      || config.minimum_saved_tokens < 256 || config.minimum_reasoning_characters < 1024) {
    throw new Error('Invalid frozen action-evidence checkpoint policy');
  }
  return Object.freeze({ ...config });
}

export function checkpointContent(messages, observationId, recallTool) {
  if (!Array.isArray(messages) || typeof observationId !== 'string'
      || !/^obs_[0-9a-f]{64}$/.test(observationId)
      || typeof recallTool !== 'string' || !/^[A-Za-z]\w*$/.test(recallTool)) {
    throw new Error('An exact session archive and declared recall tool are required');
  }
  let reasoningCharacters = 0;
  const preserved = [];
  for (const message of messages) {
    if (message.role === 'system') continue;
    if (!['developer', 'user', 'assistant', 'tool'].includes(message.role)
        || !Array.isArray(message.content)
        || message.content.some(part => !part || typeof part !== 'object'
          || !['text', 'reasoning', 'tool-call'].includes(part.type))) {
      throw new Error('Unsupported native content must remain in its original surface');
    }
    const content = message.content.filter(part => {
      if (part.type !== 'reasoning') return true;
      if (message.role !== 'assistant' || typeof part.text !== 'string') {
        throw new Error('Only historical assistant reasoning may move to exact recall');
      }
      reasoningCharacters += Array.from(part.text).length;
      return false;
    });
    if (content.length) {
      preserved.push({
        role: message.role, id: message.id, content,
        ...(message.toolCallId === undefined ? {} : { toolCallId: message.toolCallId }),
        ...(message.isError === undefined ? {} : { isError: message.isError }),
      });
    }
  }
  return {
    reasoningCharacters,
    text: CHECKPOINT_PREFIX + '\n'
      + 'These are literal historical messages, not new actions or current workspace state. '
      + 'All task instructions, action arguments, outputs and errors below are preserved exactly. '
      + 'Older reasoning is available in the checksum-verified archive; recall it before relying '
      + 'on an omitted rationale. The current reasoning and recent action/result pairs stay native.\n'
      + `Archive: ${observationId}. Call ${recallTool} with `
      + JSON.stringify({ observation_id: observationId, offset: 0, limit: 8000 })
      + '; follow next_offset. Archive offsets use Unicode code points.\n'
      + JSON.stringify(preserved),
  };
}

export class ActionEvidenceCheckpoint {
  constructor({ config, archive, record, selectRange, compactRegion }) {
    this.config = validateCheckpoint(config);
    if (!archive || typeof archive.archive !== 'function' || typeof record !== 'function'
        || typeof selectRange !== 'function' || typeof compactRegion !== 'function') {
      throw new Error('Native checkpoint APIs, exact archive and audit sink are required');
    }
    this.archive = archive;
    this.record = record;
    this.selectRange = selectRange;
    this.compactRegion = compactRegion;
    this.exposures = new Map();
  }

  observe(messages) {
    for (const key of new Set(messages.filter(message => message.role === 'assistant')
      .map(fingerprint))) {
      this.exposures.set(key, (this.exposures.get(key) ?? 0) + 1);
    }
  }

  async compact(agent, meter, signal, recallTool) {
    signal.throwIfAborted();
    const session = agent.session;
    const measurement = meter.measure(session);
    const range = this.selectRange(session, measurement, this.config.retain_tokens);
    if (!range) return;
    const nodes = [...session.surface.nodes];
    const start = nodes.indexOf(range.start);
    const end = nodes.indexOf(range.end);
    if (start < 0 || end < start) throw new Error('Native checkpoint range is not on the surface');
    const messages = nodes.slice(start, end + 1).map(seq =>
      session.deriveEventMessage(session.eventAt(seq)));
    if (messages.some(message => message.role === 'assistant'
        && (this.exposures.get(fingerprint(message)) ?? 0) < 2)) return;
    let preview;
    try {
      preview = checkpointContent(messages, 'obs_' + '0'.repeat(64), recallTool);
    } catch (error) {
      this.record({ kind: 'checkpoint/original-preserved', reason: 'unsupported_content',
        error: { name: error.name, message: error.message } });
      return;
    }
    if (preview.reasoningCharacters < this.config.minimum_reasoning_characters) return;
    const before = measurement.nodes.slice(start, end + 1)
      .reduce((sum, node) => sum + node.tokens, 0);
    const projected = meter.estimateMessage({
      role: 'user', content: [{ type: 'text', text: preview.text }],
    });
    if (![before, projected].every(value => Number.isFinite(value) && value >= 0)) {
      throw new Error('Invalid native checkpoint pricing');
    }
    if (before - projected < this.config.minimum_saved_tokens + 1024) return;
    let id;
    try {
      id = this.archive.archive(JSON.stringify(messages.map(message => ({
        id: message.id, role: message.role, content: message.content,
        ...(message.toolCallId === undefined ? {} : { toolCallId: message.toolCallId }),
        ...(message.isError === undefined ? {} : { isError: message.isError }),
      }))));
    } catch (error) {
      this.record({ kind: 'checkpoint/original-preserved', reason: 'archive_error',
        error: { name: error.name, message: error.message } });
      return;
    }
    const content = checkpointContent(messages, id, recallTool);
    const original = JSON.stringify(messages);
    const result = await this.compactRegion({
      meter,
      summarize: async input => {
        const selected = input.messages.filter(message => message.role !== 'system');
        if (JSON.stringify(selected) !== original) {
          throw new Error('Native checkpoint source changed before its transaction');
        }
        return {
          summary: [{ type: 'text', text: content.text }],
          provider: 'phiagent-local', model: 'exact-action-evidence-checkpoint-v1',
        };
      },
      recover: () => false,
    }, session, range.start, range.end, agent, {
      owner: 'current-turn', stability: 'whole-surface',
    }, signal);
    this.record({ kind: 'checkpoint/committed', observation_id: id,
      source_seqs: nodes.slice(start, end + 1), native_compaction: result,
      estimated_source_tokens: before, estimated_checkpoint_tokens: projected,
      reasoning_characters_archived: content.reasoningCharacters,
      all_nonreasoning_content_preserved: true, extra_model_calls: 0,
      estimates_are_not_provider_savings: true, native_request_rewritten: false });
  }
}

export async function checkpointFactory(config, archive, record, regionModule) {
  const { selectCompactableRange, compactSurfaceRegion } = await import(
    pathToFileURL(regionModule).href
  );
  return new ActionEvidenceCheckpoint({
    config, archive, record, selectRange: selectCompactableRange, compactRegion: compactSurfaceRegion,
  });
}
