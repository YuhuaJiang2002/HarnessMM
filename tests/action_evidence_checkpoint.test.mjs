import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ActionEvidenceCheckpoint, CHECKPOINT_PREFIX, checkpointContent, validateCheckpoint,
} from '../integrations/deepseek_harness/action_evidence_checkpoint.mjs';

const CONFIG = {
  version: 1, retain_tokens: 2048, minimum_saved_tokens: 256,
  minimum_reasoning_characters: 4096,
};
const ID = 'obs_' + 'a'.repeat(64);

function messages() {
  return [
    { id: 'original-task', role: 'user', content: [{ type: 'text', text: 'Keep exact public paths.' }] },
    { id: 'old-assistant', role: 'assistant', source: {
      kind: 'model', provider: 'native', model: 'native', replayState: { opaque: 'native replay' },
    }, content: [
      { type: 'reasoning', text: 'old public reasoning\n'.repeat(500) },
      { type: 'text', text: 'Observed result.' },
      { type: 'tool-call', id: 'call-1', name: 'terminal',
        arguments: JSON.stringify({ command: 'synthetic' }) },
    ] },
    { id: 'native-result', role: 'tool', toolCallId: 'call-1',
      content: [{ type: 'text', text: '{"return_code":0,"stdout":"observed evidence"}' }] },
    { id: 'failed-result', role: 'tool', toolCallId: 'call-2', isError: true,
      content: [{ type: 'text', text: 'ERROR: exact failure evidence' }] },
  ];
}

function fixture(changes = {}) {
  const history = messages();
  const records = [];
  const archived = [];
  const committed = [];
  const archive = { archive(value) { archived.push(value); return ID; } };
  const session = {
    surface: { nodes: [1, 2, 3, 4, 5] },
    eventAt(seq) { return { seq }; },
    deriveEventMessage({ seq }) {
      return seq <= 4 ? history[seq - 1] : {
        id: 'recent', role: 'assistant', content: [{ type: 'reasoning', text: 'active reasoning' }],
      };
    },
  };
  const meter = {
    measure() { return { nodes: [1, 2, 3, 4, 5].map(seq => ({ seq, tokens: 5000 })) }; },
    estimateMessage() { return 500; },
  };
  const policy = new ActionEvidenceCheckpoint({
    config: CONFIG, archive, record: value => records.push(value),
    selectRange: () => ({ start: 1, end: 4 }),
    compactRegion: async (dependencies, selectedSession, start, end, agent, options) => {
      assert.equal(selectedSession, session);
      assert.equal(agent.session, session);
      assert.deepEqual(options, { owner: 'current-turn', stability: 'whole-surface' });
      assert.equal(dependencies.recover(new Error('native failure')), false);
      const summary = await dependencies.summarize({ messages: [
        { role: 'system', content: [{ type: 'text', text: 'native system remains unchanged' }] },
        ...history,
      ] });
      assert.equal(summary.llmStreamCall, undefined);
      assert.equal(summary.usage, undefined);
      committed.push({ start, end, summary });
      return { shadowedSeqs: [1, 2, 3, 4] };
    },
    ...changes,
  });
  return { policy, history, records, archived, committed, meter, agent: { session } };
}

test('checkpoint policy keeps complete frozen keys and valid retention', () => {
  assert.deepEqual(validateCheckpoint(CONFIG), CONFIG);
  for (const config of [{}, { ...CONFIG, extra: true }, { ...CONFIG, retain_tokens: 2 },
    { ...CONFIG, minimum_saved_tokens: true }, { ...CONFIG, version: 2 }]) {
    assert.throws(() => validateCheckpoint(config), /Invalid frozen/);
  }
});

test('exact action/evidence card omits only reasoning and preserves every other content block', () => {
  const original = messages();
  const before = structuredClone(original);
  const result = checkpointContent(original, ID, 'CHISELMemory_recall');
  assert(result.text.startsWith(CHECKPOINT_PREFIX));
  assert.equal(result.reasoningCharacters, original[1].content[0].text.length);
  const preserved = JSON.parse(result.text.split('\n').at(-1));
  assert.deepEqual(preserved[0].content, original[0].content);
  assert.deepEqual(preserved[1].content, original[1].content.slice(1));
  assert.deepEqual(preserved[2].content, original[2].content);
  assert.equal(preserved[2].toolCallId, 'call-1');
  assert.equal(preserved[3].isError, true);
  assert.deepEqual(preserved[3].content, original[3].content);
  assert.equal(preserved[1].source, undefined);
  assert(result.text.includes(ID));
  assert.deepEqual(original, before);
});

test('unsupported rich evidence is protected rather than reduced', () => {
  const original = messages();
  original[2].content.push({ type: 'image', data: 'native image' });
  assert.throws(() => checkpointContent(original, ID, 'CHISELMemory_recall'), /Unsupported/);
});

test('native transaction follows two paid exposures, keeps tail and archives exact original content', async () => {
  const f = fixture();
  const signal = new AbortController().signal;
  await f.policy.compact(f.agent, f.meter, signal, 'CHISELMemory_recall');
  assert.equal(f.committed.length, 0);
  f.policy.observe(f.history);
  await f.policy.compact(f.agent, f.meter, signal, 'CHISELMemory_recall');
  assert.equal(f.committed.length, 0);
  f.policy.observe([...f.history, f.history[1]]);
  await f.policy.compact(f.agent, f.meter, signal, 'CHISELMemory_recall');
  assert.equal(f.committed.length, 1);
  const archive = JSON.parse(f.archived[0]);
  assert.deepEqual(archive[1].content, f.history[1].content);
  assert.equal(archive[1].source, undefined);
  assert.equal(f.committed[0].end, 4);
  assert.equal(f.records.at(-1).extra_model_calls, 0);
  assert.equal(f.records.at(-1).native_request_rewritten, false);
  assert.equal(f.history[1].source.replayState.opaque, 'native replay');
});

test('unsupported content and archive failure preserve originals with explicit diagnostics', async () => {
  const f = fixture();
  f.policy.observe(f.history);
  f.policy.observe(f.history);
  f.policy.archive.archive = () => { throw new Error('synthetic disk failure'); };
  await f.policy.compact(f.agent, f.meter, new AbortController().signal, 'CHISELMemory_recall');
  assert.equal(f.committed.length, 0);
  assert.equal(f.records.at(-1).reason, 'archive_error');
});

test('insufficient savings never creates a speculative checkpoint', async () => {
  const f = fixture();
  f.policy.observe(f.history);
  f.policy.observe(f.history);
  f.meter.estimateMessage = () => 20000;
  await f.policy.compact(f.agent, f.meter, new AbortController().signal, 'CHISELMemory_recall');
  assert.equal(f.archived.length, 0);
  assert.equal(f.committed.length, 0);
});

test('changed native source and cancellation fail explicitly', async () => {
  const f = fixture({
    compactRegion: async dependencies => dependencies.summarize({ messages: [] }),
  });
  f.policy.observe(f.history);
  f.policy.observe(f.history);
  await assert.rejects(f.policy.compact(
    f.agent, f.meter, new AbortController().signal, 'CHISELMemory_recall',
  ), /source changed/);
  const abort = new AbortController();
  abort.abort();
  await assert.rejects(f.policy.compact(
    f.agent, f.meter, abort.signal, 'CHISELMemory_recall',
  ), /abort/i);
});
