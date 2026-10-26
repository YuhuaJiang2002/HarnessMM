import { createHash, randomUUID } from 'node:crypto';
import { appendFileSync, closeSync, constants, fstatSync, openSync, readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { request as nativeHttpRequest } from 'node:http';
import { pathToFileURL } from 'node:url';
import { NativeCotJev, planInstruction, structuredAuthorTool } from './cot_jev.mjs';
import { JevV2 } from './jev_v2.mjs';
import { PersistentBank, ExecutableSubdag, controllerProjection } from './sar_v2.mjs';
import { SubstitutionAwareRouter } from './sar_admission.mjs';

// Native Harbor and the SDK cancellation signal own deadlines. Node fetch's
// independent HTTP header/body deadlines can otherwise abandon a live command.
export function nativeLoopbackPost(url, { payload, token, signal }) {
  const endpoint = new URL(url);
  if (endpoint.protocol !== 'http:' || endpoint.hostname !== '127.0.0.1'
      || !endpoint.port || endpoint.username || endpoint.password || endpoint.hash) {
    return Promise.reject(new Error('Native bridge requires literal loopback HTTP'));
  }
  const body = JSON.stringify(payload);
  return new Promise((resolve, reject) => {
    const request = nativeHttpRequest(endpoint, {
      method: 'POST', agent: false, signal,
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}`,
        'Content-Length': Buffer.byteLength(body) },
    }, response => {
      if (response.statusCode < 200 || response.statusCode >= 300) {
        response.resume();
        reject(new Error(`Bridge ${endpoint.pathname} returned HTTP ${response.statusCode}`));
        return;
      }
      const chunks = [];
      response.on('data', chunk => chunks.push(chunk));
      response.once('aborted', () => reject(new Error('Native bridge response aborted')));
      response.once('error', reject);
      response.once('end', () => {
        try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
        catch (error) { reject(error); }
      });
    });
    request.setTimeout(0);
    request.once('error', reject);
    request.end(body);
  });
}

export const name = 'phiagent-benchmark-bridge';
export const inject = ['tools', 'systemPrompt', 'tokenMeter'];

function secretFromFile(path) {
  const fd = openSync(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try {
    const info = fstatSync(fd);
    if (!info.isFile() || info.uid !== process.getuid() || (info.mode & 0o077)) {
      throw new Error('Bridge credential must be an owner-only regular file');
    }
    const value = readFileSync(fd, 'utf8').trim();
    if (!value || value.length > 16384 || /[\r\n]/.test(value)) {
      throw new Error('Bridge credential must be one nonempty line');
    }
    return value;
  } finally {
    closeSync(fd);
  }
}

export function canonicalJSON(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJSON).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key =>
      `${JSON.stringify(key)}:${canonicalJSON(value[key])}`).join(',')}}`;
  }
  const encoded = JSON.stringify(value);
  if (encoded === undefined) throw new Error('Bridge returned a non-JSON observation');
  return encoded;
}

export function nativeObservationText(value, encoding = 'canonical_json') {
  if (encoding === 'canonical_json') return canonicalJSON(value);
  if (encoding !== 'native_text' || typeof value !== 'string') {
    throw new Error('Native text observations require the declared encoding and original string');
  }
  return value;
}

export function nativeToolGuard(names) {
  const allowed = new Set(names);
  return execution => allowed.has(execution.name)
    ? undefined : 'Only the explicit native benchmark bridge catalog may execute.';
}

export function notificationContent(message) {
  if (typeof message === 'string') return [{ type: 'text', text: message }];
  if (!Array.isArray(message) || !message.length) {
    throw new Error('Bridge notification must be text or nonempty official inline content parts');
  }
  for (const part of message) {
    if (!part || typeof part !== 'object' || Array.isArray(part)) {
      throw new Error('Bridge notification content part must be an object');
    }
    const fields = Object.keys(part).sort().join(',');
    if (part.type === 'text') {
      if (fields !== 'text,type' || typeof part.text !== 'string') {
        throw new Error('Invalid bridge notification text part');
      }
    } else if (part.type === 'image') {
      if (fields !== 'data,mimeType,type'
          || !['image/jpeg', 'image/png', 'image/gif', 'image/webp'].includes(part.mimeType)
          || typeof part.data !== 'string' || !part.data
          || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(part.data)) {
        throw new Error('Invalid official inline image part in bridge notification');
      }
    } else {
      throw new Error(`Unsupported bridge notification content type: ${String(part.type)}`);
    }
  }
  return message;
}

function withoutSecrets(value, secrets) {
  if (typeof value === 'string') {
    for (const secret of secrets) {
      if (secret) value = value.split(secret).join('[REDACTED_CREDENTIAL]');
    }
    return value;
  }
  if (Array.isArray(value)) return value.map(item => withoutSecrets(item, secrets));
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [
      key, /^(authorization|x-api-key|api_key|apiKey)$/i.test(key)
        ? '[REDACTED_CREDENTIAL]' : withoutSecrets(item, secrets),
    ]));
  }
  return value;
}

export function apply(ctx, config) {
  const job = JSON.parse(readFileSync(config.jobFile, 'utf8'));
  const observationEncoding = job.native_observation_encoding ?? 'canonical_json';
  if (!['canonical_json', 'native_text'].includes(observationEncoding)) {
    throw new Error('Unsupported native observation encoding');
  }
  if (config.benchmarkContext) {
    const bytes = readFileSync(config.benchmarkContext.path);
    if (createHash('sha256').update(bytes).digest('hex') !== config.benchmarkContext.sha256) {
      throw new Error('Frozen public benchmark context SHA256 mismatch');
    }
    const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    if (text.trim()) {
      ctx.systemPrompt.section({
        name: 'benchmark:context', order: 10200, text, interpolate: false,
      });
    }
  }
  const bridge = job.bridge;
  const token = bridge ? secretFromFile(bridge.secret_file) : '';
  const secrets = [token, process.env.DEEPSEEK_API_KEY ?? ''];
  let sequence = 0;
  const record = value => {
    appendFileSync(config.receiptsFile, `${JSON.stringify(withoutSecrets({
      seq: sequence++, at: new Date().toISOString(), ...value,
    }, secrets))}\n`, { mode: 0o600 });
  };
  const cotJev = job.cot_jev ? new (job.cot_jev.version >= 8 ? SubstitutionAwareRouter : job.cot_jev.version >= 6 ? ExecutableSubdag : job.cot_jev.version === 5 ? PersistentBank : job.jev_v2 ? JevV2 : NativeCotJev)({
    policy: job.cot_jev, model: job.model,
    ...(job.jev_v2 ? { v2:job.jev_v2 } : {}),
    apiKey: process.env.DEEPSEEK_API_KEY ?? '', record,
    episodeId: job.session_id,
  }) : null;
  if (cotJev) {
    ctx.systemPrompt.section({
      name: 'chisel:cot-jev', order: 10400,
      text: planInstruction(job.cot_jev), interpolate: false,
    });
    record({ kind: 'jev/identity', policy: job.cot_jev,
      implementation: job.cot_jev.version === 3
        ? 'same-backbone one-token selection of CoT-authored native app actions'
        : 'same-backbone one-token selection of CoT-authored native terminal actions',
      calibrated_probability: false, zero_decoding_readout: false,
      original_native_tools_and_observations_unchanged: true });
  }
  let origin;
  if (bridge) {
    origin = new URL(bridge.url);
    if (origin.protocol !== 'http:' || origin.hostname !== '127.0.0.1'
        || !origin.port || origin.username || origin.password || origin.search
        || origin.hash || origin.pathname !== '/') {
      throw new Error('Bridge must be a literal loopback HTTP origin');
    }
  }
  const post = async (path, payload, signal) => {
    if (!origin) throw new Error('Bridge endpoint is not configured');
    if (config.bridgeTransport === 'native-http') {
      return nativeLoopbackPost(new URL(path, origin), { payload, token, signal });
    }
    const response = await fetch(new URL(path, origin), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(payload),
      redirect: 'error',
      signal,
    });
    if (!response.ok) throw new Error(`Bridge ${path} returned HTTP ${response.status}`);
    return response.json();
  };

  const names = job.tools.map(tool => tool.name);
  if (job.observation_policy) names.push('CHISELMemory_recall');
  if (new Set(names).size !== names.length) throw new Error('Duplicate bridge tool names');
  ctx.tools.guard(nativeToolGuard(names));
  for (const tool of job.tools) {
    ctx.tools.register({
      name: tool.name,
      description: tool.description,
      parameters: job.cot_jev?.version >= 10 ? structuredAuthorTool(tool).parameters : tool.parameters,
      output: {
        schema: { type: 'string' },
        render: (_args, value) => [{ type: 'text', text: value }],
      },
      async execute(args, exec) {
        const payload = {
          session_id: exec.agent.session.id,
          call_id: exec.callId,
          name: tool.name,
          arguments: args,
        };
        record({ kind: 'bridge/tool/start', ...payload });
        try {
          const result = await post('/tool', payload, exec.signal);
          if (!Object.hasOwn(result, 'observation')
              || typeof result.conclude_turn !== 'boolean') {
            throw new Error('Bridge response requires observation and conclude_turn');
          }
          const observation = nativeObservationText(result.observation, observationEncoding);
          record({ kind: 'bridge/tool/end', ...payload, result });
          if (job.jev_v2 || job.cot_jev?.version >= 5) cotJev?.observeTool({ ...payload, result });
          if (result.conclude_turn) exec.concludeTurn();
          return observation;
        } catch (error) {
          record({ kind: 'bridge/tool/error', ...payload,
            error: { name: error.name, message: error.message } });
          throw error;
        }
      },
    });
  }

  ctx.on('agent/created', ({ agent }) => {
    agent.ctx.tools.restrict({ allow: names });
    record({ kind: 'agent/restricted', session_id: agent.session.id, allowed_tools: names });
  });

  if (bridge?.notifications) {
    ctx.on('agent/pre-step', async (payload, next) => {
      const decision = await next();
      if (decision.kind === 'reject') return decision;
      const result = await post('/notifications', {
        session_id: payload.agent.session.id,
      }, payload.signal);
      if (!Array.isArray(result.messages)
          || (result.stop_requested !== undefined && typeof result.stop_requested !== 'boolean')) {
        throw new Error('Bridge notifications require messages and a boolean stop_requested');
      }
      const contents = result.messages.map(notificationContent);
      record({ kind: 'bridge/notifications', session_id: payload.agent.session.id,
        turn: payload.turn, step: payload.step, messages: result.messages,
        stop_requested: result.stop_requested ?? false });
      if (result.stop_requested) {
        record({ kind: 'bridge/environment-stop', session_id: payload.agent.session.id,
          turn: payload.turn, step: payload.step });
        return { kind: 'reject' };
      }
      if (contents.length) {
        const { createUserMessage } = await import(pathToFileURL(config.messageModule).href);
        const messages = [];
        for (const content of contents) {
          messages.push(createUserMessage({
            content,
            source: { kind: 'phiagent-benchmark-bridge', form: 'notice',
              summary: 'Benchmark environment notification' },
          }));
        }
        return { ...decision, messages: [...decision.messages, ...messages] };
      }
      return decision;
    });
  }

  if (job.cot_jev?.version === 5) {
    ctx.inject(['sessions'], sessionCtx => {
      sessionCtx.sessions.registerMessageProjection(controllerProjection);
    });
    ctx.on('agent/pre-step', async (payload, next) => {
      const { freezeMessage } = await import(pathToFileURL(config.messageModule).href);
      cotJev.compactSession(payload.agent.session, freezeMessage, ctx.tokenMeter);
      return next();
    });
  }

  if (job.jev_v2 && ['A5','A6'].includes(job.jev_v2.arm)) {
    ctx.on('agent/pre-step', async (payload, next) => {
      const { freezeMessage } = await import(pathToFileURL(config.messageModule).href);
      cotJev.compactSession(payload.agent.session, freezeMessage, ctx.tokenMeter);
      const reason = cotJev.guardReason();
      if (reason || cotJev.stopped) {
        if (!cotJev.stopped) record({ kind:'jev-v2/guard-termination', reason,
          counts:{...cotJev.counts}, native_success_claimed:false,
          finalization:'No task-semantic basis for a fabricated completion' });
        cotJev.stopped=true;
        return { kind:'reject' };
      }
      return next();
    });
  }

  ctx.on('llm/stream', (options, next) => (async function* () {
    const attempt_id = randomUUID();
    const started = performance.now();
    const session_id = options.sessionId ?? job.session_id;
    const payload = { session_id, attempt_id };
    record({ kind: 'llm/start', ...payload,
      provider: options.provider, model: options.model, purpose: options.purpose ?? 'conversation',
      tool_names: options.tools?.map(tool => tool.name),
      request: { messages: options.messages, system: options.system,
        systemPrompt: options.systemPrompt, maxTokens: options.maxTokens,
        ...(job.cot_jev?.version >= 2 ? { tools: options.tools } : {}) },
    });
    let clockStarted = false;
    let completed = false;
    let errorRecord;
    try {
      if (bridge?.generation_timing) {
        await post('/generation/start', payload, options.signal);
        clockStarted = true;
      }
      for await (const chunk of (cotJev ? cotJev.stream(options, next, attempt_id) : next())) {
        record({ kind: 'llm/chunk', ...payload, chunk });
        yield chunk;
      }
      completed = true;
    } catch (error) {
      errorRecord = { name: error.name, message: error.message };
      throw error;
    } finally {
      const elapsed_seconds = (performance.now() - started) / 1000;
      try {
        if (clockStarted) {
          // Cleanup must still reach the native clock when the model signal aborts.
          await post('/generation/end', { ...payload, elapsed_seconds });
        }
      } finally {
        record({ kind: 'llm/end', ...payload, elapsed_seconds, completed, error: errorRecord });
      }
    }
  })());
}
