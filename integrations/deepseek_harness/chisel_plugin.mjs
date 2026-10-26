import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

export const name = 'phiagent-chisel-frozen';
export const inject = ['systemPrompt'];

export function apply(ctx, config) {
  const bytes = readFileSync(config.path);
  const digest = createHash('sha256').update(bytes).digest('hex');
  if (digest !== config.sha256) {
    throw new Error('Frozen CHISEL SHA256 mismatch');
  }
  const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  if (!text.trim()) throw new Error('Frozen CHISEL artifact must not be empty');
  ctx.systemPrompt.section({
    name: 'chisel:frozen',
    order: 10300,
    text,
    interpolate: false,
  });
}
