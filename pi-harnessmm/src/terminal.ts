/**
 * Native terminal adapter: the single tool the authored bank may call.
 *
 * The paper's guards read `{return_code, stdout, stderr}`, so this adapter keeps
 * stdout and stderr in separate pipes instead of merging them, and always returns
 * that receipt as JSON text. Arguments and receipt shape match the frozen
 * `terminal` contract (`command`, `cwd`, `timeout_sec`).
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";

/** Maximum bytes of each stream kept in the receipt; the rest is dropped with a marker. */
const MAX_STREAM_BYTES = 96 * 1024;
/** Bytes of output the model sees; the full receipt still goes to the guards. */
export const MODEL_OUTPUT_BYTES = 8 * 1024;
/** Default wall clock before a command is killed, so a hung command cannot hold the turn. */
export const DEFAULT_TIMEOUT_SEC = 120;

/**
 * Environment variables a bank command must not see. The reference setup strips
 * names containing KEY, PASSWORD, SECRET or TOKEN for the same reason: a command
 * can otherwise echo a credential into the transcript.
 */
const SECRET_NAME = /(?:key|password|secret|token|credential|passwd)/i;

/** What a native terminal call produced. This is the only evidence guards read. */
export interface TerminalReceipt {
  return_code: number;
  stdout: string;
  stderr: string | null;
}

export interface TerminalRequest {
  command: string;
  cwd?: string;
  timeout_sec?: number;
}

/** Run one command and return its closed receipt. Never throws for a failing command. */
export async function runTerminal(request: TerminalRequest, options: { cwd: string; signal?: AbortSignal; shell?: string }): Promise<TerminalReceipt> {
  const cwd = request.cwd ?? options.cwd;
  if (!existsSync(cwd)) {
    return { return_code: 1, stdout: "", stderr: `working directory does not exist: ${cwd}` };
  }
  const shell = options.shell ?? "/bin/bash";
  const child = spawn(shell, ["-c", request.command], {
    cwd,
    env: scrubbedEnvironment(),
    stdio: ["ignore", "pipe", "pipe"],
    detached: process.platform !== "win32",
  });

  const out: Buffer[] = [];
  const err: Buffer[] = [];
  let outBytes = 0;
  let errBytes = 0;
  let truncated = false;
  const collect = (chunks: Buffer[], chunk: Buffer, used: number) => {
    if (used >= MAX_STREAM_BYTES) {
      truncated = true;
      return used;
    }
    const room = MAX_STREAM_BYTES - used;
    chunks.push(chunk.subarray(0, room));
    if (chunk.length > room) truncated = true;
    return used + Math.min(chunk.length, room);
  };
  child.stdout?.on("data", (chunk: Buffer) => {
    outBytes = collect(out, chunk, outBytes);
  });
  child.stderr?.on("data", (chunk: Buffer) => {
    errBytes = collect(err, chunk, errBytes);
  });

  let timedOut = false;
  const timeoutMs = (request.timeout_sec ?? DEFAULT_TIMEOUT_SEC) * 1000;
  const timer = timeoutMs > 0
    ? setTimeout(() => {
        timedOut = true;
        if (child.pid) killTree(child.pid);
      }, timeoutMs)
    : null;
  const onAbort = () => {
    if (child.pid) killTree(child.pid);
  };
  options.signal?.addEventListener("abort", onAbort, { once: true });

  const exitCode = await new Promise<number>(resolve => {
    child.on("error", error => {
      err.push(Buffer.from(String(error)));
      resolve(127);
    });
    child.on("close", code => resolve(code ?? (timedOut ? 124 : 1)));
  });
  if (timer) clearTimeout(timer);
  options.signal?.removeEventListener("abort", onAbort);

  const stdout = tail(Buffer.concat(out).toString("utf8")) + (truncated && outBytes >= MAX_STREAM_BYTES ? "\n[truncated]" : "");
  const stderrText = Buffer.concat(err).toString("utf8");
  return {
    return_code: timedOut ? 124 : exitCode,
    stdout,
    stderr: stderrText.length > 0 ? tail(stderrText) : null,
  };
}

/** Keep the tail so a long log still ends with the line the guard greps for. */
function tail(text: string): string {
  return text.length > MAX_STREAM_BYTES ? text.slice(-MAX_STREAM_BYTES) : text;
}

function killTree(pid: number): void {
  try {
    process.kill(-pid, "SIGKILL");
  } catch {
    try {
      process.kill(pid, "SIGKILL");
    } catch {
      /* already gone */
    }
  }
}

/** The exact receipt, unchanged: guards evaluate this, never a preview. */
export function receiptText(receipt: TerminalReceipt): string {
  return JSON.stringify(receipt);
}

/** The same receipt, shortened for the model so one dump cannot flood the context. */
export function modelReceiptText(receipt: TerminalReceipt): string {
  const budget = MODEL_OUTPUT_BYTES;
  const clip = (text: string): string =>
    text.length <= budget ? text : `[${text.length - budget} bytes omitted]\n${text.slice(-budget)}`;
  const shortened = { ...receipt, stdout: clip(receipt.stdout), stderr: receipt.stderr === null ? null : clip(receipt.stderr) };
  return JSON.stringify(shortened);
}

/** Copy of the environment without anything that looks like a credential. */
export function scrubbedEnvironment(source: NodeJS.ProcessEnv = process.env): Record<string, string> {
  const clean: Record<string, string> = {};
  for (const [name, value] of Object.entries(source)) {
    if (value === undefined || SECRET_NAME.test(name)) continue;
    clean[name] = value;
  }
  return clean;
}
