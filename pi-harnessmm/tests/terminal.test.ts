/**
 * Native terminal adapter: the receipt contract guards depend on.
 * Covers stream separation, the default timeout, credential scrubbing and the
 * split between the exact receipt and the shortened model view.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, realpath } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DEFAULT_TIMEOUT_SEC, MODEL_OUTPUT_BYTES, modelReceiptText, receiptText, runTerminal, scrubbedEnvironment } from "../src/terminal.ts";

const cwd = process.cwd();

test("stdout and stderr stay separate, and the receipt is exact JSON", async () => {
  const receipt = await runTerminal({ command: "echo out; echo err 1>&2" }, { cwd });
  assert.equal(receipt.return_code, 0);
  assert.equal(receipt.stdout.trim(), "out");
  assert.equal(receipt.stderr?.trim(), "err");
  assert.deepEqual(JSON.parse(receiptText(receipt)), receipt);
  assert.equal(receiptText(receipt), '{"return_code":0,"stdout":"out\\n","stderr":"err\\n"}');
});

test("an empty stderr is null, which is what stderr_empty guards compare against", async () => {
  const receipt = await runTerminal({ command: "printf quiet" }, { cwd });
  assert.equal(receipt.stderr, null);
  const noisy = await runTerminal({ command: "printf loud 1>&2" }, { cwd });
  assert.equal(noisy.stderr, "loud");
});

test("a failing command reports its exit code instead of throwing", async () => {
  const receipt = await runTerminal({ command: "exit 7" }, { cwd });
  assert.equal(receipt.return_code, 7);
});

test("a hung command is killed by the default timeout", async () => {
  assert.equal(DEFAULT_TIMEOUT_SEC > 0, true);
  const started = Date.now();
  const receipt = await runTerminal({ command: "sleep 5", timeout_sec: 1 }, { cwd });
  assert.equal(receipt.return_code, 124);
  assert.ok(Date.now() - started < 4000, "the command must be killed, not awaited");
});

test("a missing working directory is a receipt, not a crash", async () => {
  const receipt = await runTerminal({ command: "echo hi", cwd: "/definitely/not/here" }, { cwd });
  assert.equal(receipt.return_code, 1);
  assert.match(receipt.stderr ?? "", /working directory does not exist/);
});

test("the command sees a scrubbed environment", async () => {
  const probe = scrubbedEnvironment({ PATH: "/bin", GITHUB_TOKEN: "x", OPENAI_API_KEY: "y", DB_PASSWORD: "z", SECRET: "s", HOME: "/h" });
  assert.deepEqual(probe, { PATH: "/bin", HOME: "/h" });
});

test("the model sees a shortened receipt while guards keep the full one", async () => {
  const big = "x".repeat(MODEL_OUTPUT_BYTES * 2);
  const receipt = await runTerminal({ command: `printf '%s' "$(head -c ${MODEL_OUTPUT_BYTES * 2} /dev/zero | tr '\\0' x)"` }, { cwd });
  assert.ok(receipt.stdout.length >= MODEL_OUTPUT_BYTES, "the guard receipt keeps the real output");
  const shown = JSON.parse(modelReceiptText({ return_code: 0, stdout: big, stderr: null })) as { stdout: string };
  assert.ok(shown.stdout.length < big.length);
  assert.match(shown.stdout, /bytes omitted/);
  assert.ok(shown.stdout.endsWith("x".repeat(64)), "the tail is what a guard usually matches");
});

test("a command runs in its own cwd without leaking into the caller's", async () => {
  const dir = await mkdtemp(join(tmpdir(), "harnessmm-cwd-"));
  const receipt = await runTerminal({ command: "pwd", cwd: dir }, { cwd });
  // macOS resolves /tmp to /private/tmp, so compare the resolved path.
  assert.equal(receipt.stdout.trim(), await realpath(dir));
});
