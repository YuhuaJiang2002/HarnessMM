# DeepSeek Harness desktop smoke test

Use DeepSeek Harness 0.2.1-rc.5 with this checkout's plugin installed in the
desktop profile, an enabled policy, and a configured model. Open a fresh
session in an empty disposable workspace. The test uses three foreground
shell calls and creates one file, `hmm-smoke.txt`.

Send the prompt below. Set `TOOL` to the profile's native shell tool name,
`bash` on macOS/Linux or `pwsh` on Windows.

```text
Start with exactly one TOOL call with arguments
{"description":"Print root marker","command":"echo HMM-ROOT"}.
Immediately before that call, emit the following block in visible assistant
text, outside a code fence, replacing TOOL with the same native tool name.

<CHISEL_PLAN>{"candidates":[{"candidate_id":"A","predecessor_id":"root","tool":"TOOL","arguments":{"description":"Print and append substitution marker","command":"echo HMM-SUBSTITUTED; echo HMM-SUBSTITUTED >> hmm-smoke.txt"},"guard":{"exit_code":0,"stdout_equals":"HMM-ROOT"},"required_receipt_fields":["exit_code","stdout"],"expected_local_cost":{"selector_upper_tokens":0},"dependency_ids":[]}]}</CHISEL_PLAN>

When you see HMM-SUBSTITUTED in a tool result, call TOOL once with
{"description":"Print final marker","command":"echo HMM-DONE"}. Then call
harnessmm_status with {} and finish. Do not declare another bank or manually
run candidate A. If control returns to you before A runs, report that and stop.
```

Approve the harmless foreground calls if the profile prompts. After the turn
settles, find the session's JSONL event log and the plugin's receipts file
(`receiptsFile` in the profile, or `~/.dsh/harnessmm/receipts.jsonl`). If the
session log ends in `.zstd`, decompress a copy with
`zstd -dc /path/to/session.jsonl.zstd > /tmp/hmm-session.jsonl`.

From the plugin directory, run the verifier against the live artifacts.

```sh
node test/live-session.mjs /path/to/session.jsonl /path/to/receipts.jsonl /path/to/workspace/hmm-smoke.txt
```

The verifier checks the bank, the decision admitting `A`, the native call and
its certificate, one `tool/call` for the substitution, one line written by
the shell, and the actual result of `harnessmm_status`. The status tool returns
read-only counters for the calling session from the current plugin process.
They reset when the plugin restarts or evicts that session's state.
