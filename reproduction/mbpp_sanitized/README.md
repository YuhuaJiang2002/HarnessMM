# MBPP-sanitized aggregate reproduction

```bash
python reproduction/mbpp_sanitized/run.py
```

Requires Python 3.10–3.14 and only the standard library. This command recomputes
published metrics from archived integer counters. It makes no model calls and
does not rerun task generation or the native verifier.

`aggregate.json` retains the complete three-arm aggregate for the selected
Qwen3.5-4B / DeepSeek SDK v66 comparison, including the native arm and CoT-only
control. The source filename, original digest and deployment-path removals are
recorded in `release_provenance`; audit, source, partition and selection digests
remain attached. Every original numeric field and gate flag is retained.
`chisel` is the historical internal name for HarnessMM in this record.

The runner verifies that all 257 test executions have settled in every arm,
that usage is known, and that input plus output equals the recorded total.
It uses every execution in the denominator, including unsuccessful ones:

```text
Success (%) = 100 × successes / 257
Tokens/task (k) = (input_tokens + output_tokens) / 257 / 1000
Token saving (%) = 100 × (1 - arm_tokens / native_tokens)
```

| Arm | Success (%) | Tokens/task (k) | Saving vs native (%) |
| --- | ---: | ---: | ---: |
| Native | 90.27 | 53.98 | 0.00 |
| CoT-only | 93.39 | 15.04 | 72.14 |
| HarnessMM | 91.83 | 10.61 | 80.35 |

`expected.json` fixes these outputs. Execution costs include authoring,
selection, finalization, retries, fallback and unsuccessful executions.
Offline optimization and validation are separate. The preserved campaign
flags describe the original additional acceptance goals; this recomputation
makes no all-in RSI or mechanism-isolation claim.

The public release supplies aggregate evidence, not per-request logs, task
answers or verifier tests. A full rerun needs the upstream MBPP-sanitized data,
isolated evaluator, pinned SDK and actor, frozen 85/85/257 custom partition,
and complete provider metering under a new run identity. See
[deployment](../../docs/REPRODUCIBILITY.md) and the
[SDK configuration notes](../../docs/DEEPSEEK_REPRODUCTION.md) for the host and
actor distinction. The HumanEval v23 author is a separate archived adapter.
