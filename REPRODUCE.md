# Reproduce released results and mechanisms

Run commands from the repository root. Use Python 3.10–3.14 and Node.js 22 or
later. These commands run offline and do not charge a provider account.

| Result or mechanism | Command | Output scope |
| --- | --- | --- |
| MBPP-sanitized DeepSeek Harness comparison, paper §5.3 | `python reproduction/mbpp_sanitized/run.py` | Recomputes archived 257-task aggregates: 90.27 → 91.83% success, 53.98 → 10.61 Tokens/task (k), 80.35% execution saving |
| HumanEval SDK v23 author, paper §5.3 | `node --test tests/humaneval_reference_author.test.mjs` | Frozen source digests and synthetic author/schema checks |
| Codex native callback boundary | `python reproduction/codex_native_bank/run.py` | Synthetic dispatch, receipt propagation and abstention |
| GAIA2 public-receipt binding | `python reproduction/gaia2_continuation/run.py` | Synthetic typed item binding and consumed-action exclusion |
| Visual baseline screenshot boundary | `python reproduction/visual_baseline/run.py` | Synthetic screenshot and parsing-fallback flow |
| Executable graph and callback certificates | `node examples/guarded_reuse.mjs` | Three synthetic action substitutions and return to slow execution |
| Frozen task partitioning | `python examples/partition.py` | Synthetic 20%/20%/60% partition |

Run all four new cases with `python reproduction/run_all.py`. Each compares its
output against its checked-in expected JSON. [Case READMEs](reproduction/README.md)
describe inputs, implementation dependencies and benchmark prerequisites.

Only the MBPP aggregate case recomputes measured benchmark metrics. The other
commands check implementation behavior with synthetic fixtures. The remaining
paper comparisons and ablations require external runtimes and evidence beyond
this release; the [research-runner index](docs/REPRODUCIBILITY.md#research-runner-entry-points)
identifies their entry points. New benchmark execution requires fresh frozen
sources and run identities, native scoring, protected partitions and full cost
accounting.
