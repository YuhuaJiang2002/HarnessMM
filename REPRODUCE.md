# Reproduce released results and mechanisms

Run commands from the repository root with Python 3.10–3.14 and Node.js 22 or
later.

## README benchmark results

**Local** means the repository includes the counts needed to recompute the
reported metrics offline. **External** means the metrics require the original
evaluation and metering evidence, which is outside this release.

| README comparison | Reported execution results | Scope | Check or research entry point |
| --- | --- | --- | --- |
| HumanEval, Codex (98 tasks) | 90.82% → 90.82% success, 66.72 → 47.02 Tokens/task (k), 29.54% saving | External | [Codex research runner](scripts/run_codex_chisel_terminal.py) |
| HumanEval, DeepSeek Harness (98 tasks) | 71.43% → 85.71% success, 96.28 → 11.62 Tokens/task (k), 87.93% saving | External | [HumanEval reference setup](docs/DEEPSEEK_REPRODUCTION.md) and [DeepSeek SDK runner](integrations/deepseek_harness/runner.py) |
| MBPP-sanitized, DeepSeek Harness (257 tasks) | 90.27% → 91.83% success, 53.98 → 10.61 Tokens/task (k), 80.35% saving | Local | `python reproduction/mbpp_sanitized/run.py` recomputes the archived aggregates |

The HumanEval links document the configuration or point to research code.
A fresh evaluation requires the native runtime, model access, frozen task
partition, scorer and complete cost accounting. The [research-runner
index](docs/REPRODUCIBILITY.md#research-runner-entry-points) covers the other
paper studies.

## Local mechanism checks

These checks use synthetic inputs and make no provider calls.

| Mechanism | Command | What it checks |
| --- | --- | --- |
| HumanEval SDK v23 author | `node --test tests/humaneval_reference_author.test.mjs` | Frozen source digests and synthetic authoring and schema checks |
| Codex native callback boundary | `python reproduction/codex_native_bank/run.py` | Synthetic dispatch, receipt propagation and abstention |
| GAIA2 public-receipt binding | `python reproduction/gaia2_continuation/run.py` | Synthetic typed item binding and consumed-action exclusion |
| Visual baseline screenshot boundary | `python reproduction/visual_baseline/run.py` | Synthetic screenshot and parsing-fallback flow |
| Executable graph and callback certificates | `node examples/guarded_reuse.mjs` | Three synthetic action substitutions and return to slow execution |
| Frozen task partitioning | `python examples/partition.py` | Synthetic 20%/20%/60% partition |
| DeepSeek Harness desktop controller | `node --test dsh-plugin/harnessmm-desktop/test/controller.test.mjs` | Synthetic bank, guard, receipt and dispatch checks with no extra dependencies |
| DeepSeek Harness desktop plugin hooks | `node --test dsh-plugin/harnessmm-desktop/test/*.test.mjs` | Controller tests plus stream interception, native dispatch checks and certificate generation in a simulated host. [Test dependency setup](dsh-plugin/harnessmm-desktop/README.md#tests). |

Run `python reproduction/run_all.py` for the four [JSON reproduction
cases](reproduction/README.md) and the dependency-free desktop controller suite.
For a real desktop session, use the separate [live verification
checklist](dsh-plugin/harnessmm-desktop/README.md#live-verification-checklist).
