# MiniWoB++ visual-extension comparison

HarnessMM-V extends guarded decision reuse to screenshot-based agents.
The table compares ten harnesses on MiniWoB++. The [HarnessMM paper](https://arxiv.org/abs/2610.32813)
reports the coding experiments.

## Shared protocol

All methods use Qwen3.5-9B on the same 96 held-out MiniWoB++ tasks, with
native screenshots, public screenshot OCR and geometry, mouse/keyboard actions
and native reward scoring. The partition contains 32 training tasks, 32 validation tasks and 96 test tasks;
task families are separated before optimization. Training supplies revisions,
validation selects a candidate, and testing evaluates the frozen configuration.
Each arm uses one execution per test task, a 15-action limit and a 180-second
episode allowance. All compared methods receive token-efficiency prompts and
the objective of reducing complete-task input-plus-output tokens while
preserving performance.

HarnessPro is a configured visual reproduction of the balanced Medium tier,
using tiered evolution and generated-test selection with eight optimization
rounds before protected validation and testing. SoL-Pi uses the released Pi
v0.85.1 runtime with action fusion, observation packing, evidence-preserving
reduction and online context compaction. Its primary actor and reducer use
the same Qwen3.5-9B backbone. Codex Harness retains its official CLI v0.142.5 core with
a configured native visual adapter; DeepSeek Harness retains its native
multimodal policy and tool loop.

## Full test execution comparison

| Method | Successes / tasks | Success (%) | Tokens/task (k) |
| --- | ---: | ---: | ---: |
| baseline harness | 58 / 96 | 60.42 | 7.36 |
| SkillOpt | 62 / 96 | 64.58 | 7.97 |
| AutoSaddler | 58 / 96 | 60.42 | 7.87 |
| SoL-Pi | 58 / 96 | 60.42 | 41.99 |
| Prompt Compaction | 57 / 96 | 59.38 | 7.39 |
| DeepSeek Harness | 59 / 96 | 61.46 | 20.03 |
| Codex Harness | 57 / 96 | 59.38 | 44.77 |
| RHO | 56 / 96 | 58.33 | 9.10 |
| HarnessPro | 59 / 96 | 61.46 | 10.03 |
| HarnessMM-V | 60 / 96 | 62.50 | 5.07 |

## Cost and comparison definitions

Tokens/task (k) equals the complete test execution input-plus-output token
count divided by all 96 test tasks and by 1,000. Every actor, author, selector,
control, reducer, compaction, fallback and unsuccessful execution call within
the test scope is counted. Token saving against a comparator is
`100 * (1 - HarnessMM-V execution tokens / comparator execution tokens)`.
Savings use exact totals before rounding. Offline setup, optimization,
training-checkpoint assessment and validation are accounted for separately.
HarnessPro has a complete test execution total and a partial pretest cost total.

HarnessMM-V used the fewest execution tokens (5.07 Tokens/task (k)). SkillOpt solved
the most tasks (62 of 96, or 64.58%).

## Figure data and provenance

[visual-comparison.json](assets/readme/visual-comparison.json) records the
complete integer execution-token totals, successes, task counts, source
record names and content bindings for all ten methods. The original evidence
uses the internal name VIGRACE for the visual method; its current public name
is HarnessMM-V.

The figures can be regenerated with
[generate.py](assets/readme/generate.py). The source evidence is the visual
study's `visual_rsi_comparisons_provenance.json`, reconciled against each
named native admission record.
