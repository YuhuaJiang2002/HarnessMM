# Reproduction cases

Start with the case closest to your harness. Run commands from the repository
root with Python 3.10–3.14 and Node.js 22 or later. The offline cases use the
released implementation and make no provider calls.

| Case | What it reproduces | Command or entry point |
| --- | --- | --- |
| [HumanEval v23](humaneval_v23/) | Frozen DeepSeek SDK authoring adapter and source digests | `node --test tests/humaneval_reference_author.test.mjs` |
| [MBPP-sanitized](mbpp_sanitized/) | Success, execution tokens per task and savings from archived integer aggregates | `python reproduction/mbpp_sanitized/run.py` |
| [Codex native bank](codex_native_bank/) | Root dispatch, one stored native action, selector abstention and return to slow execution | `python reproduction/codex_native_bank/run.py` |
| [GAIA2 continuation](gaia2_continuation/) | Typed arguments bound to public list receipts, consumed-action exclusion and unknown-state deferral | `python reproduction/gaia2_continuation/run.py` |
| [Visual baseline](visual_baseline/) | Screenshot delivery, literal pixel actions and the metered parsing-fallback boundary | `python reproduction/visual_baseline/run.py` |
| [Desktop plugin controller](../dsh-plugin/harnessmm-desktop/) | Synthetic bank admission, receipt guards and dispatch checks | `node --test dsh-plugin/harnessmm-desktop/test/controller.test.mjs` |

Each Python case includes its input JSON and an `expected.json` checked by the
runner. The MBPP case recomputes an archived result. The other Python cases
replay synthetic mechanism fixtures. Full benchmark execution additionally
requires the native runtime, datasets, model access, protected evaluator and a
new frozen run identity. Case READMEs link the relevant integration or runner.

Run the four JSON cases and the desktop controller suite, then verify the
released source digests.

```bash
python reproduction/run_all.py
python scripts/verify_release.py
```

Run the [HumanEval author tests](humaneval_v23/) separately to check the
archived adapter. The [desktop plugin test instructions](../dsh-plugin/harnessmm-desktop/README.md#tests)
cover the additional simulated-host tests. [REPRODUCE.md](../REPRODUCE.md)
maps the published results and mechanisms to these commands.
