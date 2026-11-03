# Reproduction cases

Start with the case closest to your harness. Run commands from the repository
root with Python 3.10–3.14; the Codex case also requires Node.js 22 or later.
The offline cases use the released implementation and make no provider calls.

| Case | What it reproduces | Command or entry point |
| --- | --- | --- |
| [HumanEval v23](humaneval_v23/) | Frozen DeepSeek SDK authoring adapter and source digests | `node --test tests/humaneval_reference_author.test.mjs` |
| [MBPP-sanitized](mbpp_sanitized/) | Success, execution tokens per task and savings from archived integer aggregates | `python reproduction/mbpp_sanitized/run.py` |
| [Codex native bank](codex_native_bank/) | Root dispatch, one stored native action, selector abstention and return to slow execution | `python reproduction/codex_native_bank/run.py` |
| [GAIA2 continuation](gaia2_continuation/) | Typed arguments bound to public list receipts, consumed-action exclusion and unknown-state deferral | `python reproduction/gaia2_continuation/run.py` |
| [Visual baseline](visual_baseline/) | Screenshot delivery, literal pixel actions and the metered parsing-fallback boundary | `python reproduction/visual_baseline/run.py` |

Each new case contains `run.py`, its input JSON and an `expected.json` checked
by the runner. The MBPP case recomputes an archived result; the other three
replay synthetic mechanism fixtures. Full benchmark execution additionally
requires the native runtime, datasets, model access, protected evaluator and a
new frozen run identity. Case READMEs link the relevant integration or runner.

Run all offline cases and verify their released input/output digests:

```bash
python reproduction/run_all.py
python scripts/verify_release.py
```

The HumanEval modules remain byte-for-byte archived sources. The new cases use
current released modules and preserve that archive. [REPRODUCE.md](../REPRODUCE.md)
maps the published results and mechanisms to these commands.
