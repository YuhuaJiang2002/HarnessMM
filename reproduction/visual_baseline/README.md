# Screenshot baseline case

```bash
python reproduction/visual_baseline/run.py
```

Requires Python 3.10–3.14, with no third-party dependencies or provider access.

The runner supplies a complete 1×1 PNG to the released `VisualBaseline`.
`fixture.json` returns one malformed actor response followed by two valid
normalized center clicks. The adapter makes its parsing-fallback call and
converts each valid click to the literal pixel coordinate `(0, 0)`. A second
screenshot boundary makes a fresh actor call.

Expected: two clicks, three actor callback invocations including one parsing
fallback, and zero substitutions. `expected.json` fixes the callback purposes
and IDs. All responses are synthetic; no browser, vision model, benchmark
scorer or measured token ledger runs here.

For a real visual episode, the host supplies current PNG bytes, original image
dimensions, metered actor access, native execution, limits and scorer. Charge
the fallback as well as the initial call. See the
[baseline integration](../../integrations/baseline_harness/README.md) and the
[visual experiment protocol](../../docs/VISUAL_COMPARISON.md).
