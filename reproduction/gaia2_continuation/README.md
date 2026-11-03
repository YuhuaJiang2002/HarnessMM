# GAIA2 public-receipt continuation case

```bash
python reproduction/gaia2_continuation/run.py
```

Requires Python 3.10–3.14, with no third-party dependencies or model access.

`fixture.json` defines two synthetic public items, the native tool argument
types and a receipt-bound continuation bank. The runner parses the bank using
`gaia2_cot_jev`, binds each item ID to the list receipt and excludes the first
already-consumed action. Opaque current feedback defers with `UNKNOWN`.

Expected: candidates `visible-a` and `visible-b`, then only `visible-b` after
the first action is consumed. `expected.json` records these outputs. This case
checks argument construction and candidate eligibility; it dispatches no GAIA2
tools and does not measure benchmark success, fast-execution rate or savings.

Full evaluation needs the native GAIA2 environment and scorer, public tasks,
frozen disjoint partitions and complete metering. Start with
`scripts/run_gaia2_202060.py` and [deployment notes](../../docs/REPRODUCIBILITY.md).
The extended receipt/type checks are in
`tests/test_gaia2_typed_continuation_bank.py`.
