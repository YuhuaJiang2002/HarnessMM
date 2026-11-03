# Codex native action-bank case

```bash
python reproduction/codex_native_bank/run.py
```

Requires Python 3.10–3.14 and Node.js 22 or later on `PATH`. No package install
or provider credentials are needed when running from this checkout.

`fixture.json` supplies a synthetic Codex response with a literal root and two
successors. The released `NativeBankBoundary` validates that response through
the Node.js bank contract. An in-memory dispatcher returns public `READY`
receipts. The fixture selector chooses A, then abstains at B. The next tool
call returns to slow execution. Private reasoning is excluded from selector
packets.

Expected: three native dispatcher calls, two selector calls, one reused action
and an abstention. `expected.json` fixes the exact actions and callback order.
Selector usage in this fixture is synthetic test data; it is not a token-saving
measurement. No filesystem tools or native Codex process are launched.

For deployment, replace the callbacks with the original native dispatcher and
metered selector. Retain the slow actor, tools, task constraints and native
scorer. See [Codex integration](../../integrations/codex_harness/README.md) and
`scripts/run_codex_native_app_bank_campaign.py`. The complete contract checks
are in `tests/test_codex_native_app_bank.py`.
