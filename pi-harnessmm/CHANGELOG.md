# Changelog

## [0.1.0] - 2026-11-02

Initial port of HarnessMM (arXiv 2610.32813) to pi.

- `terminal` tool returning native `{return_code, stdout, stderr}` receipts with separated streams.
- `<CHISEL_PLAN>` Executable Decision Graph contract, byte-identical authoring instruction for policy versions 1-11.
- Guarded decision reuse: `localGuard` and `receiptGuard`, deterministic admission with the 1024-token net-saving
  threshold, priors-based admission for versions 8 and 10 that fails closed without frozen TRAIN medians.
- Certified callback substitution: `certified-native-substitution/v1` certificates binding arguments, receipt and
  the bypassed reasoning callback.
- Bank removal from the slow context, preserving reasoning and native actions.
- `/harnessmm` ledger command.
