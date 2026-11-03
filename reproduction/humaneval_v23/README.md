# Archived HumanEval SDK authoring configuration v23

These JavaScript modules are copied byte-for-byte from the frozen HumanEval
DeepSeek Harness test configuration. `SOURCE_MANIFEST.json` records their
archive member names, SHA-256 digests and the complete source archive digest.
They retain the repository's Apache-2.0 license. They contain no provider
credentials, task answers, model weights or paid request logs.

`structured_bank_author.mjs` exports `structuredBankAuthor(ctx, record, policy)`
and `compactPublicHistory(messages)`. The former wraps a metered SDK
`ctx.llm.stream` service. For the first task boundary it supplies the actual
`author_bank` tool schema, compiles the model's literal fields to a native
`terminal` root and guarded bank, and preserves usage. Invalid or absent
authorship uses a further paid native fallback. Subsequent boundaries use
public-history compaction. The archived bridge also configures paid
public-receipt finalization.

One isolated SDK worker process owns one task. The module's process-scoped
author-attempt flag deliberately reflects that historical lifecycle. Do not
share it across unrelated sessions or silently use it as a stock-dsh plugin.
The workflow has fixed Python commands and is not a general Node.js authoring
policy. The historical contract includes MBPP wording about public assertions;
the HumanEval task instruction limits checks to public docstring examples.
Protected verifier tests must remain outside the actor's context.

These historical modules are separate from current runtime repairs. Their
synthetic tests check schema flow, root/successor projection, abstention costs
and source digests; they do not execute a benchmark or establish a new saving.
See [the full configuration notes](../../docs/DEEPSEEK_REPRODUCTION.md) for
native-runtime prerequisites, model identity, policy versions and result scope.
