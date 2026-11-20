# Archived HumanEval SDK authoring configuration v23

These JavaScript modules are copied byte-for-byte from the frozen HumanEval
DeepSeek Harness test configuration. `SOURCE_MANIFEST.json` records their
archive member names, SHA-256 digests and the complete source archive digest.
They retain the repository's Apache-2.0 license.

`structured_bank_author.mjs` exports `structuredBankAuthor(ctx, record, policy)`
and `compactPublicHistory(messages)`. The former wraps a metered SDK
`ctx.llm.stream` service. For the first task boundary it supplies the actual
`author_bank` tool schema, compiles the model's literal fields to a native
`terminal` root and guarded bank, and preserves usage. Invalid or absent
authorship uses a further paid native fallback. Subsequent boundaries use
public-history compaction. The archived bridge also configures paid
public-receipt finalization.

One isolated SDK worker process owns one task. The author-attempt flag lasts for that worker process. Start a fresh worker
for each task. The SDK adapter runs fixed Python commands. Its historical
contract mentions MBPP public assertions. For HumanEval, the actor gets the
public docstring examples and the native evaluator holds the verifier tests.

The synthetic tests check schema flow, root and successor projection, metered
fallback on abstention, and the digests of the archived sources.

See [the full configuration notes](../../docs/DEEPSEEK_REPRODUCTION.md) for
native-runtime prerequisites, model identity, policy versions and result scope.
