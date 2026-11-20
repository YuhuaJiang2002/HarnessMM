/**
 * Conformance with the reference implementation.
 * The fixture was exported directly from HarnessMM's cot_jev.mjs / sar_admission.mjs;
 * these tests fail if a single byte of the authoring or selector text drifts.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { CHOICE_SYSTEM, SELECTOR_SYSTEM, planInstruction, validateCotJev, type Policy } from "../src/policy.ts";

interface Fixture {
  choice: string;
  selector: string;
  plans: Record<string, string>;
}

const fixture: Fixture = JSON.parse(readFileSync(new URL("./fixtures/upstream-contracts.json", import.meta.url), "utf8"));
const policy = (version: number): Policy => ({ version, enabled: true, maximum_candidates: 16, maximum_fast_steps: 8, maximum_public_bytes: 65536 });

test("the authoring instruction is byte-identical to the reference implementation", () => {
  for (const [version, expected] of Object.entries(fixture.plans)) {
    assert.equal(planInstruction(policy(Number(version))), expected, `policy v${version} instruction drifted`);
  }
});

test("selector and choice prompts are byte-identical", () => {
  assert.equal(CHOICE_SYSTEM, fixture.choice);
  assert.equal(SELECTOR_SYSTEM, fixture.selector);
});

test("an invalid policy is refused before any bank is parsed", () => {
  assert.throws(() => validateCotJev({ ...policy(6), version: 12 } as Policy), /Invalid frozen CoT\/Jev policy/);
  assert.throws(() => validateCotJev({ ...policy(6), maximum_candidates: 17 } as Policy), /Invalid frozen CoT\/Jev policy/);
  assert.throws(() => validateCotJev({ ...policy(6), maximum_public_bytes: 512 } as Policy), /Invalid frozen CoT\/Jev policy/);
  assert.throws(() => validateCotJev({ ...policy(6), extra: true } as unknown as Policy), /Invalid frozen CoT\/Jev policy/);
  assert.equal(validateCotJev(policy(6)).version, 6);
});

test("the instruction for a version carries the same contract as its neighbours", () => {
  // Upstream derives v6 and v7 from one text, and v8/v9 from another, as the fixture records.
  assert.equal(planInstruction(policy(6)), planInstruction(policy(7)));
  assert.equal(planInstruction(policy(8)), planInstruction(policy(9)));
  assert.notEqual(planInstruction(policy(5)), planInstruction(policy(6)));
});
