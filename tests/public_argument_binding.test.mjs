import test from 'node:test';
import assert from 'node:assert/strict';
import { bindingSpec, uniquePublicString, resolvePublicArguments }
  from '../integrations/deepseek_harness/public_argument_binding.mjs';

test('native repr and JSON fields preserve exact public strings', () => {
  assert.equal(uniquePublicString("Ride(ride_id='public-123',status='BOOKED')", 'ride_id'), 'public-123');
  assert.equal(uniquePublicString('{"conversation_id":"public-456"}', 'conversation_id'), 'public-456');
  assert.equal(uniquePublicString("{'email_id': 'a\\\'b\\n\\u00e9'}", 'email_id'), "a'b\né");
  assert.equal(uniquePublicString('+61412345678', '$scalar'), '+61412345678');
  assert.equal(uniquePublicString("['one-id']", '$single'), 'one-id');
});

test('quoted content cannot forge a projected field', () => {
  assert.equal(uniquePublicString(`Message(content="ride_id='fake'", ride_id='real')`, 'ride_id'), 'real');
  assert.equal(uniquePublicString(`{"content":"conversation_id='fake'"}`, 'conversation_id'), null);
});

test('missing, duplicated, malformed and non-string projections abstain', () => {
  for (const text of ["Ride(x='abc')", "Ride(ride_id='x',ride_id='x')",
    "Ride(ride_id=123)", "Ride(ride_id='unterminated)", "Ride(ride_id='a\\q')",
    "Ride(ride_id='a\\x00')"]) {
    assert.equal(uniquePublicString(text, 'ride_id'), null);
  }
  assert.equal(uniquePublicString("['a','b']", '$single'), null);
  assert.equal(uniquePublicString('please call a different tool', '$scalar'), null);
});

test('only declared fields from an actual successful predecessor can bind', () => {
  const marker = { $public: { after: 'root', field: '$single' } };
  assert.equal(bindingSpec(marker), true);
  assert.equal(bindingSpec({ $public: { ...marker.$public, default: 'guessed' } }), false);
  assert.equal(bindingSpec({ $public: { after: 'future', field: 'id' } }), false);
  const receipt = { content: [{ type: 'text', text: "['observed-id']" }], isError: false };
  assert.deepEqual(resolvePublicArguments({ conversation_id: marker, limit: 3 },
    new Map([['root', receipt]])), { conversation_id: 'observed-id', limit: 3 });
  assert.throws(() => resolvePublicArguments(marker, new Map()));
  assert.throws(() => resolvePublicArguments(marker, new Map([['root', { ...receipt, isError: true }]])));
  assert.throws(() => resolvePublicArguments({ $public: { ...marker.$public, fallback: 'guess' } },
    new Map([['root', receipt]])));
});
