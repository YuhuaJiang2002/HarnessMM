import assert from 'node:assert/strict';
import test from 'node:test';
import {parseBank, planInstruction, structuredAuthorStream, structuredAuthorTool}
  from '../integrations/deepseek_harness/cot_jev.mjs';

const policy = {version:10, enabled:true, maximum_candidates:16,
  maximum_fast_steps:8, maximum_public_bytes:65536};
const native = name => ({name, parameters:{type:'object', properties:{
  command:{type:'string'}, description:{type:'string'}},
  required:['command','description'], additionalProperties:false}});
const collect = async stream => {const out=[]; for await (const c of stream) out.push(c); return out;};
const block = b => ({type:'block-end', index:0, block:b});
const nodes = name => 'ABC'.split('').map((id,i) => ({
  candidate_id:id, predecessor_id:i ? 'ABC'[i-1] : 'root', tool:name,
  arguments:{command:'printf '+id, description:'Print '+id},
  guard:{return_code:0}, required_receipt_fields:['return_code'],
  expected_local_cost:{selector_upper_tokens:2048}, dependency_ids:[],
  args_resolved:true, requires_new_reasoning:false,
  depends_only_on_public_receipt:true, native_action_is_literal:true,
}));

for (const name of ['terminal','bash','pwsh']) {
  test(`${name}: schema keeps native required arguments and projects metadata only`, async () => {
    const original=native(name), snapshot=structuredClone(original);
    const registered=structuredAuthorTool(original);
    assert.deepEqual(original,snapshot);
    assert.deepEqual(registered.parameters.required,['command','description']);
    const nodeSchema=registered.parameters.properties.chisel_plan.items;
    assert.deepEqual(nodeSchema.properties.tool.enum,[name]);
    assert.deepEqual(nodeSchema.properties.arguments,original.parameters);
    assert.strictEqual(structuredAuthorTool(registered),registered);
    const args={command:'printf ROOT',description:'Print root'};
    const root={type:'tool-call',id:'root',name,arguments:JSON.stringify({...args,chisel_plan:nodes(name)})};
    const usage={type:'usage',usage:{totalTokens:10000}};
    const finish={type:'finish',reason:{kind:'tool-calls'}};
    const events=[]; let calls=0;
    const chunks=await collect(structuredAuthorStream({tools:[registered]},async function*(){
      calls++; yield block(root); yield usage; yield finish;
    }, e=>events.push(e),policy));
    const content=chunks.filter(c=>c.type==='block-end').map(c=>c.block);
    assert.deepEqual(JSON.parse(content.find(b=>b.type==='tool-call').arguments),args);
    assert.equal(parseBank({role:'assistant',content},policy,[original]).candidates.length,3);
    assert.equal(calls,1);
    assert.strictEqual(chunks.at(-2),usage);
    assert.strictEqual(chunks.at(-1),finish);
    assert.equal(events[0].tools[0].has_chisel_plan,true);
    assert.match(planInstruction(policy,[original]),new RegExp('tool '+name));
    assert.match(planInstruction(policy,[original]),/description/);
  });
}

test('missing shell or schema preserves the original stream and invokes slow exactly once', async () => {
  for (const tools of [undefined,[],[{name:'read',parameters:{type:'object'}}],[{name:'bash'}]]) {
    const original=[block({type:'text',text:'Native response'}),{type:'finish'}];
    const events=[];let calls=0;
    const got=await collect(structuredAuthorStream({tools},async function*(){calls++;yield*original;},e=>events.push(e),policy));
    assert.deepEqual(got,original);assert.equal(calls,1);
    assert.ok(events.some(e=>e.reason==='no_supported_registered_shell'));
  }
});

test('plan-only final output is recorded without inventing a native call', async () => {
  const original=[block({type:'text',text:'Plan: write files, test, commit.'}),{type:'finish'}];
  const events=[];
  const got=await collect(structuredAuthorStream({tools:[structuredAuthorTool(native('bash'))]},
    async function*(){yield*original;},e=>events.push(e),policy));
  assert.deepEqual(got,original);
  assert.ok(events.some(e=>e.kind==='sar/author-no-tool-call' && e.task_completion_verified===false));
});

test('registered bash without metadata schema is reported truthfully', async () => {
  const events=[];
  await collect(structuredAuthorStream({tools:[native('bash')]},async function*(){},e=>events.push(e),policy));
  assert.equal(events[0].tools[0].has_chisel_plan,false);
});

test('root repetition and missing native descriptions remain invalid', () => {
  const tool=native('bash');
  const root={type:'tool-call',id:'root',name:'bash',arguments:JSON.stringify(nodes('bash')[0].arguments)};
  const message=candidates=>({role:'assistant',content:[
    {type:'text',text:'<CHISEL_PLAN>'+JSON.stringify({candidates})+'</CHISEL_PLAN>'},root]});
  assert.throws(()=>parseBank(message(nodes('bash')),policy,[tool]),/already executed/);
  const missing=nodes('bash');delete missing[0].arguments.description;
  assert.throws(()=>parseBank(message(missing),policy,[tool]),/candidate/);
});
