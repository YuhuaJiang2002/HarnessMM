import test from 'node:test';
import assert from 'node:assert/strict';
import {ExecutableSubdag,localGuard} from '../integrations/deepseek_harness/sar_v2.mjs';
import {parseBank,planInstruction} from '../integrations/deepseek_harness/cot_jev.mjs';
const policy={version:6,enabled:true,maximum_candidates:16,maximum_fast_steps:8,maximum_public_bytes:65536};
const tool={name:'terminal',parameters:{type:'object',properties:{command:{type:'string'}},required:['command'],additionalProperties:false}};
const nodes='ABC'.split('').map((id,i)=>({candidate_id:id,predecessor_id:i?'ABC'[i-1]:'root',tool:'terminal',arguments:{command:'printf '+id},guard:{return_code:0},required_receipt_fields:['return_code'],expected_local_cost:{selector_upper_tokens:8192},dependency_ids:[]}));
const root={type:'tool-call',id:'root-call',name:'terminal',arguments:'{"command":"printf ROOT"}'};
const receipt=id=>({role:'tool',toolCallId:id,content:[{type:'text',text:JSON.stringify({return_code:0,stdout:'OK',stderr:null})}]});
const message={role:'assistant',content:[{type:'reasoning',text:'Native reasoning retained'},{type:'text',text:'<CHISEL_PLAN>'+JSON.stringify({candidates:nodes})+'</CHISEL_PLAN>'},root]};
const collect=async g=>{const a=[];for await(const c of g)a.push(c);return a;};
test('three real native boundaries chain directly without slow calls and receive complete certificates',async()=>{
 const events=[];let slow=0;
 const p=new ExecutableSubdag({policy,model:'deepseek-flash',record:e=>events.push(e),request:()=>{throw Error('unnecessary fast request');}});
 p.slowTokens=40000;
 const o={model:'deepseek-flash',tools:[tool],messages:[{role:'user',content:[{type:'text',text:'Complete native task'}]},message,receipt(root.id)]};
 for(let i=0;i<3;i++){
  const chunks=await collect(p.stream(o,async function*(){slow++;},'boundary-'+i));
  const call=chunks.find(c=>c.type==='block-end'&&c.block.type==='tool-call').block;
  assert.deepEqual(JSON.parse(call.arguments),nodes[i].arguments);
  p.observeTool({call_id:call.id,name:'terminal',arguments:JSON.parse(call.arguments),result:{observation:receipt(call.id).content[0].text}});
  o.messages.push({role:'assistant',content:[call]},receipt(call.id));
 }
 assert.equal(slow,0);const cert=events.filter(e=>e.kind==='sar/substitution-certificate');assert.equal(cert.length,3);
 assert.ok(cert.every(e=>e.slow_requests_at_boundary===0&&e.predecessor_callback_id&&e.bypassed_slow_callback_id&&e.native_receipt_hash));
 await collect(p.stream(o,async function*(){slow++;},'exhausted'));assert.equal(slow,1);
});
test('unknown receipt fields, semantic guards and future edges cannot invoke selector',()=>{
 assert.equal(localGuard({...nodes[0],guard:{stdout_equals:'OK'},required_receipt_fields:['stdout']},{value:{return_code:0}}).known,false);
 assert.equal(localGuard({...nodes[0],guard:{task_solved:true}},{value:{return_code:0}}).known,false);
 for(const bad of [{...nodes[0],predecessor_id:'B'},{...nodes[0],guard:{code_correct:true}}])
  assert.throws(()=>parseBank({...message,content:[{type:'text',text:'<CHISEL_PLAN>'+JSON.stringify({candidates:[bad]})+'</CHISEL_PLAN>'},root]},policy,[tool]));
});
test('receipt-local stderr and exact JSON values are deterministic and missing values fail closed',()=>{
 const n={...nodes[0],guard:{stderr_empty:true,json_equals:{path:['count'],value:3}},required_receipt_fields:['stderr','stdout']};
 assert.equal(localGuard(n,{value:{stderr:null,stdout:'{"count":3}'}}).eligible,true);
 assert.equal(localGuard(n,{value:{stderr:null,stdout:'{}'}}).known,false);
});
test('CoT-only and CoT+Jev share identical authoring instruction with SAR',()=>{
 assert.equal(planInstruction(policy),planInstruction({...policy,version:7}));
 assert.equal(planInstruction(policy),planInstruction({...policy,enabled:false}));
});
