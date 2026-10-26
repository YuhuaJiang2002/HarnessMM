import test from 'node:test';
import assert from 'node:assert/strict';
import { PersistentBank,receiptGuard } from '../integrations/deepseek_harness/sar_v2.mjs';
import { planInstruction } from '../integrations/deepseek_harness/cot_jev.mjs';
const policy={version:5,enabled:true,maximum_candidates:8,maximum_fast_steps:8,maximum_public_bytes:32768};
const tool={name:'terminal',description:'native',parameters:{type:'object',properties:{command:{type:'string'}},required:['command'],additionalProperties:false}};
const bank={candidates:['A','B','C'].map((label,i)=>({label,after:i?'ABC'[i-1]:'root',condition:'{"return_code":0}',tool:'terminal',arguments:{command:`echo ${label}`}}))};
const root={type:'tool-call',id:'r1',name:'terminal',arguments:'{"command":"echo root"}'};
const result=id=>({role:'tool',toolCallId:id,isError:false,content:[{type:'text',text:JSON.stringify({return_code:0,stdout:'READY',stderr:null})}]});
const original={role:'assistant',content:[{type:'reasoning',text:'original reasoning'}, {type:'text',text:'<CHISEL_PLAN>'+JSON.stringify(bank)+'</CHISEL_PLAN>'},root]};
const options=()=>({model:'deepseek-flash',tools:[tool],messages:[{role:'user',content:[{type:'text',text:'Perform the specified native operations.'}]},original,result(root.id)]});
const collect=async g=>{const a=[];for await(const x of g)a.push(x);return a;};
test('three persistent native actions bypass three slow callbacks, make zero selector requests, and certify once',async()=>{
 let slow=0,fast=0;const events=[];
 const p=new PersistentBank({policy,model:'deepseek-flash',apiKey:'fake',record:e=>events.push(e),request:async()=>{fast++;throw Error('unexpected selector');}});
 const o=options();p.originals.set(root.id,original);o.messages[1]={...original,content:original.content.filter(x=>x.type!=='text')};
 for(let i=0;i<3;i++) {
  const chunks=await collect(p.stream(o,async function*(){slow++;},'a'+i));
  const call=chunks.find(x=>x.type==='block-end'&&x.block.type==='tool-call')?.block;assert.ok(call);
  o.messages.push({role:'assistant',content:[call]},result(call.id));
  p.observeTool({call_id:call.id,name:call.name,arguments:JSON.parse(call.arguments),result:{observation:result(call.id).content[0].text}});
 }
 assert.equal(slow,0);assert.equal(fast,0);assert.equal(events.filter(x=>x.kind==='sar/substitution-certificate').length,3);
 await collect(p.stream(o,async function*(){slow++;},'exhausted'));assert.equal(slow,1);
});
test('single-action bank is rejected before paying any selector request',async()=>{
 const o=options();o.messages[1]={...original,content:[{type:'text',text:'<CHISEL_PLAN>'+JSON.stringify({candidates:bank.candidates.slice(0,1)})+'</CHISEL_PLAN>'},root]};
 let slow=0;
 const p=new PersistentBank({policy,model:'deepseek-flash',apiKey:'fake',record:()=>{},request:async()=>{throw Error('paid fast call');}});
 await collect(p.stream(o,async function*(){slow++;},'a'));assert.equal(slow,1);
});
test('guards reject uncertain receipts and unexpected errors; exact receipt markers are checked',()=>{
 assert.equal(receiptGuard('{"return_code":0,"stdout_contains":"READY"}',{value:{return_code:0,stdout:'READY'}}).eligible,true);
 assert.equal(receiptGuard('{"return_code":0}',{value:{return_code:1,stdout:'READY'}}).eligible,false);
 assert.equal(receiptGuard('looks good',{value:{return_code:0}}).known,false);
 assert.equal(receiptGuard('{"stdout_contains":"READY"}',{value:{return_code:0,stdout:'NOT_READY?'}}).eligible,true);
 assert.equal(receiptGuard('{"stdout_equals":"READY"}',{value:{return_code:0,stdout:'NOT_READY'}}).eligible,false);
});
test('instruction contains a valid closed-schema bank example',()=>{
 const example=planInstruction(policy).match(/<CHISEL_PLAN>(.*?)<\/CHISEL_PLAN>/)[1];
 const b=JSON.parse(example);assert.deepEqual(JSON.parse(b.candidates[0].condition),{return_code:0});
});
test('native session extraction preserves authentic reasoning and receipts while removing bank from slow context',()=>{
 const p=new PersistentBank({policy,model:'deepseek-flash',apiKey:'fake',record:()=>{}});
 const events=new Map([[1,{type:'assistant/message',data:{message:original}}]]);
 const replacements=[];const session={surface:{nodes:[1]},eventAt:i=>events.get(i),deriveEventMessage:e=>e.data.message,
  append:(kind,data,meta)=>{if(kind==='sar/controller-message')replacements.push(data.message);return{seq:2};}};
 p.compactSession(session,x=>x,{estimateMessage:()=>100});
 assert.equal(replacements.length,1);assert.equal(replacements[0].content.some(x=>x.type==='text'&&x.text.includes('CHISEL_PLAN')),false);
 assert.deepEqual(replacements[0].content.find(x=>x.type==='reasoning'),original.content[0]);
 assert.deepEqual(replacements[0].content.find(x=>x.type==='tool-call'),root);
 assert.deepEqual(p.originals.get(root.id),original);
});
test('projection binds original event and preserves native action arguments',async()=>{
 const {controllerProjection}=await import('../integrations/deepseek_harness/sar_v2.mjs');
 const {createHash}=await import('node:crypto');const hash=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
 const source={seq:1,type:'assistant/message',data:{message:{...original,id:'msg'}}};
 const message={...source.data.message,content:source.data.message.content.filter(x=>x.type!=='text')};
 const e={data:{source_seq:1,original_message_sha256:hash(source.data.message),source_event_sha256:hash(source),policy_version:5,message}};
 const ctx={nodes:[1],events:[{},source],baseSeq:0,messages:new Map()};
 assert.equal(controllerProjection.project(e,ctx).get(1),message);
 const altered={...e,data:{...e.data,message:{...message,content:[{...root,arguments:'{"command":"unexpected"}'}]}}};
 assert.throws(()=>controllerProjection.project(altered,ctx),/altered native actions/);
});
