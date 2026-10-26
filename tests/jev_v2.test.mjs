import test from 'node:test';
import assert from 'node:assert/strict';
import { JevV2, minimalState, compactMessages, substitutionAdmission } from '../integrations/deepseek_harness/jev_v2.mjs';
const cfg={version:2,arm:'A3',fast_families:['Files__cat'],max_fast_bytes:8192,
 max_user_bytes:4096,train_audit_sha256:'a'.repeat(64),guards:{
 max_physical_requests:100,max_slow_requests:100,max_fast_requests:100,
 max_total_input_tokens:1e7,max_total_tokens:1e7,max_recovery_attempts:10,
 max_identical_action_retries:10,max_no_progress_turns:10}};
const tool={name:'Files__cat',description:'Native file read',parameters:{type:'object',
 properties:{path:{type:'string'}},required:['path'],additionalProperties:false}};
const bank={candidates:[{label:'A',after:'root',condition:'Receipt says READY',tool:tool.name,arguments:{path:'/done'}}]};
const root={type:'tool-call',id:'root_1',name:tool.name,arguments:'{"path":"/ready"}'};
const options=()=>({model:'deepseek-flash',tools:[tool],messages:[
 {role:'user',content:[{type:'text',text:'Read /done after READY.'}]},
 {role:'assistant',content:[{type:'reasoning',text:'slow original reasoning',signature:'signature'},
 {type:'text',text:'<CHISEL_PLAN>'+JSON.stringify(bank)+'</CHISEL_PLAN>'},root]},
 {role:'tool',toolCallId:root.id,content:[{type:'text',text:"b'READY'"}]}]});
const response=label=>({model:'deepseek-flash',usage:{prompt_tokens:100,completion_tokens:1,
 total_tokens:101,prompt_cache_hit_tokens:64,prompt_cache_miss_tokens:36},choices:[{
 message:{content:label},logprobs:{content:[{token:label,logprob:-.1}]}}]});
async function collect(gen){const result=[];for await(const c of gen)result.push(c);return result;}
test('numeric selector selects complete arguments and never invokes the replaced slow request',async()=>{
 let slow=0,payload;const events=[];
 const router=new JevV2({v2:cfg,policy:{version:4,enabled:true,maximum_candidates:16,
 maximum_fast_steps:16,maximum_public_bytes:32768},model:'deepseek-flash',apiKey:'synthetic',
 record:e=>events.push(e),request:async(_url,init)=>{payload=JSON.parse(init.body);return{ok:true,status:200,text:async()=>JSON.stringify(response('0'))};}});
 const chunks=await collect(router.stream(options(),async function*(){slow++;},'attempt'));
 assert.equal(slow,0);assert.equal(router.counts.slow,0);assert.equal(router.counts.fast,1);
 const packet=JSON.parse(payload.messages[1].content);
 assert.equal(packet.candidates[0].id,0);assert.equal(packet.native_tool_contract,undefined);
 assert.equal(packet.public_user_requests,undefined);assert.equal(packet.slow_root_action,undefined);
 assert.deepEqual(chunks.find(c=>c.type==='block-end'&&c.block.type==='tool-call').block.arguments,'{"path":"/done"}');
 assert.equal(events.filter(e=>e.kind==='sar/substitution-certificate').length,0);
 const call=chunks.find(c=>c.type==='block-end'&&c.block.type==='tool-call').block;
 router.observeTool({call_id:call.id,name:call.name,arguments:JSON.parse(call.arguments),
   result:{observation:'native result',conclude_turn:false}});
 assert.equal(events.filter(e=>e.kind==='sar/substitution-certificate').length,1);
});
test('SAR rejects additive fast paths, missing regret evidence and high regret before dispatch',()=>{
 const estimate={phase:'train',independent_tasks:12,slow_lower:1000,fast_upper:100,
  fallback_upper:1000,fallback_probability_upper:.1,candidate_overhead_upper:20,
  context_growth_upper:30,regret_upper:0};
 const p={families:{[tool.name]:estimate},minimum_gain_tokens:0,maximum_regret:.03};
 assert.equal(substitutionAdmission(bank.candidates,p).admitted,true);
 assert.equal(substitutionAdmission(bank.candidates,{...p,families:{}}).admitted,false);
 assert.equal(substitutionAdmission(bank.candidates,{...p,families:{[tool.name]:{...estimate,
  fallback_probability_upper:1}}}).admitted,false);
 assert.equal(substitutionAdmission(bank.candidates,{...p,families:{[tool.name]:{...estimate,
  regret_upper:.04}}}).admitted,false);
});
test('SAR does not make a paid selector request when its TRAIN bounds are unavailable',async()=>{
 let fast=0,slow=0;
 const router=new JevV2({v2:{...cfg,sar:{version:1,phase:'train',train_audit_sha256:cfg.train_audit_sha256,
  families:{},minimum_gain_tokens:0,maximum_regret:.03}},policy:{version:4,enabled:true,
  maximum_candidates:16,maximum_fast_steps:16,maximum_public_bytes:32768},
  model:'deepseek-flash',apiKey:'synthetic',record:()=>{},request:async()=>{fast++;throw Error('unexpected');}});
 await collect(router.stream(options(),async function*(){slow++;yield{type:'usage',
  usage:{inputTokens:200,outputTokens:5,totalTokens:205}};},'a'));
 assert.equal(fast,0);assert.equal(slow,1);
});
test('TRAIN selector admission supports a singleton public guard without bypassing whitelist or receipt errors',()=>{
 const context={bank:{sha256:'a'},state:{public_user_requests:options().messages.slice(0,1),
  public_results:[{value:{native_text:"b'READY'"},is_error:false}],candidates:bank.candidates}};
 const configured={...cfg,arm:'A5',singleton_guard:'selector'};
 assert.equal(minimalState(context,{}, {...configured,singleton_guard:undefined}),null);
 assert.ok(minimalState(context,{},configured));
 assert.equal(minimalState(context,{}, {...configured,fast_families:[]}),null);
 assert.equal(minimalState({...context,state:{...context.state,public_results:[{is_error:true}]}},{},configured),null);
});
test('receipt dependencies retain actual producing calls and only their native contracts',()=>{
 const context={bank:{sha256:'a'},state:{public_user_requests:options().messages.slice(0,1),
  public_results:[{call_id:root.id,value:{native_text:"b'READY'"},is_error:false}],candidates:bank.candidates}};
 const configured={...cfg,receipt_dependencies:'native_producers'};
 const state=minimalState(context,options(),configured).selectorState;
 assert.deepEqual(state.receipt_producers,[{call_id:root.id,tool:tool.name,args:{path:'/ready'}}]);
 assert.deepEqual(state.receipt_tool_contracts,[tool]);
 assert.equal(state.native_tool_contract,undefined);
 assert.equal(minimalState(context,{...options(),tools:[]},configured),null);
 assert.equal(minimalState(context,{...options(),messages:[]},configured),null);
});
test('high risk or oversized state goes straight to slow without a paid fast call',async()=>{
 let fast=0,slow=0;
 const router=new JevV2({v2:{...cfg,arm:'A4',fast_families:[]},policy:{version:4,enabled:true,
 maximum_candidates:16,maximum_fast_steps:16,maximum_public_bytes:32768},model:'deepseek-flash',apiKey:'synthetic',
 record:()=>{},request:async()=>{fast++;throw Error('unexpected fast');}});
 await collect(router.stream(Object.freeze(options()),async function*(){slow++;yield{type:'usage',usage:{inputTokens:200,outputTokens:5,totalTokens:205}};},'attempt'));
 assert.equal(fast,0);assert.equal(slow,1);assert.equal(router.counts.tokens,205);
});
test('abstention costs are charged exactly once, including cached input',async()=>{
 const router=new JevV2({v2:cfg,policy:{version:4,enabled:true,maximum_candidates:16,
 maximum_fast_steps:16,maximum_public_bytes:32768},model:'deepseek-flash',apiKey:'synthetic',record:()=>{},
 request:async()=>({ok:true,status:200,text:async()=>JSON.stringify(response('?'))})});
 const chunks=await collect(router.stream(options(),async function*(){yield{type:'usage',usage:{inputTokens:200,cacheReadTokens:100,cacheWriteTokens:0,outputTokens:5,totalTokens:305}};},'attempt'));
 assert.equal(router.counts.tokens,406);assert.equal(router.counts.input,400);
 assert.equal(chunks.find(c=>c.type==='usage').usage.totalTokens,406);
});
test('history compaction preserves root reasoning and actual receipts',()=>{
 const ms=options().messages;const inherited={role:'assistant',content:[{type:'reasoning',text:'duplicate',signature:'signature'},
 {type:'tool-call',id:'chisel_abc_A',name:tool.name,arguments:'{"path":"/done"}'}]};
 const compact=compactMessages([...ms,inherited]);assert.equal(compact[1],ms[1]);
 assert.equal(compact[2],ms[2]);assert.equal(compact[3].content.length,1);assert.equal(inherited.content.length,2);
});
test('execution guards cannot dispatch another paid request after the declared limit',async()=>{
 let dispatched=0;const events=[];
 const router=new JevV2({v2:{...cfg,arm:'A6',guards:{...cfg.guards,max_physical_requests:1}},
 policy:{version:4,enabled:true,maximum_candidates:16,maximum_fast_steps:16,maximum_public_bytes:32768},
 model:'deepseek-flash',apiKey:'synthetic',record:e=>events.push(e),request:async()=>{dispatched++;return{ok:true,status:200,text:async()=>JSON.stringify(response('0'))};}});
 await collect(router.stream(options(),async function*(){yield{type:'usage',usage:{inputTokens:200,outputTokens:5,totalTokens:205}};},'a'));
 await assert.rejects(()=>collect(router.stream(options(),async function*(){dispatched++;},'b')),/CHISEL_EXECUTION_GUARD/);
 assert.equal(dispatched,0);assert.equal(events.filter(e=>e.kind==='jev-v2/guard-termination').length,1);
});
