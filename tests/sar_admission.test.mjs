import test from 'node:test';
import assert from 'node:assert/strict';
import {admit, compactState, SubstitutionAwareRouter} from '../integrations/deepseek_harness/sar_admission.mjs';
import {parseBank,planInstruction,addControlUsage} from '../integrations/deepseek_harness/cot_jev.mjs';
const policy={version:8,enabled:true,maximum_candidates:8,maximum_fast_steps:8,maximum_public_bytes:65536};
const tool={name:'terminal',parameters:{type:'object',properties:{command:{type:'string'}},required:['command'],additionalProperties:false}};
const n={candidate_id:'A',predecessor_id:'root',tool:'terminal',arguments:{command:'cat /app/config.toml'},
  guard:{return_code:0},required_receipt_fields:['return_code'],expected_local_cost:{selector_upper_tokens:2048},
  dependency_ids:[],args_resolved:true,requires_new_reasoning:false,depends_only_on_public_receipt:true,native_action_is_literal:true};
const root={type:'tool-call',id:'root',name:'terminal',arguments:'{"command":"pwd"}'};
const message={role:'assistant',content:[{type:'reasoning',text:'<CHISEL_PLAN>'+JSON.stringify({candidates:[n]})+'</CHISEL_PLAN>'},root]};
const priors={phase:'train',source_sha256:'f'.repeat(64),categories:{inspect:{phase:'train',median_tokens:10000}}};
const receipt={call_id:'root',value:{return_code:0,stdout:'X'.repeat(100000),stderr:null}};
const context={bank:{sha256:'b'},state:{public_results:[receipt],public_user_requests:[{content:'SECRET_FULL_HISTORY'}],
  candidates:[{label:'A',tool:n.tool,arguments:n.arguments,node:n}]}};
test('local Qwen fast input is charged in full when cache subcounts are unavailable',()=>{
 const u=addControlUsage({inputTokens:100,outputTokens:20,totalTokens:170,cacheReadTokens:50,cacheWriteTokens:0},
   {inputTokens:200,outputTokens:1,totalTokens:201});
 assert.equal(u.totalTokens,371);assert.equal(u.inputTokens,300);assert.equal(u.cacheReadTokens,50);
});
test('reasoning-channel banks parse without inventing arguments, unresolved banks reject',()=>{
 assert.equal(parseBank(message,policy,[tool]).candidates[0].node.args_resolved,true);
 const bad=structuredClone(message);bad.content[0].text=bad.content[0].text.replace('"args_resolved":true','"args_resolved":false');
 assert.throws(()=>parseBank(bad,policy,[tool]));
 assert.equal(planInstruction(policy),planInstruction({...policy,version:9}));
});
test('missing TRAIN median defers, quarter-cost boundary and high-risk paths are enforced',()=>{
 assert.equal(admit(context,priors,'task').eligible.length,1);
 assert.equal(admit(context,{categories:{}},'task').eligible.length,0);
 const cost=admit(context,priors,'task').fastUpper;
 assert.equal(admit(context,{categories:{inspect:{phase:'train',median_tokens:cost*4}}},'task').eligible.length,1);
 assert.equal(admit(context,{categories:{inspect:{phase:'train',median_tokens:cost*4-1}}},'task').eligible.length,0);
 const bad=structuredClone(context);bad.state.candidates[0].arguments.command='cat /tests/test_outputs.py';
 assert.equal(admit(bad,priors,'task').eligible.length,0);
});
test('fast state excludes trajectory, complete receipt and slow rationale',()=>{
 const state=JSON.stringify(compactState(context,context.state.candidates,'task'));
 assert.ok(state.length<1000);assert.ok(!state.includes('SECRET_FULL_HISTORY'));assert.ok(!state.includes('XXX'));
});
test('accepted one-token selection bypasses slow and receives exact native certificate',async()=>{
 const old=process.env.PHIAGENT_DEEPSEEK_FAST_URL;
 process.env.PHIAGENT_DEEPSEEK_FAST_URL='http://127.0.0.1:1234/actor/chat/completions';
 try {
 const events=[];let slow=0,fast=0;
 const p=new SubstitutionAwareRouter({policy,model:'deepseek-flash',priors,episodeId:'task',record:e=>events.push(e),
   request:async()=>{fast++;return {ok:true,status:200,text:async()=>JSON.stringify({model:'deepseek-flash',
     choices:[{message:{content:'A'},logprobs:{content:[{token:'A',logprob:-.1}]}}],usage:{prompt_tokens:200,completion_tokens:1,total_tokens:201,prompt_cache_hit_tokens:0,prompt_cache_miss_tokens:200}})}}});
 const o={model:'deepseek-flash',tools:[tool],messages:[{role:'user',content:[{type:'text',text:'task'}]},message,
   {role:'tool',toolCallId:'root',content:[{type:'text',text:JSON.stringify(receipt.value)}]}]};
 const chunks=[];for await(const c of p.stream(o,async function*(){slow++;},'boundary'))chunks.push(c);
 assert.equal(slow,0);assert.equal(fast,1);
 const call=chunks.find(c=>c.type==='block-end'&&c.block.type==='tool-call').block;
 const result={observation:JSON.stringify({return_code:0,stdout:'config',stderr:null})};
 p.observeTool({call_id:call.id,name:'terminal',arguments:n.arguments,result});
 const c=events.find(e=>e.kind==='sar/substitution-certificate');
 assert.equal(c.certified_substitution,1);assert.equal(c.slow_callback_eliminated,true);
 assert.deepEqual(c.executed_native_receipt,result);assert.deepEqual(c.literal_args,n.arguments);
 assert.equal(c.selector_tokens.totalTokens,201);
 }finally{if(old===undefined)delete process.env.PHIAGENT_DEEPSEEK_FAST_URL;else process.env.PHIAGENT_DEEPSEEK_FAST_URL=old;}
});
