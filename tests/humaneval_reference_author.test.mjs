import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {parseBank} from '../reproduction/humaneval_v23/cot_jev.mjs';

const directory=new URL('../reproduction/humaneval_v23/',import.meta.url);
const policy={version:6,enabled:true,maximum_candidates:16,maximum_fast_steps:8,maximum_public_bytes:65536};
const tool={name:'terminal',parameters:{type:'object',properties:{command:{type:'string'}},required:['command'],additionalProperties:false}};
const collect=async stream=>{const out=[];for await(const c of stream)out.push(c);return out;};
const usage={type:'usage',usage:{totalTokens:10000}};

test('reference adapter modules match the archived source digests',()=>{
  const manifest=JSON.parse(readFileSync(new URL('SOURCE_MANIFEST.json',directory)));
  for(const file of manifest.files) {
    const bytes=readFileSync(new URL(file.path,directory));
    assert.equal(createHash('sha256').update(bytes).digest('hex'),file.sha256,file.path);
  }
});

test('archived author emits the supplied root and two guarded native successors, then uses fallback',async()=>{
  const {structuredBankAuthor}=await import(new URL('structured_bank_author.mjs?valid',directory));
  const authored={root_command:'printf ROOT'};
  for(const [p,id,after,command]of [['a','A','root','python3 -m py_compile /app/solution.py'],
    ['b','B','A','python3 /app/public_examples.py && echo CHISEL_PUBLIC_OK']]) {
    Object.assign(authored,{[p+'_id']:id,[p+'_after']:after,[p+'_tool']:'terminal',
      [p+'_command']:command,[p+'_required_receipt_field']:'return_code',
      [p+'_return_code']:0,[p+'_selector_upper_tokens']:0});
  }
  const requests=[],events=[];
  const ctx={llm:{async *stream(options){
    requests.push(options);
    if(options.purpose==='sar-bank-author')yield {type:'block-end',block:{type:'tool-call',name:'author_bank',id:'authored',arguments:JSON.stringify(authored)}};
    else yield {type:'block-end',block:{type:'text',text:'Native fallback'}};
    yield usage;
  }}};
  const options={model:'synthetic-no-provider',tools:[tool],messages:[{role:'user',content:[{type:'text',text:'Synthetic contract check'}]}]};
  const stream=structuredBankAuthor(ctx,e=>events.push(e),policy);
  const output=await collect(stream(options,()=>{throw Error('unused native callback');},'first'));
  assert.equal(requests[0].tools[0].name,'author_bank');
  const content=output.filter(c=>c.type==='block-end').map(c=>c.block);
  assert.deepEqual(JSON.parse(content.find(b=>b.type==='tool-call').arguments),{command:authored.root_command});
  const bank=parseBank({role:'assistant',content},policy,[tool]);
  assert.deepEqual(bank.candidates.map(c=>c.arguments.command),[authored.a_command,authored.b_command]);
  assert.ok(events.some(e=>e.kind==='sar/structured-author-native-compile'));
  assert.strictEqual(output.at(-2),usage);
  await collect(stream(options,()=>{},'second'));
  assert.deepEqual(requests.map(r=>r.purpose),['sar-bank-author','sar-compact-fallback']);
});

test('archived author abstention retains author usage and invokes paid fallback',async()=>{
  const {structuredBankAuthor}=await import(new URL('structured_bank_author.mjs?abstain',directory));
  const purposes=[],events=[];
  const ctx={llm:{async *stream(o){purposes.push(o.purpose);yield {type:'block-end',block:{type:'text',text:'No literal bank'}};yield usage;}}};
  const out=await collect(structuredBankAuthor(ctx,e=>events.push(e),policy)(
    {model:'synthetic-no-provider',tools:[tool],messages:[]},()=>{},'boundary'));
  assert.deepEqual(purposes,['sar-bank-author','sar-compact-fallback']);
  assert.equal(out.filter(c=>c.type==='usage').length,2);
  assert.ok(events.some(e=>e.kind==='sar/structured-author-abstain'));
});
