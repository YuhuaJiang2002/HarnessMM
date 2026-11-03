import {createHash,randomUUID} from 'node:crypto';
import {parseBank} from './cot_jev.mjs';
const hash=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');

export function compactPublicHistory(messages){
 const clean=messages.map(m=>({...m,content:(m.content??[]).filter(p=>p.type!=='reasoning').flatMap(p=>p.type==='text'?[{...p,text:p.text.replace(/<CHISEL_PLAN>[\s\S]*?<\/CHISEL_PLAN>/g,'').trim()}].filter(p=>p.text):[p])}));
 const first=clean.findIndex(m=>m.role==='assistant'&&(m.content??[]).some(p=>p.type==='tool-call'));
 if(first<0)return clean;
 const keep=new Set(clean.slice(0,first).map((_,i)=>i));const pairs=[];
 for(let i=first;i<clean.length;i++){
  const calls=(clean[i].content??[]).filter(p=>p.type==='tool-call');if(!calls.length)continue;
  const results=calls.map(c=>clean.findIndex((m,j)=>j>i&&m.role==='tool'&&m.toolCallId===c.id));
  if(results.some(j=>j<0)){keep.add(i);continue;}
  pairs.push({i,results,calls});
 }
 for(const p of pairs.slice(-2)){keep.add(p.i);p.results.forEach(j=>keep.add(j));}
 for(const target of ['/app/solution.py','/app/public_examples.py']){
  const anchors=pairs.filter(p=>p.calls.some(c=>{
   let a;try{a=JSON.parse(c.arguments);}catch{return false;}
   if(typeof a.command!=='string'||!a.command.includes(target)||!/(>|tee\s)/.test(a.command))return false;
   const j=clean.findIndex((m,k)=>k>p.i&&m.role==='tool'&&m.toolCallId===c.id);let r=clean[j]?.content?.find(b=>b.type==='text')?.text;
   for(let n=0;n<3&&typeof r==='string';n++){try{r=JSON.parse(r);}catch{return false;}}
   return r?.return_code===0;
  }));
  const p=anchors.at(-1);if(p){keep.add(p.i);p.results.forEach(j=>keep.add(j));}
 }
 // Preserve trailing public messages and any unmatched tool context.
 const end=pairs.length?Math.max(...pairs.at(-1).results):first-1;
 for(let i=end+1;i<clean.length;i++)keep.add(i);
 return clean.filter((_,i)=>keep.has(i));
}

let authorAttempted=false; // One isolated SDK worker process owns exactly one task.
export function structuredBankAuthor(ctx,record,policy){
 return async function*(options,next,attemptId){
  const fallback=async function*(){
   const messages=Object.freeze(compactPublicHistory(options.messages));
   record({kind:'sar/compact-public-fallback',native_attempt_id:attemptId,original_messages:options.messages,compact_messages:messages,original_tools:options.tools,original_system:options.system,original_systemPrompt:options.systemPrompt,original_messages_sha256:hash(options.messages),compact_messages_sha256:hash(messages),system_unchanged:true,tools_unchanged:true,actual_actions_unchanged:true,reasoning_removed:true,retained_recent_pairs:2,write_receipt_anchors:true,paid:true});
   yield*ctx.llm.stream(Object.freeze({...options,messages,purpose:'sar-compact-fallback'}));
  };
  if(authorAttempted){yield*fallback();return;}
  authorAttempted=true;
  const contract="Use author_bank once. Before emitting the literal bank, mentally evaluate your implementation on EVERY unchanged public assertion. Check indexing conventions, recurrence initial values, empty or sorted inputs, and boundary cases when relevant. Correct the general implementation before emitting the bank; preserve every assertion exactly. Use these public checks to settle task ambiguity instead of a lengthy exploration of unrelated definitions. root_command must WRITE BOTH files in the same shell call: first /app/solution.py with the general implementation, then /app/public_examples.py with from solution import the exact required names and EVERY unchanged released assertion. Use two separate closed heredocs with actual newlines. Do not place writes or checks in candidate commands. Emit a flat object with ALL 15 keys. Do not nest candidates, rename keys, omit constants, or add keys. Exact object form: {\"root_command\":\"<write both requested files using the public task>\",\"a_id\":\"A\",\"a_after\":\"root\",\"a_tool\":\"terminal\",\"a_command\":\"python3 -m py_compile /app/solution.py\",\"a_required_receipt_field\":\"return_code\",\"a_return_code\":0,\"a_selector_upper_tokens\":0,\"b_id\":\"B\",\"b_after\":\"A\",\"b_tool\":\"terminal\",\"b_command\":\"python3 /app/public_examples.py && echo CHISEL_PUBLIC_OK\",\"b_required_receipt_field\":\"return_code\",\"b_return_code\":0,\"b_selector_upper_tokens\":0}. Replace only the root_command placeholder with your literal two-file write; emit every other field explicitly in the tool arguments. Candidate A compiles the written solution; B runs the written public examples and echoes CHISEL_PUBLIC_OK only on success. No reference solutions or protected cases.";
  const authorTools=Object.freeze([{"name": "author_bank", "description": "Author one literal paired-file root and two explicit future actions. All arguments are emitted by the slow model; the compiler supplies no missing commands.", "parameters": {"type": "object", "additionalProperties": false, "required": ["root_command", "a_id", "a_after", "a_tool", "a_command", "a_required_receipt_field", "a_return_code", "a_selector_upper_tokens", "b_id", "b_after", "b_tool", "b_command", "b_required_receipt_field", "b_return_code", "b_selector_upper_tokens"], "properties": {"root_command": {"type": "string", "description": "One shell command with TWO heredocs: cat > /app/solution.py <<'SOLUTION_EOF'\n<general implementation>\nSOLUTION_EOF\ncat > /app/public_examples.py <<'PUBLIC_EOF'\nfrom solution import <required function names>\n<every original released assertion verbatim>\nPUBLIC_EOF\n. Substitute only the implementation, import names, and released assertions. No compilation/check in this root."}, "a_id": {"type": "string", "enum": ["A"]}, "a_after": {"type": "string", "enum": ["root"]}, "a_tool": {"type": "string", "enum": ["terminal"]}, "a_command": {"type": "string", "enum": ["python3 -m py_compile /app/solution.py"]}, "a_required_receipt_field": {"type": "string", "enum": ["return_code"]}, "a_return_code": {"type": "integer", "enum": [0]}, "a_selector_upper_tokens": {"type": "integer", "enum": [0]}, "b_id": {"type": "string", "enum": ["B"]}, "b_after": {"type": "string", "enum": ["A"]}, "b_tool": {"type": "string", "enum": ["terminal"]}, "b_command": {"type": "string", "enum": ["python3 /app/public_examples.py && echo CHISEL_PUBLIC_OK"]}, "b_required_receipt_field": {"type": "string", "enum": ["return_code"]}, "b_return_code": {"type": "integer", "enum": [0]}, "b_selector_upper_tokens": {"type": "integer", "enum": [0]}}}}]);
  const messages=Object.freeze([...options.messages,Object.freeze({role:'user',content:Object.freeze([Object.freeze({type:'text',text:contract})])})]);
  const chunks=[],blocks=[];let usage;
  record({kind:'sar/structured-author-start',native_attempt_id:attemptId,original_messages_sha256:hash(options.messages),author_messages_sha256:hash(messages),tools:authorTools,literal_actions_required:true,paid:true});
  for await(const c of ctx.llm.stream(Object.freeze({...options,messages,tools:authorTools,purpose:'sar-bank-author'}))){
   chunks.push(c);if(c.type==='block-end')blocks.push(c.block);if(c.type==='usage')usage=c;
  }
  const calls=blocks.filter(b=>b.type==='tool-call'&&b.name==='author_bank');
  const physicalArguments=calls.length===1?calls[0].arguments:'';
  const text=physicalArguments;
  let content;
  try{
   if(calls.length!==1)throw Error('One explicit paid author_bank tool call required');
   const flat=JSON.parse(text.trim());
   const keys=['root_command',...['a','b'].flatMap(p=>['id','after','tool','command','return_code','required_receipt_field','selector_upper_tokens'].map(k=>p+'_'+k))];
   if(Object.keys(flat).sort().join()!==keys.sort().join())throw Error('Exact flat literal author fields required');
   const obj={root:{command:flat.root_command},candidates:['a','b'].map(p=>({candidate_id:flat[p+'_id'],predecessor_id:flat[p+'_after'],tool:flat[p+'_tool'],arguments:{command:flat[p+'_command']},guard:{return_code:flat[p+'_return_code']},required_receipt_fields:[flat[p+'_required_receipt_field']],expected_local_cost:{selector_upper_tokens:flat[p+'_selector_upper_tokens']}}))};
   if(typeof flat.root_command!=='string'||obj.candidates.some(n=>typeof n.arguments.command!=='string'||n.guard.return_code!==0||n.required_receipt_fields[0]!=='return_code'||!Number.isInteger(n.expected_local_cost.selector_upper_tokens)))throw Error('Typed explicit literal author fields required');
   if(!obj||typeof obj!=='object'||Object.keys(obj).sort().join()!=='candidates,root'||!obj.root||Object.keys(obj.root).join()!=='command'||typeof obj.root.command!=='string'||!obj.root.command.trim()||!Array.isArray(obj.candidates))throw Error('Author did not supply exact literal root/bank');
   const marker='<CHISEL_PLAN>'+JSON.stringify({candidates:obj.candidates})+'</CHISEL_PLAN>';
   const call={type:'tool-call',id:'author_'+randomUUID(),name:'terminal',arguments:JSON.stringify(obj.root)};
   content=[{type:'text',text:marker},call];
   const bank=parseBank({role:'assistant',content},policy,options.tools);if(!bank)throw Error('Native bank parser did not accept author literal bank');
   record({kind:'sar/structured-author-native-compile',native_attempt_id:attemptId,physical_author_tool_arguments:physicalArguments,physical_author_text:JSON.stringify(obj),physical_author_tool_name:calls[0].name,author_once_per_task:true,physical_author_text_sha256:hash(text),literal_root:obj.root,literal_root_sha256:hash(obj.root),literal_candidates:obj.candidates,compiled_native_call:call,bank_sha256:bank.sha256,commands_or_guards_invented:false});
  }catch(e){
   record({kind:'sar/structured-author-abstain',native_attempt_id:attemptId,error:e.message,physical_author_text_sha256:hash(text),fallback:'unchanged native model request; both paid calls retained'});
   // Preserve the real author usage before a further real paid native fallback.
   if(usage)yield usage;
   yield* fallback();return;
  }
  // Re-emit the exact paid slow reasoning; author JSON is recorded separately.
  let i=0;
  for(const b of blocks.filter(b=>b.type==='reasoning')){
   yield {type:'block-start',index:i,blockType:'reasoning'};yield {type:'reasoning-delta',index:i,text:b.text};yield {type:'block-end',index:i,block:b};i++;
  }
  for(const b of content){
   yield {type:'block-start',index:i,blockType:b.type};
   if(b.type==='text')yield {type:'text-delta',index:i,text:b.text};
   else yield {type:'tool-call-delta',index:i,id:b.id,name:b.name,argumentsDelta:b.arguments};
   yield {type:'block-end',index:i,block:b};i++;
  }
  if(usage)yield usage;
  yield {type:'finish',reason:{kind:'tool-calls'}};
 };
}
