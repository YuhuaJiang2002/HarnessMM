import { createHash } from 'node:crypto';
import { NativeCotJev, decisionState, parseBank, structuredAuthorStream } from './cot_jev.mjs';
const hash = x => createHash('sha256').update(JSON.stringify(x)).digest('hex');
const bytes = x => Buffer.byteLength(JSON.stringify(x));
export function receiptGuard(condition, receipt) {
  if (receipt?.is_error) return { eligible: false, known: true };
  let guard;
  try { guard = JSON.parse(condition); } catch { return { eligible: false, known: false }; }
  if (!guard || typeof guard !== 'object' || Array.isArray(guard)
      || !Object.keys(guard).length || Object.keys(guard).some(k =>
        !['return_code','stdout_contains','stdout_equals'].includes(k))) return { eligible:false, known:false };
  let value = receipt?.value?.native_text ?? receipt?.value;
  for (let i=0; i<3 && typeof value==='string'; i++) {
    try { value=JSON.parse(value); } catch { return {eligible:false,known:false}; }
  }
  if (!value || typeof value!=='object' || !Number.isSafeInteger(value.return_code)) return {eligible:false,known:false};
  // Unexpected native errors always return to slow; failed-result recovery is authored anew.
  if (value.return_code !== 0 || receipt?.is_error) return {eligible:false,known:true};
  const publicFields = {};
  for (const [key, expected] of Object.entries(guard)) {
    if (key==='return_code') {
      if (!Number.isSafeInteger(expected) || expected!==0) return {eligible:false,known:false};
      publicFields.return_code=value.return_code;
      if(value.return_code!==expected)return {eligible:false,known:true};
    } else {
      if(typeof expected!=='string' || !expected.length || typeof value.stdout!=='string')return {eligible:false,known:false};
      const ok=key==='stdout_contains'?value.stdout.includes(expected):value.stdout===expected;
      publicFields[key]={literal:expected,matched:ok};
      if(!ok)return {eligible:false,known:true};
    }
  }
  return {eligible:true,known:true,publicFields,receipt_sha256:hash(receipt)};
}
function chainDepth(bank, after='root', seen=new Set()) {
  if(seen.has(after))return 0;
  return Math.max(0,...bank.candidates.filter(c=>c.after===after)
    .map(c=>1+chainDepth(bank,c.label,new Set([...seen,after]))));
}
export class PersistentBank {
  constructor(args) {
    this.record=args.record; this.policy=args.policy;
    this.originals=new Map();this.admitted=new Set();this.pending=new Map();this.completed=new Set();
    const record=e=>{
      if(e.kind==='jev/compiled-native-action') {
        if(this.pending.has(e.call.id)||this.completed.has(e.call.id))throw Error('Duplicate compiled substitution');
        this.pending.set(e.call.id,e);
      }
      args.record(e);
    };
    this.router=new NativeCotJev({...args,record,resolveContext:(options,policy)=> {
      // Reconstruct banks only in controller memory; never in the paid slow request.
      const messages=options.messages.map(m=> {
        const root=(m.content??[]).find(p=>p.type==='tool-call'&&!p.id?.startsWith('chisel_'));
        return root && this.originals.has(root.id)?this.originals.get(root.id):m;
      });
      return decisionState({...options,messages},policy);
    },prepareContext:context=>this.prepare(context)});
  }
  prepare(context) {
    const {bank,state}=context;
    if(!this.admitted.has(bank.sha256)) {
      const depth=chainDepth(bank);
      const admitted=depth>=3 && bytes(bank.candidates)<=8192;
      this.record({kind:'sar/admission',bank_sha256:bank.sha256,expected_replaceable_callbacks:depth,
        admitted,rule:'at least 3 literal future callbacks; bank <=8192 bytes; no learned estimator'});
      if(!admitted)return null;
      this.admitted.add(bank.sha256);
    }
    const receipt=state.public_results.at(-1);
    const checked=state.candidates.map(c=>({candidate:c,...receiptGuard(c.condition,receipt)}));
    // A guard that cannot be evaluated requests new CoT, even if another branch appears eligible.
    if(checked.some(c=>!c.known)) { this.record({kind:'sar/defer',bank_sha256:bank.sha256,reason:'guard_unknown'}); this.fallbackPending=true; return null; }
    const eligible=checked.filter(c=>c.eligible);
    if(!eligible.length) { this.record({kind:'sar/defer',bank_sha256:bank.sha256,reason:'no_eligible_or_unexpected_receipt'}); this.fallbackPending=true; return null; }
    const users=state.public_user_requests.flatMap(m=>(m.content??[]).filter(p=>p.type==='text').map(p=>p.text));
    const selectorState={objective:users,candidates:eligible.map(x=>x.candidate),
      receipt:{sha256:hash(receipt),guards:eligible.map(x=>({id:x.candidate.label,fields:x.publicFields}))}};
    if(bytes(selectorState)>8192)return null;
    return {...context,state:{...state,candidates:eligible.map(x=>x.candidate)},selectorState,
      selectorSystem:'Choose one eligible slow-authored action ID respecting the supplied objective; output its single letter or ?. Guards were checked deterministically against the hash-bound immediate native receipt. Never invent arguments.',
      ...(eligible.length===1?{deterministicLabel:eligible[0].candidate.label}:{})};
  }
  async *stream(options,next,attemptId) {
    let slow=false;
    const wrapped=async function* (self) {
      slow=true;
      self.record({kind:'sar/slow-call',native_attempt_id:attemptId});
      if(self.fallbackPending)self.record({kind:'sar/fallback-slow-call',native_attempt_id:attemptId});
      self.fallbackPending=false;
      for await(const c of next())yield c;
    };
    yield* this.router.stream(options,()=>wrapped(this),attemptId);
    if(slow)this.admitted.clear();
  }
  observeTool(e) {
    const item=this.pending.get(e.call_id);if(!item)return;
    if(e.name!==item.call.name || hash(e.arguments)!==hash(JSON.parse(item.call.arguments))
      || !Object.hasOwn(e.result??{},'observation'))throw Error('Missing or mismatched native substitution receipt');
    this.pending.delete(e.call_id);this.completed.add(e.call_id);
    this.record({kind:'sar/substitution-certificate',decision_id:item.decision_id,call_id:e.call_id,
      bank_sha256:item.bank_sha256,tool:e.name,arguments:e.arguments,
      native_result_sha256:hash(e.result),slow_counterfactual:1,slow_intervention:0,
      valid_substitution:true,scope:'one actually bypassed native callback; no claim of episode counterfactual tokens'});
  }
  compactSession(session,freezeMessage,meter) {
    if(!session.surface?.nodes || typeof session.deriveEventMessage!=='function')throw Error('Native compaction API missing');
    const entries=[...session.surface.nodes].map(seq=>({seq,event:session.eventAt(seq)}))
      .filter(e=>e.event?.type==='assistant/message')
      .map(e=>({...e,message:session.deriveEventMessage(e.event)})).filter(e=>e.message);
    for(let i=0;i<entries.length;i++) {
      const e=entries[i], m=e.message;
      const hasBank=(m.content??[]).some(p=>p.type==='text'&&p.text.includes('<CHISEL_PLAN>'));
      if(hasBank) {
        const root=m.content.find(p=>p.type==='tool-call'&&!p.id?.startsWith('chisel_'));
        if(root && !this.originals.has(root.id)) {
          this.originals.set(root.id,structuredClone(m));
          try {const bank=parseBank(m,this.policy,[{name:'terminal',parameters:{type:'object'}}]);
            if(bank)this.record({kind:'sar/bank-authored',bank_sha256:bank.sha256,root_id:root.id,
              candidates:bank.candidates.length,bank_bytes:bytes(bank.candidates)});
          } catch { /* The normal router records invalid banks and falls back. */ }
        }
      }
      const synthetic=m.content.some(p=>p.type==='tool-call'&&p.id?.startsWith('chisel_'));
      const content=m.content.flatMap(p=> {
        if(p.type!=='text')return [p];
        const text=p.text.replace(/<CHISEL_PLAN>[\s\S]*?<\/CHISEL_PLAN>/g,'').trim();
        return text?[{...p,text}]:[];
      });
      if(JSON.stringify(content)===JSON.stringify(m.content))continue;
      const replacement=freezeMessage({...m,content});
      const landed=session.append('sar/controller-message',{
        source_seq:e.seq,original_message_sha256:hash(m),message:replacement,
        source_event_sha256:hash(e.event),policy_version:5,
      });
      this.record({kind:'sar/controller-bank-extracted',source_seq:e.seq,replacement_seq:landed.seq,
        original_message_sha256:hash(m),native_actions_and_receipts_preserved:true});
    }
  }
}

export const controllerProjection = {
  type:'sar/controller-message',
  project(event,context) {
    const d=event.data,seq=d.source_seq;
    if(!context.nodes.includes(seq))throw Error('SAR projection source is not live');
    const source=context.events[seq-context.baseSeq];
    const original=context.messages.get(seq)??source?.data?.message;
    if(!original || original.role!=='assistant' || d.policy_version!==5
        || hash(original)!==d.original_message_sha256 || hash(source)!==d.source_event_sha256
        || d.message.id!==original.id || d.message.role!==original.role)throw Error('SAR projection source binding differs');
    const originalCalls=original.content.filter(p=>p.type==='tool-call');
    const replacementCalls=d.message.content.filter(p=>p.type==='tool-call');
    if(hash(originalCalls)!==hash(replacementCalls))throw Error('SAR projection altered native actions');
    return new Map([[seq,Object.freeze(d.message)]]);
  },
};

// Prospective executable sub-DAG. No context projection or compaction is used.
export function localGuard(node, receipt) {
  const guard = node.guard;
  if (!guard || typeof guard !== 'object' || Array.isArray(guard) || !Object.keys(guard).length)
    return {known:false, eligible:false, reason:'invalid_guard'};
  let r=receipt?.value?.native_text ?? receipt?.value;
  for(let i=0;i<3 && typeof r==='string';i++) { try {r=JSON.parse(r);} catch {return {known:false,eligible:false,reason:'non_json_receipt'};} }
  if(!r || typeof r!=='object')return {known:false,eligible:false,reason:'missing_receipt'};
  if(receipt?.is_error)return {known:true,eligible:false,reason:'native_error'};
  const fields={return_code:'return_code',stdout_contains:'stdout',stdout_equals:'stdout',stderr_empty:'stderr',json_equals:'stdout'};
  const required=node.required_receipt_fields;
  if(!Array.isArray(required) || required.some(k=>!Object.hasOwn(r,k))
    || Object.keys(guard).some(k=>!fields[k] || !required.includes(fields[k])))
    return {known:false,eligible:false,reason:'required_field_missing'};
  for(const [k,v]of Object.entries(guard)) {
    let ok;
    if(k==='return_code') {if(!Number.isSafeInteger(v)||!Number.isSafeInteger(r.return_code))return {known:false,eligible:false};ok=r.return_code===v;}
    if(k==='stdout_contains'||k==='stdout_equals') {if(typeof v!=='string'||!v.length||typeof r.stdout!=='string')return {known:false,eligible:false};ok=k==='stdout_contains'?r.stdout.includes(v):r.stdout===v;}
    if(k==='stderr_empty') {if(typeof v!=='boolean'||!(r.stderr===null||typeof r.stderr==='string'))return {known:false,eligible:false};ok=(r.stderr===null||r.stderr==='')===v;}
    if(k==='json_equals') {
      if(!v||!Array.isArray(v.path)||v.path.some(x=>typeof x!=='string'||['__proto__','prototype','constructor'].includes(x))||!Object.hasOwn(v,'value'))return {known:false,eligible:false};
      let x;try{x=JSON.parse(r.stdout);}catch{return {known:false,eligible:false};}
      for(const key of v.path){if(!x||typeof x!=='object'||!Object.hasOwn(x,key))return {known:false,eligible:false};x=x[key];}
      ok=hash(x)===hash(v.value);
    }
    if(!ok)return {known:true,eligible:false,reason:'guard_false'};
  }
  return {known:true,eligible:true,fields:Object.fromEntries(required.map(k=>[k,r[k]]))};
}
export function deterministicAdmission(context, slowTokens, fullSAR=true) {
  const receipt=context.state.public_results.at(-1);
  const done=new Set(['root',...context.state.public_results.map(r=>r.call_id.split('_').at(-1))]);
  const eligible=[]; const rejected=[];
  for(const c of context.state.candidates) {
    const node=c.node;const check=localGuard(node,receipt);
    const missingDependency=(node.dependency_ids??[]).some(id=>!done.has(id));
    // Provider calls are the same backbone. The estimate is deliberately visible,
    // uses only paid TRAIN/earlier slow usage, and is not a guaranteed bound.
    const locallyValid=context.state.candidates.filter(x=>localGuard(x.node,receipt).eligible && (x.node.dependency_ids??[]).every(id=>done.has(id)));
    const deterministic=fullSAR&&locallyValid.length===1;
    const fastUpper=deterministic?0:Math.max(node.expected_local_cost.selector_upper_tokens,bytes({objective:context.state.public_user_requests,candidates:context.state.candidates}));
    const net=slowTokens-fastUpper;
    const risk=receipt?.is_error || (check.known&&check.fields?.return_code!==undefined&&check.fields.return_code!==0);
    const admitted=check.known&&check.eligible&&!missingDependency&&!risk&&(!fullSAR||net>1024);
    (admitted?eligible:rejected).push({candidate:c,check,estimated_net_saving:net,fast_upper:fastUpper,reason:missingDependency?'dependency_missing':risk?'native_error':!check.known?'guard_unknown':!check.eligible?'guard_false':net<=1024?'net_saving_threshold':'admitted'});
  }
  return {eligible,rejected,threshold_tokens:1024,identity:'prospective prior from preceding paid TRAIN slow usage; candidate estimate charged by complete wire ledger'};
}
export class ExecutableSubdag {
  constructor(args) {
    this.record=args.record;this.policy=args.policy;this.pending=new Map();this.done=new Set();this.slowSequence=[];this.actorSequence=[];this.authors=0;this.slowTokens=0;
    const record=e=>{
      if(e.kind==='jev/request')this.actorSequence.push({kind:'fast',id:e.decision_id});
      if(e.kind==='jev/compiled-native-action') {
        const boundary=this.boundary;
        if(!boundary || this.pending.has(e.call.id)||this.done.has(e.call.id))throw Error('Invalid substitution boundary');
        this.pending.set(e.call.id,{...e,boundary,slow_snapshot:this.slowSequence.length,actor_sequence:structuredClone(this.actorSequence)});
      }
      args.record(e);
    };
    this.router=new NativeCotJev({...args,record,prepareContext:context=>this.prepare(context)});
  }
  prepare(context) {
    const admission=deterministicAdmission(context,this.slowTokens,this.policy.version===6);
    this.record({kind:'sar/pre-admission',bank_sha256:context.bank.sha256,
      eligible:admission.eligible.map(x=>({id:x.candidate.label,estimated_net_saving:x.estimated_net_saving})),
      rejected:admission.rejected.map(x=>({id:x.candidate.label,reason:x.reason})),threshold_tokens:admission.threshold_tokens,estimate_identity:admission.identity});
    if(!admission.eligible.length){this.fallback=true;return null;}
    const candidates=admission.eligible.map(x=>x.candidate);
    return {...context,state:{...context.state,candidates},
      selectorState:{objective:context.state.public_user_requests,candidates:candidates.map(c=>c.node),receipt:context.state.public_results.at(-1)},
      selectorSystem:'Select one already executable, locally supported candidate ID or ?. Respect the task constraints. Do not judge cost or create arguments. Return a single A-P letter or ?.',
      ...(this.policy.version===6&&candidates.length===1?{deterministicLabel:candidates[0].label}:{})};
  }
  async *stream(options,next,attemptId) {
    this.boundary={bypassed_slow_callback_id:attemptId,predecessor_callback_id:options.messages?.at(-1)?.toolCallId};
    const wrapped=async function*(self) {
      self.slowSequence.push(attemptId);self.actorSequence.push({kind:'slow',id:attemptId});
      self.record({kind:'sar/slow-call',native_attempt_id:attemptId});
      if(self.fallback)self.record({kind:'sar/fallback-slow-call',native_attempt_id:attemptId});self.fallback=false;
      const content=[];
      for await(const c of (self.policy.version>=10 ? structuredAuthorStream(options,next,self.record,self.policy) : next())) {
        if(c.type==='block-end')content.push(c.block);
        if(c.type==='usage')self.slowTokens=c.usage.totalTokens;
        yield c;
      }
      try {
        const bank=parseBank({role:'assistant',content},self.policy,options.tools);
        if(bank){self.authors++;self.record({kind:'sar/bank-authored',bank_sha256:bank.sha256,root_id:bank.root.id,
          author_callback_id:attemptId,candidates:bank.candidates.length,bank_author_call_index:self.authors});}
        if(self.policy.version>=8)self.record({kind:'sar/author-coverage',native_attempt_id:attemptId,
          bank_size:bank?.candidates.length??0,
          executable_candidates:bank?.candidates.filter(c=>c.node.args_resolved&&!c.node.requires_new_reasoning).length??0,
          unresolved_candidates:0,
          reason_candidates_cannot_be_emitted:!bank?'no_bank_declared':!bank.candidates.length?'author_deferred_unknown_state':null});
      }catch(error){self.record({kind:'sar/author-invalid-bank',native_attempt_id:attemptId,error:error.message});
        if(self.policy.version>=8)self.record({kind:'sar/author-coverage',native_attempt_id:attemptId,bank_size:0,
          executable_candidates:0,unresolved_candidates:null,reason_candidates_cannot_be_emitted:'invalid_bank:'+error.message});}
    };
    yield* this.router.stream(options,()=>wrapped(this),attemptId);
  }
  observeTool(e) {
    const item=this.pending.get(e.call_id);if(!item)return;
    if(e.name!==item.call.name || hash(e.arguments)!==hash(item.candidate.arguments)
      || !Object.hasOwn(e.result??{},'observation') || this.slowSequence.length!==item.slow_snapshot)
      throw Error('Substitution lacks exact native receipt or has intervening slow request');
    this.pending.delete(e.call_id);this.done.add(e.call_id);
    let observation=e.result.observation;
    try{if(typeof observation==='string')observation=JSON.parse(observation);}catch{}
    if(observation?.executed===false){
      this.record({kind:'sar/uncertified-execution',call_id:e.call_id,reason:'shared_parent_suppressed_action',
        certified_substitution:0,executed_native_receipt:e.result});return;
    }
    this.record({kind:'sar/substitution-certificate',schema:'certified-native-substitution/v1',
      bank_hash:item.bank_sha256,bank_sha256:item.bank_sha256,candidate_id:item.candidate.label,
      ...item.boundary,exact_native_tool:e.name,exact_arguments:e.arguments,
      native_receipt_hash:hash(e.result),native_result_sha256:hash(e.result),call_id:e.call_id,
      native_receipt:e.result,guard:item.candidate.node?.guard,
      native_action_hash:hash([e.name,e.arguments]),actor_request_sequence:item.actor_sequence,
      slow_requests_at_boundary:0,slow_sequence_before:item.slow_snapshot,slow_sequence_after:this.slowSequence.length,
      valid_substitution:true,counterfactual_identity:'unchanged native SDK llm/stream boundary; no claim of paired-parent trajectory equality'});
  }
}
