"""Complete TRAIN-only public feedback for one native action-bank revision."""
from pathlib import Path
from collections import Counter
import hashlib,json,subprocess,re

def native_bank_contract():
 module=Path(__file__).resolve().parents[1]/'integrations/deepseek_harness/cot_jev.mjs'
 program='import {planInstruction} from '+json.dumps(module.as_uri())+';process.stdout.write(JSON.stringify(planInstruction('+json.dumps({'version':4,'enabled':True,'maximum_candidates':16,'maximum_fast_steps':16,'maximum_public_bytes':32768})+')));'
 result=subprocess.run(['node','--input-type=module','-e',program],capture_output=True,text=True,timeout=10,check=True)
 return json.loads(result.stdout)

def decode_author_json(text):
 text=text.strip()
 if text.startswith('```json\n')and text.endswith('\n```'):
  text=text[len('```json\n'):-len('\n```')]
 return json.loads(text)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def public_user_activity(messages):
 """Preserve actual TRAIN task updates without runtime or private messages."""
 out=[]
 for message in messages:
  if message.get('role')!='user'or message.get('source',{}).get('kind')!='user':continue
  text='\n'.join(p['text']for p in message.get('content',[])if p.get('type')=='text'and isinstance(p.get('text'),str))
  out.append({'public_text_excerpt':text[:1200]+('...'+text[-400:]if len(text)>1600 else text[1200:]),
   'public_text_characters':len(text),'public_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
   'environment_notification':'[ENVIRONMENT_NOTIFICATION' in text})
 return {'delivered_user_messages':len(out),'delivered_environment_notifications':sum(x['environment_notification']for x in out),
  'messages':out,'selection':'All actual public TRAIN user messages; bounded text, full hashes; runtime snapshots and private reasoning excluded'}
def public_bridge_activity(records):
 """Include the last executed native action, absent from its preceding LLM input.

 Only original public bridge tool events are read. Provider reasoning, judge
 prompts and scenario/oracle records are not part of this projection.
 """
 calls=[];seen=set()
 for record in records:
  if record.get('kind')!='bridge/tool/end':continue
  call_id=record.get('call_id')
  if not isinstance(call_id,str)or call_id in seen:
   raise ValueError('Each original public native tool completion needs a unique call ID')
  seen.add(call_id);result=record.get('result',{})
  text=result.get('observation')
  if not isinstance(text,str):
   raise ValueError('Original public native observations must remain textual')
  calls.append({'tool':record['name'],'arguments':record['arguments'],
   'public_receipt_excerpt':text[:600]+('...'+text[-200:]if len(text)>800 else text[600:]),
   'public_receipt_characters':len(text),'public_receipt_sha256':hashlib.sha256(text.encode()).hexdigest(),
   'conclude_turn':result['conclude_turn'],'is_error':result.get('is_error',False)})
 indices=list(range(min(6,len(calls))))+list(range(max(6,len(calls)-6),len(calls)))
 return {'completed_native_calls':len(calls),
  'first_and_last_executed_public_calls':[calls[i]for i in indices],
  'native_turn_concluding_tools':[{'tool':c['tool'],'arguments':c['arguments']}for c in calls if c['conclude_turn']],
  'selection':'All original public bridge completions counted; deterministic first6/last6, full receipt hashes; no judge, oracle or private reasoning'}
def public_tool_activity(messages):
 """Summarize all paid public tool calls without private reasoning or user text."""
 calls=[];by_id={};bank_sizes=Counter()
 for message in messages:
  if message.get('role')=='assistant':
   for part in message.get('content',[]):
    if part.get('type')=='tool-call':
     call={'tool':part.get('name'),'arguments':part.get('arguments')}
     calls.append(call);by_id[part.get('id')]=call
    elif part.get('type')=='text':
     for raw in re.findall(r'<CHISEL_PLAN>(.*?)</CHISEL_PLAN>',part.get('text',''),re.S):
      try:
       bank=json.loads(raw)
       if isinstance(bank.get('candidates'),list):bank_sizes[len(bank['candidates'])]+=1
      except (ValueError,TypeError,AttributeError):pass
  elif message.get('role')=='tool'and message.get('toolCallId')in by_id:
   text='\n'.join(p['text']for p in message.get('content',[])if p.get('type')=='text'and isinstance(p.get('text'),str))
   by_id[message['toolCallId']].update(public_receipt_excerpt=text[:600]+('...'+text[-200:]if len(text)>800 else text[600:]),public_receipt_characters=len(text),public_receipt_sha256=hashlib.sha256(text.encode()).hexdigest())
 pairs=Counter((c['tool'],json.dumps(c['arguments'],sort_keys=True))for c in calls)
 selected=list(range(min(6,len(calls))))+list(range(max(6,len(calls)-6),len(calls)))
 return {'public_tool_calls':len(calls),'declared_bank_sizes':dict(sorted(bank_sizes.items())),
  'repeated_exact_calls':[{'tool':tool,'arguments':json.loads(args),'occurrences':n}for(tool,args),n in pairs.most_common(8)if n>1],
  'first_and_last_public_calls':[calls[i]for i in selected],
  'trace_selection':'Deterministic first6/last6; every public call counted; receipt excerpts hash-bound; private reasoning excluded'}
def public_sdk(summary, row):
 if summary.get('actor_result'):
  return Path(summary['actor_result']).parent, {}
 recovery=summary['native_model_failure_scoring'];original_path=Path(recovery['original_summary'])
 assert recovery['new_actor_calls']==0 and recovery['tool_actions_replayed']is False
 assert sha(original_path)==recovery['original_summary_sha256']==row['original_summary_sha256']
 original=load(original_path)
 assert original['phase']=='train'and original['task_id']==row['task_id']and original['seed']==row['seed']
 assert original['model']==summary['model']and original['runtime']==summary['runtime']
 assert original['provider_usage']['actor']['complete']and original['provider_usage']['actor']['usage']==summary['provider_usage']['actor']['usage']
 assert Path(original['output']).resolve()==original_path.parent.resolve()
 sdk=original_path.parent/'sdk'
 manifest=sdk/'ARTIFACT_SHA256.json';assert load(manifest)['llm-streams.jsonl']==sha(sdk/'llm-streams.jsonl')
 return sdk, {'original_summary_sha256':sha(original_path),'sdk_artifact_manifest_sha256':sha(manifest),'recovered_native_verdict_only':True}
def feedback(root, train_ids):
 root=Path(root);report=load(root/'TRAIN_COMPARISON.json');rows=load(root/'TRAIN_OUTCOMES.json')
 if not report['complete']or report['settled']!=96 or len(rows)!=96:
  raise ValueError('Complete known TRAIN32 in all three arms is required')
 if report['arms']['cot_jev']['registered_40_percent_goal_passed']:
  raise ValueError('A completed passing iteration needs no new bank revision')
 expected={(i,arm)for i in train_ids for arm in ['native','cot_only','cot_jev']}
 if {(r['task_id'],r['arm'])for r in rows}!=expected or any(r['phase']!='train'or not r['valid']for r in rows):
  raise ValueError('Only the identical complete frozen TRAIN population supplies feedback')
 candidate=load(root/'TRAIN_FROZEN_CANDIDATE.json')['descriptor'];previous=Path(candidate['path']);assert sha(previous)==candidate['sha256']
 profiles=[]
 for row in sorted(rows,key=lambda r:(r['task_id'],r['arm'])):
  if row['arm']=='native':continue
  summary_path=Path(row['summary']);assert sha(summary_path)==row['summary_sha256'];summary=load(summary_path)
  assert summary['phase']=='train'and summary['valid_for_comparison']and summary['native_score']==row['native_score']and summary['provider_usage']['actor']['complete']and summary['provider_usage']['actor']['usage']['total_tokens']==row['actor_tokens']
  sdk,retention=public_sdk(summary,row);stream=sdk/'llm-streams.jsonl';records=[json.loads(x)for x in stream.read_text().splitlines()];counts=Counter(x.get('kind','')for x in records)
  public=[];starts=[x for x in records if x.get('kind')=='llm/start']
  if starts:
   for message in starts[-1].get('request',{}).get('messages',[]):
    if message.get('role')not in ['assistant','tool']:continue
    parts=[x for x in message.get('content',[])if x.get('type')in ['text','tool-call']]
    if parts:public.append({'role':message['role'],'content':parts})
  excerpt=json.dumps(public[-8:],ensure_ascii=False)
  diagnostics=[{k:x[k]for k in ['kind','reason','error']if k in x}for x in records if x.get('kind')in ['jev/invalid-bank','jev/full-cot','jev/slow-fallback']]
  profiles.append({'task_id':row['task_id'],'arm':row['arm'],'phase':'train','native_score':row['native_score'],'actor_tokens':row['actor_tokens'],'sdk_requests':counts['llm/start'],'fast_requests':counts['jev/request'],'compiled_actions':counts['jev/compiled-native-action'],'missing_or_ineligible_bank':counts['jev/full-cot'],'invalid_banks':counts['jev/invalid-bank'],'slow_fallbacks':counts['jev/slow-fallback'],'recent_public_trace':excerpt[-2400:],'trace_truncated':len(excerpt)>2400,'public_compiler_diagnostics':diagnostics[-8:],'native_tool_activity':public_tool_activity(starts[-1].get('request',{}).get('messages',[])if starts else []),'public_user_activity':public_user_activity(starts[-1].get('request',{}).get('messages',[])if starts else []),'executed_public_bridge_activity':public_bridge_activity(records),'source':{'summary_sha256':sha(summary_path),'sdk_stream_sha256':sha(stream),**retention}})
 return {'phase':'train','validation_or_test_feedback_used':False,'complete_train_comparison':report,'source_comparison_sha256':sha(root/'TRAIN_COMPARISON.json'),'source_outcomes_sha256':sha(root/'TRAIN_OUTCOMES.json'),'previous_instruction':previous.read_text(),'previous_instruction_sha256':sha(previous),'public_train_profiles':profiles}

def public_catalog_feedback(evidence, root):
 """Expose exact agent-visible TRAIN schemas, never scenario or judge state."""
 root=Path(root);rows=load(root/'TRAIN_OUTCOMES.json')
 if evidence['phase']!='train' or evidence['validation_or_test_feedback_used']is not False or len(rows)!=96:
  raise ValueError('Native catalogs require complete TRAIN-only evidence')
 catalogs={};references=[]
 for row in rows:
  if row['arm']=='native':continue
  if row['phase']!='train' or not row['valid']:
   raise ValueError('Only valid TRAIN jobs supply public catalogs')
  path=Path(row['summary']);assert sha(path)==row['summary_sha256'];summary=load(path)
  sdk,_=public_sdk(summary,row);job=sdk/'job.json';tools=load(job)['tools']
  if not isinstance(tools,list)or not tools or any(not isinstance(t,dict)or set(t)!= {'name','description','parameters'}for t in tools):
   raise ValueError('A native catalog needs its exact public name, description and schema')
  key=hashlib.sha256(json.dumps(tools,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  catalogs[key]=tools;references.append({'task_id':row['task_id'],'arm':row['arm'],'phase':'train','public_catalog_sha256':key,'sdk_job_sha256':sha(job)})
 if len(references)!=64 or len({(r['task_id'],r['arm'])for r in references})!=64:
  raise ValueError('Catalog references must cover all64 matched TRAIN jobs exactly once')
 return {**evidence,'public_native_tool_catalogs':catalogs,'public_native_catalog_references':references}

def revise_prompt(prompt, evidence, *, completion_cost_revision=False, retrieval_chain_revision=False, event_lifecycle_revision=False, public_confirmation_revision=False, compiler_first_revision=False, predicate_cost_revision=False, catalog_turn_revision=False, public_delta_revision=False):
 result = {**prompt,'predecessor_complete_train':evidence,
  'accepted_native_bank_declaration_contract':native_bank_contract(),'objective':(
  'Author exactly ONE revised reusable native-app CHISEL skill from the complete TRAIN evidence. '
  'Preserve the native task success of both the industrial parent and the identical fast-disabled CoT-only control while targeting 40 percent fewer complete actor input/output tokens than BOTH, including the slow author. '
  'The accepted policy4 compiler and model are fixed. Slow CoT authors literal future native-app action banks, and a charged one-token fast judge chooses a declared action from the current public receipt or abstains. '
  'Diagnose the supplied complete TRAIN32 parent and all64 matched bank trajectories. Address actual missing/ineligible or invalid banks and the native quality losses; avoid banking guessed future IDs, content or tool arguments. '
  'Give a precise visible CHISEL_PLAN emission discipline, exact native tool names/schemas and declared predecessor labels. Use root or an earlier label in the current bank. Do not refer to a label from a previous bank or duplicate an already executed root action. '
  'Follow the supplied compiler-generated declaration contract exactly: the single root is the native tool-call OUTSIDE the candidates array; every candidate is a FUTURE conditional action with exactly label, after, condition, tool and arguments. Do not put the root itself in candidates, add bank-wide fields, or invent a different JSON syntax. '
  'Make useful multi-action continuations conditional on observed public results, and preserve required empty kwargs. Act on stable arguments already established by the public task or prior receipts; unknown arguments require slow CoT. '
  'For asynchronous operations, use native public time and notifications and avoid repeating a mutation or writing a different artifact after a successful check. Abstain on unexpected receipts or uncertainty. Keep public outputs focused while preserving native verification and the slow history. '
  'The native common prompt already covers admissibility and tool usage. Focus this skill on executable bank coverage and reliable completion, guided by the actual complete TRAIN profiles rather than changing the parent or protocol. '
  'Never read oracle files, use validation/test feedback, hardcode scenario answers, skip verification, or impose token/turn ceilings. '
  'Return only JSON with instruction (nonempty, at most3000 characters) and supporting_task_ids (nonempty unique subset of supplied TRAIN IDs).')}
 if completion_cost_revision or retrieval_chain_revision or event_lifecycle_revision or public_confirmation_revision or compiler_first_revision or predicate_cost_revision or catalog_turn_revision or public_delta_revision:
  profiles=evidence['public_train_profiles']
  if evidence['phase']!='train' or evidence['validation_or_test_feedback_used'] is not False or len(profiles)!=64:
   raise ValueError('Completion revision requires every matched TRAIN bank trajectory')
  result['complete_train_execution_diagnosis']={arm:{field:sum(p[field]for p in profiles if p['arm']==arm)
   for field in ['actor_tokens','sdk_requests','fast_requests','compiled_actions','missing_or_ineligible_bank','invalid_banks','slow_fallbacks']}
   for arm in ['cot_only','cot_jev']}
  result['objective']=result['objective'].replace(
   'The native common prompt already covers admissibility and tool usage. Focus this skill on executable bank coverage and reliable completion, guided by the actual complete TRAIN profiles rather than changing the parent or protocol. ',
   'Use the complete execution diagnosis and all supplied public TRAIN profiles to revise completion and retrieval discipline. '
   'Successful fast actions alone do not reduce total cost when repeated retrieval, re-verification, empty banks and slow restarts dominate. '
   'Keep an explicit small set of remaining user obligations and consume each observed receipt before issuing another call. '
   'After the required mutation and its native verification succeed, remove that obligation; do not re-scan unrelated conversations, repeat a completed mutation or keep verifying the same evidence. '
   'For retrieval, use task-relevant known participants, keywords and IDs first. Reuse already observed metadata; paginate only when the receipt shows that required evidence is missing. '
   'After a root response establishes all literal arguments, bank the remaining known independent reads or required verifications together so the fast chain replaces successive full-history slow calls. '
   'A bank should contain useful future actions with decisive receipt conditions, rather than ceremonial get_current_time calls, duplicate root actions or repeated empty plan annotations. '
   'Preserve every requested continuing obligation: use native time and wait_for_notification until its specified deadline, process a genuinely new receipt once, and finish when its public completion condition is met. '
   'Do not keep waiting after a task is complete or revisit historical messages merely because they are present in context. '
   'Do not change the shared parent, policy4 compiler, scorer, native clock, required checks, native limits or fast probability mechanism. '
   'Write one concise reusable skill, with no scenario-specific IDs or answers. There are NO author tools; emit final JSON directly, never DSML or shell/tool text. ')
  result['objective']=result['objective'].replace('instruction (nonempty, at most3000 characters)',
   'instruction (nonempty; no artificial character ceiling, keep only executable guidance)')
  result['prospective_instruction_character_limit']=None
 if retrieval_chain_revision:
  profiles=evidence['public_train_profiles']
  if any('native_tool_activity'not in p for p in profiles):
   raise ValueError('Retrieval chain needs every public TRAIN tool activity summary')
  diagnosis=result['complete_train_execution_diagnosis']['cot_jev']
  result['objective']+=(
   '\nThis prospective revision addresses the complete predecessor failure: its '
   +str(diagnosis['compiled_actions'])+' fast actions still left '
   +str(diagnosis['missing_or_ineligible_bank'])+' slow restarts. The complete raw costs are in the supplied comparison. '
   +
   'Use every supplied native_tool_activity summary and public trace to locate long retrieval detours, repeated exact calls and banks that replace only one call. '
   'Author a substantially wider reusable retrieval-and-action chain, rather than repeating the prior general completion advice. '
   'When a public search has already supplied several required IDs, slow CoT must declare all justified reads with those literal IDs in one bank, using one read as root and the remaining reads as sequential candidates. '
   'Likewise, predeclare task-relevant searches with already known literal keywords across the relevant app catalogs and any independent checks whose arguments are fixed. '
   'Use the actual native schema to choose global regex/search/filter tools before unrelated historical pagination; paginate only to recover still-missing required evidence. '
   'A receipt may justify an already declared next independent read even when interpretation of the full collection waits for slow CoT after the chain. '
   'Use multiple useful same-bank labels to replace successive full-history slow calls, not a succession of root-plus-one banks. Never invent an ID or bind a future unknown argument. '
   'Declare plans only for useful eligible continuations; do not require an empty plan annotation on every ordinary slow call, and omit repetitive narration of completed work. '
   'After the declared chain finishes, slow CoT consumes all its unchanged original receipts together and handles interpretation, the remaining obligations and any new user notifications. '
   'Keep all native mutations, necessary checks, deadline handling, shared parent bytes and policy4 execution semantics unchanged. '
   'Write exactly one TRAIN-derived skill; do not hardcode the supplied cases, add tool/turn/token limits, or propose another candidate. Emit final JSON directly without author tools.')
  result['revision_identity']='public-literal-retrieval-chain/20261005-v15'
 if event_lifecycle_revision:
  profiles=evidence['public_train_profiles']
  if any('public_user_activity'not in p for p in profiles):
   raise ValueError('Event lifecycle revision requires all delivered TRAIN public user updates')
  result['objective']+=(
   '\nThe complete predecessor lost native quality (8/32 versus13/32 for both controls). '
   'Several complete paired TRAIN traces show accelerated arms finishing initial mutations before later requested follow-up notifications, while the CoT-only arm remained active and received them. '
   'Revise the event lifecycle and completion discipline first. Completing the initial action or sending the requested initial confirmation does not discharge an explicit if/when/monitor/respond obligation. '
   'Maintain separate immediate obligations and live event-triggered obligations from the actual public request. Never declare an ongoing task complete merely because the first fast chain finished. '
   'Use the unchanged native wait_for_notification tool to keep live obligations open, honoring the user-specified deadline or native environment stop; do not invent a wait ceiling, busy-poll catalogs, or prematurely return a final answer. '
   'Consume genuine notifications once, read their indicated thread or object, perform the requested follow-up with native verification, and keep any still-live obligations open. '
   'Do not predeclare a final-answer/completion notice that assumes future events cannot occur. A bank may predeclare known reads, independent mutations and necessary checks, then an explicitly justified native wait with already known literal arguments. '
   'Use actual public native tool schemas; unknown event contents and dependent writes require slow CoT after their receipt. '
   'Keep full native action receipts so the next slow turn can evaluate all original obligations, including triggered conditions and identity ambiguity. '
   'The supplied all64 public_user_activity records contain only real TRAIN user requests and notifications, with hashes, alongside every tool activity summary. Diagnose these jointly; do not treat runtime snapshots or candidate conditions as event evidence. '
   'Keep useful literal chains and task-relevant global retrieval, while removing unrelated catalog scans and repeated verifications. Preserve policy4, the common parent, native clock/scorer/limits and the same authored skill in both bank arms. '
   'Write exactly one reusable skill without copied scenario IDs, answers or new limits. Emit final JSON directly without author tools.')
  result['revision_identity']='public-event-lifecycle-bank/20261005-v16'
 if public_confirmation_revision:
  profiles=evidence['public_train_profiles']
  if any('executed_public_bridge_activity'not in p or 'public_user_activity'not in p for p in profiles):
   raise ValueError('Confirmation revision requires all64 actual public TRAIN completions and user updates')
  result['objective']+=(
   '\nThis single prospective revision follows the complete predecessor TRAIN comparison, including its quality losses. '
   'Inspect all64 executed_public_bridge_activity records jointly with actual public requests, observed receipts and the matched native success bits. '
   'These records include the final executed native tool that is absent from the preceding LLM request. They exclude judge prompts, oracle actions and private reasoning. '
   'Revise receipt-grounded user confirmations and native turn boundaries while keeping useful wide literal retrieval/action chains. '
   'A requested confirmation is an action with its own task semantics: state exactly the publicly requested outcome after its actual receipt verifies it. '
   'Keep confirmations short and factual; do not add unrelated task history, unsupported completion claims, promises, or hypothetical future outcomes to a required notice. '
   'Distinguish an initial requested notice from final completion. AgentUserInterface__send_message_to_user and SystemApp__wait_for_notification conclude the current native turn; no same-bank successor can execute after either. '
   'Place an eligible turn-concluding action last in its current chain, never declare a candidate after a turn-concluding root or candidate. '
   'Keep live follow-up obligations separate. After genuine subsequent notifications arrive, slow CoT interprets their public content and authors a new literal bank from the resulting receipts. '
   'Do not invent a tool argument to override native turn closure, suppress a required initial confirmation, or alter native clock, scoring, limits or common parent. '
   'Reuse observed IDs, retrieve only missing public evidence and predeclare independent justified reads/checks together to replace repeated full-history slow calls. '
   'Write exactly one reusable skill using only the supplied complete TRAIN public evidence; no copied case IDs, answers, oracle content or new ceilings. Emit final JSON directly without author tools.')
 result['revision_identity']='public-receipt-confirmation-bank/20261005-v17'
 if compiler_first_revision:
  profiles=evidence['public_train_profiles']
  if any('native_tool_activity'not in p or 'executed_public_bridge_activity'not in p
         or 'public_user_activity'not in p for p in profiles):
   raise ValueError('Compiler-first revision requires all64 public TRAIN execution profiles')
  diagnosis=result['complete_train_execution_diagnosis']['cot_jev']
  result['objective']=(
   'Author exactly ONE reusable compiler-first native CHISEL action-bank skill from the complete TRAIN evidence. '
   'The unchanged common parent already supplies task interpretation, app admissibility, verification, native time, confirmations and live user obligations. '
   'Keep its behavior intact. The new skill should add reliable banking mechanics and useful coverage, without duplicating or replacing those native task rules. '
   'The previous full TRAIN achieved14/32 versus13/32 native successes, but used42349097 versus28535853 complete actor tokens. '
   'Its '+str(diagnosis['sdk_requests'])+' SDK requests included '+str(diagnosis['invalid_banks'])+' invalid banks, '
   +str(diagnosis['missing_or_ineligible_bank'])+' ineligible or exhausted bank restarts, and only '
   +str(diagnosis['compiled_actions'])+' actual fast native executions. '
   'Use all64 matched public trajectories and the compiler-generated accepted declaration contract to repair these actual causes. '
   'Emit one visible CHISEL_PLAN immediately before exactly ONE registered native root tool-call in the SAME assistant response. '
   'The actual root tool-call is outside candidates. Each candidate is a FUTURE literal native action with exactly label, after, condition, tool and arguments. '
   'Do not annotate a final answer, user-facing narrative, a response with no native call, multiple simultaneous root calls, or a root with no useful justified continuation. '
   'Never repeat an executed root as a candidate. Use root or an earlier label from this SAME bank; after a chain finishes or a new user turn arrives, author a fresh bank when useful. '
   'When public task arguments or observed IDs already justify several independent required reads/checks, use one as root and declare all remaining useful actions as a same-bank literal chain. '
   'This can span the task-relevant app catalogs. Prefer useful multi-action chains over a succession of root-plus-one banks; unknown future IDs, recipients, contents and arguments require slow CoT after the original receipt. '
   'Use exact native tool names and schemas, decisive conditions on the original public receipts, and required empty kwargs. '
   'Use task-relevant native filters and batch parameters when their arguments and allowed schemas are already established; avoid unrelated catalog scans and repeated successful checks. '
   'A turn-concluding user notice or wait must be last in its chain; preserve every actual requested continuing obligation and the native environment stop. '
   'Do not manufacture extra confirmations, waits or verifications beyond the public task and common parent. Keep ordinary slow behavior available when no bank is justified; the fast judge may abstain. '
   'Keep policy4, its existing state bound, one-token selector, original parent, tools, scorer, model, clock and native limits unchanged. '
   'Count all actor input/output tokens, the slow author, invalid decisions, failures and fallbacks, and target40 percent fewer complete actor tokens than BOTH matched controls while preserving their quality. '
   'Use only the supplied complete TRAIN public evidence, never oracle/judge/private content, validation/test feedback, copied scenario answers or case IDs. '
   'There are NO author tools. Return final JSON directly with instruction (nonempty; no artificial character ceiling; retain only executable banking guidance) '
   'and supporting_task_ids (nonempty unique supplied TRAIN IDs).')
  result['revision_identity']='compiler-first-public-bank/20261005-v18'
 if predicate_cost_revision:
  profiles=evidence['public_train_profiles']
  if any('native_tool_activity'not in p or 'executed_public_bridge_activity'not in p
         or 'public_user_activity'not in p for p in profiles):
   raise ValueError('Predicate-cost revision requires all64 complete public TRAIN profiles')
  report=evidence['complete_train_comparison']['arms'];diagnosis=result['complete_train_execution_diagnosis']['cot_jev']
  result['objective']=(
   'Author exactly ONE revised reusable native-app CHISEL skill using all complete matched TRAIN public evidence. '
   'The previous compiler repair reduced invalid banks to '+str(diagnosis['invalid_banks'])+' but still required '
   +str(diagnosis['sdk_requests'])+' SDK requests, '+str(diagnosis['missing_or_ineligible_bank'])+' full-CoT restarts and '
   +str(diagnosis['compiled_actions'])+' native fast executions. Its complete actor tokens are '
   +str(report['cot_jev']['actor_tokens'])+' against '+str(report['native']['actor_tokens'])+' for native and '
   +str(report['cot_only']['actor_tokens'])+' for CoT-only. '
   'Compiler syntax is now reliable. Diagnose the long actual public TRAIN trajectories and their user requests to reduce semantic search detours and unnecessary slow calls while preserving every required outcome. '
   'Use an obligation-to-evidence discipline: maintain the currently requested public predicates, known literal arguments and missing evidence. '
   'Choose each retrieval for the specific unsatisfied predicate it can resolve. Prefer native filters and existing observed metadata over an unfiltered catalog scan. '
   'After relevant IDs and values are known, slow CoT should author a complete useful sequence of remaining independent reads, required mutations and checks as a literal bank. '
   'Do not turn one known sequence into many root-plus-one banks. Interpret the unchanged original receipts after the chain and resume slow CoT for new unknown arguments or actual user updates. '
   'Completed obligations need no further search, artifact revision or repeated verification; continuing obligations stay active until their actual public completion event or requested deadline. '
   'A new user event can change an obligation, so preserve the native environment and exact public user notifications. '
   'Abstain on missing or contradictory evidence. Never treat an observed error, an empty result or a completed tool call as proof of the requested outcome. '
   'Keep native task semantics in the unchanged common parent. Avoid duplicating its rules in the bank skill or adding ceremonial empty plans, extra confirmations, optional checks or speculative waits. '
   'Use the supplied compiler-generated contract exactly: one visible CHISEL_PLAN and one original registered root in the same response; the root is outside candidates. '
   'Every future candidate has exactly label, after, condition, tool and arguments, with known literal values, exact schemas and decisive conditions on actual public receipts. '
   'Never repeat the executed root, guess future IDs or reuse predecessor labels from a previous bank. '
   'Keep policy4, its32K public-state bound, original tools, scorer, model, parent, native clock and probability-based one-token fast judge unchanged. '
   'Target40 percent fewer complete actor tokens than BOTH controls while preserving their native success. Charge the slow author and every slow, fast, failed or fallback call. '
   'There is no cumulative token, turn, time or skill-character ceiling beyond unchanged native protocol limits. '
   'Use only these complete TRAIN public tasks, user messages and tool receipts; no private reasoning, oracle, judge prompts, validation/test feedback, copied answers or case IDs. '
   'No author tools. Emit final JSON directly with instruction (nonempty; only executable reusable guidance) and supporting_task_ids (nonempty unique supplied TRAIN IDs).')
  result['revision_identity']='public-predicate-cost-bank/20261005-v19'
 if catalog_turn_revision or public_delta_revision:
  profiles=evidence['public_train_profiles'];references=evidence.get('public_native_catalog_references',[])
  catalogs=evidence.get('public_native_tool_catalogs',{})
  if (any('native_tool_activity'not in p or 'executed_public_bridge_activity'not in p or 'public_user_activity'not in p for p in profiles)
      or not catalogs or len(references)!=64 or any(r.get('phase')!='train'or r.get('public_catalog_sha256')not in catalogs for r in references)
      or len({(r.get('task_id'),r.get('arm'))for r in references})!=64):
   raise ValueError('Catalog/turn revision needs all64 original public TRAIN catalogs and executions')
  report=evidence['complete_train_comparison'];diagnosis=result['complete_train_execution_diagnosis']['cot_jev']
  if report.get('complete')is not True or report.get('settled')!=96:
   raise ValueError('Only a complete known TRAIN comparison supplies this revision')
  result['objective']=(
   'Author exactly ONE reusable native-app CHISEL skill from the complete matched TRAIN evidence and exact agent-visible tool catalogs. '
   'Repair native task quality and costly retrieval together. The predecessor achieved '+str(report['arms']['cot_jev']['native_successes'])
   +'/32 versus '+str(report['arms']['native']['native_successes'])+'/32 native successes, with '
   +str(diagnosis['missing_or_ineligible_bank'])+' full-CoT restarts and '+str(diagnosis['compiled_actions'])+' fast executions. '
   'Use all64 public TRAIN trajectories and original schemas, not broad instructions unsupported by actual tool behavior. '
   'For each missing predicate, choose a tool whose documented searchable fields and filter semantics actually match it. '
   'Contacts search matches names, phone and email; a job, city, country or age condition may require reading contact records and complete necessary pagination instead of retrying it as a name query. '
   'Use native product-name search for a named product instead of scrolling unrelated product pages; use actual apartment filters for specified price/location/property predicates. '
   'Do not invent filter fields, increase a view limit beyond its native contract, drop eligible results, or stop exhaustive retrieval before the required population is covered. '
   'After public metadata establishes a fixed page stride, remaining offsets and relevant scope, slow CoT should declare a useful chain of native reads with literal offsets and decisive non-error receipt conditions. '
   'Likewise bank independent known-ID reads, required mutations and checks together. Preserve the original32K selector-state bound; break a large chain at genuine interpretation boundaries and consume all unchanged original receipts in slow CoT. '
   'Every known multi-action sequence should replace successive full-history slow calls where valid, rather than repeated root-plus-one banks or empty-plan narration. '
   'Repair the actual final-turn failures: native AgentUserInterface__send_message_to_user answers and closes the current user turn. '
   'Do not fast-bank an intermediate user-facing completion reply while a publicly requested conditional follow-up remains outstanding, and do not claim monitoring after terminating that turn. '
   'Use native SystemApp__wait_for_notification and actual user events to keep continuing obligations live, with the unchanged native deadline/environment stop and user-turn limits. '
   'Deliver a requested final confirmation once its public completion conditions are satisfied; preserve required notices and their contents, and distinguish a task-required initial notice from final completion using the original native semantics. '
   'Check strict inequalities, complete membership, identity and every requested mutation before confirming a result; rely on actual public receipts, never an error, an empty result or command completion alone. '
   'The same authored skill and strengthened parent are used in both bank arms. Keep policy4, original task, catalog, native scorer/model/clock/limits and one-token probability-based selector unchanged. '
   'Use the supplied compiler contract exactly: one visible CHISEL_PLAN with future candidates and one actual native root outside that array. '
   'Each candidate has exactly label, after, condition, tool and arguments with already known literal values, exact required kwargs and a predecessor in the same bank. '
   'Never repeat an executed root, guess future arguments, suppress notifications or remove necessary verification. The fast judge may abstain. '
   'Target40 percent fewer complete actor tokens than BOTH the industrial native parent and matched CoT-only control while preserving their native quality. '
   'Charge the slow author and all slow, fast, failed and fallback calls. No discretionary token, turn, time or skill-character ceilings. '
   'Use only supplied complete TRAIN public user requests, tools and native success bits; no private reasoning, oracle or judge prompts, validation/test feedback, copied case answers or IDs. '
   'Keep the skill focused on executable schema-aware retrieval, valid action chains and native turn completion. No author tools. '
   'Emit final JSON directly with instruction (nonempty) and supporting_task_ids (nonempty unique supplied TRAIN IDs).')
  result['revision_identity']='public-catalog-and-native-turn-bank/20261005-v20'
 if public_delta_revision:
  report=evidence['complete_train_comparison']['arms'];diagnosis=result['complete_train_execution_diagnosis']['cot_jev']
  result['objective']=(
   'Author exactly ONE reusable native-app CHISEL skill from this complete matched TRAIN32 comparison and all64 public trajectories. '
   'The predecessor achieved '+str(report['cot_jev']['native_successes'])+'/32 versus '+str(report['native']['native_successes'])
   +'/32 native successes, with '+str(diagnosis['sdk_requests'])+' SDK requests, '+str(diagnosis['invalid_banks'])
   +' invalid banks and '+str(diagnosis['missing_or_ineligible_bank'])+' full-CoT restarts. Its complete actor cost is '
   +str(report['cot_jev']['actor_tokens'])+' versus '+str(report['native']['actor_tokens'])+' native and '
   +str(report['cot_only']['actor_tokens'])+' CoT-only. '
   'The prior skill expanded into broad repeated task rules and forbade repeated reads even when a genuinely new notification changes an object. '
   'Replace it with focused executable banking and public-delta handling, preserving the unchanged parent and every native task requirement. '
   'Keep an obligation-to-public-evidence record: known IDs/values, unsatisfied requested predicates, pending deadline, and which specific new receipt can invalidate a completed observation. '
   'Once actual public IDs and required literal arguments are known, declare all justified independent reads or required mutations/checks as a useful same-bank chain, up to the unchanged native candidate limit. '
   'For a native notification naming one conversation or object, refresh that named object and process its genuinely new messages; do not reread all unrelated historical conversations, contacts or successfully completed mutations. '
   'A repeat read is justified when its evidence can actually change; a repeat successful mutation is not justified by an unchanged notification. '
   'Wait only for publicly requested reply/conditional/deadline obligations using native time. After those obligations finish, deliver the requested confirmation; do not add monitoring to a one-shot task. '
   'Native send_message_to_user concludes the current user turn: preserve requested initial notices and follow-ups through the documented native semantics, and never let a bank terminate before its remaining required actions. '
   'For exact time windows, evaluate the actual native timestamps and responses; respect strict versus inclusive inequalities and all required recipients/artifacts. '
   'Use each catalog schema as supplied. Search only documented fields and paginate the required population; consume earlier receipts instead of rediscovering known IDs or scanning unrelated contents. '
   'If no useful future action has already known literal arguments, make the ordinary native root call with no plan annotation. Do not emit empty plans or narrative-only banks. '
   'A useful bank has exactly one visible CHISEL_PLAN and one actual native root outside candidates, in the same assistant response. '
   'Each future candidate has exactly label, after, condition, tool and arguments; its predecessor is root or an earlier label in this SAME bank, and arguments match the exact schema including required empty kwargs. '
   'Never duplicate the root as a future candidate, refer to old-bank labels, guess IDs or merge multiple root calls. Check every declared continuation against these compiler rules before emission. '
   'Use decisive conditions on unchanged original public receipts, abstain on uncertainty, and retain ordinary slow reasoning when no bank is justified. '
   'Keep policy4, its32K public-state bound, original tool/catalog/model/scorer/clock/limits, full slow history and charged one-token probability selector unchanged. '
   'The same new authored instruction goes to both bank arms. Target40 percent fewer complete actor tokens than BOTH controls while preserving their native success. '
   'Charge authorship, failures, fast input/control tokens and all slow fallbacks. No discretionary token, turn, time or skill-character ceiling. '
   'Use only supplied complete TRAIN public evidence and native success bits; never private reasoning, oracle/judge paths, validation/test feedback, copied scenario answers or IDs. '
   'No author tools. Emit final JSON directly with instruction (nonempty; focused reusable operational guidance) and supporting_task_ids (nonempty unique supplied TRAIN IDs).')
  result['revision_identity']='public-delta-and-valid-bank/20261005-v21'
 return result
