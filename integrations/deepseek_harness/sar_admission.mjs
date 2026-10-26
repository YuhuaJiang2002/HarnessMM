import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {ExecutableSubdag, localGuard} from './sar_v2.mjs';
const hash = x => createHash('sha256').update(JSON.stringify(x)).digest('hex');
const byteUpper = x => Buffer.byteLength(typeof x === 'string' ? x : JSON.stringify(x));
export const SELECTOR_SYSTEM = 'Select an eligible literal continuation ID A-H or ?. Return exactly one character. Never invent or change any action arguments.';

export function actionType(node) {
  const command = node?.arguments?.command;
  if (node?.tool !== 'terminal' || typeof command !== 'string'
    || /\/(?:mnt\/phiagent-data|data0\/phiagent|opt\/phiagent|Users)\b/.test(node.arguments.cwd??'')
    || /\/(?:mnt\/phiagent-data|data0\/phiagent|opt\/phiagent|tests|solution)\b/.test(command)
    || /(?:rm\s+-[a-z]*r|sudo\b|docker\b|chmod\s+777|curl.*\|.*(?:sh|bash))/i.test(command)) return null;
  if (/\b(?:pytest|unittest|npm test|cargo test|go test|make check|ctest)\b/.test(command)) return 'verify';
  if (/^(?:\s*cd\s+[^;&]+\s*&&\s*)?(?:cat|head|tail|ls|find|pwd|rg|grep|sed\s+-n|stat|wc|file)\b/.test(command)) return 'inspect';
  if (/\b(?:python|python3|make|gcc|cc|cargo|npm|git|printf|echo|cp|mkdir|patch|sed)\b/.test(command)) return 'edit_or_build';
  return null;
}

export function compactState(context, candidates, episodeId) {
  const immediate = context.state.public_results.at(-1);
  return {task: episodeId, receipt_hash: hash(immediate),
    candidates:candidates.map(c=>({id:c.label,action:{tool:c.tool,arguments:c.arguments},guard:c.node.guard,
      public_guard_result:localGuard(c.node, immediate).eligible})),
    schema:{terminal:['command','cwd','timeout_sec']}};
}

export function admit(context, priors, episodeId, fullSAR=true) {
  const receipt = context.state.public_results.at(-1);
  const done = new Set(['root',...context.state.public_results.map(r=>r.call_id.split('_').at(-1))]);
  const eligible = [], rejected = [];
  const local = context.state.candidates.filter(c=>localGuard(c.node, receipt).eligible);
  const state = compactState(context, local, episodeId);
  // UTF-8 byte count is a conservative serialization estimate, not measured usage.
  const fastUpper = byteUpper(SELECTOR_SYSTEM) + byteUpper(state) + 1 + 128;
  for (const c of context.state.candidates) {
    const node = c.node, check = localGuard(node, receipt), type = actionType(node);
    const prior = priors?.categories?.[type];
    const resolved = node.args_resolved === true && node.requires_new_reasoning === false
      && node.depends_only_on_public_receipt === true && node.native_action_is_literal === true;
    const reason = !resolved ? 'unresolved_arguments' : !type ? 'not_allowlisted_or_high_risk'
      : !check.known ? 'unknown_public_receipt' : !check.eligible ? 'guard_false'
      : receipt.is_error || check.fields?.return_code !== undefined && check.fields.return_code !== 0 ? 'native_error'
      : (node.dependency_ids??[]).some(id=>!done.has(id)) ? 'dependency_missing'
      : !prior || prior.phase !== 'train' || !Number.isSafeInteger(prior.median_tokens) || prior.median_tokens <= 0 ? 'missing_train_estimate'
      : fullSAR && fastUpper * 4 > prior.median_tokens ? 'fast_cost_above_quarter_of_slow' : null;
    const decision = {candidate:c,reason,estimated_fast_total_tokens:fastUpper,
      expected_slow_calls_saved:1,estimated_replaced_slow_tokens:prior?.median_tokens??null,type};
    (reason ? rejected : eligible).push(decision);
  }
  return {eligible,rejected,state,fastUpper};
}

export class SubstitutionAwareRouter extends ExecutableSubdag {
  constructor(args) {
    const banks = new Map(), requests = new Map(), decisions = new Map();
    const episode = args.episodeId ?? 'native-task';
    const record = e => {
      if(e.kind==='sar/bank-authored')banks.set(e.bank_sha256,e.author_callback_id);
      if(e.kind==='jev/request')requests.set(e.decision_id,e);
      if(e.kind==='jev/decision')decisions.set(e.decision_id,e);
      if(e.kind==='sar/substitution-certificate') {
        const request = [...requests.values()].find(r=>r.native_attempt_id===e.bypassed_slow_callback_id);
        const decision = request && decisions.get(request.decision_id);
        e = {...e,episode_id:episode,decision_id:request?.decision_id??null,
          parent_slow_callback_id:banks.get(e.bank_sha256),literal_action:e.exact_native_tool,
          literal_args:e.exact_arguments,executed_native_receipt:e.native_receipt,
          slow_callback_eliminated:e.slow_requests_at_boundary===0,certified_substitution:1,
          selector_tokens:decision?.usage??null,fallback_occurred:false};
      }
      args.record(e);
    };
    super({...args,record});
    this.priors = args.priors ?? JSON.parse(readFileSync(process.env.PHIAGENT_SAR_TRAIN_PRIORS,'utf8'));
    if(this.priors.phase!=='train' || !this.priors.source_sha256)throw Error('Frozen TRAIN priors required');
    this.episodeId = episode;
    this.record({kind:'sar/prior-freeze',source_sha256:this.priors.source_sha256,phase:'train',
      estimate_identity:'same repaired Native TRAIN callback category median; held-out updates forbidden'});
  }
  prepare(context) {
    const a = admit(context,this.priors,this.episodeId,[8,10].includes(this.policy.version));
    this.record({kind:'sar/pre-admission',bank_sha256:context.bank.sha256,
      estimated_fast_total_tokens:a.fastUpper,eligible:a.eligible.map(x=>({id:x.candidate.label,...x,candidate:undefined})),
      rejected:a.rejected.map(x=>({id:x.candidate.label,reason:x.reason})),
      estimate_identity:'frozen same-parent TRAIN category median',threshold:'fast <= 0.25 * replaced slow',
      full_admission:[8,10].includes(this.policy.version)});
    if(!a.eligible.length){this.fallback=true;return null;}
    const candidates = a.eligible.map(x=>x.candidate);
    const state = compactState(context,candidates,this.episodeId);
    this.record({kind:'sar/fast-state',state_sha256:hash(state),serialization_bytes:byteUpper(state),
      fast_state_token_upper_bound:byteUpper(state),token_estimate_identity:'UTF8-byte estimate; actual provider input metered'});
    return {...context,state:{...context.state,candidates},selectorState:state,selectorSystem:SELECTOR_SYSTEM};
  }
}
