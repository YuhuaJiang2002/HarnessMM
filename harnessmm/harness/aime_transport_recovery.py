"""Fit optimizer tool observations to a model context, retaining source files."""
import copy
import hashlib
import json
from pathlib import Path


def fit_optimizer_context(payload, count_tokens, context):
    result=copy.deepcopy(payload)
    changes=[]
    capacity=context-32
    while count_tokens(result)+result['max_tokens']>capacity:
        available=capacity-count_tokens(result)
        if available>=min(4096,result['max_tokens']):
            original=result['max_tokens']
            result['max_tokens']=available
            changes.append({'kind':'output_capacity','original_max_tokens':original,
                            'effective_max_tokens':available,'input_unchanged':True})
            break
        candidates=[(len(m.get('content') or ''),i) for i,m in enumerate(result['messages'])
                    if m.get('role')=='tool' and len(m.get('content') or '')>2048]
        if not candidates:
            raise RuntimeError('Optimizer context exceeds model capacity after tool excerpts; source instructions preserved')
        _,index=max(candidates)
        message=result['messages'][index];original=message['content']
        message['content']=original[:1600]+'\n[Earlier tool output excerpted for model context. Full evidence remains in the workspace; use grep or view_range.]'
        changes.append({'message_index':index,'original_content':original,'replacement':message['content']})
    return result,changes


def readable_training_evidence(evidence):
    if not isinstance(evidence,dict) or evidence.get('schema_version')!='phiagent-autosaddler-text-evidence/v1':
        raise ValueError('Expected the official adapter training-evidence schema')
    records=evidence.get('case_records')
    if not isinstance(records,list) or not records:
        raise ValueError('Training evidence must retain every case record')
    files={};index=[]
    for number,row in enumerate(records):
        if not isinstance(row,dict) or not isinstance(row.get('response'),str):
            raise ValueError('Training case lacks its original textual response')
        path=f'training_cases/{number:04d}.response.txt'
        response=row['response'];files[path]=response
        index.append({**{k:v for k,v in row.items() if k!='response'},
                      'response_path':path,
                      'response_sha256':hashlib.sha256(response.encode()).hexdigest(),
                      'response_characters':len(response),'response_tail':response[-512:]})
    files['training_evidence_index.json']=json.dumps(
        {**{k:v for k,v in evidence.items() if k!='case_records'},'case_records':index},
        indent=2,ensure_ascii=False)+'\n'
    return files


def autosaddler_completion(events, iterations, *, expected_train_ids=None):
    expected=set(range(iterations));completed=set();patched=set();failures=[]
    for event in events:
        payload=event['payload'];iteration=payload.get('iteration')
        if event['event_type']=='IterationCompleted':
            completed.add(iteration)
        elif event['event_type']=='SessionCompleted' and payload.get('stage')=='proposal.patch':
            patched.add(iteration)
        elif event['event_type']=='SessionFailed' and payload.get('stage')=='proposal.patch':
            failures.append({'iteration':iteration,'error':payload.get('error'),
                             'operation_id':event['operation_id']})
    # Official engine.py returns before diagnosis when every train-before case
    # succeeded. A label alone is insufficient: require its complete evaluation.
    skipped=set()
    for event in events:
        payload=event['payload']
        if event['event_type']!='IterationCompleted' or payload.get('outcome')!='no_training_failures':
            continue
        iteration=payload.get('iteration')
        for evaluation_event in events:
            if evaluation_event['event_type']!='EvaluationCompleted':
                continue
            evaluation=evaluation_event['payload'].get('evaluation', {})
            if (evaluation.get('iteration')!=iteration or evaluation.get('split')!='train'
                    or evaluation.get('purpose')!='train_before'
                    or evaluation.get('candidate_id') not in payload.get('evaluated_candidate_ids', [])):
                continue
            requested=evaluation.get('requested_case_ids', [])
            observations=evaluation.get('observations', [])
            observed=[row.get('case_id') for row in observations]
            if (requested and len(set(requested))==len(requested)
                    and len(observed)==len(requested) and set(observed)==set(requested)
                    and (expected_train_ids is None or set(requested)==set(expected_train_ids))
                    and all(row.get('disposition')=='success' for row in observations)):
                skipped.add(iteration)
    covered=patched | skipped
    return {'status':'PASS' if completed==covered==expected else 'INCOMPLETE_OPTIMIZER',
            'expected_iterations':iterations,'completed_iterations':sorted(completed),
            'completed_patch_iterations':sorted(patched),
            'missing_patch_iterations':sorted(expected-patched),
            'official_no_training_failure_iterations':sorted(skipped),
            'missing_required_iteration_evidence':sorted(expected-covered),
            'failed_patch_sessions':failures}


def skillopt_completion(root, iterations):
    root=Path(root)
    state=json.loads((root/'runtime_state.json').read_text())
    missing=[]
    for epoch in range(2,iterations+1):
        for folder,name,field in (('slow_update','slow_result.json','slow_update_content'),
                                  ('meta_skill','meta_skill_result.json','meta_skill_content')):
            path=root/folder/f'epoch_{epoch:02d}'/name
            if not path.exists() or not json.loads(path.read_text()).get(field):
                missing.append(str(path.relative_to(root)))
    return {'status':'PASS' if state['last_completed_step']==iterations and not missing else 'INCOMPLETE_OPTIMIZER',
            'last_completed_step':state['last_completed_step'],'expected_steps':iterations,
            'missing_epoch_end_content':missing,'selected_origin':state['best_origin']}
