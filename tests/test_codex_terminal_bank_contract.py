"""Native bank study prevents residual, context, population or method relabeling."""
import hashlib
from copy import deepcopy
import pytest
from phiagent.harness.codex_terminal_bank import POLICY
from phiagent.harness.codex_terminal_bank_contract import validate_trial_contract,PARTITION_SHA256
from phiagent.harness.industrial_common_prompt import PROMPT_SHA256
from scripts.run_codex_terminal import context_control

def contract():
 return {'schema':'codex-terminal-native-bank/v1','policy':POLICY,'model':'deepseek-flash',
  'cumulative_token_limit':None,'token_usage_mode':'unlimited','heldout_feedback_to_training':False,
  'native_context_trigger_tokens':996147,'source_frozen_before_execution':True,
  'partition_sha256':PARTITION_SHA256,'shared_prompt_sha256':PROMPT_SHA256,
  'residual_sha256':hashlib.sha256(b'actual shared skill').hexdigest(),
  'populations':{'train':['tr'+str(i)for i in range(18)],'validation':['v'+str(i)for i in range(18)],'test':['te'+str(i)for i in range(53)]},
  'author':{'total_tokens':17,'receipt_sha256':'bound receipt','training_only':True},
  'native_train_parent':{'task_count':18,'successes':14,'actor_tokens':61682026},
  'interface_acceptance_status':'VERIFIED_CODEX_NATIVE_TERMINAL_BANK_INTERFACE'}

def binding(value,**kwargs):
 args={'arm':'cot_jev','method':'codex_terminal_cot_jev_bank','residual':'actual shared skill','phase':'train','task_id':'tr0','seed':20260928,'partition_sha256':PARTITION_SHA256,'shared_prompt_sha256':PROMPT_SHA256}
 args.update(kwargs);return validate_trial_contract(value,**args)

def test_shared_skill_native_limit_and_actual_method_binding():
 d=contract();assert binding(d)['enabled'] is True
 assert binding(d,arm='cot_only',method='codex_terminal_cot_bank')['enabled'] is False
 assert context_control('codex_terminal_cot_jev_bank',996147,'actual shared skill','train')['active_context_trigger_tokens']==996147
 with pytest.raises(ValueError):context_control('codex_terminal_cot_jev_bank',65536,'actual shared skill','train')

@pytest.mark.parametrize('change',['wrong_method','wrong_skill','test_task_in_train','new_seed','overlap','unknown_author','validation_feedback','changed_native_context'])
def test_frozen_contract_rejects_unmatched_or_heldout_contamination(change):
 d=contract();args={}
 if change=='wrong_method':args['method']='chisel_codex'
 elif change=='wrong_skill':args['residual']='different skill'
 elif change=='test_task_in_train':args['task_id']='te0'
 elif change=='new_seed':args['seed']=20260929
 elif change=='overlap':d['populations']['test'][0]='tr0'
 elif change=='unknown_author':d['author']['total_tokens']=None
 elif change=='validation_feedback':d['heldout_feedback_to_training']=True
 else:d['native_context_trigger_tokens']=65536
 with pytest.raises(ValueError):binding(d,**args)


def test_complete_raw_costs_charge_failures_and_authorship_and_quality_gate():
 from scripts.run_codex_terminal_bank_campaign import summary
 ids=['one','two'];rows=[{'task_id':i,'arm':a,'valid':True,'native_score':int(i=='one'),'actor_tokens':1000 if a=='native'else 500,'native_fast_actions':int(a=='cot_jev')}for i in ids for a in ['native','cot_only','cot_jev']]
 result=summary(rows,ids,100)
 assert result['arms']['cot_jev']['actor_tokens']==1100
 assert result['arms']['cot_jev']['native_success_percent']==50
 assert not result['arms']['cot_jev']['registered_40_percent_goal_passed'] # same-bank CoT-only is also charged
 rows[2]['native_score']=0
 assert not summary(rows,ids,100)['arms']['cot_jev']['admission_30_percent_gate_passed']
 rows[2]['actor_tokens']=None
 assert summary(rows,ids,100)['arms']['cot_jev']['actor_tokens']is None
 with pytest.raises(ValueError,match='Duplicate'):summary(rows+[deepcopy(rows[0])],ids,100)
