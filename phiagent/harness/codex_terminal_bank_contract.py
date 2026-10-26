"""Frozen source and partition binding for a new native Codex terminal study."""
import hashlib
from phiagent.harness.codex_terminal_bank import POLICY

CORE_METHODS = {'cot_only':'codex_terminal_cot_bank',
                'cot_jev':'codex_terminal_cot_jev_bank'}
PARTITION_SHA256 = '3c857d9adb727b23061010689ae928e0670233e2c3cbba9ed73f535b068a9f40'


def validate_trial_contract(value, *, arm, method, residual, phase, task_id, seed,
                            partition_sha256, shared_prompt_sha256):
    from phiagent.harness.industrial_common_prompt import PROMPT_SHA256
    if (value.get('schema')!='codex-terminal-native-bank/v1'
            or value.get('policy')!=POLICY or value.get('model')!='deepseek-flash'
            or value.get('cumulative_token_limit')is not None
            or value.get('token_usage_mode')!='unlimited'
            or value.get('heldout_feedback_to_training')is not False
            or value.get('native_context_trigger_tokens')!=996147
            or value.get('source_frozen_before_execution')is not True):
        raise ValueError('Preserve the frozen native terminal bank contract')
    if (arm not in CORE_METHODS or CORE_METHODS[arm]!=method
            or phase not in ('train','validation','test')
            or partition_sha256!=PARTITION_SHA256
            or value.get('partition_sha256')!=partition_sha256
            or shared_prompt_sha256!=PROMPT_SHA256
            or value.get('shared_prompt_sha256')!=shared_prompt_sha256
            or hashlib.sha256(residual.encode()).hexdigest()!=value.get('residual_sha256')):
        raise ValueError('Native terminal bank identity, prompt or residual changed')
    populations=value.get('populations',{})
    if (set(populations)!=set(('train','validation','test'))
            or [len(populations[p])for p in ('train','validation','test')]!=[18,18,53]
            or len(set(sum((populations[p]for p in ('train','validation','test')),[])))!=89
            or task_id not in populations[phase]
            or seed!=20260928):
        raise ValueError('Keep the disjoint native18/18/53 single registered seed')
    author=value.get('author',{});parent=value.get('native_train_parent',{})
    if (type(author.get('total_tokens'))is not int or author['total_tokens']<=0
            or not author.get('receipt_sha256')or not author.get('training_only')
            or parent.get('task_count')!=18 or parent.get('successes')!=14
            or parent.get('actor_tokens')!=61682026
            or value.get('interface_acceptance_status')!='VERIFIED_CODEX_NATIVE_TERMINAL_BANK_INTERFACE'):
        raise ValueError('Complete capable native parent and measured slow authorship required')
    return {'arm':arm,'enabled':arm=='cot_jev','policy':POLICY,
            'implementation':'codex-native-responses-literal-terminal-bank/v2',
            'native_execution':'original pinned Codex function dispatcher',
            'paid_fast_control_tokens':1,'slow_requests_and_responses_preserved':True,
            'native_clock':'original Harbor task.toml agent timeout includes all fast/slow calls'}
