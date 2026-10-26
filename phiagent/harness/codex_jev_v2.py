"""Independent Codex Jev-v2 contract; requires Codex TRAIN-derived routing.

Never use the Figure 3 retained noop result as a mechanism certificate.
"""
import hashlib,json
from phiagent.harness.codex_native_app_bank import NativeBankBoundary
def prepare_packet(packet,config):
 if config.get('phase')!='train' or config.get('harness')!='codex':
  raise ValueError('Require independent Codex TRAIN routing evidence')
 candidates=packet['candidates']
 if not 1<=len(candidates)<=10 or any(c['tool']not in config['fast_families']for c in candidates):return None
 observations=packet['public_results']
 if any(o.get('is_error')for o in observations):return None
 if len(candidates)==1:
  prefix='The exact original native result is '
  c=candidates[0]
  if not c['condition'].startswith(prefix)or c['condition'][len(prefix):]!=observations[-1]['value']['native_text']:return None
 result={'jev_v2':2,'state_id':hashlib.sha256(json.dumps(observations,sort_keys=True).encode()).hexdigest(),
  'constraints':packet['public_user_request'],'observation':observations,
  'candidates':[{'id':i,'tool':c['tool'],'args':c['arguments'],'guard':c['condition']}for i,c in enumerate(candidates)]}
 if len(json.dumps(result,ensure_ascii=False).encode())>config['max_fast_bytes']:return None
 return result
class CodexJevV2(NativeBankBoundary):
 def __init__(self,*args,v2,**kwargs):
  super().__init__(*args,prepare_packet=lambda p:prepare_packet(p,v2),**kwargs)
