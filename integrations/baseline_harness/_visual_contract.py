"""Extracted B0 prompt and literal action contract; see SOURCE_MANIFEST.json."""

import json

NATIVE='Solve the instruction rendered in the screenshot by mouse and keyboard. Output ONE JSON object {"reasoning":"your visual geometry and execution reasoning","action":ACTION}. Before deciding, reason about the visible control rectangle and its center in normalized coordinates. For typing, the point must be INSIDE the editable box beneath the label, not on the label. Describe the bounds and use their center. For clicking use {"kind":"click","x":0.0,"y":0.0}; for entering text use {"kind":"type","x":0.0,"y":0.0,"text":"the literal required text"}. Coordinates x and y are FRACTIONS of the screenshot dimensions, strictly between 0 and 1: left=0, right=1, top=0, bottom=1. Point to the CENTER of the control. A type action already clicks/focuses and replaces the field, so no separate click is needed. Drawing/dragging uses {"kind":"drag","x":fraction,"y":fraction,"to_x":fraction,"to_y":fraction} with literal start and end coordinates from the image. Keys use {"kind":"key","key":"Tab"} or Enter/Escape/ArrowDown/ArrowUp/Backspace. Stop uses {"kind":"stop"}. Wait uses {"kind":"wait"}. Read the actual screenshot, execute only visible actions, and never repeat a completed action.'

def compile_coordinates(p,size):
 # Fixed author-time conversion only. No observation-dependent relocation.
 import math
 def convert(a):
  if a.get('kind') in ('click','type','drag'):
   for k,d in [('x',size[0]),('y',size[1])]:
    v=a[k]
    if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<1:raise ValueError('Expected normalized literal coordinate')
    a[k]=int(v*d)
   if a['kind']=='drag':
    for k,d in [('to_x',size[0]),('to_y',size[1])]:
     v=a[k]
     if type(v)not in(int,float)or not math.isfinite(v)or not 0<=v<1:raise ValueError('Invalid normalized drag endpoint')
     a[k]=int(v*d)
 convert(p['action'])
 for n in p.get('nodes',[]):
  convert(n['action']);r=n['region']
  if len(r)!=4 or any(type(v)not in(int,float)or not math.isfinite(v)or not 0<=v<=1 for v in r):raise ValueError('Expected normalized literal ROI')
  n['region']=[int(v*size[i%2])for i,v in enumerate(r)]
 return p

def parse(text):
 # Decode maximal objects, not nested dictionaries within a valid graph.
 decoder=json.JSONDecoder();candidates=[];i=0
 while i<len(text):
  if text[i]!='{':i+=1;continue
  try:
   p,end=decoder.raw_decode(text[i:]);i+=end
   if isinstance(p,dict) and ('action' in p or 'kind' in p):candidates.append(p)
  except ValueError:i+=1
 if len(candidates)!=1:raise ValueError('Expected one action-bearing JSON object')
 p=candidates[0]
 if 'kind' in p or isinstance(p.get('action'),str):
  nodes=p.pop('nodes',None);action={('kind' if k=='action' else k):v for k,v in p.items()};p={'action':action}
  if nodes is not None:p['nodes']=nodes
 return p

def action_check(a,size):
 if not isinstance(a,dict) or a.get('kind') not in ('click','type','drag','key','wait','stop'):raise ValueError('Invalid action')
 kind=a['kind']; required={'click':{'kind','x','y'},'type':{'kind','x','y','text'},'drag':{'kind','x','y','to_x','to_y'},'key':{'kind','key'},'wait':{'kind'},'stop':{'kind'}}[kind]
 if set(a)!=required:raise ValueError('Action fields must be literal and exact')
 if kind in ('click','type','drag'):
  if any(type(a[k]) is not int for k in ('x','y')) or not (0<=a['x']<size[0] and 0<=a['y']<size[1]):raise ValueError('Invalid coordinates')
 if kind=='drag' and (any(type(a[k]) is not int for k in ('to_x','to_y')) or not(0<=a['to_x']<size[0] and 0<=a['to_y']<size[1])):raise ValueError('Invalid drag endpoints')
 if kind=='type' and (not isinstance(a['text'],str) or len(a['text'])>500):raise ValueError('Invalid text')
 if kind=='key' and a['key'] not in ('Tab','Enter','Escape','ArrowDown','ArrowUp','Backspace'):raise ValueError('Invalid key')
 return json.loads(json.dumps(a))
