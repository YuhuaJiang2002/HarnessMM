"""Exact public tool semantics and the returned call for finite-choice execution."""
from copy import deepcopy
import hashlib,json


def ground_fast_messages(messages,agent,tools,fresh):
    result=deepcopy(messages)
    state=json.loads(result[1]['content'])
    for row in state['choices'].values():
        name=row['action']['action'];tool=tools[name]
        row['native_tool_semantics']={'name':name,'description':tool.description,'output_type':tool.output_type}
    calls=[row for row in fresh if row.get_type()=='tool_call']
    observations=[row for row in fresh if row.get_type()=='observation' and isinstance(getattr(row,'content',None),str)]
    if len(calls)==len(observations)==1:
        args=calls[0].tool_arguments
        if isinstance(args,str):args=json.loads(args)
        state['latest_native_return']={'actual_tool_name':calls[0].tool_name,'actual_tool_arguments':deepcopy(args),
                                      'actual_observation':observations[0].content,
                                      'observation_sha256':hashlib.sha256(observations[0].content.encode()).hexdigest(),
                                      'observed_tool_call_has_returned':True,'write_success_not_inferred':True}
    result[1]['content']=json.dumps(state,ensure_ascii=False,separators=(',',':'))
    result[0]['content']+='\nThe exact native_tool_semantics explain each available tool. latest_native_return pairs an actual native tool call with its returned public observation; it is not an issued-action prediction. This pair establishes that this call returned, while a write effect still requires its actual success evidence. For a requested read with supported exact inputs and a satisfied ordering prerequisite, select its label; its unknown future output is the reason to perform the read. Keep ? for unsupported inputs, unmet conditions, changed state, uncertain effects or interpretation requiring CoT.'
    return result
