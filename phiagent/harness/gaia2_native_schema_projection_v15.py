"""Exact on-demand read-tool contracts for the native CoT/fallback actor."""
from copy import deepcopy
import hashlib
from phiagent.harness.gaia2_lazy_tool_schemas import (
    LOOKUP, read_schema_can_be_deferred, project_catalogue,
)

class NativeSchemaProjection:
    def __init__(self,agent,record):
        from are.simulation.tools import Tool
        from are.simulation.tool_box import get_tool_description_with_args
        self.agent,self.record=agent,record
        original=dict(agent.action_executor.tools)
        self.metadata_available=all(hasattr(t, "description") for t in original.values())
        if not self.metadata_available:
            return
        self.descriptions={n:get_tool_description_with_args(t) for n,t in original.items()}
        self.summaries={n:t.description for n,t in original.items()}
        self.read_only={n for n,t in original.items() if read_schema_can_be_deferred(n,
            write_operation=getattr(getattr(t,'app_tool',None),'write_operation',None))}
        self.hot=set();self.exposures=0;self.first_log=None
        projection=self
        class Lookup(Tool):
            name=LOOKUP
            description=('Return the exact original description and argument schema of a '
                         'native tool. Reads metadata only, with no scenario state access.')
            inputs={'tool_name':{'type':'string','description':'Exact native tool name'}}
            output_type='string'
            def forward(self,tool_name):
                if tool_name not in projection.descriptions:raise ValueError('Unknown native tool')
                projection.hot.add(tool_name)
                text=projection.descriptions[tool_name]
                projection.record({'source':'exact_native_schema_lookup','tool_name':tool_name,
                    'schema_sha256':hashlib.sha256(text.encode()).hexdigest(),
                    'new_model_calls':0,'native_action_iteration_consumed':True})
                return text
        if LOOKUP in original:raise ValueError('Duplicate native schema lookup')
        lookup=Lookup();lookup.validate_arguments()
        agent.action_executor.tools[LOOKUP]=lookup
        self.declaration=get_tool_description_with_args(lookup)

    def project(self,messages):
        if not self.metadata_available:
            self.record({"source":"native_schema_projection", "applied":False,
                         "reason":"native_tool_metadata_unavailable", "new_model_calls_for_projection":0})
            return messages
        first=self.agent.logs[0] if self.agent.logs else None
        if first is not self.first_log:
            self.first_log=first;self.exposures=0;self.hot.clear()
        for row in self.agent.logs:
            if row.get_type()=='tool_call':
                name=getattr(row,'tool_name',None)
                if name in self.descriptions:self.hot.add(name)
        original=deepcopy(messages)
        if self.exposures<2:
            projected,audit=messages,{'applied':False,'reason':'first_two_full_catalogue_exposures'}
        else:
            # Native prompts may already contain the added exact-lookup declaration.
            stripped=deepcopy(messages)
            stripped[0]['content']=stripped[0]['content'].replace('\n'+self.declaration,'')
            projected,audit=project_catalogue(stripped,self.descriptions,self.summaries,
                                             self.hot,self.read_only)
            if not audit['applied']:projected=messages
        if audit['applied']:
            projected[0]['content']+='\n'+self.declaration
        else:self.exposures+=1
        if messages!=original:raise ValueError('Native history was mutated')
        self.record({'source':'native_schema_projection','agent_id':self.agent.agent_id,
                     'completed_full_exposures':self.exposures,
                     'task_and_observation_messages_unchanged':projected[1:]==messages[1:],
                     'new_model_calls_for_projection':0,**audit})
        return projected
