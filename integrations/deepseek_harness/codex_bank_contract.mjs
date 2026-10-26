// A pure contract bridge: no model calls, tools, credentials, or benchmark reads.
import { parseBank, chatDecision, planInstruction, CHOICE_SYSTEM } from './cot_jev.mjs';
let raw='';for await(const chunk of process.stdin) raw+=chunk;
try {
 const input=JSON.parse(raw);
 const result=input.operation==='bank'
   ? parseBank(input.message,input.policy,input.tools)
   : input.operation==='decision'
     ? chatDecision(input.response,input.labels,input.model)
     : input.operation==='instruction'
       ? planInstruction(input.policy)
       : input.operation==='choice-system'
       ? CHOICE_SYSTEM + ' First test each candidate condition against the exact latest receipt. Reject a candidate if that condition is false, even if the proposed tool call is valid. For example: condition requires RED but actual receipt says BLUE => ?. Condition requires RED and actual receipt says RED => its declared label. For native app receipts, value.native_text is the exact original tool output, including literal bytes notation when present. The root object identifies the executed tool and its literal arguments. public_user_request contains only already delivered native notifications. Give ? priority over an unsupported action. Emit exactly one bare declared label or ?. Never emit CHISEL_PLAN, XML, JSON, analysis or a tool call.'
     : (()=>{throw new Error('Unknown contract operation');})();
 process.stdout.write(JSON.stringify({valid:true,result}));
} catch(error) {
 process.stdout.write(JSON.stringify({valid:false,error:error.message}));
}
