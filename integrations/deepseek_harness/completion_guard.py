"""Prospective shared-parent recovery for an empty reasoning-only SDK finish.

This requests a new slow turn in the same episode and session. It never
creates tool calls, edits arguments or asserts task completion.
"""
RECOVERY_PROMPT = (
    'The last turn ended with reasoning but no visible answer or valid tool call. '
    'Text inside reasoning is not an executed action. Continue the original task '
    'through the declared terminal tool with literal arguments. If the work is '
    'complete, provide a visible final answer grounded in actual public receipts.'
)


def reasoning_only_finish(turn, stream_events):
    if turn.get('status') != 'completed' or turn.get('stop_requested'):
        return False
    if str(turn.get('final_response') or '').strip():
        return False
    ends = [e for e in stream_events if e.get('kind') == 'llm/end' and e.get('completed')]
    if not ends:
        return False
    attempt = ends[-1]['attempt_id']
    blocks = [e['chunk']['block'] for e in stream_events
              if e.get('attempt_id') == attempt and e.get('kind') == 'llm/chunk'
              and e.get('chunk', {}).get('type') == 'block-end']
    return bool(blocks) and all(b.get('type') == 'reasoning' for b in blocks)


def run_with_completion_recovery(session, prompt, read_stream_events, *, maximum_recoveries=2):
    """Same native session/episode clock; a bounded malformed-finish recovery."""
    if maximum_recoveries != 2:
        raise ValueError('Freeze the registered two-recovery protocol in a new run')
    result = session.run(prompt)
    recoveries = 0
    while recoveries < maximum_recoveries and reasoning_only_finish(result, read_stream_events()):
        result = session.run(RECOVERY_PROMPT)
        recoveries += 1
    return result, {'identity': 'reasoning-only-completion-recovery-v1',
                    'additional_native_turns': recoveries,
                    'same_session': True, 'same_native_episode_limits': True,
                    'invented_tool_calls': 0, 'cumulative_token_limit': None}
