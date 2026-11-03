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


def run_with_task_completion_recovery(session, prompt, task_complete, *, maximum_recoveries=2):
    """Opt-in execution-task guard using a host-owned public completion check.

    ``task_complete(result)`` returns True, False or None (unknown). Only an
    explicit False requests continuation. The check must use public artifacts
    or receipts, not protected benchmark answers/tests. It does not execute
    plan text or assume every conversational answer requires a tool call.
    All native turns, including recovery, belong in the physical cost ledger.
    """
    if not callable(task_complete):
        raise TypeError('A host-owned public task completion check is required')
    if type(maximum_recoveries) is not int or not 0 <= maximum_recoveries <= 2:
        raise ValueError('Use zero, one or two recovery turns within native limits')
    result = session.run(prompt)
    recoveries = 0
    complete = None
    while result.get('status') == 'completed' and not result.get('stop_requested'):
        complete = task_complete(result)
        if complete is not True and complete is not False and complete is not None:
            raise TypeError('Completion check must return True, False or None')
        if complete is not False or recoveries >= maximum_recoveries:
            break
        result = session.run(
            'The public task completion check reports that the requested work '
            'is unfinished. A written plan is not an executed action. Continue '
            'the original task through the declared native tools, respecting '
            'permissions and the remaining episode deadline. Ground completion '
            'in actual public artifacts and receipts.'
        )
        recoveries += 1
    return result, {'identity': 'public-task-completion-recovery-v1',
                    'additional_native_turns': recoveries,
                    'task_complete': complete,
                    'same_session': True, 'same_native_episode_limits': True,
                    'invented_tool_calls': 0, 'cumulative_token_limit': None}
