import pytest

from integrations.deepseek_harness.completion_guard import (
    reasoning_only_finish,
    run_with_task_completion_recovery,
)


class Session:
    def __init__(self, results):
        self.results = iter(results)
        self.prompts = []

    def run(self, prompt):
        self.prompts.append(prompt)
        return next(self.results)


def result(text='Plan: write a file, test it, then commit.', **extra):
    return dict(status='completed', final_response=text, **extra)


def test_plan_only_final_needs_explicit_public_completion_check():
    initial = result()
    assert not reasoning_only_finish(initial, [])
    session = Session([initial, result('Executed', artifacts_exist=True)])
    final, audit = run_with_task_completion_recovery(
        session, 'Create the requested files', lambda r: r.get('artifacts_exist', False))
    assert final['artifacts_exist']
    assert audit['additional_native_turns'] == 1
    assert audit['task_complete'] is True
    assert audit['invented_tool_calls'] == 0
    assert 'declared native tools' in session.prompts[1]


@pytest.mark.parametrize('state', [True, None])
def test_complete_or_unknown_does_not_request_another_turn(state):
    session = Session([result()])
    _, audit = run_with_task_completion_recovery(session, 'Task', lambda r: state)
    assert audit['additional_native_turns'] == 0


@pytest.mark.parametrize('extra', [{'stop_requested': True}, {'status': 'cancelled'}, {'status': 'failed'}])
def test_native_stops_and_failures_are_not_overridden(extra):
    initial = result()
    initial.update(extra)
    session = Session([initial])
    def check(_):
        pytest.fail('Stopped native task must not be checked or continued')
    _, audit = run_with_task_completion_recovery(session, 'Task', check)
    assert audit['additional_native_turns'] == 0


def test_recovery_is_bounded_and_unfinished_is_reported():
    session = Session([result(), result(), result()])
    _, audit = run_with_task_completion_recovery(session, 'Task', lambda r: False)
    assert len(session.prompts) == 3
    assert audit['additional_native_turns'] == 2
    assert audit['task_complete'] is False


def test_non_boolean_completion_result_is_rejected():
    with pytest.raises(TypeError, match='True, False or None'):
        run_with_task_completion_recovery(Session([result()]), 'Task', lambda r: 'yes')
