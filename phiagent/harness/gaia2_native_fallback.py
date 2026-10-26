"""Public-receipt compilation with an exact, unprompted native actor fallback."""
from __future__ import annotations

from copy import deepcopy
import json
import re
import time

from phiagent.harness import gaia2_202060_progress as progress
from phiagent.harness.gaia2_202060_ranges import coverage
from phiagent.harness.gaia2_read_limit_recovery import recover_read_limit


INTERRUPTIONS = {
    "task", "agent_user_interface", "environment_notifications", "hint", "task_reminder",
}


class NativeFallbackEngine(progress.ReceiptEngine):
    def __call__(self, prompt, **kwargs):
        started = time.monotonic()
        logs = self.agent.logs
        task = "\n".join(
            row.content for row in logs
            if row.get_type() == "task" and isinstance(getattr(row, "content", None), str)
        )
        reset = task != self.graph.task or self.cursor > len(logs)
        if reset:
            self.graph = progress.ReceiptGraph(
                execute_followups=self.graph.execute_followups, task=task,
            )
            self.cursor = 0
        fresh = logs[self.cursor:]
        kinds = [row.get_type() for row in fresh]
        observations = [
            row.content for row in fresh
            if row.get_type() == "observation" and isinstance(getattr(row, "content", None), str)
        ]
        errors = [
            row.get_content_for_llm()
            if callable(getattr(row, "get_content_for_llm", None)) else None
            for row in fresh if row.get_type() == "error"
        ]
        observation = "\n".join(observations)
        error = "error" in kinds or bool(re.search(
            r'(?i)(?:error:|traceback \(|"error"\s*:)', observation,
        ))
        intervened = bool(INTERRUPTIONS.intersection(kinds))
        legal = set(self.agent.action_executor.tools)
        correction = None
        if (self.graph.execute_followups and not reset and len(errors) == 1
                and not intervened and "observation" not in kinds):
            correction = recover_read_limit(
                self.graph.previous, errors[0], self.agent.action_executor.tools,
            )
        if correction is not None and progress.key(correction) not in self.graph.attempted:
            failed = deepcopy(self.graph.previous)
            self.graph.observe(errors[0], error=True)
            self.cursor = len(logs)
            text = self.graph.remember(
                "Thought: Retry the rejected read with its public pagination limit.\nAction:\n"
                + json.dumps(correction, ensure_ascii=False), legal,
            )
            checks = {
                "public_read_limit_recovery": True, "typed_native_error": True,
                "public_schema_default_matches_bound": True, "only_read_limit_changed": True,
            }
            self.record({
                "source": "receipt_graph_execution", "agent_id": self.agent.agent_id,
                "executed_response": text, "failed_action": failed,
                "native_error_sha256": progress.key(errors[0]), "checks": checks,
                "provider_response": False, "new_model_calls": 0,
            })
            return text, self._compiled_usage(started)

        self.graph.observe(observation, error=error, intervened=intervened)
        self.cursor = len(logs)
        planned, checks = self.graph.continuation(
            observation, error=error, intervened=intervened, legal=legal,
        )
        self.record({
            "source": "native_fallback_state", "agent_id": self.agent.agent_id,
            "enumeration_coverage": {
                scope: coverage(pages) for scope, pages in self.graph.pages.items()
            },
            "checks": checks, "compiled_action_available": planned is not None,
            "state_sent_to_actor": False,
        })
        if planned is not None:
            self.record({
                "source": "receipt_graph_execution", "agent_id": self.agent.agent_id,
                "executed_response": planned, "checks": checks,
                "provider_response": False, "new_model_calls": 0,
            })
            return planned, self._compiled_usage(started)

        response = self.provider(prompt, **kwargs)
        text, metadata = response if isinstance(response, tuple) else (response, {})
        if isinstance(text, str):
            # Record public edges without replacing the native model's action text.
            self.graph.remember(text, legal)
        self.record({
            "source": "receipt_graph_model", "agent_id": self.agent.agent_id,
            "original_response": text, "executed_response": text,
            "plan_edges_queued": len(self.graph.pending), "checks": checks,
            "metadata": metadata, "native_fallback_unchanged": True,
            "new_model_calls": 1, "native_messages_forwarded_by_identity": True,
            "native_kwargs_unmodified": True,
        })
        return response

    @staticmethod
    def _compiled_usage(started):
        return {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "completion_duration": time.monotonic() - started,
        }


def install_adapter(config, directory, role, mode, skill):
    progress.ReceiptEngine = NativeFallbackEngine
    progress.install_adapter(config, directory, role, mode, skill)
