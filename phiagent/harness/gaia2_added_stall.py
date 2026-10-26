"""Prospective stall gate on the frozen receipt graph; public actor history only."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from phiagent.harness import gaia2_202060_progress as progress
from phiagent.harness.gaia2_jev import fingerprint


CANONICAL_GRAPH = progress.ReceiptGraph
CANONICAL_INSTALL = progress.install_adapter
POLICY = "receipt_graph_added_public_stall_v1"


@dataclass
class StallReceiptGraph(CANONICAL_GRAPH):
    require_stall: bool = False
    stalled: bool = False
    previous_feedback: str | None = None
    stall_cause: str | None = None
    public_state_updates: int = 0

    def __post_init__(self):
        if type(self.require_stall) is not bool:
            raise ValueError("The prospective stall gate must be a Boolean")
        if self.execute_followups is not True:
            raise ValueError("Both arms retain the complete canonical receipt graph")

    def observe(self, observation, *, error=False, intervened=False):
        current = (fingerprint(self.previous, observation)
                   if self.previous is not None and isinstance(observation, str) else None)
        repeated = current is not None and current == self.previous_feedback
        if not self.stalled and (error or repeated):
            self.stalled = True
            self.stall_cause = "public_error" if error else "repeated_action_and_feedback"
        self.previous_feedback = current
        self.public_state_updates += 1
        super().observe(observation, error=error, intervened=intervened)

    def continuation(self, observation, *, error, intervened, legal):
        arguments = {"error": error, "intervened": intervened, "legal": legal}
        if self.require_stall and not self.stalled:
            # Eligibility inspection must not mark an unissued edge as executed.
            preview = deepcopy(self)
            eligible, checks = CANONICAL_GRAPH.continuation(preview, observation, **arguments)
            if eligible is not None:
                return None, self.annotate(checks, eligible=True, rejected=True, candidate=eligible)
        returned, checks = super().continuation(observation, **arguments)
        return returned, self.annotate(checks, eligible=returned is not None, rejected=False)

    def annotate(self, checks, *, eligible, rejected, candidate=None):
        return {
            **checks,
            "prospective_stall_gate": {
                "policy": POLICY, "require_stall": self.require_stall,
                "latched": self.stalled, "first_stall_cause": self.stall_cause,
                "public_feedback_sha256": self.previous_feedback,
                "public_state_updates": self.public_state_updates,
                "eligible_without_stall": eligible, "rejected_only_by_stall": rejected,
                "blocked_native_response": candidate,
            },
        }


def install_adapter(config, directory, role, mode, skill):
    gate = config["prospective_stall"]
    if (gate["policy"] != POLICY or type(gate["require_stall"]) is not bool
            or config["receipt_execution"] is not True or mode != "jev" or skill != ""):
        raise ValueError("Only a gate on the intact, empty-skill canonical receipt graph is allowed")

    class BoundGraph(StallReceiptGraph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, require_stall=gate["require_stall"], **kwargs)

    progress.ReceiptGraph = BoundGraph
    CANONICAL_INSTALL(config, directory, role, mode, skill)
