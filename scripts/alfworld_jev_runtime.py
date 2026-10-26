"""Opt-in Jev HTTP transport and auditable CoT/decision interleaving."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, urlopen

from phiagent.learning import supported_local_decision as local_control
from phiagent.learning.alfworld_jev import (
    Gate, JEV_MODEL, SCHEMA, decision_request, native_action_response,
    provider_usage, routing_reason, select_decision,
    LOCAL_BACKEND, LOCAL_DEFER, LOCAL_LABELS, LocalGate, local_choice_messages, select_local_decision,
    RULE_MEMORY_MODE, RuleMemory, grounded_interaction,
    PROGRESS_RECOVERY_MODE, RECOVERY_ABLATIONS, progress_choices, progress_choice_messages,
)
from scripts import run_chisel_repair_alfworld as repair


ENDPOINT = "https://api.typesafe.ai/v1/systemone"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        return None


def readiness(protocol: Mapping, environment: Mapping | None = None) -> dict:
    if protocol.get("backend") == LOCAL_BACKEND:
        blockers = []
        try:
            LocalGate(**protocol.get("gate", {}))
        except (TypeError, ValueError):
            blockers.append("INVALID_LOCAL_GATE")
        if protocol.get("model") != "qwen35-9b@main" or protocol.get("model_revision") != "c202236235762e1c871ad0ccb60c8ee5ba337b9a":
            blockers.append("LOCAL_MODEL_NOT_MATCHED_TO_CHISEL")
        for key in ("request_byte_limit", "cot_note_characters"):
            if type(protocol.get(key)) is not int or protocol[key] < 1:
                blockers.append(f"INVALID_{key.upper()}")
        if protocol.get("logprobs_mode") != "processed_logprobs":
            blockers.append("PROCESSED_LOGPROBS_REQUIRED")
        if protocol.get("context_mode") not in (None, RULE_MEMORY_MODE, PROGRESS_RECOVERY_MODE):
            blockers.append("UNSUPPORTED_LOCAL_CONTEXT_MODE")
        ablation = protocol.get("recovery_ablation", "none")
        if (ablation not in RECOVERY_ABLATIONS
                or ablation != "none" and protocol.get("context_mode") != PROGRESS_RECOVERY_MODE):
            blockers.append("UNSUPPORTED_RECOVERY_ABLATION")
        deterministic = protocol.get("deterministic_action_selection", False)
        if (type(deterministic) is not bool
                or deterministic and protocol.get("context_mode") != PROGRESS_RECOVERY_MODE):
            blockers.append("INVALID_DETERMINISTIC_SELECTION")
        periodic = protocol.get("periodic_cot_refresh", True)
        if (type(periodic) is not bool
                or not periodic and protocol.get("context_mode") != PROGRESS_RECOVERY_MODE):
            blockers.append("INVALID_PERIODIC_COT_REFRESH")
        return {"status": "READY_FOR_LOCAL_MODEL_ADMISSION" if not blockers else "BLOCKED",
                "blockers": blockers, "backend": LOCAL_BACKEND, "typesafe_api_used": False,
                "external_credentials_required": False, "true_jev_used": False,
                "model_calls": 0, "benchmark_episodes": 0}
    environment = os.environ if environment is None else environment
    blockers = []
    if not environment.get("TYPESAFE_API_KEY", "").strip():
        blockers.append("TYPESAFE_API_KEY_MISSING")
    if protocol.get("model") != JEV_MODEL:
        blockers.append("PINNED_JEV_MODEL_MISMATCH")
    if protocol.get("endpoint") != ENDPOINT:
        blockers.append("OFFICIAL_JEV_ENDPOINT_REQUIRED")
    if protocol.get("external_public_state_transfer_authorized") is not True:
        blockers.append("PUBLIC_STATE_TRANSFER_NOT_AUTHORIZED")
    try:
        Gate(**protocol.get("gate", {}))
    except (ValueError, TypeError):
        blockers.append("INVALID_CONFIDENCE_GATE")
    for key in ("request_timeout_seconds", "request_byte_limit", "response_byte_limit",
                "maximum_jev_reserved_tokens", "cot_note_characters"):
        value = protocol.get(key)
        if type(value) is not int or value <= 0:
            blockers.append(f"INVALID_{key.upper()}")
    return {"status": "READY_FOR_AUTHENTICATED_PREFLIGHT" if not blockers else "BLOCKED",
            "blockers": blockers, "credential_present": bool(environment.get("TYPESAFE_API_KEY", "").strip()),
            "model": protocol.get("model"), "live_access_verified": False,
            "model_calls": 0, "benchmark_episodes": 0}


def access_preflight(protocol: dict, hostname: str) -> dict:
    repair.remote_only(hostname)
    checked = readiness(protocol)
    if protocol.get("backend") == LOCAL_BACKEND:
        return checked
    if checked["blockers"]:
        return checked
    api_key = os.environ["TYPESAFE_API_KEY"]
    request = Request(ENDPOINT.removesuffix("/systemone") + "/models", headers={"Authorization": "Bearer " + api_key})
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=protocol["request_timeout_seconds"]) as response:
            raw = response.read(protocol["response_byte_limit"] + 1)
        if len(raw) > protocol["response_byte_limit"] or api_key.encode() in raw:
            raise ValueError("Invalid model-list response")
        models = json.loads(raw).get("models")
        if not isinstance(models, list) or not models:
            raise ValueError("No accessible TypeSafe models")
        names = [row["name"] for row in models if isinstance(row, dict) and isinstance(row.get("name"), str)]
        if not {protocol["model"], "jev-latest", "jev-preview"}.intersection(names):
            raise ValueError("Account does not list Jev access")
        return {**checked, "status": "READY_FOR_REAL_MODEL_ADMISSION", "model_list_access_verified": True,
                "available_model_names": names, "versioned_inference_access_verified": False,
                "api_requests": 1, "model_calls": 0}
    except Exception as error:
        if isinstance(error, HTTPError):
            error.close()
        return {**checked, "status": "BLOCKED", "blockers": ["JEV_ACCESS_PREFLIGHT_FAILED"],
                "exception_type": type(error).__name__,
                "http_status": error.code if isinstance(error, HTTPError) else None,
                "api_requests": 1, "model_calls": 0}


def jev_call(payload: dict, output: Path, protocol: dict, hostname: str) -> dict:
    repair.remote_only(hostname)
    checked = readiness(protocol)
    if checked["blockers"]:
        raise RuntimeError("Jev preflight blocked: " + ",".join(checked["blockers"]))
    if payload.get("model") != protocol["model"]:
        raise ValueError("Jev payload model differs from frozen protocol")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    if len(encoded) > protocol["request_byte_limit"]:
        raise ValueError("Jev request exceeds the fixed byte allowance; never silently truncate")
    output.mkdir(parents=True, exist_ok=False)
    repair.write_json(output / "request.json", {"endpoint": ENDPOINT, "stage": "jev", "body": payload})
    started = time.monotonic()
    api_key = os.environ["TYPESAFE_API_KEY"]
    request = Request(ENDPOINT, encoded, {"Content-Type": "application/json", "Authorization": "Bearer " + api_key})
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=protocol["request_timeout_seconds"]) as response:
            raw = response.read(protocol["response_byte_limit"] + 1)
        if len(raw) > protocol["response_byte_limit"]:
            raise ValueError("Jev response exceeds the frozen byte allowance")
        text = raw.decode("utf-8")
        if api_key in text:
            raise ValueError("Unexpected credential reflection; response withheld")
        (output / "response.json").write_text(text, encoding="utf-8")
        value = json.loads(text)
        usage = provider_usage(value)
        repair.write_json(output / "accounting.json", {
            "provider": "typesafe", "model": value.get("model"), "usage": usage,
            "calls": 1, "valid": True, "elapsed_seconds": time.monotonic() - started,
            "output_tokens_charged_to_token_total_even_when_free": True,
        })
        return value
    except Exception as error:
        if isinstance(error, HTTPError):
            try:
                body = error.read(protocol["response_byte_limit"]).decode(errors="replace")
                (output / "http_error_body.txt").write_text(body.replace(api_key, "[REDACTED]"), encoding="utf-8")
            finally:
                error.close()
        repair.write_json(output / "failure.json", {
            "exception_type": type(error).__name__,
            "http_status": error.code if isinstance(error, HTTPError) else None,
            "cost_unknown": not (output / "accounting.json").exists(), "retries": 0,
            "elapsed_seconds": time.monotonic() - started,
        })
        raise


def require_local_binding(binding: Mapping) -> None:
    if binding["model"]["id"] != "qwen35-9b@main" or binding["model"]["revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a":
        raise ValueError("Local decision and CoT must share the exact frozen CHISEL model")
    for endpoint in binding["model"]["endpoints"]:
        parsed = urlparse(endpoint)
        if (parsed.scheme not in ("http", "https") or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Local discriminator may only call the owned loopback model service")


def bind_local_labels(binding: dict, output: Path) -> dict[str, int]:
    require_local_binding(binding)
    endpoint = binding["model"]["endpoints"][0].removesuffix("/v1") + "/tokenize"
    tokens = {}
    receipts = []
    for label in LOCAL_LABELS + LOCAL_DEFER:
        payload = {"model": binding["model"]["id"], "prompt": label, "add_special_tokens": False}
        request = Request(endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=30) as response:
                value = json.load(response)
        except HTTPError as error:
            error.close()
            raise
        ids = value.get("tokens")
        if (value.get("count") != 1 or not isinstance(ids, list) or len(ids) != 1
                or type(ids[0]) is not int or ids[0] < 0 or ids[0] in tokens.values()):
            raise ValueError("Each local action label must be one distinct native token")
        tokens[label] = ids[0]
        receipts.append({"label": label, "request": payload, "response": value})
    repair.write_json(output, {"model": binding["model"], "label_token_ids": tokens,
                              "metadata_requests": len(tokens), "model_inference_calls": 0,
                              "receipts": receipts})
    return tokens


def episode_ledger(output: Path) -> dict:
    requests = sorted((output / "providers").glob("*/*/request.json"))
    accounted = sorted((output / "providers").glob("*/*/accounting.json"))
    by_provider = {name: {"calls": 0, "accounted_calls": 0, "usage": dict.fromkeys(repair.USAGE_KEYS, 0)}
                   for name in ("cot", "jev", "local")}
    for filename in requests:
        by_provider[filename.parent.name]["calls"] += 1
    for filename in accounted:
        value = repair.read_json(filename)
        usage = value["usage"]
        if (any(type(usage.get(key)) is not int or usage[key] < 0 for key in repair.USAGE_KEYS)
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
            raise ValueError("Invalid hybrid provider accounting")
        row = by_provider[filename.parent.name]
        row["accounted_calls"] += 1
        for key in repair.USAGE_KEYS:
            row["usage"][key] += usage[key]
    complete = len(requests) == len(accounted) and all(
        (filename.parent / "accounting.json").exists() for filename in requests)
    return {"usage": {key: sum(row["usage"][key] for row in by_provider.values()) for key in repair.USAGE_KEYS},
            "calls": len(requests), "attempted_calls": len(requests), "accounted_calls": len(accounted),
            "cost_complete": complete, "provider_usage": by_provider,
            "router_model_calls": by_provider["jev"]["calls"] + by_provider["local"]["calls"],
            "actor_cot_calls": by_provider["cot"]["calls"],
            "provider_model_ids": {"cot": "binding.model.id", "jev": JEV_MODEL,
                                   "local": "binding.model.id"}}


class InterleavedActor:
    def __init__(self, protocol: dict, *, cot, jev):
        self.protocol = protocol
        self.local = protocol.get("backend") == LOCAL_BACKEND
        self.gate = LocalGate(**protocol["gate"]) if self.local else Gate(**protocol["gate"])
        self.cot = cot
        self.jev = jev
        self.last_cot_note = ""
        self.consecutive_fast_steps = 0
        self.memory_mode = self.local and protocol.get("context_mode") == RULE_MEMORY_MODE
        self.rule_memory = None
        self.progress_mode = self.local and protocol.get("context_mode") == PROGRESS_RECOVERY_MODE
        self.recovery_ablation = protocol.get("recovery_ablation", "none")
        if self.recovery_ablation not in RECOVERY_ABLATIONS or self.recovery_ablation != "none" and not self.progress_mode:
            raise ValueError("Recovery ablations require the typed public-progress configuration")
        self.deterministic_action_selection = protocol.get("deterministic_action_selection", False)
        if (type(self.deterministic_action_selection) is not bool
                or self.deterministic_action_selection and not self.progress_mode):
            raise ValueError("Deterministic action selection requires public-progress recovery")
        self.periodic_cot_refresh = protocol.get("periodic_cot_refresh", True)
        if (type(self.periodic_cot_refresh) is not bool
                or not self.periodic_cot_refresh and not self.progress_mode):
            raise ValueError("Disabling periodic CoT refresh requires public-progress recovery")
        self.initial_observation = None
        self.recovery_active = False

    def memory_for(self, skill: str) -> RuleMemory:
        if self.rule_memory is None:
            self.rule_memory = RuleMemory(skill)
        if self.rule_memory.source != skill:
            raise ValueError("Source rule memory cannot change within an episode")
        return self.rule_memory

    def step(self, *, turn: int, task: str, prompt: str, skill: str, observation: str,
             actions: list[str], history: list[dict], output: Path) -> dict:
        reason = routing_reason(turn=turn, actions=actions, history=history, last_cot_note=self.last_cot_note,
                                consecutive_fast_steps=self.consecutive_fast_steps, gate=self.gate)
        if self.local:
            if len(actions) > len(LOCAL_LABELS):
                reason = "cot_local_label_capacity"
            elif reason == "jev_candidate":
                reason = "local_candidate"
        progress = None
        if self.progress_mode:
            if turn == 0:
                self.initial_observation = observation
            elif self.initial_observation is None:
                raise ValueError("Typed recovery requires the recorded initial public observation")
            else:
                progress = progress_choices(task=task, initial_observation=self.initial_observation,
                                            actions=actions, history=history)
                if (reason in ("cot_repeated_action_and_feedback", "cot_public_tool_error")
                        or progress.get("no_progress_actions", 0) >= 2
                        or self.recovery_ablation == "no_stall_gate"):
                    self.recovery_active = True
                if self.recovery_ablation == "no_progress_filter" and progress["eligible"]:
                    progress = {**progress, "allowed_actions": list(actions), "reason": "unfiltered_native_actions"}
            if not self.recovery_active:
                reason = "cot_before_public_stall"
            elif (reason != "cot_periodic_refresh" or self.recovery_ablation == "no_periodic_refresh"
                    or not self.periodic_cot_refresh):
                if progress["eligible"] and len(progress["allowed_actions"]) <= len(LOCAL_LABELS):
                    reason = "local_candidate"
                else:
                    reason = progress["reason"] if not progress["eligible"] else "cot_progress_label_capacity"
        memory = self.memory_for(skill) if self.memory_mode else None
        routine = None
        retrieved = []
        if memory is not None and reason == "local_candidate":
            routine = grounded_interaction(task=task, observation=observation, actions=actions, history=history)
            if routine["eligible"]:
                retrieved = memory.retrieve(routine["stage"])
                if not retrieved:
                    reason = "cot_memory_no_relevant_rule"
            else:
                reason = routine["reason"]
        usage = dict.fromkeys(repair.USAGE_KEYS, 0)
        decision = {"schema_version": SCHEMA, "turn": turn, "gate": asdict(self.gate),
                    "pre_route": reason, "cot_calls": 0, "jev_calls": 0,
                    "thresholds_calibrated_on_alfworld": False}
        if self.local:
            decision.update(backend=LOCAL_BACKEND, local_calls=0, true_jev_used=False,
                            typesafe_api_used=False, calibration_claimed=False)
        if memory is not None:
            decision["rule_memory"] = {"source_sha256": memory.source_sha256,
                                       "retrieved_rule_ids": [entry["id"] for entry in retrieved],
                                       "prefilter": routine}
        if self.progress_mode:
            decision["typed_progress"] = {"recovery_active": self.recovery_active, "public_constraint": progress}
        supported = reason in ("jev_candidate", "local_candidate")
        deterministic = (
            self.recovery_ablation == "no_typed_selector" or self.deterministic_action_selection
        ) and reason == "local_candidate"
        candidate_actions = progress["allowed_actions"] if self.progress_mode and supported else actions
        support = local_control.SupportedCandidates(
            supported=supported, actions=tuple(candidate_actions), reason=reason,
        )
        stage = "local" if self.local else "jev"

        def select(_actions: tuple[str, ...]) -> str | None:
            nonlocal reason
            if self.local:
                if self.progress_mode:
                    messages, choices = progress_choice_messages(task=task, skill=skill, observation=observation,
                                                                 progress=progress)
                else:
                    messages, choices = local_choice_messages(task=task, skill=skill, observation=observation,
                        actions=actions, history=history, last_cot_note=self.last_cot_note,
                        rule_memory=memory, routine=routine)
                request = {"model": self.protocol["model"], "messages": messages, "choices": choices}
            else:
                request = decision_request(task=task, skill=skill, observation=observation, actions=actions, history=history,
                                           last_cot_note=self.last_cot_note, model=self.protocol["model"])
            byte_count = len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode())
            if byte_count > self.protocol["request_byte_limit"]:
                reason = "cot_local_request_too_large" if self.local else "cot_jev_request_too_large"
                decision["pre_route"] = reason
                return None
            response = self.jev(request, output / stage)
            if self.local:
                selected = select_local_decision(response, request["choices"], expected_model=self.protocol["model"], gate=self.gate)
                if self.progress_mode:
                    selected = {**selected, "probability_scope": (
                        "next_token_distribution_over_complete_native_actions" if self.recovery_ablation == "no_progress_filter"
                        else "next_token_distribution_over_public_progress_filtered_native_actions")}
                if memory is not None and selected["route"] == "local" and selected["action"] != routine["grounded_action"]:
                    selected = {**selected, "route": "cot", "action": None,
                                "proposed_action": selected["action"], "reason": "local_choice_failed_code_prerequisite"}
            else:
                selected = select_decision(response, request, gate=self.gate)
            decision[stage + "_calls"] = 1
            decision[stage + "_decision"] = selected
            for key in usage:
                usage[key] += selected["usage"][key]
            reason = selected["reason"]
            return selected["action"] if selected["route"] in ("jev", "local") else None

        def commit(action: str) -> dict:
            self.consecutive_fast_steps += 1
            if deterministic:
                return {"text": native_action_response(action), "usage": usage,
                        "decision": {**decision, "route": "deterministic",
                                     "reason": "first_progress_native_action",
                                     "deterministic_action": action}}
            return {"text": native_action_response(action), "usage": usage,
                    "decision": {**decision, "route": stage, "reason": reason}}

        def fallback() -> dict:
            text, cot_usage = self.cot(prompt, output / "cot")
            for key in usage:
                usage[key] += cot_usage[key]
            self.consecutive_fast_steps = 0
            action = re.search(r"<action>(.*?)</action>", text, re.S)
            self.last_cot_note = text[:action.start()].strip()[-self.protocol["cot_note_characters"]:] if action else ""
            return {"text": text, "usage": usage,
                    "decision": {**decision, "route": "cot", "reason": reason, "cot_calls": 1,
                                 "note_is_partial_actor_deliberation_not_observed_fact": True}}

        outcome = local_control.execute_supported_decision(
            support, select=None if deterministic else select, commit=commit, fallback=fallback,
        )
        outcome.value["decision"]["supported_local_decision"] = outcome.trace.to_dict()
        return outcome.value


class EpisodeDriver:
    def __init__(self, *, binding: dict, budget: dict, protocol: dict, output: Path, seed: int, cot_provider=None):
        checked = readiness(protocol)
        if checked["blockers"]:
            raise RuntimeError("Jev preflight blocked: " + ",".join(checked["blockers"]))
        self.binding = binding
        self.budget = budget
        self.protocol = protocol
        self.output = output
        self.seed = seed
        if cot_provider is None:
            from scripts.run_alfworld_modular_rsi import PROVIDER_CALL

            cot_provider = PROVIDER_CALL
        self.cot_provider = cot_provider
        local = protocol.get("backend") == LOCAL_BACKEND
        if local:
            require_local_binding(binding)
            if set(binding.get("local_label_token_ids", {})) != set(LOCAL_LABELS + LOCAL_DEFER):
                raise ValueError("Local action-label token identity was not verified")
        self.actor = InterleavedActor(protocol, cot=self._cot, jev=self._local if local else self._jev)
        self.memory_recorded = False

    def _charged(self, output: Path, amount: int, invoke):
        from scripts.run_alfworld_modular_rsi import locked_ledger, reserve_request, settle_request

        ledger = Path(self.binding["ledger"])
        with locked_ledger(ledger) as state:
            reserve_request(state, amount, current_time=time.time())
        usage = None
        try:
            return invoke()
        finally:
            filename = output / "accounting.json"
            if filename.exists():
                usage = repair.read_json(filename)["usage"]
            with locked_ledger(ledger) as state:
                settle_request(state, amount, usage)

    def _cot(self, prompt: str, output: Path):
        from scripts.run_alfworld_modular_rsi import native_count

        messages = [{"role": "system", "content": repair.SYSTEM}, {"role": "user", "content": prompt}]
        count = native_count(self.binding, messages=messages)
        amount = count + self.budget["max_completion_tokens"]
        if amount > self.binding["max_model_len"]:
            raise ValueError("Unchanged CoT prompt exceeds the frozen context allowance")
        def invoke():
            text, usage = self.cot_provider(self.binding, endpoint_index=0, messages=messages,
                max_tokens=self.budget["max_completion_tokens"], seed=self.seed, output=output,
                timeout=self.budget["request_timeout_seconds"], stage="cot")
            if usage["prompt_tokens"] != count:
                raise ValueError("Actual CoT usage differs from native tokenizer")
            return text, usage
        return self._charged(output, amount, invoke)

    def _jev(self, payload: dict, output: Path):
        return self._charged(output, self.protocol["maximum_jev_reserved_tokens"],
            lambda: jev_call(payload, output, self.protocol, self.binding["hostname"]))

    def _local(self, payload: dict, output: Path):
        from scripts.run_alfworld_modular_rsi import native_count

        count = native_count(self.binding, messages=payload["messages"])
        if count + 1 > self.binding["max_model_len"]:
            raise ValueError("Local decision exceeds the frozen context length")
        tokens = [self.binding["local_label_token_ids"][label] for label in payload["choices"]]
        def invoke():
            _, usage = self.cot_provider(self.binding, endpoint_index=0, messages=payload["messages"],
                max_tokens=1, seed=self.seed, output=output, timeout=self.budget["request_timeout_seconds"],
                stage="local", decision_token_ids=tokens)
            if usage["prompt_tokens"] != count:
                raise ValueError("Local decision native input tokens differ from charged usage")
            repair.write_json(output / "choice-map.json", {"choices": payload["choices"], "token_ids": tokens,
                "backend": LOCAL_BACKEND, "true_jev_used": False, "calibration_claimed": False})
            return repair.read_json(output / "response.json")
        return self._charged(output, count + 1, invoke)

    def step(self, **kwargs) -> dict:
        turn = kwargs["turn"]
        if self.actor.memory_mode and not self.memory_recorded:
            repair.write_json(self.output / "RULE_MEMORY.json", self.actor.memory_for(kwargs["skill"]).record())
            self.memory_recorded = True
        result = self.actor.step(**kwargs, output=self.output / "providers" / f"{turn:03d}")
        repair.write_json(self.output / "decisions" / f"{turn:03d}.json", {
            **result["decision"], "usage": result["usage"], "native_actions": kwargs["actions"],
            "native_prompt_sha256": repair.digest(kwargs["prompt"]),
            "native_prompt": kwargs["prompt"], "public_observation": kwargs["observation"],
            "skill_sha256": repair.digest(kwargs["skill"]),
        })
        return result

    def ledger(self) -> dict:
        result = episode_ledger(self.output)
        result["provider_model_ids"]["cot"] = self.binding["model"]["id"]
        result["provider_model_ids"]["local"] = self.binding["model"]["id"]
        if self.protocol.get("backend") == LOCAL_BACKEND:
            result["provider_model_ids"].pop("jev")
            result.update(true_jev_used=False, typesafe_api_used=False,
                          discriminator_model=self.binding["model"]["id"])
        return result
