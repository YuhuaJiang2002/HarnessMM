#!/usr/bin/env python3
"""Independently verify a completed training pilot from saved raw response bytes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time

from scripts.prepare_chisel_component_quality_data import canonical_hash, write_json
from scripts.run_compact_harness_training_pilot import (
    BASELINES,
    CANDIDATES,
    atomic_json,
    checked,
    controlled_context,
    require_cpu_client,
    scorer_runtime_mode,
    sha,
    validate_model,
)
from scripts.run_chisel_component_quality import build_messages, official_scorer, score_rows, strict_response


def await_retirement(root: Path, observer: Path) -> None:
    require_cpu_client()
    observer.mkdir(parents=True, exist_ok=False)
    campaign = root.parent
    config = json.loads((root / "config.json").read_text())
    binding = json.loads((campaign / "binding.admitted.json").read_text())
    deadline = datetime.fromisoformat(binding["allocation_deadline_utc"]) + timedelta(minutes=5)
    if deadline.tzinfo is None:
        raise RuntimeError("An absolute timezone-aware retirement deadline is required")
    write_json(observer / "STARTED.json", {
        "hostname": socket.gethostname(), "pid": os.getpid(), "command": sys.argv,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "deadline_utc": deadline.isoformat(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_sha256": sha(Path(__file__)), "config_sha256": sha(root / "config.json"),
        "seed": config["model"]["request_seed"], "model_calls": 0, "quality_outcomes_inspected": False,
    })
    tick = threading.Event()
    next_capacity = 0.0
    try:
        while True:
            if any(path.exists() for path in (root / "FAILURE.json", root / "CLIENT_STOPPED.json", campaign / "RUN_FAILURE.json")):
                raise RuntimeError("Generation failure evidence exists; do not score or restart a partial matrix")
            if datetime.now(timezone.utc) >= deadline:
                raise RuntimeError("Retirement observation deadline expired; no audit launched")
            retirement = campaign / "RETIREMENT.json"
            if retirement.is_file():
                retired = json.loads(retirement.read_text())
                if retired["status"] != "RETIRED" or retired["owned_live_processes"] or not retired["leases_released"]:
                    raise RuntimeError("Owned service retirement is not verified")
                drained = checked(root / "GENERATION_DRAINED.json")
                if (not drained["generation_complete"] or drained["pending_http"] != 0
                        or drained["unknown_cost_requests"] != 0 or not (root / "RESULT.json").exists()):
                    raise RuntimeError("Service retired without a complete valid result")
                write_json(observer / "READY.json", {
                    "at_utc": datetime.now(timezone.utc).isoformat(), "completed": drained["completed"],
                    "retirement_sha256": sha(retirement), "result_sha256": sha(root / "RESULT.json"),
                    "new_model_calls": 0,
                })
                print("COMPLETE_GENERATION_AND_RETIREMENT_OBSERVED_STARTING_INDEPENDENT_AUDIT", flush=True)
                return
            if time.monotonic() >= next_capacity:
                gpu_command = ["nvidia-smi", "-i", str(binding["physical_gpu_index"]),
                               "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader"]
                write_json(observer / f"GPU-{time.monotonic_ns()}.json", {
                    "at_utc": datetime.now(timezone.utc).isoformat(), "command": gpu_command,
                    "stdout": subprocess.check_output(gpu_command, text=True, timeout=30),
                    "model_calls": 0,
                })
                next_capacity = time.monotonic() + 300
            heartbeat = json.loads((root / "HEARTBEAT.json").read_text())
            snapshot = {key: heartbeat[key] for key in (
                "at_utc", "attempted", "completed", "pending_http", "known_real_tokens", "unknown_cost_requests",
            )}
            atomic_json(observer / "HEARTBEAT.json", {**snapshot, "state": "AWAITING_COMPLETE_GENERATION_AND_RETIREMENT"})
            print(json.dumps(snapshot), flush=True)
            tick.wait(30)
    except BaseException as error:
        write_json(observer / "FAILURE.json", {"at_utc": datetime.now(timezone.utc).isoformat(), "error": repr(error), "new_model_calls": 0})
        raise


def independent_eligible(methods: dict, config: dict) -> list[str]:
    candidates = tuple(config.get("candidate_arms", CANDIDATES))
    if (not candidates or len(set(candidates)) != len(candidates)
            or set(candidates) & set(BASELINES)
            or config["arms"] != list(BASELINES + candidates)
            or set(methods) != set(config["arms"])):
        raise RuntimeError("The complete declared baseline/candidate inventory is required")
    references = tuple(config.get("comparison_arms", BASELINES))
    if (not references or len(set(references)) != len(references)
            or not set(BASELINES) <= set(references) <= set(methods)):
        raise RuntimeError("Invalid frozen comparison references")
    return [
        candidate for candidate in candidates
        if all(
            methods[candidate][metric] > methods[baseline][metric] + 1e-12
            for baseline in references if baseline != candidate for metric in ("exact_match", "f1")
        ) and all(
            methods[candidate]["total_tokens"] < methods[baseline]["total_tokens"]
            for baseline in references if baseline != candidate
        )
    ]


def diagnostic_plan(config: dict, candidate: str) -> dict:
    if config.get("context_selection"):
        statistics = config["statistics"]
        references = [arm for arm in config["arms"] if arm != candidate]
        if (candidate != config["primary_candidate"]
                or candidate != statistics["primary_candidate"]
                or statistics["primary_references"] != references
                or statistics["primary_metrics"] != ["exact_match", "f1", "tokens"]
                or statistics["factorial_quality_metrics"] != ["exact_match", "f1"]):
            raise RuntimeError("Context diagnostics must retain all comparisons and frozen endpoints")
        primary = [("primary", candidate, reference, metric)
                   for reference in references for metric in statistics["primary_metrics"]]
        factorial = [(name, pair[0], pair[1], metric)
                     for name, pair in config["factorial"]["contrasts"].items()
                     for metric in statistics["factorial_quality_metrics"]]
        unique = {}
        for entry in primary + factorial:
            _name, left, right, metric = entry
            if left == right or left not in config["arms"] or right not in config["arms"]:
                raise RuntimeError("Invalid context factorial comparison")
            unique.setdefault((left, right, metric), entry)
        if len(unique) != statistics["simultaneous_bounds"]:
            raise RuntimeError("Context diagnostic family differs from the preregistered unique endpoints")
        return {
            "primary": primary, "factorial": factorial, "references": references,
            "contrasts": list(unique.values()), "family_size": len(unique),
            "alpha": statistics["familywise_alpha"],
        }
    if config.get("rule_compression"):
        statistics = config["statistics"]
        references = statistics["primary_references"]
        metrics = statistics["primary_metrics"]
        if (candidate != config["primary_candidate"] or candidate != statistics["primary_candidate"]
                or references != list(BASELINES) + [config["retained_method"], "no_skill_answer_only"]
                or metrics != ["exact_match", "f1", "tokens"] or statistics["secondary_metrics"] != metrics):
            raise RuntimeError("Compression diagnostics require all fixed references and the simple answer-only control")
        primary = [("primary", candidate, reference, metric) for reference in references for metric in metrics]
        secondary = [(name, pair[0], pair[1], metric)
                     for name, pair in statistics["secondary_contrasts"].items() for metric in metrics]
        contrasts = primary + secondary
        identities = {(left, right, metric) for _name, left, right, metric in contrasts}
        if (len(identities) != len(contrasts) or len(contrasts) != statistics["simultaneous_bounds"]
                or any(left == right or left not in config["arms"] or right not in config["arms"]
                       for left, right, _metric in identities)):
            raise RuntimeError("The complete compression diagnostic family is required")
        return {"primary": primary, "factorial": secondary, "references": references, "contrasts": contrasts,
                "family_size": len(contrasts), "alpha": statistics["familywise_alpha"]}
    if config.get("cost_source_design"):
        statistics = config["statistics"]
        if candidate != config["primary_candidate"]:
            raise RuntimeError("Only the prospectively declared primary candidate may use this family")
        named = config["cost_source_design"]["primary_contrasts"]
        pairs = {tuple(pair): name for name, pair in named.items()}
        if len(pairs) != len(named):
            raise RuntimeError("Duplicate mechanism contrasts are not separate tests")
        for left in statistics["quality_candidates"]:
            for right in statistics["quality_references"]:
                if left != right:
                    pairs.setdefault((left, right), "quality_floor")
        contrasts = [(name, left, right, metric) for (left, right), name in pairs.items()
                     for metric in statistics["effect_metrics"]]
        if (statistics["effect_metrics"] != ["exact_match", "f1", "tokens"]
                or any(left not in config["arms"] or right not in config["arms"] or left == right for left, right in pairs)
                or len(contrasts) != statistics["simultaneous_bounds"]):
            raise RuntimeError("The complete preregistered cost-source confidence family is required")
        return {"primary": contrasts, "factorial": [], "contrasts": contrasts,
                "references": statistics["quality_references"], "family_size": len(contrasts),
                "alpha": statistics["familywise_alpha"]}
    declared = config.get("statistics")
    if declared is None:
        references, metrics, alpha = list(BASELINES), ["exact_match", "f1", "tokens"], 0.05
        factorial = []
    else:
        if candidate != declared["primary_candidate"]:
            raise RuntimeError("Only the prospectively declared primary candidate may use this family")
        references, metrics, alpha = declared["primary_references"], declared["primary_metrics"], declared["familywise_alpha"]
        if references != list(BASELINES) + [config["retained_method"]] or metrics != ["exact_match", "f1", "tokens"]:
            raise RuntimeError("The complete primary references and endpoints are required")
        factorial = [
            (name, pair[0], pair[1], metric)
            for name, pair in config["factorial"]["contrasts"].items()
            for metric in declared["factorial_quality_metrics"]
        ]
    primary = [("primary", candidate, reference, metric) for reference in references for metric in metrics]
    all_contrasts = primary + factorial
    keys = [(left, right, metric) for _name, left, right, metric in all_contrasts]
    if (len(keys) != len(set(keys)) or any(
        left not in config["arms"] or right not in config["arms"] or left == right
        or metric not in ("exact_match", "f1", "tokens") for left, right, metric in keys
    )):
        raise RuntimeError("Duplicate or invalid declared contrast")
    if declared is not None and len(all_contrasts) != declared["simultaneous_bounds"]:
        raise RuntimeError("The complete preregistered confidence family is required")
    return {"primary": primary, "factorial": factorial, "references": references,
            "contrasts": all_contrasts, "family_size": len(all_contrasts), "alpha": alpha}


def context_receipt_roundoff(frozen: dict, replayed: dict) -> float:
    """Tolerate only a legacy informational float, never input or decision differences."""
    if frozen == replayed:
        return 0.0
    field = "lexical_query_coverage"
    if (frozen.get("schema_version") != "phiagent-qa-context-selection/v1"
            or frozen.get("numeric_reduction") is not None
            or replayed.get("numeric_reduction") is not None
            or {key: value for key, value in frozen.items() if key != field}
            != {key: value for key, value in replayed.items() if key != field}):
        raise RuntimeError("Material or canonical context receipt fields differ")
    left, right = frozen.get(field), replayed.get(field)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12
        for value in (left, right)
    ) or not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Legacy coverage annotation differs beyond floating-point roundoff")
    return abs(left - right)


def verify_context_sources(
    source: Path, origin: dict, planned: list[dict], public: list[dict], config: dict, spec: dict,
) -> dict:
    from transformers import AutoTokenizer
    from phiagent.learning import qa_context_evidence as context_module
    from phiagent.learning.qa_context_evidence import reconstruct_context

    for relative in (
        "phiagent/learning/qa_context_evidence.py",
        "phiagent/learning/public_evidence_retention.py",
        "phiagent/learning/sparse_retrieval.py",
        "scripts/run_compact_harness_training_pilot.py",
    ):
        if sha(source / relative) != origin["source_sha256"][relative]:
            raise RuntimeError("Frozen context-selector source changed")
    if (
        sha(Path(context_module.__file__))
        != origin["source_sha256"]["phiagent/learning/qa_context_evidence.py"]
        or sha(Path(controlled_context.__code__.co_filename))
        != origin["source_sha256"]["scripts/run_compact_harness_training_pilot.py"]
    ):
        raise RuntimeError("The auditor must load the exact frozen generator version, not a newer selector")
    validate_model(config, spec)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["tokenizer_path"], local_files_only=True, trust_remote_code=False,
    )
    items = {item["id"]: item for item in public}
    prompts = {}
    for arm in config["arms"]:
        relative = f"prompts/{arm}.system.txt"
        if sha(source / relative) != origin["source_sha256"][relative]:
            raise RuntimeError("Context study changed a frozen system prompt")
        prompts[arm] = (source / relative).read_text()
    if any(prompts[arm] != prompts["delete"] for arm in config["candidate_arms"]):
        raise RuntimeError("Context candidate changed the DELETE harness as well as its context")
    cache = {}
    counts = {arm: {"tasks": 0, "changed_tasks": 0, "source_tokens": 0, "selected_tokens": 0}
              for arm in config["arms"]}
    seen = set()
    roundoff_cells, maximum_roundoff = 0, 0.0
    for cell in planned:
        identity = (cell["task"], cell["arm"])
        if identity in seen or cell["task"] not in items or cell["arm"] not in counts:
            raise RuntimeError("Invalid or repeated context-study cell")
        seen.add(identity)
        original = items[cell["task"]]
        receipt = cell.get("context_evidence")
        if not isinstance(receipt, dict):
            raise RuntimeError("Missing source-bound context receipt")
        restored = reconstruct_context(original["context"], receipt)
        model_item, replayed = controlled_context(tokenizer, original, config, cell["arm"], cache)
        if restored != model_item["context"]:
            raise RuntimeError("Context selection does not reproduce from frozen question/source alone")
        difference = context_receipt_roundoff(receipt, replayed)
        roundoff_cells += int(difference > 0)
        maximum_roundoff = max(maximum_roundoff, difference)
        expected = {
            "model": config["model"]["repository"],
            "messages": build_messages(prompts[cell["arm"]], model_item),
            "temperature": 0, "max_tokens": config["model"]["max_output_tokens"],
            "seed": config["model"]["request_seed"],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if cell["body"] != expected:
            raise RuntimeError("Context candidate changed question, model settings or the common system prompt")
        row = counts[cell["arm"]]
        row["tasks"] += 1
        row["changed_tasks"] += int(not receipt["identity_transform"])
        row["source_tokens"] += receipt["source_context_tokens"]
        row["selected_tokens"] += receipt["rendered_context_tokens"]
    if len(seen) != len(items) * len(config["arms"]):
        raise RuntimeError("Context selection matrix is incomplete")
    views = {(cell["task"], cell["arm"]): cell["context_evidence"]["rendered_context_sha256"]
             for cell in planned}
    view_differences = {
        name: {
            "left": pair[0], "right": pair[1],
            "different_context_tasks": sum(
                views[task, pair[0]] != views[task, pair[1]] for task in items
            ),
            "tasks": len(items),
        }
        for name, pair in config["factorial"]["contrasts"].items()
    }
    return {
        "status": "SOURCE_OFFSETS_TOKEN_BUDGET_AND_REQUEST_REPLAY_VERIFIED",
        "methods": counts, "calls_replayed": len(planned),
        "all_material_context_receipt_fields_and_requests_exact": True,
        "legacy_annotation_roundoff": {
            "field": "lexical_query_coverage", "absolute_tolerance": 1e-12,
            "affected_cells": roundoff_cells, "maximum_absolute_difference": maximum_roundoff,
            "scope": "Informational legacy annotation only; no request, selected span, budget, prediction or metric changes.",
        },
        "factual_truth_or_answer_sufficiency_certified": False,
        "selector_uses_answers_or_responses": False,
        "factorial_materialization_diagnostic": {
            "scope": "Descriptive comparison of already frozen context hashes; no new statistical test or selection.",
            "contrasts": view_differences,
        },
    }


def context_answer_retention(
    public: list[dict], planned: list[dict], answers: dict, scores: dict, config: dict,
) -> dict:
    items = {item["id"]: item for item in public}
    rows = {arm: {
        "tasks": 0, "literal_answer_retained": 0, "em_gains_vs_delete": 0,
        "em_losses_vs_delete": 0, "em_losses_with_literal_answer_removed": 0,
        "em_losses_despite_literal_answer_retained": 0,
    } for arm in config["arms"]}
    for cell in planned:
        task, arm = cell["task"], cell["arm"]
        context = items[task]["context"]
        pieces = [context[span["start_char"]:span["end_char"]].casefold()
                  for span in cell["context_evidence"]["spans"]]
        retained = any(answer.casefold() in piece for answer in answers[task] for piece in pieces)
        difference = scores[arm][task]["exact_match"] - scores["delete"][task]["exact_match"]
        row = rows[arm]
        row["tasks"] += 1
        row["literal_answer_retained"] += int(retained)
        row["em_gains_vs_delete"] += int(difference > 0)
        row["em_losses_vs_delete"] += int(difference < 0)
        row["em_losses_with_literal_answer_removed"] += int(difference < 0 and not retained)
        row["em_losses_despite_literal_answer_retained"] += int(difference < 0 and retained)
    return {
        "status": "POST_GENERATION_DIAGNOSTIC_NOT_AN_OFFICIAL_RETRIEVAL_METRIC",
        "definition": "A case-insensitive accepted answer string occurs inside one retained original contiguous span; never join separate spans to claim retention.",
        "label_use": "Diagnostic only, after the frozen generation matrix; never an input to selection.",
        "semantic_sufficiency_or_causal_failure_attribution_claimed": False,
        "methods": rows,
    }


def verify_length_controls(planned: list[dict], config: dict) -> dict:
    controls = config.get("length_matched_controls", {})
    if not controls:
        return {}
    cells = {(cell["task"], cell["arm"]): cell for cell in planned}
    if len(cells) != len(planned):
        raise RuntimeError("Duplicate task/arm cells in the length-control matrix")
    tasks = {cell["task"] for cell in planned}
    verified = {}
    for arm, control in controls.items():
        matches = []
        for task in sorted(tasks):
            try:
                padded, base, reference = (cells[task, key] for key in (arm, control["base_arm"], control["reference_arm"]))
            except KeyError as error:
                raise RuntimeError("Incomplete length-control task coverage") from error
            match = padded["length_match"]
            units = match["padding_units"]
            if (type(units) is not int or units < 0 or match["base_arm"] != control["base_arm"]
                    or match["reference_arm"] != control["reference_arm"]
                    or match["prompt_tokens"] != padded["expected_prompt_tokens"]
                    or padded["expected_prompt_tokens"] != reference["expected_prompt_tokens"]):
                raise RuntimeError("Frozen exact prompt-length equality differs")
            messages, base_messages, reference_messages = (cell["body"]["messages"] for cell in (padded, base, reference))
            expected = (base_messages[0]["content"] + config["neutral_padding"]["padding_header"]
                        + config["neutral_padding"]["padding_unit"] * units)
            if (messages[0] != {"role": "system", "content": expected}
                    or messages[1:] != base_messages[1:] or messages[1:] != reference_messages[1:]):
                raise RuntimeError("Length control changed content beyond frozen neutral padding")
            settings = [{key: value for key, value in cell["body"].items() if key != "messages"}
                        for cell in (padded, base, reference)]
            if settings[0] != settings[1] or settings[0] != settings[2]:
                raise RuntimeError("Length-control sampling or output settings differ")
            matches.append({"task": task, "prompt_tokens": match["prompt_tokens"], "padding_units": units})
        verified[arm] = {"base_arm": control["base_arm"], "reference_arm": control["reference_arm"],
                         "tasks": len(matches), "all_exact_prompt_token_matches": True, "matches": matches,
                         "padding_behavioral_inertness_established": False}
    return verified


def output_compliance(rows: dict) -> dict:
    counts = {"tasks": len(rows), "exactly_one_answer_tag_pair": 0, "answer_only": 0,
              "outside_answer_text": 0, "missing_or_multiple_answer_tags": 0,
              "separate_reasoning_field_nonempty": 0, "empty_answers": 0, "length_stops": 0}
    for row in rows.values():
        message = row["raw"]["choices"][0]["message"]
        content = message["content"]
        matches = list(re.finditer(r"<answer>.*?</answer>", content, re.DOTALL))
        exactly_one = len(matches) == content.count("<answer>") == content.count("</answer>") == 1
        counts["exactly_one_answer_tag_pair"] += exactly_one
        counts["missing_or_multiple_answer_tags"] += not exactly_one
        if exactly_one:
            match = matches[0]
            outside = bool((content[:match.start()] + content[match.end():]).strip())
            counts["outside_answer_text"] += outside
            counts["answer_only"] += not outside
        counts["separate_reasoning_field_nonempty"] += bool(message.get("reasoning_content") or message.get("reasoning"))
        counts["empty_answers"] += row["empty_answer"]
        counts["length_stops"] += row["finish_reason"] == "length"
    return {**counts, "no_tasks_filtered": True,
            "boundary": "Text outside tags indicates output verbosity, not verified reasoning quality or correctness."}


def verify_compression_sources(source: Path, origin: dict, planned: list[dict], public: list[dict], config: dict) -> dict:
    from phiagent.learning.compact_qa_contracts import COMPACT_CONTRACTS, cost_source_contracts, delete_compression_contracts

    prompts = {arm: (source / "prompts" / f"{arm}.system.txt").read_text() for arm in config["arms"]}
    relatives = [f"prompts/{arm}.system.txt" for arm in config["arms"]] + ["DELETE_COMPRESSION_MAP.json"]
    for relative in relatives:
        if sha(source / relative) != origin["source_sha256"][relative]:
            raise RuntimeError("Frozen compression source/prompt mapping hash changed")
    for arm, expected in config["prompt_sha256"].items():
        if sha(source / "prompts" / f"{arm}.system.txt") != expected:
            raise RuntimeError("Compression study baseline bytes changed")
    compressed, mapping = delete_compression_contracts(prompts["delete"])
    if json.loads((source / "DELETE_COMPRESSION_MAP.json").read_text()) != mapping:
        raise RuntimeError("Compression mapping does not reproduce the declared source clauses")
    if mapping["source_clause_count"] != config["rule_compression"]["source_clause_count"]:
        raise RuntimeError("Compression mapping does not cover every frozen source clause")
    if (prompts["no_skill_answer_only"] != cost_source_contracts(prompts["no_skill"], prompts["delete"])["no_skill_answer_only"]
            or prompts["compact_target"] != COMPACT_CONTRACTS["compact_target"]):
        raise RuntimeError("The simple strong control or original Compact Target changed")
    for arm, prompt in compressed.items():
        if prompts[arm] != prompt or prompt.split("## Answer Contract\n", 1)[1] != prompts["delete"].split("## Answer Contract\n", 1)[1]:
            raise RuntimeError("Compression candidate or exact direct-output contract changed")
    items = {item["id"]: item for item in public}
    prompt_counts = {(cell["task"], cell["arm"]): cell["expected_prompt_tokens"] for cell in planned}
    if len(prompt_counts) != len(planned) or len(planned) != len(items) * len(config["arms"]):
        raise RuntimeError("Every compression arm must run on the complete frozen input cohort")
    for cell in planned:
        system = prompts[cell["arm"]]
        if cell["arm"] in config["length_matched_controls"]:
            control = config["length_matched_controls"][cell["arm"]]
            if system != prompts[control["base_arm"]]:
                raise RuntimeError("Padded candidate must use the same compressed base prompt")
            system += config["neutral_padding"]["padding_header"] + config["neutral_padding"]["padding_unit"] * cell["length_match"]["padding_units"]
        expected = {
            "model": config["model"]["repository"], "messages": build_messages(system, items[cell["task"]]),
            "temperature": config["model"]["temperature"], "max_tokens": config["model"]["max_output_tokens"],
            "seed": config["model"]["request_seed"], "chat_template_kwargs": {"enable_thinking": False},
        }
        if cell["body"] != expected:
            raise RuntimeError("Actual planned request does not preserve full inputs and frozen direct prompt")
    savings = {}
    for arm, minimum in config["rule_compression"]["minimum_prompt_saving_each_input"].items():
        per_task = [prompt_counts[task, "delete"] - prompt_counts[task, arm] for task in items]
        if min(per_task) < minimum:
            raise RuntimeError("A declared compressed prompt does not reduce token length")
        savings[arm] = {"minimum_prompt_tokens_saved": min(per_task), "total_prompt_tokens_saved": sum(per_task), "tasks": len(per_task)}
    return {
        "status": "VERIFIED_SOURCE_COVERAGE_AND_ACTUAL_REQUEST_IDENTITY_ONLY",
        "source_clause_count": mapping["source_clause_count"], "source_section_count": mapping["source_section_count"],
        "mapping_sha256": sha(source / "DELETE_COMPRESSION_MAP.json"),
        "answer_contract_sha256": mapping["answer_contract_sha256"],
        "simple_control_sha256": sha(source / "prompts/no_skill_answer_only.system.txt"),
        "candidate_sha256": mapping["candidate_sha256"], "prompt_savings": savings,
        "semantic_equivalence_proven": False, "individual_rule_effectiveness_established": False,
    }


def verify(root: Path, output: Path) -> dict:
    require_cpu_client()
    if output.exists():
        raise FileExistsError("Use a new independent verification directory")
    output.mkdir(parents=True, exist_ok=False)
    gpu_command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader"]
    write_json(output / "EXECUTION.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "command": sys.argv, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_physical_gpu": None, "reason": "Remote CPU-only official-score replay; no model calls",
        "gpu_observation_command": gpu_command,
        "gpu_observation": subprocess.check_output(gpu_command, text=True, timeout=30),
    })
    prepared = checked(root / "PREPARED.json")
    result = checked(root / "RESULT.json")
    drained = checked(root / "GENERATION_DRAINED.json")
    config = json.loads((root / "config.json").read_text())
    for name, expected in (
        ("config.json", prepared["config_sha256"]),
        ("model-spec.json", prepared["model_spec_sha256"]),
        ("requests.json", prepared["requests_sha256"]),
    ):
        if sha(root / name) != expected:
            raise RuntimeError(f"Frozen input hash changed: {name}")
    data_root = Path(prepared["data"])
    if sha(data_root / "manifest.json") != prepared["data_manifest_sha256"]:
        raise RuntimeError("Prepared data manifest changed")
    data = checked(data_root / "manifest.json")
    for name, expected in data["files_sha256"].items():
        if sha(data_root / name) != expected:
            raise RuntimeError(f"Prepared data file changed: {name}")
    if not all((
        drained["generation_complete"], drained["server_completion_verified"],
        drained["pending_http"] == 0, drained["unknown_cost_requests"] == 0,
        result["paper_admitted"] is False,
        result["retrieval_contribution_established"] is False,
    )):
        raise RuntimeError("Incomplete generation or invalid claim boundary")
    planned = json.loads((root / "requests.json").read_text())
    public = [json.loads(line) for line in (data_root / "public.jsonl").read_text().splitlines()]
    groups = {item["id"]: item["group"] for item in public}
    if len(groups) != len(public) or len(public) != prepared["tasks"]:
        raise RuntimeError("Public cohort repeats or omits frozen tasks")
    if len(planned) != prepared["requests"] or len({cell["cell"] for cell in planned}) != len(planned):
        raise RuntimeError("Incomplete or duplicate frozen cell inventory")
    actual_dirs = {path.name for path in (root / "calls").iterdir() if path.is_dir()}
    if actual_dirs != {cell["cell"] for cell in planned}:
        raise RuntimeError("Missing or extra request directories")
    length_controls = verify_length_controls(planned, config)
    context_selection = None
    if config.get("context_selection"):
        context_selection = verify_context_sources(
            Path(prepared["source"]), json.loads((root / "origin.json").read_text()),
            planned, public, config, json.loads((root / "model-spec.json").read_text()),
        )
        preparation = json.loads((root / "CONTEXT_PREPARATION.json").read_text())
        for arm, row in context_selection["methods"].items():
            recorded = preparation["methods"][arm]
            if (row["changed_tasks"] != recorded["changed_tasks"]
                    or row["source_tokens"] != recorded["source_context_tokens"]
                    or row["selected_tokens"] != recorded["selected_context_tokens"]):
                raise RuntimeError("Context preparation cost/source totals differ from independent replay")
        if result["all_inputs_full_context"] != all(
            cell["context_evidence"]["identity_transform"] for cell in planned
        ):
            raise RuntimeError("The result misstates whether all contexts were preserved")
        write_json(output / "CONTEXT_EVIDENCE_VERIFIED.json", context_selection)
    compression = None
    if config.get("rule_compression"):
        compression = verify_compression_sources(
            Path(prepared["source"]), json.loads((root / "origin.json").read_text()), planned, public, config,
        )
        preflight = json.loads((root / "COMPRESSION_PREFLIGHT.json").read_text())
        if preflight["candidates"] != compression["prompt_savings"] or preflight["source_map_sha256"] != compression["mapping_sha256"]:
            raise RuntimeError("Compression preflight differs from independent request replay")
        write_json(output / "SOURCE_FIDELITY.json", compression)
    rows = {arm: {} for arm in config["arms"]}
    known_tokens = 0
    hashes = {}
    for cell in planned:
        if cell["task"] not in groups or cell["group"] != groups[cell["task"]]:
            raise RuntimeError("Request task/question-group identity differs from frozen data")
        if cell["cell"] != canonical_hash([cell["task"], cell["arm"], config["model"]["request_seed"]]):
            raise RuntimeError("Cell identity changed")
        directory = root / "calls" / cell["cell"]
        request = json.loads((directory / "request.body").read_bytes())
        if request != cell["body"]:
            raise RuntimeError("Actual request body differs from frozen public-input messages")
        transport = json.loads((directory / "transport.json").read_text())
        if transport["status"] != 200:
            raise RuntimeError("A failed HTTP request entered the completed matrix")
        raw = json.loads((directory / "response.body").read_bytes())
        observed = strict_response(
            raw, model=config["model"]["repository"],
            expected_prompt_tokens=cell["expected_prompt_tokens"],
        )
        if observed["usage"]["completion_tokens"] > config["model"]["max_output_tokens"]:
            raise RuntimeError("Actual output exceeds the frozen cap")
        saved = json.loads((directory / "record.json").read_text())
        if (
            saved["raw"] != raw or saved["usage"] != observed["usage"]
            or saved["task"] != cell["task"] or saved["arm"] != cell["arm"]
            or saved["group"] != cell["group"]
            or saved["response_sha256"] != sha(directory / "response.body")
        ):
            raise RuntimeError("Saved record differs from original provider bytes")
        if cell["task"] in rows[cell["arm"]]:
            raise RuntimeError("Repeated task inside an arm")
        rows[cell["arm"]][cell["task"]] = saved
        known_tokens += observed["usage"]["total_tokens"]
        hashes[str((directory / "response.body").relative_to(root))] = sha(directory / "response.body")
    if (
        drained["attempted"] != len(planned) or drained["completed"] != len(planned)
        or drained["known_real_tokens"] != known_tokens
        or known_tokens + drained["synthetic_tokens"] > config["budget"]["maximum_total_provider_tokens"]
    ):
        raise RuntimeError("Request or token accounting differs")
    answers = json.loads((data_root / "answers.json").read_text())
    if len(answers) != prepared["tasks"] or set(answers) != set(groups):
        raise RuntimeError("Answer cohort differs")
    compliance = {arm: output_compliance(records) for arm, records in rows.items()}
    if length_controls:
        write_json(output / "LENGTH_MATCHES.json", {
            "controls": length_controls, "actual_provider_prompt_tokens_also_verified": True,
            "requests_sha256": sha(root / "requests.json"),
        })
    if config.get("cost_source_design") or config.get("rule_compression"):
        write_json(output / "OUTPUT_COMPLIANCE.json", {
            "methods": compliance, "all_declared_tasks_retained": True,
        })
    scorer = official_scorer({"scorer": {
        "path": str(data_root / "mrqa_official_eval.py"),
        "revision": config["scoring"]["revision"], "sha256": config["scoring"]["source_sha256"],
    }}, runtime=scorer_runtime_mode(config))
    recomputed = {}
    per_arm_scores = {}
    for arm in config["arms"]:
        per_task = score_rows(rows[arm], answers, scorer)
        per_arm_scores[arm] = per_task
        if per_task != json.loads((root / f"{arm}.scores.json").read_text()):
            raise RuntimeError(f"Official per-task metric replay differs: {arm}")
        predictions = {qid: item["prediction"] for qid, item in per_task.items()}
        metrics = scorer.evaluate(answers, predictions, skip_no_answer=False)
        reported = result["methods"][arm]
        if (
            reported["tasks"] != len(answers)
            or reported["exact_match"] != metrics["exact_match"]
            or reported["f1"] != metrics["f1"]
            or reported["total_tokens"] != sum(item["tokens"] for item in per_task.values())
            or reported["prompt_tokens"] != sum(item["prompt_tokens"] for item in per_task.values())
            or reported["completion_tokens"] != sum(item["completion_tokens"] for item in per_task.values())
            or reported["empty_answers"] != sum(item["empty_answer"] for item in per_task.values())
            or reported["length_stops"] != sum(item["finish_reason"] == "length" for item in per_task.values())
        ):
            raise RuntimeError(f"Aggregate score/cost differs: {arm}")
        recomputed[arm] = {
            **metrics, "total_tokens": reported["total_tokens"], "tasks": len(answers)
        }
    if context_selection is not None:
        write_json(output / "CONTEXT_RETENTION_DIAGNOSTIC.json", context_answer_retention(
            public, planned, answers, per_arm_scores, config,
        ))
    independently_eligible = independent_eligible(recomputed, config)
    reported_eligible = [
        candidate for candidate, value in result["advancement"].items()
        if value["eligible_for_larger_training_study"]
    ]
    if sorted(independently_eligible) != sorted(reported_eligible):
        raise RuntimeError("Independent all-baseline advancement gate differs")
    new_eligible = None
    if config.get("retained_method"):
        reference = recomputed[config["retained_method"]]
        new_eligible = [
            arm for arm in config["selection_eligible_arms"] if arm in independently_eligible
            and all(recomputed[arm][metric] >= reference[metric] for metric in ("exact_match", "f1"))
            and any(recomputed[arm][metric] > reference[metric] + 1e-12 for metric in ("exact_match", "f1"))
        ]
        quality = config.get("quality_matched_cost")
        if quality is not None:
            new_eligible = [arm for arm in new_eligible if all(
                recomputed[arm][metric] + 1e-12 >= recomputed[anchor][metric]
                for anchor in quality["strong_quality_references"] for metric in quality["metrics"]
            )]
        new_eligible.sort(key=lambda arm: (-recomputed[arm]["exact_match"], -recomputed[arm]["f1"], recomputed[arm]["total_tokens"], arm))
        decision = json.loads((root.parent / "DEVELOPMENT_DECISION.json").read_text())
        if (decision["eligible_candidates"] != new_eligible
                or decision["selected_candidate"] != (new_eligible[0] if new_eligible else None)
                or decision["result_sha256"] != sha(root / "RESULT.json")
                or decision["confirmation_opened"] or decision["statistical_guarantee_established"] or decision["paper_admitted"]):
            raise RuntimeError("Independent retained-method refinement gate or claim boundary differs")
    value = {
        "schema_version": "phiagent-compact-training-verification/v1",
        "status": "VERIFIED_TRAINING_ONLY_NOT_PAPER_ADMITTED",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "python": sys.version, "command": sys.argv,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": config["model"]["request_seed"], "config_sha256": sha(root / "config.json"),
        "packages": {entry.metadata["Name"]: entry.version for entry in importlib.metadata.distributions() if entry.metadata["Name"]},
        "source_sha256": sha(Path(__file__)), "result_sha256": sha(root / "RESULT.json"),
        "generation_origin_sha256": sha(root / "origin.json"),
        "audit_source_files_sha256": {
            relative: sha(Path(__file__).resolve().parents[1] / relative)
            for relative in (
                "scripts/verify_compact_harness_training_pilot.py",
                "scripts/run_chisel_component_quality.py",
                "phiagent/evaluation/chisel_component_quality.py",
                "phiagent/learning/residual_meta_harness.py",
            )
        },
        "requests_verified": len(planned), "real_provider_tokens": known_tokens,
        "length_matched_controls_verified": length_controls,
        "source_compression_verified": compression,
        **({"context_selection_verified": context_selection,
            "audit_python_hash_seed": os.environ.get("PYTHONHASHSEED")}
           if context_selection is not None else {}),
        "output_compliance": compliance,
        "response_sha256": hashes, "methods": recomputed,
        "scores_sha256": {arm: sha(root / f"{arm}.scores.json") for arm in config["arms"]},
        "requests_sha256": sha(root / "requests.json"),
        "independently_eligible_new_refinements": new_eligible,
        "independently_eligible": independently_eligible,
        "new_model_requests": 0, "new_independent_tasks": 0,
        "official_aggregate_replay_calls": len(config["arms"]),
        "paper_admitted": False, "retrieval_contribution_established": False,
        "limits": "A replay of a TRAIN subset is neither fresh confirmation nor a new benchmark result.",
    }
    value["canonical_sha256"] = canonical_hash(value)
    write_json(output / "VERIFIED.json", value)
    return value


def cost_source_report(output: Path, config: dict, verified: dict, result: dict, diagnostic: dict) -> dict:
    from phiagent.evaluation.chisel_component_quality import quality_matched_costs

    methods = result["methods"]
    settings = config["quality_matched_cost"]
    frontiers = {
        "delete_anchor": quality_matched_costs(methods, [settings["single_strong_anchor"]]),
        "fixed_baselines": quality_matched_costs(methods, settings["fixed_baseline_references"]),
        "strong_references": quality_matched_costs(methods, settings["strong_quality_references"]),
    }
    effects = {}
    for name, (candidate, reference) in config["cost_source_design"]["primary_contrasts"].items():
        left, right = methods[candidate], methods[reference]
        prompt_saved = right["prompt_tokens"] - left["prompt_tokens"]
        completion_saved = right["completion_tokens"] - left["completion_tokens"]
        total_saved = right["total_tokens"] - left["total_tokens"]
        if prompt_saved + completion_saved != total_saved:
            raise RuntimeError("Input/output token decomposition differs from independently verified totals")
        effects[name] = {
            "candidate": candidate, "reference": reference, "tasks": left["tasks"],
            "em_delta_pp": left["exact_match"] - right["exact_match"],
            "f1_delta_pp": left["f1"] - right["f1"],
            "prompt_tokens_saved": prompt_saved, "completion_tokens_saved": completion_saved,
            "total_tokens_saved": total_saved, "token_reduction_pct": 100 * total_saved / right["total_tokens"],
            "em_lower_pp": 100 * diagnostic["contrasts"][f"{candidate}_vs_{reference}_exact_match"]["one_sided_lower_bound"],
            "f1_lower_pp": 100 * diagnostic["contrasts"][f"{candidate}_vs_{reference}_f1"]["one_sided_lower_bound"],
            "mean_token_saving_lower": diagnostic["contrasts"][f"{candidate}_vs_{reference}_tokens"]["one_sided_lower_bound"],
        }
    interaction = {
        field: effects["output_style_long"][field] - effects["output_style_short"][field]
        for field in ("em_delta_pp", "f1_delta_pp", "prompt_tokens_saved", "completion_tokens_saved", "total_tokens_saved")
    }
    value = {
        "schema_version": "phiagent-cost-source-quality-matched-development/v4",
        "status": "VERIFIED_DEVELOPMENT_DIAGNOSTIC_NOT_CONFIRMATION",
        "at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "config_sha256": verified["config_sha256"], "result_sha256": verified["result_sha256"],
        "methods": methods, "cost_effects": effects, "quality_frontiers": frontiers,
        "output_compliance": verified["output_compliance"],
        "length_control_tasks_verified": {arm: record["tasks"] for arm, record in verified["length_matched_controls_verified"].items()},
        "descriptive_output_by_content_interaction_long_minus_short": interaction,
        "interaction_has_no_new_significance_test": True,
        "confidence_family_size": diagnostic["family_size"],
        "unknown_cost_requests": result["calls"]["unknown_cost_requests"],
        "acquisition": {"real_calls": verified["requests_verified"], "real_tokens": verified["real_provider_tokens"],
                        "synthetic_calls": result["calls"]["synthetic_requests"], "synthetic_tokens": result["calls"]["synthetic_tokens"]},
        "no_independent_confirmation_launched": True, "statistical_success_guarantee_established": False,
        "paper_admitted": False,
        "limits": [
            "All quality frontiers use complete matched cohort point estimates, not confidence-certified quality equality.",
            "Long/short content changes rule information and length; neutral padding is measured, not assumed inert.",
            "No response is excluded for instruction noncompliance or output cap; common official scoring applies.",
            "One model/runtime/training cohort, with uncertified historical freshness. Not a full public benchmark.",
            "Selected-path inference savings exclude the explicitly reported acquisition cost.",
        ],
    }
    write_json(output / "COST_SOURCE_REPORT.json", value)
    write_json(output / "QUALITY_MATCHED_COSTS.json", frontiers)
    with (output / "cost_effects.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("contrast", *next(iter(effects.values())).keys()))
        writer.writeheader()
        for name, effect in effects.items():
            writer.writerow({"contrast": name, **effect})
    fields = ("method", "tasks", "exact_match", "f1", "prompt_tokens", "completion_tokens", "total_tokens",
              "empty_answers", "length_stops", "answer_only", "outside_answer_text", "missing_or_multiple_answer_tags")
    with (output / "methods_detailed.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm in config["arms"]:
            writer.writerow({"method": arm, **methods[arm], **{
                field: verified["output_compliance"][arm][field]
                for field in ("answer_only", "outside_answer_text", "missing_or_multiple_answer_tags")
            }})
    lines = [
        "# Token Cost Sources and Quality-Matched Cost", "",
        "Verified training-development results only. No independent confirmation or statistical success guarantee.", "",
        f"{methods['delete']['tasks']} tasks, {verified['requests_verified']} real calls, "
        f"{verified['real_provider_tokens']:,} real provider tokens plus {result['calls']['synthetic_tokens']} synthetic tokens.",
        "Same frozen full inputs, Qwen3.8-27B, common output cap and official EM/F1 scorer. All methods and format failures are retained.",
        "", "## Complete Results", "",
        "| Method | EM (%) | F1 (%) | Prompt Tokens | Completion Tokens | Total Tokens | Length Stops |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in config["arms"]:
        row = methods[arm]
        lines.append(f"| {arm} | {row['exact_match']:.4f} | {row['f1']:.4f} | {row['prompt_tokens']:,} "
                     f"| {row['completion_tokens']:,} | {row['total_tokens']:,} | {row['length_stops']} |")
    lines.extend([
        "", "## Cost Decomposition", "",
        "Positive token values mean reference minus candidate savings. Quality differences mean candidate minus reference. "
        "Every total equals prompt plus completion exactly; negative components remain visible.",
        "", "| Contrast | EM Delta (pp) | F1 Delta (pp) | Prompt Saved | Completion Saved | Total Saved | Reduction (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, effect in effects.items():
        lines.append(f"| {name} | {effect['em_delta_pp']:+.3f} | {effect['f1_delta_pp']:+.3f} | {effect['prompt_tokens_saved']:,} "
                     f"| {effect['completion_tokens_saved']:,} | {effect['total_tokens_saved']:,} | {effect['token_reduction_pct']:+.2f} |")
    lines.extend([
        "", "## Output Compliance", "",
        "Outside-tag text is a verbosity measure, not evidence of valid reasoning. No task is removed or rescored for noncompliance.",
        "", "| Method | Answer Only | Outside-Tag Text | Missing/Multiple Tags |",
        "| --- | ---: | ---: | ---: |",
    ])
    for arm in config["arms"]:
        compliance = verified["output_compliance"][arm]
        lines.append(f"| {arm} | {compliance['answer_only']} | {compliance['outside_answer_text']} | {compliance['missing_or_multiple_answer_tags']} |")
    lines.extend(["", "## Quality-Matched Cost", "",
                  "Both observed EM and F1 must meet the componentwise maximum of a prospectively fixed reference set. "
                  "This is not a statistical noninferiority test. No quality interpolation or oracle routing is used."])
    for name, frontier in frontiers.items():
        floor = frontier["quality_floor"]
        lines.extend(["", f"### {name}", "",
                      f"References: {', '.join(frontier['references'])}.",
                      f"Quality floor: EM {floor['exact_match']:.4f}%, F1 {floor['f1']:.4f}%.",
                      f"Cheapest quality-feasible methods: {frontier['cheapest_quality_feasible_arms']}.",
                      "", "| Method | EM Gap (pp) | F1 Gap (pp) | Quality Pass | Total Tokens | Cheaper Than Every Reference |",
                      "| --- | ---: | ---: | --- | ---: | --- |"])
        for arm in config["arms"]:
            point = frontier["points"][arm]
            lines.append(f"| {arm} | {point['quality_delta_from_floor_pp']['exact_match']:+.3f} "
                         f"| {point['quality_delta_from_floor_pp']['f1']:+.3f} | {point['quality_floor_pass']} "
                         f"| {point['total_tokens']:,} | {point['cheaper_than_every_reference']} |")
    lines.extend([
        "", "## Statistical and Evidence Limits", "",
        f"All {diagnostic['family_size']} declared one-sided metric bounds are in DIAGNOSTIC_BOUNDS.json; "
        "the family and zero-loss tolerance were frozen before generation. Favorable point estimates are not proof of no loss.",
        f"Strictly advancing candidates after all gates: {verified['independently_eligible_new_refinements']}.",
        "The long/short comparison is not a pure length intervention. LENGTH_MATCHES.json verifies exact token matching; "
        "padding effects and domain-rule effects must both be inspected before attribution.",
        "The output-style by content interaction is descriptive only and receives no new post-hoc significance claim.",
        "No independent-confirmation data was opened. Fixed external prompt artifacts are not complete optimizer results.",
        "Full raw-response replay, budget and model provenance are in VERIFIED.json and the experiment archive.",
    ])
    with (output / "COST_SOURCE_SUMMARY.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return value


def compression_fidelity_report(
    output: Path, config: dict, verified: dict, result: dict, diagnostic: dict, scores: dict, groups: dict,
) -> dict:
    from phiagent.evaluation.chisel_component_quality import quality_matched_costs

    if verified["source_compression_verified"] is None:
        raise RuntimeError("Source identity and complete clause coverage must be independently verified first")
    methods, settings = result["methods"], config["quality_matched_cost"]
    frontiers = {
        "delete_anchor": quality_matched_costs(methods, ["delete"]),
        "strong_references": quality_matched_costs(methods, settings["strong_quality_references"]),
    }
    comparisons, task_deltas = {}, []
    for candidate in config["selection_eligible_arms"]:
        comparisons[candidate] = {}
        for reference in ("delete", "no_skill_answer_only"):
            left, right = methods[candidate], methods[reference]
            gains = losses = fractional_gains = fractional_losses = exact_predictions = 0
            for task in sorted(groups):
                candidate_score, reference_score = scores[candidate][task], scores[reference][task]
                delta = candidate_score["exact_match"] - reference_score["exact_match"]
                f1_delta = candidate_score["f1"] - reference_score["f1"]
                gains += delta > 0
                losses += delta < 0
                fractional_gains += f1_delta > 1e-12
                fractional_losses += f1_delta < -1e-12
                exact_predictions += candidate_score["prediction"] == reference_score["prediction"]
                task_deltas.append({
                    "candidate": candidate, "reference": reference, "task": task, "group": groups[task],
                    "candidate_prediction": candidate_score["prediction"], "reference_prediction": reference_score["prediction"],
                    "em_delta": delta, "f1_delta": f1_delta,
                    "tokens_saved": reference_score["tokens"] - candidate_score["tokens"],
                })
            contrasts = diagnostic["contrasts"]
            comparisons[candidate][reference] = {
                "em_delta_pp": left["exact_match"] - right["exact_match"], "f1_delta_pp": left["f1"] - right["f1"],
                "prompt_tokens_saved": right["prompt_tokens"] - left["prompt_tokens"],
                "completion_tokens_saved": right["completion_tokens"] - left["completion_tokens"],
                "total_tokens_saved": right["total_tokens"] - left["total_tokens"],
                "token_reduction_pct": 100 * (1 - left["total_tokens"] / right["total_tokens"]),
                "em_gains": gains, "em_regressions": losses, "f1_gains": fractional_gains,
                "f1_regressions": fractional_losses, "identical_predictions": exact_predictions,
                "em_lower_pp": 100 * contrasts[f"{candidate}_vs_{reference}_exact_match"]["one_sided_lower_bound"],
                "f1_lower_pp": 100 * contrasts[f"{candidate}_vs_{reference}_f1"]["one_sided_lower_bound"],
            }
    feasible = [arm for arm in config["selection_eligible_arms"]
                if frontiers["strong_references"]["points"][arm]["quality_floor_pass"]
                and methods[arm]["total_tokens"] < methods["delete"]["total_tokens"]]
    prior = config.get("resource_amendment", {})
    report = {
        "schema_version": "phiagent-delete-compression-fidelity-result/v5",
        "status": "VERIFIED_DEVELOPMENT_ONLY_NOT_BEHAVIORAL_EQUIVALENCE",
        "at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "config_sha256": verified["config_sha256"], "result_sha256": verified["result_sha256"],
        "methods": methods, "source_fidelity": verified["source_compression_verified"],
        "output_compliance": verified["output_compliance"], "comparisons": comparisons,
        "quality_frontiers": frontiers, "compression_feasible_point_estimates": feasible,
        "strict_all_baseline_advancement": verified["independently_eligible_new_refinements"],
        "diagnostic_bounds": diagnostic["family_size"], "simple_strong_control_always_present": True,
        "acquisition": {
            "completed_real_requests": verified["requests_verified"], "real_tokens": verified["real_provider_tokens"],
            "successful_attempt_synthetic_calls": result["calls"]["synthetic_requests"],
            "successful_attempt_synthetic_tokens": result["calls"]["synthetic_tokens"],
            "prior_failed_synthetic_calls": prior.get("prior_synthetic_request_attempts", 0),
            "prior_failed_synthetic_tokens": prior.get("prior_synthetic_token_usage", 0),
            "prior_unknown_cost_requests": prior.get("prior_unknown_cost_requests", 0),
            "all_in_exact_cost_known": prior.get("prior_unknown_cost_requests", 0) == 0,
        },
        "semantic_fidelity_proven": False, "statistical_success_guarantee_established": False,
        "paper_admitted": False, "confirmation_opened": False,
        "boundary": "Clause coverage and exact output instructions establish implementation identity only. All task gains/regressions are retained; quality matching is descriptive and not independent confirmation.",
    }
    write_json(output / "FIDELITY_REPORT.json", report)
    write_json(output / "PAIRED_COMPRESSION_TASKS.json", task_deltas)
    write_json(output / "QUALITY_MATCHED_COSTS.json", frontiers)
    with (output / "methods_detailed.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("method", *next(iter(methods.values())).keys()))
        writer.writeheader()
        for arm in config["arms"]:
            writer.writerow({"method": arm, **methods[arm]})
    lines = [
        "# DELETE Source-Bound Compression", "",
        "Development only. No independent confirmation, semantic-equivalence guarantee or paper admission.", "",
        f"{methods['delete']['tasks']} matched tasks/groups; {verified['requests_verified']} independently replayed model requests.",
        "All 33 source clauses have mapped counterparts; rule order and the DELETE direct-output contract are preserved.",
        "The no-skill answer-only control is retained unchanged. Layout compression preserves all original rule sentences.",
        "", "| Method | EM (%) | F1 (%) | Prompt Tokens | Completion Tokens | Total Tokens | Answer Only |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in config["arms"]:
        method = methods[arm]
        lines.append(f"| {arm} | {method['exact_match']:.4f} | {method['f1']:.4f} | {method['prompt_tokens']:,} "
                     f"| {method['completion_tokens']:,} | {method['total_tokens']:,} | {verified['output_compliance'][arm]['answer_only']} |")
    lines.extend(["", "## Paired Quality and Cost", "",
                  "Savings are reference minus candidate; quality deltas are candidate minus reference. No failing task is omitted.",
                  "", "| Candidate | Reference | EM Delta (pp) | F1 Delta (pp) | EM Gains/Losses | Total Saved | Reduction (%) | EM Lower (pp) |",
                  "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |"])
    for arm, references in comparisons.items():
        for reference, row in references.items():
            lines.append(f"| {arm} | {reference} | {row['em_delta_pp']:+.3f} | {row['f1_delta_pp']:+.3f} "
                         f"| {row['em_gains']}/{row['em_regressions']} | {row['total_tokens_saved']:,} "
                         f"| {row['token_reduction_pct']:+.2f} | {row['em_lower_pp']:+.3f} |")
    for name, frontier in frontiers.items():
        floor = frontier["quality_floor"]
        lines.extend(["", f"## {name}", "",
                      f"References: {', '.join(frontier['references'])}.",
                      f"Observed quality floor: EM {floor['exact_match']:.4f}%, F1 {floor['f1']:.4f}%.",
                      f"Quality-feasible methods: {frontier['quality_feasible_arms']}.",
                      f"Cheapest quality-feasible methods: {frontier['cheapest_quality_feasible_arms']}."])
    acquisition = report["acquisition"]
    lines.extend([
        "", f"Compressed candidates meeting the strong observed quality floor and cheaper than DELETE: {feasible}.",
        f"Separate strict all-baseline advancement: {report['strict_all_baseline_advancement']}.",
        "These are different gates. A positive compression point is not a statistical guarantee or a default replacement.",
        "", "## Accounting and Limits", "",
        f"Known completed acquisition: {acquisition['real_tokens']:,} real tokens plus {acquisition['successful_attempt_synthetic_tokens']} synthetic tokens.",
        f"Prior failed synthetic attempts: {acquisition['prior_failed_synthetic_calls']}; their tokens: {acquisition['prior_failed_synthetic_tokens']}; "
        f"unknown-cost requests: {acquisition['prior_unknown_cost_requests']}.",
        "Do not report an exact all-in cost when prior usage is unknown. Failed H200 admission and all resource costs remain recorded.",
        f"All {diagnostic['family_size']} predeclared bounds remain in DIAGNOSTIC_BOUNDS.json. All pairwise prediction changes are in PAIRED_COMPRESSION_TASKS.json.",
        "Preserved rule coverage is a design/implementation check, not proof that all nuances or response behavior survived compression.",
        "Whole training cohort only; unseen-input success, statistical noninferiority and paper-level superiority remain unestablished.",
    ])
    with (output / "FIDELITY_SUMMARY.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return report


def development_diagnostic(root: Path, output: Path, candidate: str) -> dict:
    from phiagent.evaluation.chisel_component_quality import fixed_input_token_saving_bound, paired_finite_sample_lower_bound

    require_cpu_client()
    verified = checked(output / "VERIFIED.json")
    if sha(root / "RESULT.json") != verified["result_sha256"]:
        raise RuntimeError("Verified result changed before descriptive statistical diagnostics")
    config = json.loads((root / "config.json").read_text())
    if candidate not in config.get("candidate_arms", CANDIDATES):
        raise RuntimeError("Diagnostic candidate is outside the frozen candidate inventory")
    plan = diagnostic_plan(config, candidate)
    prepared = checked(root / "PREPARED.json")
    data_root = Path(prepared["data"])
    manifest = checked(data_root / "manifest.json")
    if sha(data_root / "public.jsonl") != manifest["files_sha256"]["public.jsonl"]:
        raise RuntimeError("Frozen question group identities changed")
    public = [json.loads(line) for line in (data_root / "public.jsonl").read_text().splitlines()]
    groups = {item["id"]: item["group"] for item in public}
    seed = config["model"]["request_seed"]
    scores = {}
    for arm in config["arms"]:
        filename = root / f"{arm}.scores.json"
        if sha(filename) != verified["scores_sha256"][arm]:
            raise RuntimeError("Official per-task scores changed after independent verification")
        scores[arm] = json.loads(filename.read_text())
    contrasts = {}
    for name, left_arm, right_arm, metric in plan["contrasts"]:
        left = {seed: {task: row[metric] for task, row in scores[left_arm].items()}}
        right = {seed: {task: row[metric] for task, row in scores[right_arm].items()}}
        if metric == "tokens":
            left, right = right, left
        contrasts[f"{left_arm}_vs_{right_arm}_{metric}"] = {
            "contrast_type": name,
            **paired_finite_sample_lower_bound(
                left, right, groups, seeds=(seed,),
                score_upper_bound=config["model"]["context_length"] if metric == "tokens" else 1,
                family_size=plan["family_size"], alpha=plan["alpha"], binary=metric == "exact_match",
            ),
        }
    token_certificates = {}
    if config.get("statistics", {}).get("conditional_token_budget_certificate"):
        if sha(root / "requests.json") != verified["requests_sha256"]:
            raise RuntimeError("Frozen prompt lengths changed after verification")
        prompts = {arm: {} for arm in config["arms"]}
        for cell in json.loads((root / "requests.json").read_text()):
            prompts[cell["arm"]][cell["task"]] = cell["expected_prompt_tokens"]
        for baseline in plan["references"]:
            token_certificates[baseline] = fixed_input_token_saving_bound(
                prompts[candidate], prompts[baseline], output_cap=config["model"]["max_output_tokens"],
                context_length=config["model"]["context_length"],
            )
        write_json(output / "TOKEN_BUDGET_BOUNDS.json", {
            "candidate": candidate, "requests_sha256": verified["requests_sha256"],
            "certificates": token_certificates, "success_guarantee_established": False,
        })
    disjoint = config.get("data", {}).get("mode") == "disjoint_prepared_training"
    limitation = (
        "Prospectively frozen disjoint training groups, but historical freshness is uncertified and this is not a complete public evaluation."
        if disjoint else "Adaptive reuse of the exposed TRAIN64 invalidates confirmatory interpretation, regardless of whether any numerical bound passes."
    )
    value = {
        "status": "DEVELOPMENT_DIAGNOSTIC_NOT_INDEPENDENT_PUBLIC_CONFIRMATION",
        "at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "candidate": candidate, "family_size": plan["family_size"], "familywise_alpha": plan["alpha"], "contrasts": contrasts,
        "source_statistics_sha256": sha(Path(__file__).resolve().parents[1] / "phiagent/evaluation/chisel_component_quality.py"),
        "statistical_guarantee_established": False, "paper_admitted": False,
        "limitation": limitation,
    }
    write_json(output / "DIAGNOSTIC_BOUNDS.json", value)
    with (output / "methods.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("method", "tasks", "exact_match", "f1", "total_tokens"))
        writer.writeheader()
        for arm in config["arms"]:
            writer.writerow({"method": arm, **verified["methods"][arm]})
    result = checked(root / "RESULT.json")
    lines = [
        "# " + config.get("report_title", "Compact Contract Development Screen"),
        "",
        "Training-source development only. No independent confirmation, statistical guarantee, or paper admission.",
        "",
        limitation,
        f"{prepared['tasks']} tasks; {len(set(groups.values()))} question groups; "
        f"{verified['requests_verified']} verified real calls; {verified['real_provider_tokens']:,} real provider tokens.",
        f"Synthetic admission: {result['calls']['synthetic_requests']} call(s), "
        f"{result['calls']['synthetic_tokens']} tokens. Optimization cost is not deployment savings.",
        "",
        "| Method | EM (%) | F1 (%) | Total Tokens |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in config["arms"]:
        row = verified["methods"][arm]
        lines.append(f"| {arm} | {row['exact_match']:.4f} | {row['f1']:.4f} | {row['total_tokens']:,} |")
    lines.extend([
        "", "All methods were run in the same complete matched matrix. SkillOpt, AutoSaddler and DELETE "
        "are unchanged fixed prompt artifacts, not full optimizer executions.",
        "", f"Independently eligible under the strict EM/F1/cost gate: {verified['independently_eligible']}.",
        f"New refinements also improving the retained method: {verified['independently_eligible_new_refinements']}.",
        "", f"## Diagnostic Lower Bounds for {candidate}", "",
        "These training calculations are diagnostic, not valid confirmatory inference for a new complete public benchmark. "
        f"Alpha {plan['alpha']} is divided over {plan['family_size']} one-sided bounds. No seed is counted as an independent task.",
        "",
        "| Baseline | EM Difference (pp) | EM Lower (pp) | F1 Difference (pp) | F1 Lower (pp) | Mean Token Saving | Token Saving Lower |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for baseline in plan["references"]:
        exact, fractional, tokens = (contrasts[f"{candidate}_vs_{baseline}_{metric}"] for metric in ("exact_match", "f1", "tokens"))
        lines.append(
            f"| {baseline} | {100 * exact['observed_difference']:.3f} | {100 * exact['one_sided_lower_bound']:.3f} "
            f"| {100 * fractional['observed_difference']:.3f} | {100 * fractional['one_sided_lower_bound']:.3f} "
            f"| {tokens['observed_difference']:.2f} | {tokens['one_sided_lower_bound']:.2f} |"
        )
    if plan["factorial"]:
        lines.extend([
            "", "## " + ("Compression Controls" if config.get("rule_compression") else "Exact Instruction Ablations"), "",
            "| Contrast | Metric | Observed Difference | Simultaneous Lower Bound |",
            "| --- | --- | ---: | ---: |",
        ])
        for name, left_arm, right_arm, metric in plan["factorial"]:
            contrast = contrasts[f"{left_arm}_vs_{right_arm}_{metric}"]
            scale, unit = (1, "tokens/task") if metric == "tokens" else (100, "pp")
            lines.append(f"| {name} | {metric} ({unit}) | {scale * contrast['observed_difference']:.3f} | {scale * contrast['one_sided_lower_bound']:.3f} |")
    if token_certificates:
        lines.extend([
            "", "## Conditional Token Budget", "",
            "Deterministic for these frozen inputs, exact prompt tokenization, one call per method and the output cap; "
            "not a success or unseen-input guarantee. Full study acquisition cost is not included in deployment savings.",
            "", "| Reference | Total Saving Lower | Reduction Lower (%) | Every Frozen Task Cheaper |",
            "| --- | ---: | ---: | --- |",
        ])
        for baseline, certificate in token_certificates.items():
            lines.append(f"| {baseline} | {certificate['total_saving_lower']:,} | {certificate['total_reduction_pct_lower']:.2f} | {certificate['every_frozen_task_strictly_cheaper']} |")
    lines.extend([
        "", "No bound is a per-question guarantee. New independent full-public-cohort evidence, "
        "frozen candidate selection and all-baseline official-metric superiority are required before any paper claim.",
        "", "Sources: VERIFIED.json, DIAGNOSTIC_BOUNDS.json and the hash-bound generation artifacts.",
    ])
    with (output / "SUMMARY.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    if config.get("cost_source_design"):
        cost_source_report(output, config, verified, result, value)
    if config.get("rule_compression"):
        compression_fidelity_report(output, config, verified, result, value, scores, groups)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-candidate")
    parser.add_argument("--await-retirement", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("use a new verification directory")
    try:
        if args.await_retirement:
            await_retirement(args.experiment, args.output_dir.with_name(args.output_dir.name + "-observer"))
        result = verify(args.experiment, args.output_dir)
        if args.diagnostic_candidate:
            development_diagnostic(args.experiment, args.output_dir, args.diagnostic_candidate)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        if args.output_dir.is_dir():
            write_json(args.output_dir / "FAILURE.json", {
                "type": type(error).__name__, "message": str(error),
            })
        raise
    print(json.dumps({
        "status": result["status"], "requests": result["requests_verified"],
        "eligible": result["independently_eligible"],
    }, indent=2))


if __name__ == "__main__":
    main()
