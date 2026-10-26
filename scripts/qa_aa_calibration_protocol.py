"""Frozen Original-only A/A requests and descriptive repeatability diagnostics."""

from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess

from scripts.prepare_chisel_component_quality_data import canonical_hash
from scripts.qa_harness_mixture_protocol import (
    FrozenOriginalChisel, INCUMBENT, OVERLAY_FILES as INCUMBENT_FILES,
    stage_incumbent_artifacts, validate_methods,
)
from scripts.run_chisel_component_quality import (
    ProtocolError, build_messages, prompt_length, score_rows,
)


SCHEMA = "phiagent-qa-aa-calibration/v1"
RESULT_STATUS = "COMPLETE_OPENED_DEVELOPMENT_AA_CALIBRATION"
RESULT_SCOPE = (
    "Original-only repeated inference on a label-blind subset of previously opened "
    "development groups; not fresh quality confirmation or a universal determinism guarantee."
)
DEVELOPMENT_STATUS = "AA_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_PROMOTION"
RECORD_CLIENT_TIMING = True
PHASES = (
    (INCUMBENT, "serial", 1, "base"),
    ("parallel_fixed_a", "parallel_fixed", 16, "base"),
    ("parallel_shuffled_a", "parallel_shuffled", 16, "shuffled"),
    ("parallel_shuffled_b", "parallel_shuffled", 16, "shuffled"),
    ("parallel_fixed_b", "parallel_fixed", 16, "base"),
    ("serial_repeat", "serial", 1, "base"),
)
ARMS = tuple(row[0] for row in PHASES)
WITHIN_MODE_PAIRS = (
    (INCUMBENT, "serial_repeat"),
    ("parallel_fixed_a", "parallel_fixed_b"),
    ("parallel_shuffled_a", "parallel_shuffled_b"),
)
OVERLAY_FILES = (
    *INCUMBENT_FILES,
    "scripts/qa_aa_calibration_protocol.py",
    "scripts/prepare_qa_aa_calibration.py",
    "scripts/verify_qa_aa_calibration.py",
    "scripts/render_qa_aa_calibration.py",
    "tests/test_qa_aa_calibration.py",
    "tests/test_qa_harness_mixture_audit_binding.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict) -> None:
    expected_phases = [
        {"arm": arm, "mode": mode, "client_concurrency": concurrency, "order": order}
        for arm, mode, concurrency, order in PHASES
    ]
    if (
        config.get("schema_version") != SCHEMA
        or config["arms"] != list(ARMS) or config["comparison_arms"] != list(ARMS)
        or config["candidate_arms"] != [] or config["selection_eligible_arms"] != []
        or config["primary_candidate"] != INCUMBENT or config["primary_reference"] != INCUMBENT
        or config["cost_reference"] != INCUMBENT
        or config["incumbent"]["status"] != "BOUND" or config["incumbent"]["arm"] != INCUMBENT
        or config["calibration"]["phases"] != expected_phases
        or config["calibration"]["same_body_for_all_repeats"] is not True
        or config["calibration"]["historical_outputs_reused"] is not False
        or config["calibration"]["runtime_batch_invariant"] != "0"
        or config["calibration"]["server_microbatch_layout_controlled"] is not False
        or config["calibration"]["automatic_candidate_experiment"] is not False
        or config["model"]["temperature"] != 0 or config["model"]["enable_thinking"] is not False
        or config["model"]["max_output_tokens"] != 512 or config["model"]["request_seed"] != 20260918
        or config["budget"]["execution_workers"] != 16
        or config["budget"]["maximum_inflight_requests"] != 16
        or config["budget"]["maximum_physical_gpus"] != 1
        or config["budget"]["maximum_synthetic_request_attempts"] != 1
        or config["budget"]["maximum_allocation_seconds"] != 2700
        or config["budget"]["maximum_total_provider_tokens"] != 4000000
        or config["budget"]["transport_retries"] != 0
        or config["data"]["mode"] != "opened_development_aa_calibration"
        or config["data"]["question_groups"] != 128
        or config["data"]["historical_outcomes_exposed"] is not True
        or config["data"]["selection_uses_outcomes"] is not False
        or config["paper_admitted"] is not False
        or any(config.get(key) for key in (
            "capability_routing", "mixture_router", "preservation",
            "context_selection", "rule_compression", "length_matched_controls",
        ))
    ):
        raise ProtocolError("The bounded Original-only repeated-inference protocol changed")
    statistical_plan(config)


def statistical_plan(config: dict) -> dict:
    if config["statistics"] != {
        "primary_endpoint": "question_group_with_any_extracted_answer_change_across_six_requests",
        "familywise_alpha": 0.05, "simultaneous_bounds": 1,
        "statistical_unit": "question_group_not_repeat_or_request",
        "all_fifteen_pairwise_comparisons": "descriptive_only",
        "no_optional_stopping": True,
        "zero_observed_changes_is_not_population_determinism": True,
    }:
        raise ProtocolError("The AA endpoint, independent unit, or stopping rule changed")
    return {"family_size": 1, "alpha": 0.05, "reference": INCUMBENT}


def fixed_prompt_hashes(config: dict) -> dict:
    return {profile: config["prompt_sha256"][profile] for profile in ("delete", "modify")}


def prompt_artifact_names(_config: dict) -> list[str]:
    return ["delete", "modify"]


def stage_incumbent(source: Path, config: dict) -> list[str]:
    validate_config(config)
    return stage_incumbent_artifacts(source, config)


def select_public_items(public: list[dict], *, count: int, salt: str) -> list[dict]:
    if (
        not public or len({item["id"] for item in public}) != len(public)
        or any(set(item) != {"id", "group", "question", "context"} for item in public)
        or type(count) is not int or count <= 0 or not isinstance(salt, str) or not salt
    ):
        raise ProtocolError("Unique complete public inputs and a positive group budget are required")
    groups = {item["group"] for item in public}
    if len(groups) < count:
        raise ProtocolError("The opened source pool has too few whole question groups")
    chosen = set(sorted(groups, key=lambda group: (canonical_hash([salt, group]), group))[:count])
    return sorted((item for item in public if item["group"] in chosen), key=lambda item: item["id"])


def validate_prepared_data(config: dict, root: Path, config_sha256: str) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["canonical_sha256"] != canonical_hash({
            key: value for key, value in manifest.items() if key != "canonical_sha256"
        })
        or manifest["schema_version"] != "phiagent-opened-aa-data/v1"
        or manifest["config_sha256"] != config_sha256
        or manifest["parent_manifest_sha256"] != config["data"]["parent_manifest_sha256"]
        or manifest["question_group_count"] != config["data"]["question_groups"]
        or not 0 < manifest["task_count"] <= config["data"]["maximum_tasks"]
        or manifest["historical_outcomes_exposed"] is not True
        or manifest["independent_quality_confirmation"] is not False
        or manifest["selection_uses_outcomes"] is not False
        or manifest["new_model_calls"] != 0 or manifest["evaluator_calls"] != 0
    ):
        raise ProtocolError("Opened-development cohort or evidence identity changed")
    for name, expected in manifest["files_sha256"].items():
        path = root / name
        if Path(name).name != name or path.is_symlink() or sha(path) != expected:
            raise ProtocolError(f"Frozen AA data changed: {name}")
    return manifest


def copy_prepared_data(config_path: Path, source: Path, destination: Path) -> None:
    config = json.loads(config_path.read_text())
    validate_prepared_data(config, source, sha(config_path))
    shutil.copytree(source, destination)
    (destination / "answers.json").chmod(0o600)


def task_order(item: dict, order: str, config: dict) -> tuple[str, str]:
    return canonical_hash([config["calibration"]["order_salt"], order, item["id"]]), item["id"]


def prepare_cells(tokenizer, items: list[dict], prompts: dict, config: dict, source: Path):
    validate_config(config)
    if (
        not items or len({item["id"] for item in items}) != len(items)
        or any(set(item) != {"id", "group", "question", "context"} for item in items)
        or set(prompts) != {"delete", "modify"}
        or any(hashlib.sha256(prompts[key].encode()).hexdigest() != expected
               for key, expected in fixed_prompt_hashes(config).items())
    ):
        raise ProtocolError("The exact Original prompts and complete unique inputs must be frozen")
    incumbent = FrozenOriginalChisel(source, config)
    ranks = {
        order: {item["id"]: index for index, item in enumerate(
            sorted(items, key=lambda item: task_order(item, order, config))
        )}
        for order in ("base", "shuffled")
    }
    if len(items) > 1 and ranks["base"] == ranks["shuffled"]:
        raise ProtocolError("The reordered condition did not change the public input order")
    cells = []
    for item in items:
        route = incumbent.route(item)
        messages = build_messages(prompts[route["variant"]], item)
        count = prompt_length(tokenizer, messages)
        body = {
            "model": config["model"]["repository"], "messages": messages,
            "temperature": 0, "max_tokens": config["model"]["max_output_tokens"],
            "seed": config["model"]["request_seed"],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for phase, (arm, mode, concurrency, order) in enumerate(PHASES):
            cell = {
                "cell": canonical_hash([item["id"], arm, config["model"]["request_seed"]]),
                "task": item["id"], "group": item["group"], "arm": arm,
                "expected_prompt_tokens": count, "max_output_tokens": body["max_tokens"],
                "incumbent_route": route, "body": body,
                "aa_schedule": {
                    "phase_index": phase, "mode": mode, "client_concurrency": concurrency,
                    "order": order, "position": ranks[order][item["id"]],
                    "batch_index_in_phase": ranks[order][item["id"]] // concurrency,
                    "body_sha256": hashlib.sha256(json.dumps(
                        body, ensure_ascii=False, separators=(",", ":"),
                    ).encode()).hexdigest(),
                },
            }
            output_cap(cell, config)
            cells.append(cell)
    cells.sort(key=lambda cell: cell["cell"])
    batches = execution_batches(cells, config)
    return cells, {
        "status": "IDENTICAL_ORIGINAL_BODIES_AND_CLIENT_SCHEDULES_FROZEN",
        "tasks": len(items), "question_groups": len({item["group"] for item in items}),
        "requests": len(cells), "client_batches": len(batches),
        "phases": config["calibration"]["phases"],
        "prompt_tokens_per_phase": sum(
            cell["expected_prompt_tokens"] for cell in cells if cell["arm"] == INCUMBENT
        ),
        "original_routes": dict(Counter(
            cell["incumbent_route"]["variant"] for cell in cells if cell["arm"] == INCUMBENT
        )),
        "identical_body_across_all_six_repeats": True,
        "same_actor_seed_temperature_and_output_cap": True,
        "responses_or_answers_used_for_preparation": False,
        "server_microbatch_layout_controlled": False,
        "phase_order": "serial, fixed, shuffled, shuffled, fixed, serial",
        "historical_outcomes_exposed": True,
        "model_calls": 0, "router_model_calls": 0,
    }


def output_cap(cell: dict, config: dict) -> int:
    cap = config["model"]["max_output_tokens"]
    if (
        cell["arm"] not in ARMS or cell["max_output_tokens"] != cap
        or cell["body"]["max_tokens"] != cap
        or cell["expected_prompt_tokens"] + cap > config["model"]["context_length"]
    ):
        raise ProtocolError("Every repeat must retain the full Original model context and output cap")
    return cap


def execution_batches(cells: list[dict], config: dict) -> list[list[dict]]:
    if len({cell["cell"] for cell in cells}) != len(cells) or set(cell["arm"] for cell in cells) != set(ARMS):
        raise ProtocolError("The six-phase AA schedule is incomplete or duplicated")
    task_ids = {cell["task"] for cell in cells}
    batches = []
    for phase, (arm, mode, concurrency, order) in enumerate(PHASES):
        selected = sorted(
            (cell for cell in cells if cell["arm"] == arm),
            key=lambda cell: cell["aa_schedule"]["position"],
        )
        if len(selected) != len(task_ids) or {cell["task"] for cell in selected} != task_ids:
            raise ProtocolError("Every AA phase must retain exactly the same tasks")
        for position, cell in enumerate(selected):
            schedule = cell["aa_schedule"]
            if any(schedule[key] != value for key, value in {
                "phase_index": phase, "mode": mode, "client_concurrency": concurrency,
                "order": order, "position": position, "batch_index_in_phase": position // concurrency,
            }.items()):
                raise ProtocolError("Client dispatch order/concurrency differs from its phase")
        batches.extend(selected[offset:offset + concurrency]
                       for offset in range(0, len(selected), concurrency))
    if max(map(len, batches)) > config["budget"]["execution_workers"]:
        raise ProtocolError("A declared AA client batch exceeds its frozen concurrency budget")
    return batches


def attest_runtime(binding: dict, config: dict) -> dict:
    output = subprocess.check_output([
        "nvidia-smi", "--id", str(config["resource"]["physical_gpu_index"]),
        "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=30)
    worker_pids = {int(line.strip()) for line in output.splitlines() if line.strip()}
    if not worker_pids:
        raise ProtocolError("The AA model service has no physical GPU worker")
    processes = {}
    for pid in worker_pids | {binding["server_identity"]["pid"]}:
        environment = dict(
            part.split(b"=", 1)
            for part in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
            if b"=" in part
        )
        observed = {
            key: environment.get(key.encode(), b"").decode()
            for key in ("VLLM_BATCH_INVARIANT", "CUDA_VISIBLE_DEVICES")
        }
        if (
            observed["VLLM_BATCH_INVARIANT"] != config["calibration"]["runtime_batch_invariant"]
            or observed["CUDA_VISIBLE_DEVICES"] not in {
                str(config["resource"]["physical_gpu_index"]), "0",
            }
        ):
            raise ProtocolError("AA live worker batch-invariance or GPU visibility changed")
        processes[str(pid)] = observed
    return {
        "status": "UNCHANGED_RUNTIME_FLAG_ATTESTED_ON_SERVER_AND_GPU_WORKERS",
        "worker_pids": sorted(worker_pids), "server_pid": binding["server_identity"]["pid"],
        "processes": processes, "batch_invariant_runtime_changed": False,
    }


def score_records(records: dict, answers: dict, scorer) -> dict:
    result = score_rows(records, answers, scorer)
    for task, row in result.items():
        content = records[task]["raw"]["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ProtocolError("AA raw assistant output must be text")
        row["raw_content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    return result


def repeatability_summary(scores: dict, groups: dict[str, str]) -> dict:
    if set(scores) != set(ARMS) or not groups or any(set(rows) != set(groups) for rows in scores.values()):
        raise ProtocolError("Repeatability requires the complete paired six-phase matrix")
    for rows in scores.values():
        for row in rows.values():
            if (
                type(row["exact_match"]) is not int or row["exact_match"] not in (0, 1)
                or type(row["f1"]) not in (int, float) or not math.isfinite(row["f1"])
                or not 0 <= row["f1"] <= 1 or not isinstance(row["prediction"], str)
                or not isinstance(row["raw_content_sha256"], str) or len(row["raw_content_sha256"]) != 64
                or any(type(row[key]) is not int or row[key] < 0
                       for key in ("tokens", "prompt_tokens", "completion_tokens"))
                or row["tokens"] != row["prompt_tokens"] + row["completion_tokens"]
            ):
                raise ProtocolError("AA agreement uses exact finite official scores and measured tokens")
    pairs = {}
    for left, right in itertools.combinations(ARMS, 2):
        lrows, rrows = scores[left], scores[right]
        changed = {
            "raw_content": [task for task in groups if lrows[task]["raw_content_sha256"] != rrows[task]["raw_content_sha256"]],
            "extracted_answer": [task for task in groups if lrows[task]["prediction"] != rrows[task]["prediction"]],
            "exact_match": [task for task in groups if lrows[task]["exact_match"] != rrows[task]["exact_match"]],
            "f1": [task for task in groups if lrows[task]["f1"] != rrows[task]["f1"]],
            "completion_tokens": [task for task in groups if lrows[task]["completion_tokens"] != rrows[task]["completion_tokens"]],
            "prompt_tokens": [task for task in groups if lrows[task]["prompt_tokens"] != rrows[task]["prompt_tokens"]],
            "finish_reason": [task for task in groups if lrows[task]["finish_reason"] != rrows[task]["finish_reason"]],
        }
        pairs[f"{left}_vs_{right}"] = {
            "left": left, "right": right, "tasks": len(groups),
            "within_mode_repeat": (left, right) in WITHIN_MODE_PAIRS,
            "changed_tasks": {key: len(values) for key, values in changed.items()},
            "changed_question_groups": {key: len({groups[task] for task in values}) for key, values in changed.items()},
            "changed_task_ids": {key: sorted(values) for key, values in changed.items()},
            "em_gains": sum(rrows[task]["exact_match"] > lrows[task]["exact_match"] for task in groups),
            "em_losses": sum(rrows[task]["exact_match"] < lrows[task]["exact_match"] for task in groups),
            "em_difference_pp": math.fsum(rrows[task]["exact_match"] - lrows[task]["exact_match"] for task in groups) / len(groups) * 100,
            "f1_difference_pp": math.fsum(rrows[task]["f1"] - lrows[task]["f1"] for task in groups) / len(groups) * 100,
            "total_token_difference": sum(rrows[task]["tokens"] - lrows[task]["tokens"] for task in groups),
        }
    all_group_changes = {
        field: set().union(*(
            {groups[task] for task in pair["changed_task_ids"][field]}
            for pair in pairs.values()
        ))
        for field in next(iter(pairs.values()))["changed_tasks"]
    }
    count = len(set(groups.values()))
    answer_changes = len(all_group_changes["extracted_answer"])
    return {
        "schema_version": "phiagent-aa-repeatability-summary/v1",
        "status": (
            "VARIATION_OBSERVED" if any(all_group_changes.values())
            else "NO_VARIATION_OBSERVED_IN_THIS_FINITE_RUN"
        ),
        "unique_tasks": len(groups), "question_groups": count,
        "repeats_per_task": len(ARMS), "fresh_provider_requests": len(groups) * len(ARMS),
        "primary_changed_question_groups": answer_changes,
        "primary_change_fraction": answer_changes / count,
        "any_change_question_groups": {key: len(values) for key, values in all_group_changes.items()},
        "all_fifteen_pairs": pairs,
        "zero_change_iid_one_sided_95_upper_bound": (
            1 - 0.05 ** (1 / count) if answer_changes == 0 else None
        ),
        "bound_boundary": "Zero-event sensitivity only under iid question-group assumptions; not a guarantee for unseen tasks or other schedules.",
        "request_repeats_are_independent_samples": False,
        "execution_cause_identified": False,
        "paper_admitted": False, "quality_improvement_claim": False,
    }


def diagnostic_bounds(scores: dict, groups: dict[str, str], config: dict) -> dict:
    statistical_plan(config)
    return {"aa_repeatability": repeatability_summary(scores, groups)}


def target_gate(methods: dict, config: dict, bounds: dict | None = None) -> dict:
    validate_methods(methods, config)
    diagnostic = bounds["aa_repeatability"] if bounds is not None else None
    return {
        "status": "AA_DIAGNOSTIC_PENDING" if diagnostic is None else diagnostic["status"],
        "candidate": None, "primary_reference": INCUMBENT, "primary_baseline_status": "BOUND",
        "observed_all_six_output_agreement": (
            None if diagnostic is None else diagnostic["any_change_question_groups"]["raw_content"] == 0
        ),
        "observed_all_six_answer_agreement": (
            None if diagnostic is None else diagnostic["primary_changed_question_groups"] == 0
        ),
        "point_target_passed": False, "statistical_target_passed": False,
        "diagnostic_only_not_a_quality_candidate_gate": True,
        "universal_determinism_established": False, "batching_cause_established": False,
        "automatic_official_scale_up_permitted": False, "paper_admitted": False,
        "production_defaults_changed": False,
    }


def result_diagnostics(planned: list[dict], scores: dict, config: dict) -> dict:
    return {"aa_repeatability": repeatability_summary(
        scores, {cell["task"]: cell["group"] for cell in planned},
    )}


def replay_matrix(tokenizer, public, planned, source, config) -> dict:
    prompts = {key: (source / f"prompts/{key}.system.txt").read_text() for key in prompt_artifact_names(config)}
    rebuilt, summary = prepare_cells(tokenizer, public, prompts, config, source)
    if rebuilt != planned:
        raise RuntimeError("AA request, exact Original route, or client scheduling replay differs")
    for task in {cell["task"] for cell in planned}:
        bodies = {cell["aa_schedule"]["body_sha256"] for cell in planned if cell["task"] == task}
        if len(bodies) != 1:
            raise RuntimeError("An A/A task received different request bytes")
    return {
        "summary": summary, "all_six_original_request_bodies_identical_per_task": True,
        "all_original_context_and_model_caps_retained": True,
        "answers_or_outcomes_used_for_reconstruction": False,
    }
