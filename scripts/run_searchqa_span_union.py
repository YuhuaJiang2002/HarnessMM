"""Run frozen exact-span packing candidates on the complete remote TRAIN split."""
from __future__ import annotations

from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import figure2_searchqa_ranker as ranker  # noqa: E402
from scripts import figure2_searchqa_recovery as recovery  # noqa: E402
from scripts import figure2_searchqa_v4 as native  # noqa: E402
from scripts import searchqa_span_union as packing  # noqa: E402
from scripts.analyze_searchqa_accuracy_headroom import load_train, read  # noqa: E402
from scripts.figure2_searchqa_train import official_scorer, save, sha  # noqa: E402


def inherited_surface_rule(profiles: dict) -> str:
    prefix = "- Preserve the answer surface form supported by the strongest evidence"
    rules = {
        paragraph.strip().split("\n- ", 1)[0]
        for branch in ("delete", "modify")
        for paragraph in profiles[branch].split("\n\n")
        if paragraph.strip().startswith(prefix)
    }
    if len(rules) != 1:
        raise ValueError("The inherited surface-form rule is missing or ambiguous")
    rule = rules.pop()
    return "\n" + rule


def prepare(root: Path, config: dict, host: dict, *, selector=packing):
    if (config["schema_version"] not in (
            "searchqa-exact-span-union/v1", "searchqa-autosota-evidence/v1")
            or config["scope"] != "train" or config["training_tasks"] != 3395
            or config["validation_calls"] != 0 or config["test_calls"] != 0
            or config["source_selection_input_cap"] not in (640, 704, 768, 800)):
        raise ValueError("Unexpected prospective TRAIN contract")
    parent = Path(config["predecessor"])
    for relative, expected in config["predecessor_sha256"].items():
        if sha(parent / relative) != expected:
            raise ValueError("Completed reference changed: " + relative)
    old_config = read(parent / "pool-config.json")
    for field in (
        "model_repository", "model_revision", "request_seed",
        "archive_sha256", "scorer_sha256", "partition_sha256",
    ):
        if config[field] != old_config[field]:
            raise ValueError("Common execution identity differs: " + field)
    for relative in (
        "phiagent/learning/qa_typed_recovery.py", "phiagent/learning/qa_meta_rsi.py",
        "scripts/qa_harness_mixture_protocol.py",
    ):
        if sha(ROOT / relative) != sha(parent / "source" / relative):
            raise ValueError("The matched reference adapter changed: " + relative)
    audit = read(Path(config["headroom"]) / "NATIVE_TRAIN_AUDIT.json")
    if (sha(Path(config["headroom"]) / "NATIVE_TRAIN_AUDIT.json")
            != config["headroom_audit_sha256"]
            or audit["status"] != "PASS_COMPLETE_TRAIN_AUDIT"
            or audit["source_complete_sha256"] != sha(parent / "TRAIN_DATASET_COMPLETE.json")):
        raise ValueError("Missing complete native reference audit")
    items, labels, references, partition = load_train(parent)
    if len(items) != 3395 or len(items) != len(partition["train"]["ids"]):
        raise ValueError("Complete TRAIN population is required")
    predictor = read(root / "inputs/RANKER_FROZEN.json")
    profiles = read(root / "inputs/reference/PROFILES.json")
    extra = inherited_surface_rule(profiles) if config["inherit_surface_rule"] else ""
    ordered = [
        identity for group in partition["train"]["groups"]
        for identity in sorted(partition["groups"][group])
    ]
    if len(ordered) != len(set(ordered)) or set(ordered) != set(items):
        raise ValueError("Every frozen TRAIN question must occur once")
    freeze = {
        "candidate": config["family"][0], "source_selection_input_cap":
        config["source_selection_input_cap"], "extra_instruction": extra,
        "system": recovery.TARGET_ROLE_SYSTEM + extra,
        "source_sha256": sha(Path(selector.__file__)),
        "method_schema_version": config["schema_version"],
        "ranker_sha256": sha(root / "inputs/RANKER_FROZEN.json"),
        "partition_sha256": config["partition_sha256"],
        "ids": ordered, "validation_feedback": False, "test_feedback": False,
        "all_train_after_accounting_gate": True,
        "new_model_calls_for_source_selection": 0,
        "quality_cost_admission_unchanged": True, "paper_admitted": False,
    }
    save(root / "CANDIDATES_FROZEN.json", freeze)
    save(root / "REFERENCE_REUSE.json", {
        "root": str(parent), "audit_sha256": config["headroom_audit_sha256"],
        "methods": {
            arm: ranker.summary(list(rows.values())) for arm, rows in references.items()
        },
        "new_reference_model_calls": 0, "task_counts": {k: len(v) for k, v in references.items()},
        "historical_optimizer_provider_tokens": config["inherited_optimizer_provider_tokens"],
        "historical_cost_not_added_to_new_actor_execution": True,
    })
    return items, labels, references, partition, predictor, profiles, ordered, extra


def study(root: Path, config: dict, prepared: tuple, state: dict, *,
          request_builder=packing.request):
    from transformers import AutoTokenizer
    from phiagent.learning import qa_meta_rsi
    from scripts.qa_harness_mixture_protocol import FrozenOriginalChisel

    items, labels, references, partition, predictor, profiles, ordered, extra = prepared
    reference = root / "inputs/reference"
    parent = read(reference / "lineage/config.json")
    for arm in ("no_skill", "delete", "modify"):
        if ranker.digest(profiles[arm]) != parent["prompt_sha256"][arm]:
            raise ValueError("Frozen reference instructions differ")
    router = FrozenOriginalChisel(reference / "lineage", parent)
    original = qa_meta_rsi.initial_skill({k: profiles[k] for k in ("delete", "modify")})
    scorer = official_scorer({
        "scorer": str(root / "inputs/mrqa_official_eval.py"),
        "scorer_sha256": config["scorer_sha256"],
    })
    provider = native.Provider(
        root, config,
        AutoTokenizer.from_pretrained(config["model_directory"], local_files_only=True),
    )
    candidate = config["family"][0]
    rows, lock = {}, threading.Lock()
    (root / "tasks").mkdir()
    state.update(phase="FIRST_NATIVE_ACCOUNTING_GATE", target_cells=len(ordered))

    def execute(identity: str):
        item = items[identity]
        messages, receipt = request_builder(
            item, predictor, provider.count, config["source_selection_input_cap"], extra,
        )
        calls = []

        def call(request, stage):
            answer, record = provider.call(identity, candidate, stage, request)
            calls.append(record)
            return answer

        answer = call(messages, "compact")
        recovered = recovery.fallback_needed(item["question"], item["context"], answer)
        if recovered:
            answer = native.current_chisel(
                item, router, profiles, original, call, "paid_recovery",
            )
        prediction = router.campaign.extract_answer(answer)
        row = {
            "id": identity, "arm": candidate, "split": "train", "answer": answer,
            "prediction": prediction,
            "em": int(scorer.metric_max_over_ground_truths(
                scorer.exact_match_score, prediction, labels[identity],
            )),
            "f1": scorer.metric_max_over_ground_truths(
                scorer.f1_score, prediction, labels[identity],
            ),
            "usage": {
                key: sum(entry["usage"][key] for entry in calls)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "calls": calls, "paid_recovery": recovered, "source_receipt": receipt,
        }
        save(root / "tasks" / (ranker.digest(identity + ":" + candidate) + ".json"), row)
        with lock:
            if identity in rows:
                raise ValueError("Duplicate candidate TRAIN outcome")
            rows[identity] = row
            state["completed_cells"] = len(rows)

    execute(ordered[0])
    gate = rows[ordered[0]]
    save(root / "NATIVE_GATE.json", {
        "status": "PASS_NATIVE_ACCOUNTING_GATE", "id": ordered[0], "calls": len(gate["calls"]),
        "tokens": gate["usage"]["total_tokens"],
        "actual_source_union": gate["source_receipt"],
        "counts_once_in_full_train": True, "quality_claimed": False,
    })
    state["phase"] = "COMPLETE_TRAIN_ACQUISITION"
    native.bounded_execute(execute, ordered[1:], config["execution_workers"])
    if set(rows) != set(ordered) or provider.usage["unknown_usage_calls"] or provider.usage["failed_calls"]:
        raise ValueError("Complete TRAIN outcomes and fully known native costs are required")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if sum(row["usage"][key] for row in rows.values()) != provider.usage[key]:
            raise ValueError("Episode and provider token ledgers differ")
    summary = ranker.summary(list(rows.values()))
    comparisons = {}
    for arm, recorded in references.items():
        reference_summary = ranker.summary(list(recorded.values()))
        comparisons[arm] = {
            "reference": reference_summary,
            "em_delta_percentage_points": summary["em_percent"] - reference_summary["em_percent"],
            "actor_token_reduction_percent": 100 * (
                1 - summary["mean_actor_tokens"] / reference_summary["mean_actor_tokens"]
            ),
            "gains": sum(row["em"] and not recorded[key]["em"] for key, row in rows.items()),
            "losses": sum(not row["em"] and recorded[key]["em"] for key, row in rows.items()),
        }
    quality = summary["em_percent"] >= max(
        comparison["reference"]["em_percent"] for comparison in comparisons.values()
    )
    cost = summary["mean_actor_tokens"] <= (
        .8 * comparisons["no_skill"]["reference"]["mean_actor_tokens"]
    )
    save(root / "TRAIN_RESULT.json", {
        "status": "COMPLETE_3395_NATIVE_TRAIN", "candidate": candidate, "summary": summary,
        "comparisons": comparisons, "provider_usage": provider.usage,
        "source_union_used_tasks": sum(
            row["source_receipt"]["overlap_characters_not_repeated"] > 0 for row in rows.values()
        ),
        "paid_recovery_tasks": sum(row["paid_recovery"] for row in rows.values()),
        "quality_preserved": quality, "minimum_20_percent_saving": cost,
        "training_gate_passed": quality and cost, "validation_calls": 0, "test_calls": 0,
        "official_optimizer_comparison_pending": True, "paper_admitted": False,
        "historical_optimizer_provider_tokens": config["inherited_optimizer_provider_tokens"],
    })
    state.update(phase="COMPLETE_3395_NATIVE_TRAIN", completed=True)


if __name__ == "__main__":
    native.run(Path(sys.argv[1]), prepare_study=prepare, run_study=study)
