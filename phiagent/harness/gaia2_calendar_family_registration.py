"""Pure registration/fence checks for one prospective three-candidate TRAIN family."""
from copy import deepcopy
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_cas(expected, observed):
    for key in ("pid", "start_ticks", "source_sha256", "jobs_sha256", "candidates_sha256", "selections"):
        require(observed[key] == expected[key], "Stale handoff field: " + key)
    require(observed["alive"] and observed["lock_owned"], "Incumbent does not own its live lock")
    require(observed["test_jobs"] == 0 and not observed["all_method_freeze"]
            and not observed["test_inputs"], "Test release advanced")
    require(set(observed["selections"]) <= {"FROZEN_native.json"},
            "A new method selection already committed")


def family_generation(jobs, candidates, candidate, candidate_path, prefix, decision_sha):
    require(candidate["candidate_id"] not in candidates.get("chisel", {}),
            "Candidate is already registered")
    old_jobs, old_candidates = deepcopy(jobs), deepcopy(candidates)
    members = {}
    for ident, path in candidates.get("chisel", {}).items():
        training = [name for name, job in jobs.items()
                    if job["method"] == "chisel" and job["stage"] == "train-candidate"
                    and job["candidate"] == path]
        require(len(training) == 4, "Each inherited member must retain four training shards")
        members[ident] = {"candidate_path": path, "training_jobs": sorted(training), "inherited": True}
    require(len(members) == 2, "Expected the explicitly authorized two-member predecessor family")
    merged = deepcopy(jobs)
    new_names = []
    for index in range(4):
        name = f"{prefix}-s{index}"
        require(name not in jobs, "Training identity collision")
        merged[name] = {
            "name": name, "method": "chisel", "stage": "train-candidate",
            "index": index, "count": 4, "candidate": candidate_path, "dependencies": [],
            "host": "h200-2",
        }
        new_names.append(name)
    registered = deepcopy(candidates)
    registered.setdefault("chisel", {})[candidate["candidate_id"]] = candidate_path
    members[candidate["candidate_id"]] = {
        "candidate_path": candidate_path, "training_jobs": new_names, "inherited": False,
    }
    closure = [name for member in members.values() for name in member["training_jobs"]]
    descriptor = {
        "members": members, "member_count": 3, "training_shards_per_member": 4,
        "training_executions_per_member": 96, "closure_dependencies": closure,
        "main_decision_sha256": decision_sha, "selection_and_test_fence": "REQUIRES_NEW_MAIN_DECISION",
        "new_validation_authorized": False, "new_test_authorized": False,
    }
    for name, job in old_jobs.items():
        require(merged[name] == job, "Inherited job/dependencies changed")
    for method, entries in old_candidates.items():
        require(all(registered[method][ident] == path for ident, path in entries.items()),
                "Inherited candidate changed")
    require(all(set(job["dependencies"]) <= merged.keys() for job in merged.values()),
            "Missing job dependency")
    return {"jobs": merged, "candidates": registered, "family": descriptor, "new_jobs": new_names}


def publication_allowed(family, stage):
    if stage in ("validation", "selection", "test"):
        return False
    return stage == "train-candidate" and family["member_count"] == len(family["members"])


def validate_training_completion(state, candidate_id, names):
    require(not state.get("failures"), "A fresh training shard failed")
    require(set(state["runs"]) == set(names), "Incomplete or extra training population")
    identities = []
    for name in names:
        row = state["runs"][name]
        result = row["RESULT.json"]
        require(row["VERIFIED.json"]["status"] == "PASS"
                and result["candidate_id"] == candidate_id and result["executions"] == 24
                and result["split"] == "train" and result["stage"] == "train-candidate",
                "Fresh native training audit is incompatible")
        require(row["RUNTIME_OBSERVED.json"]["status"] == "PASS", "Actual runtime was not admitted")
        identities.extend(row["PARTITION_ADMISSION.json"]["assigned_scenarios"])
    require(len(identities) == len(set(identities)) == 32, "Training scenarios are not exact once")
    return {"executions": 96, "successes": sum(state["runs"][name]["RESULT.json"]["successes"]
                                             for name in names)}
