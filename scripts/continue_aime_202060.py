"""Dependency-driven remote CHISEL training, score-only selection and frozen test."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.queue_aime_202060 import (  # noqa: E402
    PARENT, PARTITION, capacity, now, remote, source_ready, status, write,
)

PREFIX = "20260928-split202060-"
INITIAL = ("train0-v3", "train1-v2", "skillopt-v2", "autosaddler-v2")
LIBRARY = PARENT + "/" + PREFIX + "chisel-learn-v2"
FREEZE = PARENT + "/" + PREFIX + "freeze-v1"
STAGES = (
    ("no-skill-val-v1", "no_skill_validation", 3, 29430, (), []),
    ("chisel-learn-v2", "chisel_learn", 2, 29472, ("train0-v3", "train1-v2"), [
        "--baseline-root", PARENT + "/" + PREFIX + "train0-v3",
        "--baseline-root", PARENT + "/" + PREFIX + "train1-v2",
        "--recover-learning-from", PARENT + "/" + PREFIX + "chisel-learn-v1",
    ]),
    ("chisel-val0-v1", "chisel_validation", 3, 29630,
     ("chisel-learn-v2",), ["--shard", "0", "--library-root", LIBRARY]),
    ("chisel-val1-v1", "chisel_validation", 7, 29670,
     ("chisel-learn-v2",), ["--shard", "1", "--library-root", LIBRARY]),
)
FINAL = (("no_skill", 3, 29730), ("skillopt", 5, 29750),
         ("autosaddler", 6, 29760), ("chisel", 7, 29770))


def path(suffix, successors=None):
    return PARENT + "/" + PREFIX + (successors or {}).get(suffix, suffix)


def rewrite_dependencies(arguments, successors):
    substitutions = {path(old): path(new) for old, new in successors.items()}
    return [substitutions.get(value, value) for value in arguments]


def next_attempt(suffix):
    match = re.fullmatch(r"(.+)-v(\d+)", suffix)
    if not match:
        raise ValueError("A resource successor requires a versioned run name")
    return f"{match[1]}-v{int(match[2]) + 1}"


def gpu2_available():
    return remote("""
import subprocess,json
gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=index,uuid,memory.used',
 '--format=csv,noheader,nounits'],text=True).strip().split(',')
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
 '--format=csv,noheader,nounits'],text=True)
print(json.dumps({'free':int(gpu[2])<128 and gpu[1].strip() not in apps,'gpu_uuid':gpu[1].strip()}))
""")["free"]


def ready_stages(done, attempted, states):
    return [
        row for row in STAGES
        if row[0] not in attempted and row[0] not in done
        and not any(key in states.get(row[0], {}) for key in ("LAUNCH.json", "FAILURE.json"))
        and set(row[4]).issubset(done)
    ]


def execute(command, *, timeout=180):
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "h200-2", shlex.join(command)],
        capture_output=True, text=True, timeout=timeout,
    )
    return {
        "at_utc": now(), "command": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
    }


def launch_stage(bundle, sha256, suffix, mode, gpu, port, extra):
    return execute([
        "python3", bundle + "/scripts/prepare_aime_202060.py",
        "--name", PREFIX + suffix, "--mode", mode, "--gpu", str(gpu), "--port", str(port),
        "--manifest-sha256", sha256, "--partition-root", PARTITION + "/aime2025", "--wait-for-gpu", *extra,
    ])


def freeze(bundle, sha256, successors=None):
    command = [
        "python3", bundle + "/scripts/freeze_aime_202060.py",
        "--output", FREEZE, "--partition-root", PARTITION + "/aime2025",
        "--manifest-sha256", sha256,
        "--no-skill", path("no-skill-val-v1"), "--skillopt", path("skillopt-v2"),
        "--autosaddler", path("autosaddler-v2"), "--library-root", LIBRARY,
        "--validation-root", path("chisel-val0-v1"), "--validation-root", path("chisel-val1-v1"),
    ]
    return execute(rewrite_dependencies(command, successors or {}))


def summarize_final(successors=None):
    roots = {method: path("test-" + method + "-v1", successors) for method, _, _ in FINAL}
    return remote(f"""
import json,pathlib,hashlib
roots={roots!r};results={{}};total=0;freeze_hashes=set();manifests=set();populations=set()
for method,location in roots.items():
 root=pathlib.Path(location);proof=json.loads((root/'VERIFIED.json').read_text())
 result=json.loads((root/'RESULT.json').read_text())
 assert proof['status']=='PASS' and proof['result_sha256']==hashlib.sha256((root/'RESULT.json').read_bytes()).hexdigest()
 assert result['tasks']==18 and result['method']==method
 freeze_hashes.add(result['policy_freeze_sha256']);manifests.add(result['manifest_sha256'])
 populations.add(tuple(result['test_ids']));results[method]=result;total+=result['usage']['total_tokens']
assert len(freeze_hashes)==len(manifests)==len(populations)==1
f=pathlib.Path({FREEZE!r});policies=json.loads((f/'POLICY_FREEZE.json').read_text())
candidate=results['chisel']['correct'];others=[results[m]['correct'] for m in ('no_skill','skillopt','autosaddler')]
if 'paired_draft_ablation' in results['chisel']:others.append(results['chisel']['paired_draft_ablation']['correct'])
report={{'status':'FOUR_METHOD_TEST_COMPLETE_VERIFIED','population':'AIME2025 archived-public20/20/60 resplit',
 'counts':{{'train':6,'validation':6,'test':18}},'manifest_sha256':next(iter(manifests)),
 'methods':results,'execution_tokens':total,'development_usage':policies['development_usage'],
 'chisel_strictly_better_than_all_compared':candidate>max(others),'paper_admitted':candidate>max(others),
 'post_test_policy_updates':False,'old_results_reused':False}}
print(json.dumps(report))
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--relocator", help="Frozen remote zero-call-only GPU2 relocation script")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with (args.destination / "OWNER.json").open("x") as output:
        json.dump({"pid": os.getpid(), "at_utc": now(), "command": os.sys.argv}, output, indent=2)
    attempted = {}
    successor_file = args.destination / "SUCCESSORS.json"
    successors = json.loads(successor_file.read_text()) if successor_file.exists() else {}
    relocation_attempted = set()
    frozen = final_started = False
    last_capacity = 0
    units = list(INITIAL) + [row[0] for row in STAGES]
    units += ["test-" + method + "-v1" for method, _, _ in FINAL]
    while True:
        try:
            if time.monotonic() - last_capacity >= 300:
                last_capacity = time.monotonic()
                capacity(args.destination)
            released = source_ready()
            if not released["ready"]:
                write(args.destination / "PROGRESS.json", {
                    "at_utc": now(), "phase": "WAITING_PARENT_PARTITION_RELEASE", "new_model_calls": 0,
                })
                time.sleep(min(120, max(1, last_capacity + 300 - time.monotonic())))
                continue
            rows = status([path(unit, successors) for unit in units])
            states = dict(zip(units, rows))
            done = {key for key, row in states.items() if row.get("VERIFIED.json", {}).get("status") == "PASS"}
            blocked = {
                key: row["FAILURE.json"] for key, row in states.items() if "FAILURE.json" in row
            }
            write(args.destination / "LATEST.json", {
                "at_utc": now(), "runs": rows, "done": sorted(done), "blockers": blocked,
            })
            ready = ready_stages(done, attempted, states)
            if ready:
                capacity(args.destination)
                free_gpu2 = gpu2_available()
                if free_gpu2:
                    first = ready[0]
                    ready[0] = (first[0], first[1], 2, first[3], first[4], first[5])
                with ThreadPoolExecutor(max_workers=len(ready)) as pool:
                    receipts = list(pool.map(
                        lambda row: launch_stage(
                            args.bundle, released["sha256"], row[0], row[1], row[2], row[3],
                            rewrite_dependencies(row[5], successors),
                        ), ready,
                    ))
                for spec, receipt in zip(ready, receipts):
                    attempted[spec[0]] = receipt
                write(args.destination / "LAUNCHES.json", attempted)
                print(json.dumps({"at_utc": now(), "stage_launches": receipts}), flush=True)
            elif args.relocator and gpu2_available():
                priority = [
                    "train1-v2", "no-skill-val-v1", "chisel-learn-v2",
                    "chisel-val0-v1", "chisel-val1-v1",
                    "skillopt-v2", "autosaddler-v2",
                    *["test-" + method + "-v1" for method, _, _ in FINAL],
                ]
                waiting = [
                    unit for unit in priority
                    if states.get(unit, {}).get("waiting_unused") and unit not in relocation_attempted
                ]
                if waiting:
                    unit = waiting[0]
                    capacity(args.destination)
                    current = successors.get(unit, unit)
                    replacement = next_attempt(current)
                    receipt = execute([
                        "python3", args.relocator, "--old-root", path(current),
                        "--name", PREFIX + replacement, "--bundle", args.bundle, "--gpu", "2",
                    ])
                    relocation_attempted.add(unit)
                    write(args.destination / (unit + "-RESOURCE_HANDOFF.json"), receipt)
                    if receipt["returncode"]:
                        write(args.destination / "RELOCATION_BLOCKED.json", receipt)
                    else:
                        successors[unit] = replacement
                        write(successor_file, successors)
                        print(json.dumps({"at_utc": now(), "resource_successor": successors}), flush=True)
            final_dependencies = {
                "no-skill-val-v1", "skillopt-v2", "autosaddler-v2", "chisel-learn-v2",
                "chisel-val0-v1", "chisel-val1-v1",
            }
            if not frozen and final_dependencies.issubset(done):
                receipt = freeze(args.bundle, released["sha256"], successors)
                write(args.destination / "FREEZE_RECEIPT.json", receipt)
                if receipt["returncode"]:
                    write(args.destination / "BLOCKED.json", receipt)
                    break
                frozen = True
            if frozen and not final_started:
                capacity(args.destination)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    receipts = list(pool.map(
                        lambda spec: launch_stage(
                            args.bundle, released["sha256"], "test-" + spec[0] + "-v1",
                            "final", spec[1], spec[2],
                            ["--freeze-root", FREEZE, "--method", spec[0]],
                        ), FINAL,
                    ))
                write(args.destination / "FINAL_LAUNCHES.json", receipts)
                final_started = True
            if all("test-" + method + "-v1" in done for method, _, _ in FINAL):
                report = summarize_final(successors)
                write(args.destination / "COMPARISON.json", report)
                print(json.dumps(report), flush=True)
                break
            failed_preparations = {
                key: receipt for key, receipt in attempted.items() if receipt["returncode"]
            }
            running = any(row["alive"] for row in rows)
            if (blocked or failed_preparations) and not running and not ready:
                write(args.destination / "BLOCKED.json", {
                    "at_utc": now(), "execution_failures": blocked,
                    "preparation_failures": failed_preparations,
                    "reason": "No unchanged-root-cause retry; diagnose before a new frozen attempt",
                })
                break
            time.sleep(min(120, max(1, last_capacity + 300 - time.monotonic())))
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
            with (args.destination / "ERRORS.jsonl").open("a") as output:
                output.write(json.dumps({"at_utc": now(), "error": str(error)}) + "\n")
            time.sleep(120)


if __name__ == "__main__":
    main()
